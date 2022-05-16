#include "bnn.h"
#include "layer.h"
#include "dimension_def.h"
// #include "conv_weights.h"
// #include "weights_fracnet_64.h"
#include <math.h>

using namespace std;

static int8 msb_fmap_tile_buffer_0[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
static int8 msb_fmap_tile_buffer_1[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
static int8 lsb_fmap_tile_buffer[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];		// shortcut activation/error on-chip

static int8 out_buf_t0[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// conv output
static int8 out_buf_t1[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// bn output
static int1 relu_mask[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// relu mask for backprop

static int8 conv_3x3_weight_tile_buffer[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
static int8 conv_1x1_weight_tile_buffer[NUM_1x1_WT][CHANNEL_OUT_T][CHANNEL_IN_T];

static int8 pool_out_buf[64];												// AvgPool buffer
static int8 linear_weight_tile_buffer[10][64];
static int8 linear_out_buf[10];												// FC buffer

static int8 error[10];
static int8 labels[10];

static int8 gamma[NUM_ACT][CHANNEL_OUT_T];					// bn weight
static int8 beta[NUM_ACT][CHANNEL_OUT_T];					// bn bias

//--------------------
//  Top Function 
//--------------------
void FracNet_T(
	int8 image[3][32][32],
	int8 output[10]

	// int8 out_buf_t0[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH],
	// int8 out_buf_t1[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH]
)
{
#pragma HLS INTERFACE m_axi depth=3072 port=image offset=slave bundle=IMG				// 1*3*32*32 = 3072
#pragma HLS INTERFACE m_axi depth=10 port=output offset=slave bundle=RESULT				// 1*10
// #pragma HLS INTERFACE m_axi depth=871200 port=out_buf_t0 offset=slave bundle=out_buf_t0	// 50*1*16*33*33 = 871200
// #pragma HLS INTERFACE m_axi depth=871200 port=out_buf_t1 offset=slave bundle=out_buf_t1
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

// instance allocation
#pragma HLS ALLOCATION function instances=bn limit=1
#pragma HLS ALLOCATION function instances=bn_bp limit=1
#pragma HLS ALLOCATION function instances=bn_relu limit=1
#pragma HLS ALLOCATION function instances=bn_relu_bp limit=1
#pragma HLS ALLOCATION function instances=shortcut limit=1

#pragma HLS ALLOCATION function instances=avgpool limit=1
#pragma HLS ALLOCATION function instances=FC limit=1

#pragma HLS ALLOCATION function instances=conv_3x3_uni limit=1
#pragma HLS ALLOCATION function instances=conv_1x1_uni limit=1
#pragma HLS ALLOCATION function instances=conv_3x3_grad limit=1
#pragma HLS ALLOCATION function instances=conv_1x1_grad limit=1

	int H_fmap_in, H_fmap_out, in_channels, in_channels_after_pack, out_channels_after_pack;
    int out_channels, out_channel_start, stride, conv_3x3_weight_ptr, conv_1x1_weight_ptr, fc_weight_ptr, ini;
	int1 ctrl_conv, ctrl_sc, ctrl_fc, ctrl_avgpool;


	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////		Forward path		//////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	ini = 0;
	ctrl_sc = 0; // if ctrl_sc=0, generate and send out_copy into DDR
	ctrl_conv = 0;

	LOOP_GetImg:
	for (int row = 0; row < 32; row ++) {
		for (int col = 0; col < 32; col ++) {
			for (int c = 0; c < 3; c ++) {
#pragma HLS PIPELINE II=1	// 3080 cycles
				msb_fmap_tile_buffer_1[0][c][row][col] = image[c][row][col];
				out_buf_t1[ini][c][row][col] = image[c][row][col];
			}
		}
	}

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 3;
	out_channels = 16;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;

	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);
	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);

	conv_3x3_weight_ptr = 0;

	LOOP_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 1
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 16;
	out_channels = 16;
	stride = 1;

	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);
	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
	LOOP_layer1_0_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 2
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],
				H_fmap_in
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	LOOP_layer1_0_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 3
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc	// generate and send out_copy into DDR
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	LOOP_layer1_1_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 4
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out],
				H_fmap_in
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
	LOOP_layer1_1_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 5
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_2 PG1 //////////////////////
	LOOP_layer1_2_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 6
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],
				H_fmap_in
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_2 PG2 //////////////////////
	LOOP_layer1_2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 7
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 16;
	in_channels = 16;
	out_channels = 32;
	stride = 2;

	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);
	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);

	conv_1x1_weight_ptr = 0;

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
	LOOP_layer2_0_ConvSC:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 8
		conv_1x1_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_1x1_uni(
				msb_fmap_tile_buffer_0[c_out], conv_1x1_weight_tile_buffer[conv_1x1_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],	// conv+bn shortcut
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 9
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 32;
	out_channels = 32;
	stride = 1;

	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);
	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
	LOOP_layer2_0_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 10
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	LOOP_layer2_1_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 11
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],
				H_fmap_in
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	LOOP_layer2_1_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 12
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_2 PG1 //////////////////////
	LOOP_layer2_2_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 13
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out],
				H_fmap_in
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_2 PG2 //////////////////////
	LOOP_layer2_2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 14
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 8;
	in_channels = 32;
	out_channels = 64;
	stride = 2;

	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);
	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
	LOOP_layer3_0_ConvSC:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 15
		conv_1x1_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_1x1_uni(
				msb_fmap_tile_buffer_1[c_out], conv_1x1_weight_tile_buffer[conv_1x1_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn(
				msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out],	// conv+bn shortcut
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 16
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 64;
	out_channels = 64;
	stride = 1;

	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);
	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
	LOOP_layer3_0_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 17
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	LOOP_layer3_1_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 18
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out],
				H_fmap_in
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
	LOOP_layer3_1_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 19
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_2 PG1 //////////////////////
	LOOP_layer3_2_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 20
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],
				H_fmap_in
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_2 PG2 //////////////////////
	LOOP_layer3_2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 21
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
				gamma[ini], beta[ini],
				H_fmap_out
			);
			shortcut(
				msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini],
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// AvgPool and FC // /////////////////

	// avgpool
	ctrl_avgpool = 0;
	LOOP_AvgPool:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		avgpool(
			msb_fmap_tile_buffer_0[c_out], pool_out_buf, ctrl_avgpool, c_out
		);
	}

	// FC
	ctrl_fc = 0;
	LOOP_FC:
	FC(
		pool_out_buf, linear_weight_tile_buffer, linear_out_buf, ctrl_fc
	);

	write_output:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		output[j] = linear_out_buf[j];
	}

	////////////////////////////////////////////////
	//////////// CrossEntropy loss /////////////////

	// error_calc:
	// MSE for simplicity
	// for (int j = 0; j < 10; j ++) {
// #pragma HLS PIPELINE II=1
	//	error[j] = 2 * (linear_out_buf[j] - labels[j]);
	// }

	// error = (e^xi - yi*(sum{e^xi})) / sum{e^xi}
	// CrossEntropy Loss
	int8 sum = 0;
	error_calc_1:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		sum += 1 + linear_out_buf[j] + linear_out_buf[j]*linear_out_buf[j]/2;
	}
	error_calc_2:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		error[j] = (1 + linear_out_buf[j] + linear_out_buf[j]*linear_out_buf[j]/2 - labels[j]*sum)/sum;
	}

//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// AvgPool and FC // /////////////////

	// FC_bp
	ctrl_fc = 1;
	LOOP_FC_bp:
	FC(
		pool_out_buf, linear_weight_tile_buffer, error, ctrl_fc
	);

	// avgpool_bp
	ctrl_avgpool = 1;
	LOOP_AvgPool_bp:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		avgpool(
			msb_fmap_tile_buffer_1[c_out], pool_out_buf, ctrl_avgpool, c_out
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 64;
	out_channels = 64;
	stride = 1;

	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);
	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);

	ctrl_sc = 1;
	ctrl_conv = 1;

	////////////////////////////////////////////////
	//////////// layer3_2 PG2 //////////////////////
	LOOP_layer3_2_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 21
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in],
				H_fmap_in
			);
			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_2 PG1 //////////////////////
	LOOP_layer3_2_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 20
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
	LOOP_layer3_1_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 19
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],
				H_fmap_in
			);
			bn_relu_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	LOOP_layer3_1_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 18
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
	LOOP_layer3_0_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 17
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], lsb_fmap_tile_buffer[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 16;
	in_channels = 64;
	out_channels = 32;
	stride = 2;

	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);
	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 16
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				lsb_fmap_tile_buffer[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], lsb_fmap_tile_buffer[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - 2*in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
	LOOP_layer3_0_ConvSC_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 15
		conv_1x1_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], msb_fmap_tile_buffer_0[c_in],	// conv+bn shortcut
				gamma[ini],
				H_fmap_out
			);
			conv_1x1_uni(	// note the index of shortcut input
				msb_fmap_tile_buffer_0[c_in], conv_1x1_weight_tile_buffer[conv_1x1_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_1x1_weight_grad_cal
			conv_1x1_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_1x1_weight_tile_buffer[conv_1x1_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 32;
	out_channels = 32;
	stride = 1;

	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);
	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);

	////////////////////////////////////////////////
	//////////// layer2_2 PG2 //////////////////////
	LOOP_layer2_2_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 14
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],
				H_fmap_in
			);
			bn_relu_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_2 PG1 //////////////////////
	LOOP_layer2_2_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 13
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	LOOP_layer2_1_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 12
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in],
				H_fmap_in
			);
			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	LOOP_layer2_1_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 11
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
	LOOP_layer2_0_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 10
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], lsb_fmap_tile_buffer[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 32;
	in_channels = 32;
	out_channels = 16;
	stride = 2;

	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);
	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 9
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				lsb_fmap_tile_buffer[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], lsb_fmap_tile_buffer[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini-1 - 2*in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
	LOOP_layer2_0_ConvSC_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 8
		conv_1x1_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], msb_fmap_tile_buffer_1[c_in],	// conv+bn shortcut
				gamma[ini],
				H_fmap_out
			);
			conv_1x1_uni(	// note the index of shortcut input
				msb_fmap_tile_buffer_1[c_in], conv_1x1_weight_tile_buffer[conv_1x1_weight_ptr], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_1x1_weight_grad_cal
			conv_1x1_grad(
				out_buf_t1[ini -in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_1x1_weight_tile_buffer[conv_1x1_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 16;
	out_channels = 16;
	stride = 1;

	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);
	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);

	////////////////////////////////////////////////
	//////////// layer1_2 PG2 //////////////////////
	LOOP_layer1_2_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 7
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in],
				H_fmap_in
			);
			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_2 PG1 //////////////////////
	LOOP_layer1_2_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 6
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
	LOOP_layer1_1_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 5
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],
				H_fmap_in
			);
			bn_relu_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	LOOP_layer1_1_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 4
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	LOOP_layer1_0_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 3
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
			identity_shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in],
				H_fmap_in
			);
			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
	LOOP_layer1_0_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 2
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
			shortcut(
				msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],	// DDR not used
				H_fmap_out, ctrl_sc
			);
		}
	}

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 16;
	out_channels = 3;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;

	out_channels_after_pack = max(1, out_channels/CHANNEL_OUT_T);
	in_channels_after_pack = max(1, in_channels/CHANNEL_IN_T);

	LOOP_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >=0; c_in --) {
		ini -= 1;	// ini = 1
		conv_3x3_weight_ptr -= 1;
		for (int c_out = out_channels_after_pack - 1; c_out >=0; c_out --) {

			bn_relu_bp(
				msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
				gamma[ini],
				H_fmap_out
			);
			conv_3x3_uni(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_out, ctrl_conv
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			conv_3x3_grad(
				out_buf_t1[ini - in_channels/CHANNEL_IN_T + 1], msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}
}	// end FracBNN_T
