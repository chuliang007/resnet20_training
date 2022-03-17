#include "bnn.h"
#include "layer.h"
#include "dimension_def.h"

using namespace std;

static  int8 msb_fmap_tile_buffer_0[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
static  int8 msb_fmap_tile_buffer_1[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
static  int8 lsb_fmap_tile_buffer[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];		// shortcut activation/error on-chip
static	int8 conv_3x3_weight_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
static	int8 conv_1x1_weight_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T];

static	int8 grad_buf_t0[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];					// weight_3x3 gradient on-chip
static	int8 grad_buf_t1[CHANNEL_OUT_T][CHANNEL_IN_T];							// weight_1x1 gradient on-chip

static  int8 pool_out_buf[BATCH_SIZE][CHANNEL_OUT_T];							// AvgPool buffer
static  int8 linear_weight_tile_buffer[10][CHANNEL_OUT_T];
static  int8 linear_out_buf[BATCH_SIZE][10];									// FC buffer

static	int8 gamma[CHANNEL_OUT_T];
static	int8 beta[CHANNEL_OUT_T];
static	int8 grad_gamma[CHANNEL_OUT_T];
static	int8 grad_beta[CHANNEL_OUT_T];

//--------------------
//  Top Function 
//--------------------
void FracNet_T(
	int8 image[BATCH_SIZE][3][32][32],
	int8 output[BATCH_SIZE][10],

	int8 conv_3x3_weight_all[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 conv_1x1_weight_all[NUM_1x1_WT][CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 linear_weight[8][10][CHANNEL_OUT_T],

	int8 out_buf_t0[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// output activation
    int8 out_buf_t1[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	int8 out_buf_sc[NUM_SC][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// shortcut output activation

    int1 relu_mask[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH]		// relu mask for backprop
)
{
#pragma HLS INTERFACE m_axi depth=12288 port=image offset=slave bundle=IMG	// 4*3*32*32 = 12288
#pragma HLS INTERFACE m_axi depth=40 port=output offset=slave bundle=RESULT	// 4*10

#pragma HLS INTERFACE m_axi depth=2755584 port=conv_3x3_weight_all offset=slave bundle=conv_3x3_weight_all	// 1196*4*64*3*3 = 2755584
#pragma HLS INTERFACE m_axi depth=43008 port=conv_1x1_weight_all offset=slave bundle=conv_1x1_weight_all	// 168*4*64 = 43008
#pragma HLS INTERFACE m_axi depth=5120 port=linear_weight offset=slave bundle=linear_weight					// 8*10*64 = 5120

#pragma HLS INTERFACE m_axi depth=333425664 port=out_buf_t0 offset=slave bundle=out_buf_t0					// 1196*4*64*33*33 = 333425664
#pragma HLS INTERFACE m_axi depth=333425664 port=out_buf_t1 offset=slave bundle=out_buf_t1
#pragma HLS INTERFACE m_axi depth=46835712 port=out_buf_sc offset=slave bundle=out_buf_sc					// 168*4*64*33*33 = 46835712
/* #pragma HLS INTERFACE m_axi depth=46835712 port=relu_mask offset=slave bundle=relu_mask */

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

// instance allocation
#pragma HLS ALLOCATION function instances=bn_relu limit=1
#pragma HLS ALLOCATION function instances=bn_relu_bp limit=1
#pragma HLS ALLOCATION function instances=shortcut limit=1

#pragma HLS ALLOCATION function instances=avgpool limit=1
#pragma HLS ALLOCATION function instances=avgpool_bp limit=1
#pragma HLS ALLOCATION function instances=FC limit=1
#pragma HLS ALLOCATION function instances=FC_bp limit=1

#pragma HLS ALLOCATION function instances=conv_3x3 limit=1
#pragma HLS ALLOCATION function instances=conv_1x1 limit=1

#pragma HLS ALLOCATION function instances=conv_3x3_rot_bp limit=1
#pragma HLS ALLOCATION function instances=conv_1x1_rot_bp limit=1
#pragma HLS ALLOCATION function instances=conv_3x3_grad limit=1
#pragma HLS ALLOCATION function instances=conv_1x1_grad limit=1

#pragma HLS ALLOCATION function instances=SGD_WU_3x3 limit=1
#pragma HLS ALLOCATION function instances=SGD_WU_1x1 limit=1
/*
// array partition
#pragma HLS ARRAY_PARTITION variable=msb_fmap_tile_buffer_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=msb_fmap_tile_buffer_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=msb_fmap_tile_buffer_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=msb_fmap_tile_buffer_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=lsb_fmap_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=lsb_fmap_tile_buffer complete dim=2

#pragma HLS ARRAY_PARTITION variable=conv_3x3_weight_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_3x3_weight_tile_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=conv_1x1_weight_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_1x1_weight_tile_buffer complete dim=2

#pragma HLS ARRAY_PARTITION variable=pool_out_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=linear_out_buf complete dim=2

#pragma HLS ARRAY_PARTITION variable=relu_mask complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu_mask complete dim=2

#pragma HLS ARRAY_PARTITION variable=grad_buf_t0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_buf_t0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=grad_buf_t1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_buf_t1 complete dim=2

#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_beta complete dim=1
*/
	// Initialize the buffers to 0
	fmap_buf_init:
	for (int c = 0; c < CHANNEL_IN_T; c ++) {
		for (int b = 0; b < BATCH_SIZE; b ++) {
			pool_out_buf[b][c] = 0;
#pragma HLS pipeline
			for (int ii=0; ii < 10; ii++) {
				linear_out_buf[b][ii] = 0;
				linear_weight_tile_buffer[ii][c] = 0;
			}
			for (int i = 0; i < WIDTH; i ++) {
				for (int j = 0; j < WIDTH; j ++) {
					msb_fmap_tile_buffer_0[b][c][i][j] = 0;
					msb_fmap_tile_buffer_1[b][c][i][j] = 0;
					lsb_fmap_tile_buffer[b][c][i][j] = 0;
				}
			}
		}
	}
	
	grad_buf_init:
	for (int c_out = 0; c_out < CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < CHANNEL_IN_T; c_in ++) {
			grad_buf_t1[c_out][c_in] = 0;
#pragma HLS pipeline
			for (int row = 0; row < 3; row ++){
				for (int col = 0; col < 3; col ++){
					grad_buf_t0[c_out][c_in][row][col] = 0;
				}
			}
		}
	}

	int H_fmap_in, H_fmap_out, in_channels, in_channels_after_pack; 
    int out_channels, out_channel_start, stride, conv_3x3_weight_ptr, conv_1x1_weight_ptr, ini, ini_sc;
	int1 ctrl_sc;


//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////		Forward path		//////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	LOOP_GetImg:
	for (int row = 0; row < 32; row ++) {
		for (int col = 0; col < 32; col ++) {
#pragma HLS pipeline
			for (int b = 0; b < BATCH_SIZE; b ++) {
				for (int c = 0; c < 3; c ++) {
					msb_fmap_tile_buffer_1[b][c][row][col] = image[b][c][row][col];
				}
			}
		}
	}
	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 64;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;
	conv_3x3_weight_ptr = 0;

	ini = 0;
	ini_sc = 0;
	ctrl_sc = 1; // if ctrl_sc=1, generate and send out_copy into DDR

    LOOP_Conv1:	 // 4 outermost for-loops
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// ini = 0

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 64;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
	LOOP_layer1_0_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 1
				conv_3x3_weight_ptr += 1;

				// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer,
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	LOOP_layer1_0_Conv2:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 2
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t1[ini],
					H_fmap_out, ctrl_sc	// generate and send out_copy into DDR
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	LOOP_layer1_1_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 3
				conv_3x3_weight_ptr += 1;

				// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1, msb_fmap_tile_buffer_0, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
	LOOP_layer1_1_Conv2:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 4
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1, msb_fmap_tile_buffer_0, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 2 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 16;
	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 128;
	stride = 2;
	conv_1x1_weight_ptr = 0;

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
	LOOP_layer2_0_ConvSC:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 5
				// ini_sc = 0;

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				conv_1x1(
					msb_fmap_tile_buffer_1, conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_sc[ini_sc],
					stride, H_fmap_in, H_fmap_out
				);
				bn(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,	// conv+bn shortcut
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {
				// ini = 5
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 128;
	in_channels_after_pack = 1;
	out_channels = 128;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
	LOOP_layer2_0_Conv2:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 6
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	LOOP_layer2_1_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 7
				conv_3x3_weight_ptr += 1;

				// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1, msb_fmap_tile_buffer_0, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	LOOP_layer2_1_Conv2:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 8
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1, msb_fmap_tile_buffer_0, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);
			}
		}
    }
	////////////////////////////////////////////////
	//////////// LAYER 3 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 8;
	in_channels = 128;
	in_channels_after_pack = 1;
	out_channels = 256;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
	LOOP_layer3_0_ConvSC:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 9
				ini_sc += 1; // ini_sc = 1
				conv_1x1_weight_ptr += 1;

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				conv_1x1(
					msb_fmap_tile_buffer_1, conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_sc[ini_sc],
					stride, H_fmap_in, H_fmap_out
				);
				bn(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer,	// conv+bn shortcut
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// ini = 9
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 256;
	in_channels_after_pack = 1;
	out_channels = 256;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
	LOOP_layer3_0_Conv2:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 10
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	LOOP_layer3_1_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 11
				conv_3x3_weight_ptr += 1;

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1, msb_fmap_tile_buffer_0, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
	LOOP_layer3_1_Conv2:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 12
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1, msb_fmap_tile_buffer_0, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 4 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 4;
	in_channels = 256;
	in_channels_after_pack = 1;
	out_channels = 512;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer4_0 shortcut (conv+bn) ///////
	LOOP_layer4_0_ConvSC:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 13
				ini_sc += 1; // ini_sc = 2
				conv_1x1_weight_ptr += 1;

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				conv_1x1(
					msb_fmap_tile_buffer_1, conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_sc[ini_sc],
					stride, H_fmap_in, H_fmap_out
				);
				bn(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,	// conv+bn shortcut
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_0 PG1 //////////////////////
	LOOP_layer4_0_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// ini = 13
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 4 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 4;
	in_channels = 512;
	in_channels_after_pack = 1;
	out_channels = 512;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer4_0 PG2 //////////////////////
	LOOP_layer4_0_Conv2:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 14
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1,  msb_fmap_tile_buffer_0, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_1 PG1 //////////////////////
	LOOP_layer4_1_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 15
				conv_3x3_weight_ptr += 1;
				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer,
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
	LOOP_layer4_1_Conv2:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 16
				conv_3x3_weight_ptr += 1;

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],
					stride, H_fmap_in, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

    // Initialize the buffers for pooling and FC layer
    /*
	int8 pool_out_buf[BATCH_SIZE][CHANNEL_OUT_T];
	int8 linear_out_buf[BATCH_SIZE][10];
	int8 linear_weight[8][10][CHANNEL_OUT_T];	// FC weight: out_channels/CHANNEL_OUT_T
	*/
	out_buf_init:
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
			pool_out_buf[b][c] = 0;
		}
		for (int i = 0; i < 10; i ++) {
			linear_out_buf[b][i] = 0;
		}
	}

	// avgpool
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		avgpool(msb_fmap_tile_buffer_0, pool_out_buf);
	}

	// FC
	int fc_weight_ptr = 0;
	load_fc_weights(
		linear_weight_tile_buffer, linear_weight[fc_weight_ptr]
	);
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		fc_weight_ptr += 1;
		FC(pool_out_buf, linear_weight[fc_weight_ptr], linear_out_buf);
	}

	write_output:
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int i = 0; i < 10; i ++) {
			output[b][i] = linear_out_buf[b][i];
		}
	}

	////////////////////////////////////////////////
	//////////// CrossEntropy loss /////////////////
	int8 error[BATCH_SIZE][10];
	int8 labels[BATCH_SIZE][10];

	error_calc:
	// MSE for simplicity
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int i = 0; i < 10; i ++) {
			error[b][i] = 2 * (linear_out_buf[b][i] - labels[b][i]);
		}
	}


//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////


	// int8 linear_weight_transpose[CHANNEL_OUT_T][10];	// FC weight transposed

	// FC_bp
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		FC_bp(error, linear_weight[fc_weight_ptr], pool_out_buf);
	}

	// avgpool_bp
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		avgpool_bp(pool_out_buf, msb_fmap_tile_buffer_0);
	}

	printf("ini: %d", ini);
	printf("conv_3x3_weight_ptr: %d", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 4 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 4;
	in_channels = 512;
	in_channels_after_pack = 1;
	out_channels = 512;
	stride = 1;

	ctrl_sc = 0; // do not generate output_copy of shortcut

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
	LOOP_layer4_1_Conv2_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,
					H_fmap_in
				);

				// ini = 16

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				//////////////////////////
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_1 PG1 //////////////////////
	LOOP_layer4_1_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 15
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				///////////////////////////
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_0 PG2 //////////////////////
	LOOP_layer4_0_Conv2_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1; // ini = 14
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_1, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_0, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 4 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 8;
	in_channels = 512;
	in_channels_after_pack = 1;
	out_channels = 256;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer4_0 shortcut (conv+bn) ///////
	LOOP_layer4_0_ConvSC_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// ini = 14
				// ini_sc = 2
				// conv_1x1_weight_ptr -= 1;

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				bn_bp(
					msb_fmap_tile_buffer_1, out_buf_sc[ini_sc], msb_fmap_tile_buffer_0,	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap_tile_buffer_0, conv_1x1_weight_tile_buffer, lsb_fmap_tile_buffer,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_1x1_weight_grad_cal
				conv_1x1_grad(
					out_buf_t1[ini - 2], msb_fmap_tile_buffer_0, grad_buf_t1,
					stride, H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_0 PG1 //////////////////////
	LOOP_layer4_0_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 13
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_1, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_0, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 256;
	in_channels_after_pack = 1;
	out_channels = 256;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
	LOOP_layer3_1_Conv2_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,
					H_fmap_in
				);

				ini -= 1;	// ini = 12
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				//////////////////////////
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	LOOP_layer3_1_Conv1_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 11
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
	LOOP_layer3_0_Conv2_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1; // ini = 10
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_1, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_0, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 16;
	in_channels = 256;
	in_channels_after_pack = 1;
	out_channels = 128;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
	LOOP_layer3_0_ConvSC_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// ini = 10
				ini_sc -= 1;// ini_sc = 1
				conv_1x1_weight_ptr -= 1;

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				bn_bp(
					msb_fmap_tile_buffer_1, out_buf_sc[ini_sc], msb_fmap_tile_buffer_0,	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap_tile_buffer_0, conv_1x1_weight_tile_buffer, lsb_fmap_tile_buffer,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_1x1_weight_grad_cal
				conv_1x1_grad(
					out_buf_t1[ini - 2], msb_fmap_tile_buffer_0, grad_buf_t1,
					stride, H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 9
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_1, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_0, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 128;
	in_channels_after_pack = 1;
	out_channels = 128;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	LOOP_layer2_1_Conv2_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,
					H_fmap_in
				);

				ini -= 1;	// ini = 8
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	LOOP_layer2_1_Conv1_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 7
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
	LOOP_layer2_0_Conv2_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1; // ini = 6
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_1, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_0, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 32;
	in_channels = 128;
	in_channels_after_pack = 1;
	out_channels = 64;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
	LOOP_layer2_0_ConvSC_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// ini = 6
				ini_sc -= 1;// ini_sc = 0
				conv_1x1_weight_ptr -= 1;

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				bn_bp(
					msb_fmap_tile_buffer_1, out_buf_sc[ini_sc], msb_fmap_tile_buffer_0,	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap_tile_buffer_0, conv_1x1_weight_tile_buffer, lsb_fmap_tile_buffer,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_1x1_weight_grad_cal
				conv_1x1_grad(
					out_buf_t1[ini - 2], msb_fmap_tile_buffer_0, grad_buf_t1,
					stride, H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 5
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_1, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_0, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 64;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
	LOOP_layer1_1_Conv2_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer,
					H_fmap_in
				);

				ini -= 1; // ini = 4
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	LOOP_layer1_1_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 3
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_0, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_1, out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	LOOP_layer1_0_Conv2_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				ini -= 1;	// ini = 2
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_1, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_0, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
	LOOP_layer1_0_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 1
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_1, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - 1], msb_fmap_tile_buffer_0, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_1, lsb_fmap_tile_buffer, msb_fmap_tile_buffer_0, out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 64;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;

    LOOP_Conv1_bp:
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 0
				conv_3x3_weight_ptr -= 1;

				bn_relu_bp(
					msb_fmap_tile_buffer_0, out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1,
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1, conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0,
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
/*
				// get image first
					for (int c = 0; c < 3; c ++) {
						for (int row = 0; row < 32; row ++) {
							for (int col = 0; col < 32; col ++) {
								msb_fmap_tile_buffer_0[b][c][row][col] = image[b][c][row][col];
							}
						}
					}
				}
*/
				conv_3x3_grad(
					msb_fmap_tile_buffer_0, msb_fmap_tile_buffer_1, grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
			}
		}
    }
}	// end FracBNN_T
