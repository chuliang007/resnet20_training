#include "bnn.h"
#include "layer.h"
#include "dimension_def.h"
#include "conv_weights.h"
// #include "weights_fracnet_64.h"
#include <math.h>

using namespace std;

static float msb_fmap_tile_buffer_0[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];
static float msb_fmap_tile_buffer_1[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];
static float msb_fmap_tile_buffer_2[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];
static float msb_fmap_tile_buffer_dataflow[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];

static float out_buf_t0[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];
static float out_buf_t1[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];
static uint1 relu_mask[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];

//static float conv_3x3_weight_tile_buffer[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
//static float conv_1x1_weight_tile_buffer[NUM_1x1_WT][CHANNEL_OUT_T][CHANNEL_IN_T];
static float conv_3x3_weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];

static float pool_out_buf[64];
static float pool_out_buf_copy[64];
//static float linear_weight_tile_buffer[10][64];
//static float linear_bias[10];
static float linear_out_buf[10];											// FC buffer

static float sum;

//--------------------
//  Top Function 
//--------------------
void FracNet_T(
	float image[3][32][32],
	float ctrl_tl,
	float &loss,
	float error[10]
)
{
#pragma HLS INTERFACE m_axi depth=3072 port=image offset=slave bundle=IMG		// 1*3*32*32 = 3072
#pragma HLS INTERFACE m_axi depth=1 port=loss offset=slave bundle=LOSS			// 1
#pragma HLS INTERFACE m_axi depth=10 port=error offset=slave bundle=ERROR		// 1*10
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

// instance allocation
#pragma HLS ALLOCATION function instances=bn_relu limit=1
#pragma HLS ALLOCATION function instances=bn_relu_bp limit=1

#pragma HLS ALLOCATION function instances=avgpool limit=1
#pragma HLS ALLOCATION function instances=FC limit=1

#pragma HLS ALLOCATION function instances=conv_3x3 limit=1
#pragma HLS ALLOCATION function instances=conv_3x3_grad_v2 limit=1

	int H_fmap_in, H_fmap_out, in_channels, in_channels_after_pack, out_channels_after_pack;
    int out_channels, out_channel_start, stride, conv_3x3_weight_ptr, conv_1x1_weight_ptr, fc_weight_ptr, ini, ini_act_bias, ini_wt_bias;
	uint1 ctrl_conv, ctrl_frz, ctrl_fc, ctrl_avgpool;

	/* block minifloat tile size 32x32
	 *
	 * 32 x 32 * 32, at most 32 shared bias
	 * 64 x 16 * 16
	 * act_bias_shift update by channel tile dimension
	 * 
	 * wt_bias tiled by (channel_in_t * channel_out_t * kernel_height * kernel_width)
	 */

	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////		Forward path		//////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	ini = 0;
	ini_act_bias = 0;

	ctrl_conv = 0;
	ctrl_frz = 1;		// 0 for frozen, 1 for update

	LOOP_GetImg:
	for (int row = 0; row < 32; row ++) {
		for (int col = 0; col < 32; col ++) {
			for (int c = 0; c < 3; c ++) {
#pragma HLS PIPELINE II=1	// 3080 cycles
				msb_fmap_tile_buffer_2[0][c][row][col] = image[c][row][col];
				out_buf_t1[ini][c][row][col] = image[c][row][col];
			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	in_channels = 3;
	out_channels = 32;
	H_fmap_in = 32;
	H_fmap_out = 32;
	stride = 1;

	in_channels_after_pack = 1;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;

	conv_3x3_weight_ptr = 0;

	////////////////////////////////////////////////
	//////////// layer1 PG1 ////////////////////////

	LOOP_layer1_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 1
		// conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
			conv_3x3_weight_ptr += 1;

			conv_3x3(
				msb_fmap_tile_buffer_2[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
		}
		bn_relu(
			msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	in_channels = 32;
	out_channels = 32;
	H_fmap_in = 32;
	H_fmap_out = 32;
	stride = 1;

	in_channels_after_pack = in_channels/CHANNEL_IN_T;;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer1 PG2 ////////////////////////

	LOOP_layer1_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 2
		// conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
			conv_3x3_weight_ptr += 1;

			conv_3x3(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
		}
		bn_relu(
			msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_2[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_2[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 Downsample ////////////////
	////////////////////////////////////////////////

	in_channels = 32;
	out_channels = 64;
	H_fmap_in = 32;
	H_fmap_out = 16;
	stride = 2;

	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer1 shortcut ///////////////////

	LOOP_layer1_ConvSC:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 3
		// conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
			conv_3x3_weight_ptr += 1;

			conv_3x3(
				msb_fmap_tile_buffer_2[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
		}
		bn_relu(
			msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_0[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	in_channels = 64;
	out_channels = 64;
	H_fmap_in = 16;
	H_fmap_out = 16;
	stride = 1;

	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer2 PG1 ////////////////////////

	LOOP_layer2_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 4
		// conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
			conv_3x3_weight_ptr += 1;

			conv_3x3(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_2[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
		}
		bn_relu(
			msb_fmap_tile_buffer_2[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	////////////////////////////////////////////////
	//////////// layer2 PG2 ////////////////////////

	LOOP_layer2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 5
		// conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
			conv_3x3_weight_ptr += 1;

			conv_3x3(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
		}
		bn_relu(	// wrong output (beyond H_fmap_out)
			msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_2[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_2[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Downsample ////////////////
	////////////////////////////////////////////////

	in_channels = 64;
	out_channels = 64;
	H_fmap_in = 16;
	H_fmap_out = 8;
	stride = 2;

	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer2 shortcut ///////////////////

	LOOP_layer2_ConvSC:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 6
		// conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
			conv_3x3_weight_ptr += 1;

			conv_3x3(
				msb_fmap_tile_buffer_2[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv
			);
		}
		bn_relu(
			msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_0[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	////////////////////////////////////////////////
	//////////// AvgPool and FC ////////////////////

	// avgpool
	ctrl_avgpool = 0;
	LOOP_AvgPool:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		avgpool(
			msb_fmap_tile_buffer_0[c_out], pool_out_buf, ctrl_avgpool, c_out, pool_out_buf_copy
		);
	}

	// FC
	ctrl_fc = 0;
	LOOP_FC:
	if (ctrl_tl != 0) {
		FC(
			pool_out_buf, pool_out_buf_copy, linear_weight_tile_buffer, linear_bias, linear_out_buf, ctrl_fc
		);
	} else {
		FC(
			pool_out_buf, pool_out_buf_copy, linear_weight_tile_buffer_tl, linear_bias_tl, linear_out_buf, ctrl_fc
		);
	}

	////////////////////////////////////////////////
	/////////////////   FC   ///////////////////////

	sum = 0;
	softmax_sum_approx:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		sum += 1 + linear_out_buf[j] + 0.5*(linear_out_buf[j])*(linear_out_buf[j]) + 0.1667*(linear_out_buf[j])*(linear_out_buf[j])*(linear_out_buf[j]);
	}
	// printf("sum: %f \n", sum);

	loss = 0;
	CE_loss:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		if (ctrl_tl != 0) {
			loss += -labels[j] * hls::log2((1 + linear_out_buf[j] + 0.5*(linear_out_buf[j])*(linear_out_buf[j]) + 0.1667*(linear_out_buf[j])*(linear_out_buf[j])*(linear_out_buf[j]))/sum);
		} else {
			loss += -labels_tl[j] * hls::log2((1 + linear_out_buf[j] + 0.5*(linear_out_buf[j])*(linear_out_buf[j]) + 0.1667*(linear_out_buf[j])*(linear_out_buf[j])*(linear_out_buf[j]))/sum);
		}
	}

	CE_error:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		if (ctrl_tl != 0) {
			error[j] = (1 + linear_out_buf[j] + 0.5*(linear_out_buf[j])*(linear_out_buf[j]) + 0.1667*(linear_out_buf[j])*(linear_out_buf[j])*(linear_out_buf[j]))/sum - labels[j];
		} else {
			error[j] = (1 + linear_out_buf[j] + 0.5*(linear_out_buf[j])*(linear_out_buf[j]) + 0.1667*(linear_out_buf[j])*(linear_out_buf[j])*(linear_out_buf[j]))/sum - labels_tl[j];
		}
	}


//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// AvgPool and FC ////////////////////

	// FC_bp
	ctrl_fc = 1;
	LOOP_FC_bp:
	if (ctrl_tl != 0) {
		FC(
			pool_out_buf, pool_out_buf_copy, linear_weight_tile_buffer, linear_bias, error, ctrl_fc
		);
	} else {
		FC(
			pool_out_buf, pool_out_buf_copy, linear_weight_tile_buffer_tl, linear_bias_tl, error, ctrl_fc
		);
	}

	// avgpool_bp
	ctrl_avgpool = 1;
	LOOP_AvgPool_bp:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		avgpool(
			msb_fmap_tile_buffer_1[c_out], pool_out_buf, ctrl_avgpool, c_out, pool_out_buf_copy
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Upsample //////////////////
	////////////////////////////////////////////////

	in_channels = 64;
	out_channels = 64;
	H_fmap_in = 8;
	H_fmap_out = 16;
	stride = 2;

	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;

	ctrl_conv = 1;

	////////////////////////////////////////////////
	//////////// layer2 shortcut ///////////////////

	LOOP_layer2_ConvSC_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >= 0; c_in --) {
		ini -= 1;	// ini = 6
		// conv_3x3_weight_ptr -= 1;

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		bn_relu_bp(
			msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in],
			bn_wt[ini], bn_bias[ini], vel_bn_wt[ini], vel_bn_bias[ini], ctrl_frz,
			H_fmap_in
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr -= 1;

			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			conv_3x3_backward(
				msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_2[c_out], out_buf_t0[0],
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr],
				stride, H_fmap_in, H_fmap_out, c_in
			);
			quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
			quant_act(msb_fmap_tile_buffer_2[c_out], H_fmap_out);
//			conv_3x3(
//				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_2[c_out], out_buf_t0[ini],
//				stride, H_fmap_out, H_fmap_in, c_in, ctrl_conv
//			);
//			///////////////////////////
//			// conv_3x3_weight_grad_cal
//			conv_3x3_grad_v2(
//				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr], ctrl_frz,
//				stride, H_fmap_in
//			);
//			// end gradient calculation
//			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	in_channels = 64;
	out_channels = 64;
	H_fmap_in = 16;
	H_fmap_out = 16;
	stride = 1;

	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer2 PG2 ////////////////////////

	LOOP_layer2_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >= 0; c_in --) {
		ini -= 1;	// ini = 5
		// conv_3x3_weight_ptr -= 1;

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_2[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		bn_relu_bp(
			msb_fmap_tile_buffer_2[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in],
			bn_wt[ini], bn_bias[ini], vel_bn_wt[ini], vel_bn_bias[ini], ctrl_frz,
			H_fmap_in
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {		
			conv_3x3_weight_ptr -= 1;

			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			conv_3x3_backward(
				msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_1[c_out], out_buf_t0[0],
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr],
				stride, H_fmap_in, H_fmap_out, c_in
			);
			quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
			quant_act(msb_fmap_tile_buffer_1[c_out], H_fmap_out);
//			conv_3x3(	// MODIFIED HERE
//				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
//				stride, H_fmap_out, H_fmap_in, c_in, ctrl_conv
//			);
//			///////////////////////////
//			// conv_3x3_weight_grad_cal
//			conv_3x3_grad_v2(
//				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr], ctrl_frz,
//				stride, H_fmap_in
//			);
//			// end gradient calculation
//			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer2 PG1 ////////////////////////

	LOOP_layer2_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >= 0; c_in --) {
		ini -= 1;	// ini = 4
		// conv_3x3_weight_ptr -= 1;
		
		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		bn_relu_bp(
			msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in],
			bn_wt[ini], bn_bias[ini], vel_bn_wt[ini], vel_bn_bias[ini], ctrl_frz,
			H_fmap_in
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr -= 1;

			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			conv_3x3_backward(
				msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_2[c_out], out_buf_t0[0],
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr],
				stride, H_fmap_in, H_fmap_out, c_in
			);
			quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
			quant_act(msb_fmap_tile_buffer_2[c_out], H_fmap_out);
//			conv_3x3(
//				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_2[c_out], out_buf_t0[ini],
//				stride, H_fmap_out, H_fmap_in, c_in, ctrl_conv
//			);
//			///////////////////////////
//			// conv_3x3_weight_grad_cal
//			if (c_out <= out_channels_after_pack/2 - 1) {
//				conv_3x3_grad_v2(
//					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr], ctrl_frz,
//					stride, H_fmap_in
//				);
//			}
//			// end gradient calculation
//			///////////////////////////
		}
	}

	ctrl_frz = ctrl_tl;
	
	////////////////////////////////////////////////
	//////////// LAYER 1 Upsample //////////////////
	////////////////////////////////////////////////

	in_channels = 64;
	out_channels = 32;
	H_fmap_in = 16;
	H_fmap_out = 32;
	stride = 2;

	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer1 shortcut ///////////////////

	LOOP_layer1_ConvSC_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >= 0; c_in --) {
		ini -= 1;	// ini = 3
		// conv_3x3_weight_ptr -= 1;

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_2[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		bn_relu_bp(
			msb_fmap_tile_buffer_2[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in],
			bn_wt[ini], bn_bias[ini], vel_bn_wt[ini], vel_bn_bias[ini], ctrl_frz,
			H_fmap_in
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr -= 1;

			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			conv_3x3_backward(
				msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_1[c_out], out_buf_t0[0],
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr],
				stride, H_fmap_in, H_fmap_out, c_in
			);
			quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
			quant_act(msb_fmap_tile_buffer_1[c_out], H_fmap_out);
//			conv_3x3(
//				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
//				stride, H_fmap_out, H_fmap_in, c_in, ctrl_conv
//			);
//			///////////////////////////
//			// conv_3x3_weight_grad_cal
//			if (c_in <= in_channels_after_pack/2 - 1) {
//				conv_3x3_grad_v2(
//					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr], ctrl_frz,
//					stride, H_fmap_in
//				);
//			}
//			// end gradient calculation
//			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	in_channels = 32;
	out_channels = 32;
	H_fmap_in = 32;
	H_fmap_out = 32;
	stride = 1;

	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer1 PG2 ////////////////////////

	LOOP_layer1_Conv2_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >= 0; c_in --) {
		ini -= 1;	// ini = 2
		// conv_3x3_weight_ptr -= 1;

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		bn_relu_bp(
			msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in],
			bn_wt[ini], bn_bias[ini], vel_bn_wt[ini], vel_bn_bias[ini], ctrl_frz,
			H_fmap_in
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {		
			conv_3x3_weight_ptr -= 1;

			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			conv_3x3_backward(
				msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_dataflow[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_2[c_out], out_buf_t0[0],
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr],
				stride, H_fmap_in, H_fmap_out, c_in
			);
			quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
			quant_act(msb_fmap_tile_buffer_2[c_out], H_fmap_out);
//			conv_3x3(	// MODIFIED HERE
//				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_2[c_out], out_buf_t0[ini],
//				stride, H_fmap_out, H_fmap_in, c_in, ctrl_conv
//			);
//			///////////////////////////
//			// conv_3x3_weight_grad_cal
//			if (c_out <= 0) {
//				conv_3x3_grad_v2(
//					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr], ctrl_frz,
//					stride, H_fmap_in
//				);
//			}
//			// end gradient calculation
//			///////////////////////////
		}
	}

	in_channels = 32;
	out_channels = 3;
	H_fmap_in = 32;
	H_fmap_out = 32;
	stride = 1;

	out_channels_after_pack = 1;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer1 PG1 ////////////////////////

	LOOP_layer1_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >= 0; c_in --) {
		ini -= 1;	// ini = 1
		// conv_3x3_weight_ptr -= 1;
		
		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_2[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		bn_relu_bp(
			msb_fmap_tile_buffer_2[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_dataflow[c_in],
			bn_wt[ini], bn_bias[ini], vel_bn_wt[ini], vel_bn_bias[ini], ctrl_frz,
			H_fmap_in
		);
		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr -= 1;
			
			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			conv_3x3_backward(
				msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_dataflow[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_0[c_out], out_buf_t0[0],
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr],
				stride, H_fmap_in, H_fmap_out, c_in
			);
			quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
			quant_act(msb_fmap_tile_buffer_0[c_out], H_fmap_out);
//			conv_3x3(
//				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
//				stride, H_fmap_out, H_fmap_in, c_in, ctrl_conv
//			);
//			///////////////////////////
//			// conv_3x3_weight_grad_cal
//			if (c_in <= 0) {
//				conv_3x3_grad_v2(
//					out_buf_t1[ini - out_channels/CHANNEL_OUT_T], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], vel_conv_3x3[conv_3x3_weight_ptr], ctrl_frz,
//					stride, H_fmap_in
//				);
//			}
//			// end gradient calculation
//			///////////////////////////
		}
	}
}	// end FracBNN_T

