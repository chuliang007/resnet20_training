#include "bnn.h"
#include "layer.h"
#include "dimension_def.h"
#include "conv_weights.h"
// #include "weights_fracnet_64.h"
#include <math.h>

using namespace std;

static float msb_fmap_tile_buffer_0[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
static float msb_fmap_tile_buffer_1[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
static float msb_fmap_tile_buffer_s2[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip

static float out_buf_t0[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// conv output
static float out_buf_t1[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// generic_bn output
static uint1 relu_mask[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// relu mask for backprop

//static float conv_3x3_weight_tile_buffer[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
//static float conv_1x1_weight_tile_buffer[NUM_1x1_WT][CHANNEL_OUT_T][CHANNEL_IN_T];
static float conv_3x3_weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];

static float pool_out_buf[64];
static float pool_out_buf_copy[64];	// generic_avgpool buffer
//static float linear_weight_tile_buffer[10][64];
static float linear_out_buf[10];											// generic_FC buffer

static float sum;
//static float labels[10];

//static float bn_wt[NUM_ACT][CHANNEL_OUT_T];								// generic_bn weight
//static float bn_bias[NUM_ACT][CHANNEL_OUT_T];								// generic_bn bias
//static float mu[NUM_ACT][CHANNEL_OUT_T];									// generic_bn input mu
//static float std_var[NUM_ACT][CHANNEL_OUT_T];								// generic_bn input standard variance

//static float act_bias[ACT_BIAS][CHANNEL_OUT_T];							// activation's shared exponent bias of block minifloat
//static float wt_bias[WT_BIAS][CHANNEL_OUT_T];								// weight's shared exponent bias of block minifloat
//static float wt_bias_1x1[WT_BIAS_1x1][CHANNEL_OUT_T];						// weight's shared exponent bias of block minifloat
//static float act_bias_shift[ACT_BIAS][CHANNEL_OUT_T];						// used for activation's shared exponent bias update
static float bias_gap4add[CHANNEL_OUT_T];									// shared exponent gap between act & wt in bm addition

//--------------------
//  Top Function 
//--------------------
void FracNet_T(
	float image[3][32][32],
	float &loss,
	float error[10]
)
{
#pragma HLS INTERFACE m_axi depth=3072 port=image offset=slave bundle=IMG		// 1*3*32*32 = 3072
#pragma HLS INTERFACE m_axi depth=1 port=loss offset=slave bundle=LOSS			// 1
#pragma HLS INTERFACE m_axi depth=10 port=error offset=slave bundle=ERROR		// 1*10
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

// instance allocation
#pragma HLS ALLOCATION function instances=generic_bn limit=1
#pragma HLS ALLOCATION function instances=generic_bn_bp limit=1
#pragma HLS ALLOCATION function instances=generic_bn_relu limit=1
#pragma HLS ALLOCATION function instances=generic_bn_relu_bp limit=1
#pragma HLS ALLOCATION function instances=generic_shortcut limit=1

#pragma HLS ALLOCATION function instances=generic_avgpool limit=1
#pragma HLS ALLOCATION function instances=generic_FC limit=1

#pragma HLS ALLOCATION function instances=generic_conv_3x3 limit=1
#pragma HLS ALLOCATION function instances=generic_deconv_3x3 limit=1
#pragma HLS ALLOCATION function instances=generic_conv_1x1 limit=1
#pragma HLS ALLOCATION function instances=generic_deconv_1x1 limit=1
#pragma HLS ALLOCATION function instances=generic_conv_3x3_grad limit=1
#pragma HLS ALLOCATION function instances=generic_conv_1x1_grad limit=1

	int H_fmap_in, H_fmap_out, in_channels, in_channels_after_pack, out_channels_after_pack;
    int out_channels, out_channel_start, stride, conv_3x3_weight_ptr, conv_1x1_weight_ptr, fc_weight_ptr, ini, ini_act_bias, ini_wt_bias;
	uint1 ctrl_conv, ctrl_fc, ctrl_avgpool;

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

	LOOP_GetImg:
	for (int row = 0; row < 32; row ++) {
		for (int col = 0; col < 32; col ++) {
			for (int c = 0; c < 3; c ++) {
#pragma HLS PIPELINE II=1	// 3080 cycles
				msb_fmap_tile_buffer_s2[0][c][row][col] = image[c][row][col];
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
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			generic_conv_3x3(
				msb_fmap_tile_buffer_s2[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
		}
		generic_bn_relu(
			msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	float layer1_Conv1_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_in; row ++){
				for(int col = 0; col < H_fmap_in; col ++){
					if (abs(layer1_Conv1_out_max) < abs(msb_fmap_tile_buffer_1[c_out][c_in][row][col])) {
						layer1_Conv1_out_max = msb_fmap_tile_buffer_1[c_out][c_in][row][col];
					}
				}
			}
		}
	}
	// printf("layer1_Conv1_out_max: %f \n", layer1_Conv1_out_max);

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
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			generic_conv_3x3(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
		}
		generic_bn_relu(
			msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_s2[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_s2[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	float layer1_Conv2_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_in; row ++){
				for(int col = 0; col < H_fmap_in; col ++){
					if (abs(layer1_Conv2_out_max) < abs(msb_fmap_tile_buffer_s2[c_out][c_in][row][col])) {
						layer1_Conv2_out_max = msb_fmap_tile_buffer_s2[c_out][c_in][row][col];
					}
				}
			}
		}
	}
	// printf("layer1_Conv2_out_max: %f \n", layer1_Conv2_out_max);

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
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			generic_conv_3x3(
				msb_fmap_tile_buffer_s2[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
		}
		generic_bn_relu(
			msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_0[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	float layer1_ConvSC_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_out; row ++){
				for(int col = 0; col < H_fmap_out; col ++){
					if (abs(layer1_ConvSC_out_max) < abs(msb_fmap_tile_buffer_0[c_out][c_in][row][col])) {
						layer1_ConvSC_out_max = msb_fmap_tile_buffer_0[c_out][c_in][row][col];
					}
//					if (msb_fmap_tile_buffer_0[c_out][c_in][row][col] != 0)
//						// printf("layer1_ConvSC_out_max[%d][%d][%d][%d]: %f \n", c_out, c_in, row, col, msb_fmap_tile_buffer_0[c_out][c_in][row][col]);
				}
			}
		}
	}
	// printf("layer1_ConvSC_out_max: %f \n", layer1_ConvSC_out_max);

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
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			generic_conv_3x3(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_s2[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
		}
		generic_bn_relu(
			msb_fmap_tile_buffer_s2[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	float layer2_Conv1_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_out; row ++){
				for(int col = 0; col < H_fmap_out; col ++){
					if (abs(layer2_Conv1_out_max) < abs(msb_fmap_tile_buffer_1[c_out][c_in][row][col])) {
						layer2_Conv1_out_max = msb_fmap_tile_buffer_1[c_out][c_in][row][col];
					}
//					if (msb_fmap_tile_buffer_1[c_out][c_in][row][col] != 0)
//						// printf("layer2_Conv1_out_max[%d][%d][%d][%d]: %f \n", c_out, c_in, row, col, msb_fmap_tile_buffer_1[c_out][c_in][row][col]);
				}
			}
		}
	}
	// printf("layer2_Conv1_out_max: %f \n", layer2_Conv1_out_max);

	////////////////////////////////////////////////
	//////////// layer2 PG2 ////////////////////////

	LOOP_layer2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		ini += 1;	// ini = 5
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			generic_conv_3x3(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
		}
		generic_bn_relu(	// wrong output (beyond H_fmap_out)
			msb_fmap_tile_buffer_0[c_out], msb_fmap_tile_buffer_s2[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_s2[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	float layer2_Conv2_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_out; row ++){		//32
				for(int col = 0; col < H_fmap_out; col ++){	//32
					if (abs(layer2_Conv2_out_max) < abs(msb_fmap_tile_buffer_s2[c_out][c_in][row][col])) {
						layer2_Conv2_out_max = msb_fmap_tile_buffer_s2[c_out][c_in][row][col];
					}
//					if (msb_fmap_tile_buffer_s2[c_out][c_in][row][col] != 0)
//						// printf("layer2_Conv2_out_max[%d][%d][%d][%d]: %f \n", c_out, c_in, row, col, msb_fmap_tile_buffer_s2[c_out][c_in][row][col]);
				}
			}
		}
	}
	// printf("layer2_Conv2_out_max: %f \n", layer2_Conv2_out_max);

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
		conv_3x3_weight_ptr += 1;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {

			generic_conv_3x3(
				msb_fmap_tile_buffer_s2[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
		}
		generic_bn_relu(
			msb_fmap_tile_buffer_1[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t1[ini], relu_mask[ini],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_0[c_out], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);
	}

	float layer2_ConvSC_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_out; row ++){
				for(int col = 0; col < H_fmap_out; col ++){
					if (abs(layer2_ConvSC_out_max) < abs(msb_fmap_tile_buffer_0[c_out][c_in][row][col])) {
						layer2_ConvSC_out_max = msb_fmap_tile_buffer_0[c_out][c_in][row][col];
					}
//					if (msb_fmap_tile_buffer_0[c_out][c_in][row][col] != 0)
//						// printf("layer2_ConvSC_out_max[%d][%d][%d][%d]: %f \n", c_out, c_in, row, col, msb_fmap_tile_buffer_0[c_out][c_in][row][col]);
				}
			}
		}
	}
	// printf("layer2_ConvSC_out_max: %f \n", layer2_ConvSC_out_max);

//	printf("ini = %d, conv_3x3_weight_ptr = %d \n", ini, conv_3x3_weight_ptr);

	////////////////////////////////////////////////
	//////////// AvgPool and FC ////////////////////

	// avgpool
	ctrl_avgpool = 0;
	LOOP_AvgPool:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		generic_avgpool(
			msb_fmap_tile_buffer_0[c_out], pool_out_buf, ctrl_avgpool, c_out, pool_out_buf_copy, bias_gap4add
		);
	}

//	for (int i = 0; i < 64; i ++) {
//		cout << "avg_pool_out: " << pool_out_buf[i] << "  " << endl;
//	}

	// FC
	ctrl_fc = 0;
	LOOP_FC:
	generic_FC(
		pool_out_buf, pool_out_buf_copy, linear_weight_tile_buffer, linear_out_buf, ctrl_fc
	);

//	for (int i = 0; i < 10; i ++) {
//		cout << "linear_out_buf: " << linear_out_buf[i] << "  " << endl;
//	}


	////////////////////////////////////////////////
	/////////////////   FC   ///////////////////////

	sum = 0;
	softmax_sum_approx:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		sum += 1 + linear_out_buf[j] + 0.5*(linear_out_buf[j])*(linear_out_buf[j]) + 0.1667*(linear_out_buf[j])*(linear_out_buf[j])*(linear_out_buf[j]);
	}
//	printf("sum: %f \n", sum);

	loss = 0;
	CE_loss:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		loss += -labels[j] * hls::log2((1 + linear_out_buf[j] + 0.5*(linear_out_buf[j])*(linear_out_buf[j]) + 0.1667*(linear_out_buf[j])*(linear_out_buf[j])*(linear_out_buf[j]))/sum);
	}

	CE_error:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		error[j] = (1 + linear_out_buf[j] + 0.5*(linear_out_buf[j])*(linear_out_buf[j]) + 0.1667*(linear_out_buf[j])*(linear_out_buf[j])*(linear_out_buf[j]))/sum - labels[j];
	}


//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// AvgPool and FC ////////////////////

	// FC_bp
	ctrl_fc = 1;
	LOOP_FC_bp:
	generic_FC(
		pool_out_buf, pool_out_buf_copy, linear_weight_tile_buffer, error, ctrl_fc
	);

	// avgpool_bp
	ctrl_avgpool = 1;
	LOOP_AvgPool_bp:
	for (int c_out = 0; c_out < out_channels_after_pack; c_out ++) {
		generic_avgpool(
			msb_fmap_tile_buffer_1[c_out], pool_out_buf, ctrl_avgpool, c_out, pool_out_buf_copy, bias_gap4add
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
		conv_3x3_weight_ptr -= 1;

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		generic_bn_relu_bp(
			msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {
			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			generic_deconv_3x3(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_s2[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad(
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	float bp_layer2_ConvSC_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_in; row ++){
				for(int col = 0; col < H_fmap_in; col ++){
					if (abs(bp_layer2_ConvSC_out_max) < abs(msb_fmap_tile_buffer_s2[c_out][c_in][row][col])) {
						bp_layer2_ConvSC_out_max = msb_fmap_tile_buffer_s2[c_out][c_in][row][col];
					}
				}
			}
		}
	}
	// printf("bp_layer2_ConvSC_out_max: %f \n", bp_layer2_ConvSC_out_max);

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
		conv_3x3_weight_ptr -= 1;

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_s2[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		generic_bn_relu_bp(
			msb_fmap_tile_buffer_s2[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {		
			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			generic_deconv_3x3(	// MODIFIED HERE
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad(
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
				stride, H_fmap_in
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	float bp_layer2_Conv2_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_in; row ++){
				for(int col = 0; col < H_fmap_in; col ++){
					if (abs(bp_layer2_Conv2_out_max) < abs(msb_fmap_tile_buffer_1[c_out][c_in][row][col])) {
						bp_layer2_Conv2_out_max = msb_fmap_tile_buffer_1[c_out][c_in][row][col];
					}
				}
			}
		}
	}
	// printf("bp_layer2_Conv2_out_max: %f \n", bp_layer2_Conv2_out_max);

	////////////////////////////////////////////////
	//////////// layer2 PG1 ////////////////////////

	LOOP_layer2_Conv1_bp:
	for (int c_in = in_channels_after_pack - 1; c_in >= 0; c_in --) {
		ini -= 1;	// ini = 4
		conv_3x3_weight_ptr -= 1;
		
		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		generic_bn_relu_bp(
			msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {
			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			generic_deconv_3x3(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_s2[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_out <= out_channels_after_pack/2 - 1) {
				generic_conv_3x3_grad(
					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
					stride, H_fmap_in
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	float bp_layer2_Conv1_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_in; row ++){
				for(int col = 0; col < H_fmap_in; col ++){
					if (abs(bp_layer2_Conv1_out_max) < abs(msb_fmap_tile_buffer_s2[c_out][c_in][row][col])) {
						bp_layer2_Conv1_out_max = msb_fmap_tile_buffer_s2[c_out][c_in][row][col];
					}
				}
			}
		}
	}
	// printf("bp_layer2_Conv1_out_max: %f \n", bp_layer2_Conv1_out_max);

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
		conv_3x3_weight_ptr -= 1;

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_s2[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		generic_bn_relu_bp(
			msb_fmap_tile_buffer_s2[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {
			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			generic_deconv_3x3(
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_in <= in_channels_after_pack/2 - 1) {
				generic_conv_3x3_grad(
					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
					stride, H_fmap_in
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	float bp_layer1_ConvSC_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_in; row ++){
				for(int col = 0; col < H_fmap_in; col ++){
					if (abs(bp_layer1_ConvSC_out_max) < abs(msb_fmap_tile_buffer_1[c_out][c_in][row][col])) {
						bp_layer1_ConvSC_out_max = msb_fmap_tile_buffer_1[c_out][c_in][row][col];
					}
				}
			}
		}
	}
	// printf("bp_layer1_ConvSC_out_max: %f \n", bp_layer1_ConvSC_out_max);

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
		conv_3x3_weight_ptr -= 1;

		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_1[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		generic_bn_relu_bp(
			msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_in],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);

		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {		
			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			generic_deconv_3x3(	// MODIFIED HERE
				msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_s2[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_out <= 0) {
				generic_conv_3x3_grad(
					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
					stride, H_fmap_in
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	float bp_layer1_Conv2_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_in; row ++){
				for(int col = 0; col < H_fmap_in; col ++){
					if (abs(bp_layer1_Conv2_out_max) < abs(msb_fmap_tile_buffer_s2[c_out][c_in][row][col])) {
						bp_layer1_Conv2_out_max = msb_fmap_tile_buffer_s2[c_out][c_in][row][col];
					}
				}
			}
		}
	}
	// printf("bp_layer1_Conv2_out_max: %f \n", bp_layer1_Conv2_out_max);

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
		conv_3x3_weight_ptr -= 1;
		
		// block minifloat quant
		quant_wt(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr]);
		quant_act(msb_fmap_tile_buffer_s2[c_in], H_fmap_out);
		quant_act(out_buf_t0[ini], H_fmap_out);
		quant_act(out_buf_t1[ini], H_fmap_out);

		generic_bn_relu_bp(
			msb_fmap_tile_buffer_s2[c_in], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_in],
			bn_wt[ini], bn_bias[ini],
			H_fmap_out, bias_gap4add
		);
		for (int c_out = out_channels_after_pack - 1; c_out >= 0; c_out --) {
			rot180_3x3(conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr], conv_3x3_weight_rot);
			generic_deconv_3x3(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_rot, msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],
				stride, H_fmap_in, H_fmap_out, c_in, ctrl_conv, bias_gap4add
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_in <= 0) {
				generic_conv_3x3_grad(
					out_buf_t1[ini - out_channels/CHANNEL_OUT_T], msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer[conv_3x3_weight_ptr],
					stride, H_fmap_in
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	float bp_layer1_Conv1_out_max = 0;
	for(int c_in = 0; c_in < 8; c_in ++){
		for(int c_out = 0; c_out < 8; c_out ++){
			for(int row = 0; row < H_fmap_in; row ++){
				for(int col = 0; col < H_fmap_in; col ++){
					if (abs(bp_layer1_Conv1_out_max) < abs(msb_fmap_tile_buffer_0[c_out][c_in][row][col])) {
						bp_layer1_Conv1_out_max = msb_fmap_tile_buffer_0[c_out][c_in][row][col];
					}
				}
			}
		}
	}
	// printf("bp_layer1_Conv1_out_max: %f \n", bp_layer1_Conv1_out_max);
	
}	// end FracBNN_T

