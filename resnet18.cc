#include "bnn.h"
#include "layer.h"
#include "dimension_def.h"

using namespace std;

static  int8 msb_fmap_tile_buffer_0[8][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
static  int8 msb_fmap_tile_buffer_1[8][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
static  int8 lsb_fmap_tile_buffer[8][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];	// shortcut activation/error on-chip
static	int8 conv_3x3_weight_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
static	int8 conv_1x1_weight_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T];

static	int8 grad_buf_t0[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];					// weight_3x3 gradient on-chip
static	int8 grad_buf_t1[CHANNEL_OUT_T][CHANNEL_IN_T];							// weight_1x1 gradient on-chip

static  int8 pool_out_buf[BATCH_SIZE][512];										// AvgPool buffer
static  int8 linear_weight_tile_buffer[10][512];
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
	int8 linear_weight[8][512],

	int8 out_buf_t0[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// BN output activation
    int8 out_buf_t1[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// Conv output activation

    int1 relu_mask[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH]		// relu mask for backprop
)
{
#pragma HLS INTERFACE m_axi depth=12288 port=image offset=slave bundle=IMG	// 4*3*32*32 = 12288
#pragma HLS INTERFACE m_axi depth=40 port=output offset=slave bundle=RESULT	// 4*10

#pragma HLS INTERFACE m_axi depth=11022336 port=conv_3x3_weight_all offset=slave bundle=conv_3x3_weight_all	// 299*64*64*3*3 = 11022336
#pragma HLS INTERFACE m_axi depth=172032 port=conv_1x1_weight_all offset=slave bundle=conv_1x1_weight_all	// 42*64*64 = 172032
#pragma HLS INTERFACE m_axi depth=5120 port=linear_weight offset=slave bundle=linear_weight					// 8*10*64 = 5120

#pragma HLS INTERFACE m_axi depth=17005824 port=out_buf_t0 offset=slave bundle=out_buf_t0					// 61*4*64*33*33 = 17005824
#pragma HLS INTERFACE m_axi depth=17005824 port=out_buf_t1 offset=slave bundle=out_buf_t1
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

// array partition
#pragma HLS ARRAY_PARTITION variable=msb_fmap_tile_buffer_0 complete dim=3
#pragma HLS ARRAY_PARTITION variable=msb_fmap_tile_buffer_1 complete dim=3
#pragma HLS ARRAY_PARTITION variable=lsb_fmap_tile_buffer complete dim=3

#pragma HLS ARRAY_PARTITION variable=conv_3x3_weight_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_3x3_weight_tile_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=conv_1x1_weight_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_1x1_weight_tile_buffer complete dim=2

#pragma HLS ARRAY_PARTITION variable=pool_out_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=linear_weight_tile_buffer complete dim=2

#pragma HLS ARRAY_PARTITION variable=grad_buf_t0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_buf_t0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=grad_buf_t1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_buf_t1 complete dim=2

#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_beta complete dim=1

	// Initialize the buffers to 0
	fmap_buf_init:
	for(int i=0; i < 8; i++) {
		for (int c = 0; c < CHANNEL_IN_T; c ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {
				pool_out_buf[i][b][c] = 0;
				for (int i = 0; i < WIDTH; i ++) {
					for (int j = 0; j < WIDTH; j ++) {
						msb_fmap_tile_buffer_0[i][b][c][i][j] = 0;
						msb_fmap_tile_buffer_1[i][b][c][i][j] = 0;
						lsb_fmap_tile_buffer[i][b][c][i][j] = 0;
					}
				}
#pragma HLS pipeline
				for (int ii=0; ii < 10; ii++) {
					linear_out_buf[i][b][ii] = 0;
					linear_weight_tile_buffer[ii][c] = 0;
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
    int out_channels, out_channel_start, stride, conv_3x3_weight_ptr, conv_1x1_weight_ptr, fc_weight_ptr, ini; //ini_sc;
	int1 ctrl_sc;


//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////		Forward path		//////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	ini = 0;
	ctrl_sc = 1; // if ctrl_sc=1, generate and send out_copy into DDR

	LOOP_GetImg:
	for (int row = 0; row < 32; row ++) {
		for (int col = 0; col < 32; col ++) {
#pragma HLS pipeline
			for (int b = 0; b < BATCH_SIZE; b ++) {
				for (int c = 0; c < 3; c ++) {
					msb_fmap_tile_buffer_1[0][b][c][row][col] = image[b][c][row][col];
					out_buf_t1[ini][b][c][row][col] = msb_fmap_tile_buffer_1[0][b][c][row][col];	// store image as the first input activation
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

    LOOP_Conv1:	 // 4 outermost for-loops
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 1
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
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
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 2
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in],
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	LOOP_layer1_0_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 3
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini],
					H_fmap_out, ctrl_sc	// generate and send out_copy into DDR
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	LOOP_layer1_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 4
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
	LOOP_layer1_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 5
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini],
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
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 6
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_1x1_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				conv_1x1(
					msb_fmap_tile_buffer_1[c_in], conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],	// conv+bn shortcut
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 7
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
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
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 8
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	LOOP_layer2_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 9
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini]; 	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	LOOP_layer2_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 10
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini],
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
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 11
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_1x1_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				conv_1x1(
					msb_fmap_tile_buffer_1[c_in], conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],	// conv+bn shortcut
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 12
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
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
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 13
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	LOOP_layer3_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 14
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
	LOOP_layer3_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 15
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini],
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
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 16
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_1x1_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				conv_1x1(
					msb_fmap_tile_buffer_1[c_in], conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],	// conv+bn shortcut
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_0 PG1 //////////////////////
	LOOP_layer4_0_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 17
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
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
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 18
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in],  msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_1[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_1 PG1 //////////////////////
	LOOP_layer4_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 19
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],
					H_fmap_in
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
	LOOP_layer4_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 20
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini],
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

    // Initialize the buffers for pooling and FC layer
	out_buf_init:
	for (int i = 0; i < 8; i ++) {
		for (int b = 0; b < BATCH_SIZE; b ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				pool_out_buf[b][c] = 0;
			}
#pragma HLS pipeline
			for (int j = 0; j < 10; j ++) {
				linear_out_buf[b][j] = 0;
			}
		}
	}

	// avgpool
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		avgpool(
			msb_fmap_tile_buffer_1[c_out], pool_out_buf, c_out
		);
	}

	// FC
	FC(
		pool_out_buf, linear_weight, linear_out_buf
	);

	write_output:
	for (int b = 0; b < BATCH_SIZE; b ++) {
#pragma HLS pipeline
		for (int j = 0; j < 10; j ++) {
			output[b][j] = linear_out_buf[b][j];
		}
	}

	////////////////////////////////////////////////
	//////////// CrossEntropy loss /////////////////
	int8 error[BATCH_SIZE][10];
	int8 labels[10];

	error_calc:
	// MSE for simplicity
	for (int b = 0; b < BATCH_SIZE; b ++) {
#pragma HLS pipeline
		for (int j = 0; j < 10; j ++) {
			error[b][j] = 2 * (linear_out_buf[b][j] - labels[j]);
		}
	}


//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////


	// FC_bp
	FC_bp(
		error, linear_weight_tile_buffer, pool_out_buf
	);

	// avgpool_bp
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0;  c_out --) {
		avgpool_bp(
			pool_out_buf, msb_fmap_tile_buffer_1[c_out], c_out
		);
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

	ctrl_sc = 0; // do not generate output_copy of shortcut

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
	LOOP_layer4_1_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 20
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],
					H_fmap_in
				);
				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 19
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				///////////////////////////
				conv_3x3_grad(
					out_buf_t1[ini - c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_0 PG2 //////////////////////
	LOOP_layer4_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 18
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini - c_out], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
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
	//////////// layer4_0 PG1 //////////////////////
	LOOP_layer4_0_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 17
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer, lsb_fmap_tile_buffer[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out-out_channels], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
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
	//////////// layer4_0 shortcut (conv+bn) ///////
	LOOP_layer4_0_ConvSC_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 16
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_1x1_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				bn_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], msb_fmap_tile_buffer_1[c_out],	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap_tile_buffer_1[c_out], conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_1x1_weight_grad_cal
				conv_1x1_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_1[c_out], grad_buf_t1,
					stride, H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],	// DDR not used
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 15
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],
					H_fmap_in
				);
				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 14
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
	LOOP_layer3_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 13
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
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
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 12
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer, lsb_fmap_tile_buffer[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out-out_channels], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
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
	//////////// layer3_0 shortcut (conv+bn) ///////
	LOOP_layer3_0_ConvSC_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 11
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_1x1_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				bn_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], msb_fmap_tile_buffer_1[c_out],	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap_tile_buffer_1[c_out], conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_1x1_weight_grad_cal
				conv_1x1_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_1[c_out], grad_buf_t1,
					stride, H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],	// DDR not used
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 10
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],
					H_fmap_in
				);
				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 9
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
	LOOP_layer2_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 8
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
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
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 7
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer, lsb_fmap_tile_buffer[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out-out_channels], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
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
	//////////// layer2_0 shortcut (conv+bn) ///////
	LOOP_layer2_0_ConvSC_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 6
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_1x1_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				bn_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], msb_fmap_tile_buffer_1[c_out],	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap_tile_buffer_1[c_out], conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_1x1_weight_grad_cal
				conv_1x1_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_1[c_out], grad_buf_t1,
					stride, H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],	// DDR not used
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 5
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out],
					H_fmap_in
				);
				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 4
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_1[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini],	// DDR not used
					H_fmap_out, ctrl_sc
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	LOOP_layer1_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 3
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				identity_shortcut(
					msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out],
					H_fmap_in
				);
				bn_relu_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 2
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_0[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_1[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_1[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation
				///////////////////////////
				shortcut(
					msb_fmap_tile_buffer_0[c_out], lsb_fmap_tile_buffer[c_out], msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini],	// DDR not used
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
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 1
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				bn_relu_bp(
					msb_fmap_tile_buffer_1[c_out], out_buf_t0[ini], relu_mask[ini], msb_fmap_tile_buffer_0[c_out],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				conv_3x3_rot_bp(
					msb_fmap_tile_buffer_0[c_out], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_out],
					stride, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				conv_3x3_grad(
					out_buf_t1[ini-1-c_out], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
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
