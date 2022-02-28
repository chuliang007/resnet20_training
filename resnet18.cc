#include "bnn.h"
#include "layer.h"
// #include "dimension_def.h"

using namespace std;
//--------------------
//  Top Function 
//--------------------
void FracNet_T(
	int16 image[BATCH_SIZE][3][32][32],
	int16 output[BATCH_SIZE][10]
)
{
#pragma HLS INTERFACE m_axi depth=12288 port=image offset=slave bundle=IMG
#pragma HLS INTERFACE m_axi depth=40 port=output offset=slave bundle=RESULT
//#pragma HLS INTERFACE m_axi depth=147456 port=conv_3x3_weight offset=slave bundle=RESULT_weight_3x3
//#pragma HLS INTERFACE m_axi depth=147456 port=conv_1x1_weight offset=slave bundle=RESULT_weight_1x1

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

#pragma HLS ALLOCATION instances=bn_relu limit=1 function
#pragma HLS ALLOCATION instances=bn_relu_bp limit=1 function
#pragma HLS ALLOCATION instances=avgpool limit=1 function
#pragma HLS ALLOCATION instances=avgpool_bp limit=1 function
#pragma HLS ALLOCATION instances=FC limit=1 function
#pragma HLS ALLOCATION instances=FC_bp limit=1 function
#pragma HLS ALLOCATION instances=shortcut limit=1 function
#pragma HLS ALLOCATION instances=conv_3x3 limit=1 function
#pragma HLS ALLOCATION instances=conv_1x1 limit=1 function
#pragma HLS ALLOCATION instances=conv_3x3_rot_bp limit=1 function
#pragma HLS ALLOCATION instances=conv_1x1_rot_bp limit=1 function
#pragma HLS ALLOCATION instances=conv_3x3_grad limit=1 function
#pragma HLS ALLOCATION instances=conv_1x1_grad limit=1 function
#pragma HLS ALLOCATION instances=SGD_WU_3x3 limit=1 function
#pragma HLS ALLOCATION instances=SGD_WU_1x1 limit=1 function

	int16 msb_fmap[NUM_ACT][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];		// input
	int16 lsb_fmap[NUM_ACT][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];		// shortcut input

	int16 conv_3x3_weight[NUM_WT_3x3][CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
	int16 conv_1x1_weight[NUM_WT_1x1][CHANNEL_OUT_T][CHANNEL_IN_T];

	float gamma[CHANNEL_OUT_T];
	float beta[CHANNEL_OUT_T];
	int16 g_gamma[CHANNEL_OUT_T];
	int16 g_beta[CHANNEL_OUT_T];
// #pragma HLS ARRAY_PARTITION variable=msb_fmap complete dim=1
// #pragma HLS ARRAY_PARTITION variable=lsb_fmap complete dim=1

	int16 out_buf_0[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
	int16 out_buf_t0[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];		// output
	int16 out_buf_t1[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];		// shortcut output
	int1  relu_mask[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];		// relu mask for backprop

	int16 grad_buf_t0[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];		// weight_3x3 gradient
	int16 grad_buf_t1[CHANNEL_OUT_T][CHANNEL_IN_T];				// weight_1x1 gradient
// #pragma HLS ARRAY_PARTITION variable=out_buf_0 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=out_buf_0 complete dim=2
// #pragma HLS ARRAY_PARTITION variable=out_buf_t0 complete dim=1
// #pragma HLS ARRAY_PARTITION variable=out_buf_t1 complete dim=1

    // Initialize the buffers to 0
	global_buffer_init_0:
	for (int i = 0; i < NUM_ACT; i ++) {
		for (int b = 0; b < BATCH_SIZE; b ++){
			for (int i = 0; i < WIDTH; i ++){
				for (int j = 0; j < WIDTH; j ++){
					for (int k = 0; k < CHANNEL_IN_T; k ++) {
						msb_fmap[i][b][k][i][j] = 0;
						lsb_fmap[i][b][k][i][j] = 0;
					}
					for (int k = 0; k < CHANNEL_OUT_T; k ++) {
						out_buf_0[i][b][k][i][j] = 0;
						out_buf_t0[i][b][k][i][j] = 0;
						out_buf_t1[i][b][k][i][j] = 0;
						relu_mask[i][b][k][i][j] = 0;
					}
				}
			}
		}
	}

	global_buffer_init:
	for (int c_out = 0; c_out < CHANNEL_OUT_T; c_out ++){
		for (int c_in = 0; c_in < CHANNEL_IN_T; c_in ++) {
			grad_buf_t1[c_out][c_in] = 0;
			for (int row = 0; row < 3; row ++){
				for (int col = 0; col < 3; col ++){
					grad_buf_t0[c_out][c_in][row][col] = 0;
				}
			}
		}
	}

	int H_fmap_in, H_fmap_out, in_channels, in_channels_after_pack; 
    int out_channels, out_channel_start, stride, conv_3x3_weight_ptr, conv_1x1_weight_ptr, ini;


//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////		Forward path		//////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	LOOP_GetImg:
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int c = 0; c < 3; c ++) {
			for (int row = 0; row < 32; row ++) {
				for (int col = 0; col < 32; col ++) {
					//#pragma HLS PIPELINE
					msb_fmap[0][b][c][row][col] = image[b][c][row][col];
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

    LOOP_Conv1:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {
				// ini = 0
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], out_buf_t1[ini],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */ 
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], out_buf_t1[ini],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
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

				conv_1x1(
					msb_fmap[ini], conv_1x1_weight[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn(
					out_buf_t1[ini], lsb_fmap[ini],	// conv+bn shortcut
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], out_buf_t1[ini],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], lsb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], out_buf_t1[ini],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
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
				conv_1x1_weight_ptr += 1;

				conv_1x1(
					msb_fmap[ini], conv_1x1_weight[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn(
					out_buf_t1[ini], lsb_fmap[ini],	// conv+bn shortcut
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], out_buf_t1[ini],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], lsb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], out_buf_t1[ini],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
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
				conv_1x1_weight_ptr += 1;

				conv_1x1(
					msb_fmap[ini], conv_1x1_weight[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn(
					out_buf_t1[ini], lsb_fmap[ini],	// conv+bn shortcut
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/*bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], out_buf_t1[ini],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], lsb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], msb_fmap[ini + 1],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini], msb_fmap[ini + 1], relu_mask[ini],
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

				conv_3x3(
					msb_fmap[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				/* bn(
					out_buf_t0[ini], out_buf_0[ini],
					gamma, beta,
					H_fmap_out
				);
				relu(
					out_buf_0[ini], out_buf_t1[ini],
					H_fmap_out
				); */
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
				);	// CHANNEL_OUT_T = CHANNEL_IN_T
			}
		}
    }

    // Initialize the buffers for pooling and FC layer
	int16 pool_out_buf[BATCH_SIZE][CHANNEL_OUT_T];
	int16 linear_out_buf[BATCH_SIZE][10];
	int16 linear_weight[10][CHANNEL_OUT_T];	// FC weight

	pool_out_buf_init:
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
			pool_out_buf[b][c] = 0;
		}
	}

	linear_out_buf_init:
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int i = 0; i < 10; i ++) {
			linear_out_buf[b][i] = 0;
		}
	}

	// avgpool
	ini += 1;	// ini = 17
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		avgpool(msb_fmap[ini], pool_out_buf, 4, 4);
	}

	// FC
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		FC(pool_out_buf, linear_weight, linear_out_buf);
	}

	write_output:
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int i = 0; i < 10; i ++) {
			output[b][i] = linear_out_buf[b][i];
		}
	}

	////////////////////////////////////////////////
	//////////// CrossEntropy loss /////////////////
	int16 error[BATCH_SIZE][10];
	int16 labels[BATCH_SIZE][10];

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



	int16 linear_weight_transpose[CHANNEL_OUT_T][10];	// FC weight transposed

	// FC_bp
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		FC_bp(error, linear_weight_transpose, pool_out_buf);
	}

	// avgpool_bp
	ini += 1;	// ini = 18
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		avgpool_bp(pool_out_buf, msb_fmap[ini], 4);
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
	//////////// layer4_1 PG2 //////////////////////
	LOOP_layer4_1_Conv2_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				ini -= 1;	// ini = 17

				/* relu_bp(
					msb_fmap[ini], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				); */
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_1 PG1 //////////////////////
	LOOP_layer4_1_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 16
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t0[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t0[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
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

				ini -= 1; // ini = 15
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
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

				// ini = 15
				conv_1x1_weight_ptr -= 1;

				/* rot180_1x1(conv_1x1_weight[conv_1x1_weight_ptr], conv_1x1_weight_rot180[conv_1x1_weight_ptr]);
				conv_1x1_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight_rot180[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				); */
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t1[ini - 2], lsb_fmap[ini],	// conv+bn shortcut
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// msb_fmap[ini+1] as error (kernel), and msb_fmap[ini-2] as fmap (input)
				///////////////////////////
				conv_1x1_grad(
					msb_fmap[ini - 2], msb_fmap[ini + 1], grad_buf_t1,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight[conv_1x1_weight_ptr], conv_1x1_weight[conv_1x1_weight_ptr]
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_0 PG1 //////////////////////
	LOOP_layer4_0_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 14
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t0[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t0[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					lsb_fmap[ini + 1], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
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
				ini -= 1; // ini = 13
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	LOOP_layer3_1_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 12
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t0[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t0[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
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

				ini -= 1; // ini = 11
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				); */
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
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

				// ini = 11
				conv_1x1_weight_ptr -= 1;

				/* rot180_1x1(conv_1x1_weight[conv_1x1_weight_ptr], conv_1x1_weight_rot180[conv_1x1_weight_ptr]);
				conv_1x1_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight_rot180[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				); */
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t1[ini - 2], lsb_fmap[ini],	// conv+bn shortcut
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// msb_fmap[ini+1] as error (kernel), and msb_fmap[ini-2] as fmap (input)
				///////////////////////////
				conv_1x1_grad(
					msb_fmap[ini - 2], msb_fmap[ini + 1], grad_buf_t1,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight[conv_1x1_weight_ptr], conv_1x1_weight[conv_1x1_weight_ptr]
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 10
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t0[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t0[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					lsb_fmap[ini + 1], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
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
				ini -= 1; // ini = 9
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	LOOP_layer2_1_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 8
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t0[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t0[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				); */
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
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

				ini -= 1; // ini = 7
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				); */
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
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

				// ini = 7
				conv_1x1_weight_ptr -= 1;

				/* rot180_1x1(conv_1x1_weight[conv_1x1_weight_ptr], conv_1x1_weight_rot180[conv_1x1_weight_ptr]);
				conv_1x1_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight_rot180[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				); */
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight[conv_1x1_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t1[ini - 2], lsb_fmap[ini],	// conv+bn shortcut
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// msb_fmap[ini+1] as error (kernel), and msb_fmap[ini-2] as fmap (input)
				///////////////////////////
				conv_1x1_grad(
					msb_fmap[ini - 2], msb_fmap[ini + 1], grad_buf_t1,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight[conv_1x1_weight_ptr], conv_1x1_weight[conv_1x1_weight_ptr]
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 6
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t0[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t0[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				); */
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					lsb_fmap[ini + 1], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
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
				ini -= 1; // ini = 5
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	LOOP_layer1_1_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 4
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t0[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t0[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
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
				ini -= 1;	// ini = 3
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
	LOOP_layer1_0_Conv1_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini -= 1;	// ini = 2
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t0[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t0[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
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

				ini -= 1;	// ini = 1
				conv_3x3_weight_ptr -= 1;

				/* relu_bp(
					msb_fmap[ini + 1], out_buf_0[ini - 1], out_buf_t1[ini],
					H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t0[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				); 
				rot180_3x3(conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight_rot180[conv_3x3_weight_ptr]);
				conv_3x3_bp(
					out_buf_0[ini], conv_3x3_weight_rot180[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);*/
				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_0[ini],
					gamma, g_gamma, g_beta,
					H_fmap_out
				);
				conv_3x3_rot_bp(
					out_buf_0[ini], conv_3x3_weight[conv_3x3_weight_ptr], msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini-1], out_buf_0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight[conv_3x3_weight_ptr], conv_3x3_weight[conv_3x3_weight_ptr]
				);
			}
		}
    }
}	// end FracBNN_T
