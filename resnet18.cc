#include "bnn.h"
#include "layer.h"
// #include "dimension_def.h"

using namespace std;

// static	int8 msb_fmap[NUM_ACT][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];			// input
// static	int8 lsb_fmap[NUM_SC][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];			// shortcut input
// static  int8 fmap_tile_buffer[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH];
static	int8 conv_3x3_weight_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
static	int8 conv_1x1_weight_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T];

// static	int8 out_buf_t0[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];		// output
// static	int8 out_buf_t1[NUM_SC][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];		// shortcut output
// static	int1 relu_mask[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];		// relu mask for backprop

static	int8 grad_buf_t0[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];					// weight_3x3 gradient
static	int8 grad_buf_t1[CHANNEL_OUT_T][CHANNEL_IN_T];							// weight_1x1 gradient

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

	int8 msb_fmap[NUM_ACT][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int8 lsb_fmap[NUM_SC][BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int8 out_buf_t0[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// output
    int8 out_buf_t1[NUM_SC][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// shortcut output

    int1 relu_mask[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH]		// relu mask for backprop
)
{
#pragma HLS INTERFACE m_axi depth=12288 port=image offset=slave bundle=IMG
#pragma HLS INTERFACE m_axi depth=40 port=output offset=slave bundle=RESULT

#pragma HLS INTERFACE m_axi depth=3184182 port=conv_3x3_weight_all offset=slave bundle=conv_3x3_weight_all	// 1382*4*64*3*3 = 3184182
#pragma HLS INTERFACE m_axi depth=43008 port=conv_1x1_weight_all offset=slave bundle=conv_1x1_weight_all	// 168*4*64 = 43008

#pragma HLS INTERFACE m_axi depth=362283008 port=msb_fmap offset=slave bundle=msb_fmap	// 1382*4*64*32*32 = 362283008
#pragma HLS INTERFACE m_axi depth=44040192 port=lsb_fmap offset=slave bundle=lsb_fmap	// 168*4*64*32*32 = 44040192
#pragma HLS INTERFACE m_axi depth=362283008 port=out_buf_t0 offset=slave bundle=out_buf_t0
#pragma HLS INTERFACE m_axi depth=44040192 port=out_buf_t1 offset=slave bundle=out_buf_t1
#pragma HLS INTERFACE m_axi depth=362283008 port=relu_mask offset=slave bundle=relu_mask

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

// instance allocation
#pragma HLS ALLOCATION instances=bn_relu limit=2 function
#pragma HLS ALLOCATION instances=bn_relu_bp limit=2 function
#pragma HLS ALLOCATION instances=shortcut limit=1 function

#pragma HLS ALLOCATION instances=avgpool limit=1 function
#pragma HLS ALLOCATION instances=avgpool_bp limit=1 function
#pragma HLS ALLOCATION instances=FC limit=1 function
#pragma HLS ALLOCATION instances=FC_bp limit=1 function

#pragma HLS ALLOCATION instances=conv_3x3 limit=2 function
#pragma HLS ALLOCATION instances=conv_1x1 limit=1 function

#pragma HLS ALLOCATION instances=conv_3x3_rot_bp limit=2 function
#pragma HLS ALLOCATION instances=conv_1x1_rot_bp limit=1 function
#pragma HLS ALLOCATION instances=conv_3x3_grad limit=2 function
#pragma HLS ALLOCATION instances=conv_1x1_grad limit=1 function

#pragma HLS ALLOCATION instances=SGD_WU_3x3 limit=2 function
#pragma HLS ALLOCATION instances=SGD_WU_1x1 limit=1 function

// array partition
// #pragma HLS ARRAY_PARTITION variable=fmap_tile_buffer complete dim=1
// #pragma HLS ARRAY_PARTITION variable=fmap_tile_buffer complete dim=2

#pragma HLS ARRAY_PARTITION variable=conv_3x3_weight_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_3x3_weight_tile_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=conv_1x1_weight_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_1x1_weight_tile_buffer complete dim=2

#pragma HLS ARRAY_PARTITION variable=grad_buf_t0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_buf_t0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=grad_buf_t1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_buf_t1 complete dim=2

#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=grad_beta complete dim=1
	/*
	global_buffer_init_0:
	for (int i = 0; i < WIDTH; i ++){
		for (int j = 0; j < WIDTH; j ++){
			for (int k = 0; k < CHANNEL_IN_T; k ++) {
				msb_fmap[k][i][j] = 0;
				lsb_fmap[k][i][j] = 0;
			}
			for (int k = 0; k < CHANNEL_OUT_T; k ++) {
				out_buf_t0[k][i][j] = 0;
				out_buf_t1[k][i][j] = 0;
				relu_mask[k][i][j] = 0;
			}
		}
	}
	*/
	// Initialize the buffers to 0
	global_buffer_init:
	for (int c_out = 0; c_out < CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < CHANNEL_IN_T; c_in ++) {
#pragma HLS pipeline
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
#pragma HLS PIPELINE
			for (int row = 0; row < 32; row ++) {
				for (int col = 0; col < 32; col ++) {
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

    LOOP_Conv1:	// 4 outermost for-loops
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {
				// ini = 0
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
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
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
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

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all,
					conv_1x1_weight_ptr
				);
				conv_1x1(
					msb_fmap[ini], conv_1x1_weight_tile_buffer, out_buf_t1[ini],
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
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], lsb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
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
				conv_1x1_weight_ptr += 1;

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all,
					conv_1x1_weight_ptr
				);
				conv_1x1(
					msb_fmap[ini], conv_1x1_weight_tile_buffer, out_buf_t1[ini],
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], lsb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all,
					conv_1x1_weight_ptr
				);
				conv_1x1(
					msb_fmap[ini], conv_1x1_weight_tile_buffer, out_buf_t1[ini],
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
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

				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3(
					msb_fmap[ini], conv_3x3_weight_tile_buffer, out_buf_t0[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_relu(
					out_buf_t0[ini],  out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				);
				shortcut(
					out_buf_t1[ini], msb_fmap[ini - 1], msb_fmap[ini + 1],
					H_fmap_out
				);
			}
		}
    }

    // Initialize the buffers for pooling and FC layer
	int8 pool_out_buf[BATCH_SIZE][CHANNEL_OUT_T];
	int8 linear_out_buf[BATCH_SIZE][10];
	int8 linear_weight[16][10][CHANNEL_OUT_T];	// FC weight: out_channels/CHANNEL_OUT_T

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
	ini += 1;	// ini = 17
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		avgpool(msb_fmap[ini], pool_out_buf);
	}

	// FC
	int fc_weight_ptr = 0;
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


	int8 linear_weight_transpose[CHANNEL_OUT_T][10];	// FC weight transposed

	// FC_bp
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		FC_bp(error, linear_weight_transpose, pool_out_buf);
	}

	// avgpool_bp
	ini += 1;	// ini = 18
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		avgpool_bp(pool_out_buf, msb_fmap[ini]);
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

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
	LOOP_layer4_1_Conv2_bp:
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				// lsb_fmap[ini] = msb_fmap[ini];	// identity branch
				ini -= 1;	// ini = 17

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all,
					conv_1x1_weight_ptr
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t1[ini - 2], lsb_fmap[ini],	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
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
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					lsb_fmap[ini + 1], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all,
					conv_1x1_weight_ptr
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t1[ini - 2], lsb_fmap[ini],	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
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
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					lsb_fmap[ini + 1], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				load_conv_1x1_weights(
					conv_1x1_weight_tile_buffer, conv_1x1_weight_all,
					conv_1x1_weight_ptr
				);
				conv_1x1_rot_bp(	// note the index of shortcut input
					msb_fmap[ini + 1], conv_1x1_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				bn_bp(
					out_buf_t1[ini], out_buf_t1[ini - 2], lsb_fmap[ini],	// conv+bn shortcut
					gamma, grad_gamma, grad_beta,
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
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					lsb_fmap[ini + 1], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, out_buf_t1[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				shortcut(
					msb_fmap[ini + 2], out_buf_t1[ini], msb_fmap[ini],
					H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini - 1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
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

				bn_relu_bp(
					msb_fmap[ini + 1], out_buf_t0[ini - 1], relu_mask[ini - 1], out_buf_t0[ini],
					gamma, grad_gamma, grad_beta,
					H_fmap_out
				);
				load_conv_3x3_weights(
					conv_3x3_weight_tile_buffer, conv_3x3_weight_all,
					conv_3x3_weight_ptr
				);
				conv_3x3_rot_bp(
					out_buf_t0[ini], conv_3x3_weight_tile_buffer, msb_fmap[ini],
					stride, in_channels, out_channels, H_fmap_in, H_fmap_out
				);
				///////////////////////////
				// conv_3x3_weight_grad_cal
				// out_buf_t0[ini] as error (kernel), and msb_fmap[ini-1] as fmap (input)
				///////////////////////////
				conv_3x3_grad(
					msb_fmap[ini-1], out_buf_t0[ini], grad_buf_t0,
					stride, in_channels, out_channels, H_fmap_in,
					H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_tile_buffer
				);
			}
		}
    }
}	// end FracBNN_T
