#include "typedefs.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "bnn.h"
// #include "weights_tb.h"
#include <fstream>
#include <hls_math.h>
#include "dimension_def.h"
#include "weights_fracnet_64.h"
#include "conv_weights.h"

//--------------------
//   Utils Function
//--------------------

// conv weight loading from DRAM
void load_conv_3x3_weights(
	int8 weight_3x3_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 conv_3x3_weight_all[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int conv_3x3_weight_ptr
)
{
#pragma HLS ARRAY_PARTITION variable=weight_3x3_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_3x3_tile_buffer complete dim=2

	for (int c_out = 0; c_out < CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < CHANNEL_IN_T; c_in ++){
#pragma HLS PIPELINE
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					weight_3x3_tile_buffer[c_out][c_in][row][col] = conv_3x3_weight_all[conv_3x3_weight_ptr][c_out][c_in][row][col];
				}
			}
		}
	}
}

void load_conv_1x1_weights(
	int8 weight_1x1_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 conv_1x1_weight_all[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T],
	int conv_1x1_weight_ptr
)
{
#pragma HLS ARRAY_PARTITION variable=weight_1x1_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_1x1_tile_buffer complete dim=2

	for (int c_out = 0; c_out < CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < CHANNEL_IN_T; c_in ++){
#pragma HLS PIPELINE
			weight_1x1_tile_buffer[c_out][c_in] = conv_1x1_weight_all[conv_1x1_weight_ptr][c_out][c_in];
		}
	}
}

/*
// rot180 for Conv weights
void rot180_3x3(
	int8 mat[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],			// in
	int8 out_buf[CHANNEL_OUT_T][CHANNEL_IN_T][3][3]			// out, mat_rot180
)
{
	for (int n = 0; n < CHANNEL_OUT_T; n ++) {
        for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					out_buf[c][n][row][col] = mat[n][c][3-row-1][3-col-1];
				}
			}
		}
	}
}

void rot180_1x1(
	int8 mat[CHANNEL_OUT_T][CHANNEL_IN_T], 					// in
	int8 out_buf[CHANNEL_OUT_T][CHANNEL_IN_T]				// out, mat_rot180
)
{
	for (int n = 0; n < CHANNEL_OUT_T; n ++) {
        for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE
        	out_buf[c][n] = mat[n][c];
		}
	}
}
*/

// Batch Norm
void bn(
	int8 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs

	int8 gamma[CHANNEL_OUT_T],
	int8 beta[CHANNEL_OUT_T],

    int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=bn_inputs complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2

#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1

	int N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
#pragma HLS ARRAY_PARTITION variable=var complete dim=1

#pragma HLS DATAFLOW

    // calc mean
	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    mu[c] = mu[c] + bn_inputs[n][c][row][col]/N;
				}
			}
		}
	}
    // calc std variance
	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
                	var[c] = var[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
		sigma[c] = hls::sqrtf(var[c]/N);
	}
    // calc affine output
	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
					out_buf[n][c][row][col] = gamma[c]*(bn_inputs[n][c][row][col]-mu[c])/sigma[c] + beta[c];
				}
			}
		}
	}			
}

// Batch Norm Back-prop
void bn_bp(
	int8 error[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	int8 bn_inputs_fw[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	int8 gamma[CHANNEL_OUT_T],										// in
	int8 g_gamma[CHANNEL_OUT_T],									// out
	int8 g_beta[CHANNEL_OUT_T],										// out

	int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=error complete dim=2
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2

#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_beta complete dim=1

	int N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
#pragma HLS ARRAY_PARTITION variable=var complete dim=1

#pragma HLS DATAFLOW

	// calc mean
	for (int row = 0; row < H_fmap; row ++){
		for (int col = 0; col < H_fmap; col ++){
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++){
				for (int n = 0; n < BATCH_SIZE; n ++){
                    mu[c] = mu[c] + bn_inputs_fw[n][c][row][col]/N;
				}
			}
		}
	}
	// calc std variance
	for (int row = 0; row < H_fmap; row ++){
		for (int col = 0; col < H_fmap; col ++){
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++){
				for (int n = 0; n < BATCH_SIZE; n ++){
					var[c] = var[c] + (bn_inputs_fw[n][c][row][col]-mu[c])*(bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
#pragma HLS PIPELINE
    	sigma[c] = hls::sqrtf(var[c]/N);
    }
	//calc g_gamma and g_beta (also used for error calc)
	for (int row = 0; row < H_fmap; row ++){
		for (int col = 0; col < H_fmap; col ++){
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++){
				for (int n = 0; n < BATCH_SIZE; n ++){
                    g_beta[c] = g_beta[c] + error[n][c][row][col];
                    g_gamma[c] = g_gamma[c] + error[n][c][row][col]*(bn_inputs_fw[n][c][row][col]-mu[c])/sigma[c];
				}
			}
		}
	}
	// calc backprop error
	for (int row = 0; row < H_fmap; row ++){
		for (int col = 0; col < H_fmap; col ++){
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++){
				for (int n = 0; n < BATCH_SIZE; n ++){
            		out_buf[n][c][row][col] = gamma[c]*error[n][c][row][col]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (bn_inputs_fw[n][c][row][col]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
				}
			}
		}
	}			
}

/*
// ReLu
void relu(
	int8 input[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],     // in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],   // out
	int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2

	for (int n = 0; n < BATCH_SIZE; n ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
			for (int row = 0; row < H_fmap; row ++) {
				for (int col = 0; col < H_fmap; col ++) {
					if (input[n][c][row][col] > 0) {
						out_buf[n][c][row][col] = input[n][c][row][col];
					} else {
						out_buf[n][c][row][col] = 0;
					}
				}
			}
		}
	}
}

// ReLu Back-prop
void relu_bp(
	int8 error[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],  			// error in
	int8 input_fw[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH], 		// activation in foward
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],    		// error out, error_relu
	int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=error complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_fw complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2

	for (int n = 0; n < BATCH_SIZE; n ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
			for (int row = 0; row < H_fmap; row ++) {
				for (int col = 0; col < H_fmap; col ++) {
					if (input_fw[n][c][row][col] > 0) {
						out_buf[n][c][row][col] = error[n][c][row][col];
					} else {
						out_buf[n][c][row][col] = 0;
					}
				}
			}
		}
	}
}
*/

// AvgPool
void avgpool(
	int8 avg_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T] 						// out, avg_outputs

	//int stride,
	//int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=avg_inputs complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2

	// int H_fmap_OUT = H_fmap/stride;
	// int stride2 = stride*stride;
	// int8 accum;

	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
		for (int n = 0; n < BATCH_SIZE; n ++) {
			for (int s = 0; s < 4; s ++) {
				for (int ss = 0; ss < 4; s ++) {
					out_buf[n][c] += avg_inputs[n][c][s][ss]/16;
				}
			}
		}
	}
}

// AvgPool Back-prop
void avgpool_bp(
	int8 error[BATCH_SIZE][CHANNEL_OUT_T],							// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH]			// out, error_avg
	// int stride
	// int H_fmap	// #(-1,512,1,1)
)
{
#pragma HLS ARRAY_PARTITION variable=error complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2

	// int H_fmap_OUT = stride;
	// int stride2 = stride*stride;

	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
		for (int n = 0; n < BATCH_SIZE; n ++) {
			for (int s = 0; s < 4; s ++) {
				for (int ss = 0; ss < 4; ss ++) {
					out_buf[n][c][s][ss] = error[n][c]/16;
				}
			}
		}
	}
}

// FC (no bias)
void FC(
	int8 inputs[BATCH_SIZE][CHANNEL_OUT_T],
	int8 linear_weight[10][CHANNEL_OUT_T],
	int8 outputs[BATCH_SIZE][10]
)
{
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=2
#pragma HLS ARRAY_PARTITION variable=linear_weight complete dim=2

	for (int cii = 0; cii < CHANNEL_OUT_T; cii++) {
#pragma HLS PIPELINE
		for (int coo = 0; coo < 10; coo ++) {
			for (int bii = 0; bii < BATCH_SIZE; bii++) {
				outputs[bii][coo] += inputs[bii][cii] * linear_weight[coo][cii];
			}
		}
	}
}

// FC Back-prop (no bias)
void FC_bp(
	int8 inputs[BATCH_SIZE][10],
	int8 linear_weight_transpose[CHANNEL_OUT_T][10],
	int8 outputs[BATCH_SIZE][CHANNEL_OUT_T]
)
{
#pragma HLS ARRAY_PARTITION variable=linear_weight_transpose complete dim=1
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=2

	for (int coo = 0; coo < CHANNEL_OUT_T; coo ++) {
#pragma HLS PIPELINE
		for (int cii = 0; cii < 10; cii++) {
			for (int bii = 0; bii < BATCH_SIZE; bii++) {
				outputs[bii][coo] += inputs[bii][cii] * linear_weight_transpose[coo][cii];
			}
		}
	}
}

// Shortcut- identity branch
void shortcut(
	int8 input_a[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	int8 input_b[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=input_a complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_b complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2

	// int8 out_feature_a[BATCH_SIZE][CHANNEL_OUT_T];
	// int8 out_feature_b[BATCH_SIZE][CHANNEL_OUT_T];

	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
		for (int row = 0; row < H_fmap; row ++) {
			for (int col = 0; col < H_fmap; col ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
					out_buf[n][c][row][col] = input_a[n][c][row][col] + input_b[n][c][row][col];
				}
			}
		}
	}
}

// ============
// Conv forward
// ============

// Conv_3x3, padding=1
void conv_3x3
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	
	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2

	// input 0-padding(1, 1, 1, 1)
	// conv
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
#pragma HLS PIPELINE
			for (int row = 0; row < H_fmap_out; row ++) {
				for (int col = 0; col < H_fmap_out; col ++) {
					for (int bi = 0; bi < BATCH_SIZE; bi ++) {
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
								int row_in = row*stride + krow - 1;		// -1 due to 0-padding
								int col_in = col*stride + kcol - 1;
								// if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									output[bi][co][row][col] += input[bi][ci][row_in][col_in] * weight[co][ci][krow][kcol];
								// }
							}
						}
					}
				}
			}
		}
	}
}

// Conv_1x1, padding=0
void conv_1x1
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T], //[1][1],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2

	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
#pragma HLS PIPELINE
			for (int row = 0; row < H_fmap_out; row++) {
				for (int col = 0; col < H_fmap_out; col++) {
					for (int bi = 0; bi < BATCH_SIZE; bi++) {
						int row_in = row*stride;	// krow = kcol = 0
						int col_in = col*stride;
						// if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							output[bi][co][row][col] += input[bi][ci][row_in][col_in] * weight[co][ci]; //[0][0];
						// }
					}
				}
			}
		}
	}
}

// =============
// Conv backward
// =============

// conv_3x3_rot_bp, padding=1
void conv_3x3_rot_bp
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2
/*
	// weight rot180
	int8 weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
	for (int n = 0; n < CHANNEL_OUT_T; n ++) {
        for (int c = 0; c < CHANNEL_IN_T; c ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					weight_rot[c][n][row][col] = weight[n][c][3-row-1][3-col-1];
				}
			}
		}
	}
*/
	// input dilation and 0-padding(1, 1+A, 1, 1+A)
	int8 input_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
	int8 weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
#pragma HLS ARRAY_PARTITION variable=input_dil complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_dil complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_rot complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_rot complete dim=2

#pragma HLS DATAFLOW

	for (int co = 0; co < CHANNEL_OUT_T; co++) {
#pragma HLS PIPELINE
		for (int row = 0; row < H_fmap_in; row++) {
			for (int col = 0; col < H_fmap_in; col++) {
				for (int bi = 0; bi < BATCH_SIZE; bi++) {
					input_dil[bi][co][row*stride + 1][col*stride + 1] = input[bi][co][row][col];	// +1 due to 0-padding
				}
			}
		}

		// weight rot180
		for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					weight_rot[c][co][row][col] = weight[co][c][3-row-1][3-col-1];
				}
			}
		}
	}

	// conv
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
#pragma HLS PIPELINE
			for (int row = 0; row < H_fmap_out; row ++) {
				for (int col = 0; col < H_fmap_out; col ++) {
					for (int bi = 0; bi < BATCH_SIZE; bi ++) {
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
								int row_in = row + krow;	// stride 1 transposed conv
								int col_in = col + kcol;
								// if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
								output[bi][co][row][col] += input_dil[bi][ci][row_in][col_in] * weight_rot[co][ci][krow][kcol];
								//}
							}
						}
					}
				}
			}
		}
	}
}

// conv_1x1_rot_bp, padding=0
void conv_1x1_rot_bp
(
	int8 input[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	int8 weight[CHANNEL_OUT_T][CHANNEL_OUT_T],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2
/*
	for (int n = 0; n < CHANNEL_OUT_T; n ++) {
		for (int c = 0; c < CHANNEL_IN_T; c ++) {
			weight_rot[c][n] = weight[n][c];
		}
	}
*/
	// weight rot180
	int8 weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T];
	// input dilation
	int8 input_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
#pragma HLS ARRAY_PARTITION variable=weight_rot complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_rot complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_dil complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_dil complete dim=2

#pragma HLS DATAFLOW

	for (int co = 0; co < CHANNEL_OUT_T; co++) {
#pragma HLS PIPELINE
		for (int row = 0; row < H_fmap_in; row++) {
			for (int col = 0; col < H_fmap_in; col++) {
				for (int bi = 0; bi < BATCH_SIZE; bi++) {
					input_dil[bi][co][row*stride][col*stride] = input[bi][co][row][col];
				}
			}
		}
		// weight rot180
		for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE
			weight_rot[c][co] = weight[co][c];
		}
	}

	// conv
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
#pragma HLS PIPELINE
			for (int row = 0; row < H_fmap_out; row ++) {
				for (int col = 0; col < H_fmap_out; col ++) {
					for (int bi = 0; bi < BATCH_SIZE; bi ++) {
						int row_in = row;	// stride 1 transposed conv, krow = kcol = 0
						int col_in = col;
						output[bi][co][row][col] += input_dil[bi][ci][row_in][col_in] * weight_rot[co][ci];
					}
				}
			}
		}
	}
}

// =============
// Conv gradient
// =============

// Conv_3x3_grad, padding=1
void conv_3x3_grad
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int8 weight[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int k_row_in	// weight size (error as weight), k_row_in = H_fmap_in
)
{
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=1
#pragma HLS ARRAY_PARTITION variable=output complete dim=2

#pragma HLS DATAFLOW

	// weight dilation, k_dil = 1 + (k-1)*s
	int KERNEL_DIL = (k_row_in-1)*stride + 1;
	int8 weight_dil[BATCH_SIZE][CHANNEL_OUT_T][2*WIDTH][2*WIDTH];	// larger buffer for dilated weights
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS PIPELINE
		for (int krow = 0; krow < k_row_in; krow ++) {
			for (int kcol = 0; kcol < k_row_in; kcol ++) {
				for (int bi = 0; bi < BATCH_SIZE; bi ++) {
					weight_dil[bi][co][krow*stride][kcol*stride] = weight[bi][co][krow][kcol];
				}
			}
		}
	}

	// input 0-padding(1, 1+A, 1, 1+A)
	// conv
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
#pragma HLS PIPELINE
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					for (int krow = 0; krow < KERNEL_DIL; krow ++) {
						for (int kcol = 0; kcol < KERNEL_DIL; kcol ++) {
							for (int bi = 0; bi < BATCH_SIZE; ci ++) {
								int row_in = row*stride + krow - 1;		// -1 due to 0-padding
								int col_in = col*stride + kcol - 1;
								// if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
								output[co][ci][row][col] += input[bi][ci][row_in][col_in] * weight_dil[bi][co][krow][kcol];
								// }
							}
						}
					}
				}
			}
		}
	}
}

// Conv_1x1_grad, padding=0
void conv_1x1_grad
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int8 weight[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T],
	
	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int k_row_in	// weight size (error as weight), k_row_in = H_fmap_in
)
{
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=1
#pragma HLS ARRAY_PARTITION variable=output complete dim=2

#pragma HLS DATAFLOW

	// weight dilation, k_dil = 1 + (k-1)*s
	int KERNEL_DIL = (k_row_in-1)*stride + 1;
	int8 weight_dil[BATCH_SIZE][CHANNEL_OUT_T][2*WIDTH][2*WIDTH];	// larger buffer for dilated weights
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS PIPELINE
		for (int krow = 0; krow < k_row_in; krow ++) {
			for (int kcol = 0; kcol < k_row_in; kcol ++) {
				for (int bi = 0; bi < BATCH_SIZE; bi ++) {
					weight_dil[bi][co][krow*stride][kcol*stride] = weight[bi][co][krow][kcol];
				}
			}
		}
	}

	// no 0-padding
	// conv
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
#pragma HLS PIPELINE
			for (int bi = 0; bi < BATCH_SIZE; ci++) {
				for (int krow = 0; krow < KERNEL_DIL; krow++) {
					for (int kcol = 0; kcol < KERNEL_DIL; kcol++) {
						// if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in) {
						output[co][ci] += input[bi][ci][krow][kcol] * weight_dil[bi][co][krow][kcol];
						// }
					}
				}
			}
		}
	}
}

// SGD conv_3x3 weight update
void SGD_WU_3x3
(
	int8 gradient[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 weight_WU[CHANNEL_OUT_T][CHANNEL_IN_T][3][3]
)
{
#pragma HLS ARRAY_PARTITION variable=gradient complete dim=1
#pragma HLS ARRAY_PARTITION variable=gradient complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_WU complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_WU complete dim=2

	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
#pragma HLS PIPELINE
			for (int krow = 0; krow < 3; krow++) {
				for (int kcol = 0; kcol < 3; kcol++) {
					weight_WU[co][ci][krow][kcol] = weight[co][ci][krow][kcol] - lr*gradient[co][ci][krow][kcol];
				}
			}
		}
	}
}

// SGD conv_1x1 weight update
void SGD_WU_1x1
(
	int8 gradient[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 weight_WU[CHANNEL_OUT_T][CHANNEL_IN_T]
)
{
#pragma HLS ARRAY_PARTITION variable=gradient complete dim=1
#pragma HLS ARRAY_PARTITION variable=gradient complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_WU complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_WU complete dim=2

	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
#pragma HLS PIPELINE
			weight_WU[co][ci] = weight[co][ci] - lr*gradient[co][ci];
		}
	}
}

/*
void load_image()
{    
	std::ifstream ifs_param("conv1_input.bin", std::ios::in | std::ios::binary);
	ifs_param.read((char*)(images), sizeof(unsigned char)*96*NUM_TESTS*32*32);
	ifs_param.close();
}

void load_label()
{    
	std::ifstream ifs_param("labels.bin", std::ios::in | std::ios::binary);
	ifs_param.read((char*)(labels), sizeof(unsigned char)*NUM_TESTS);
	ifs_param.close();
}

void get_image(unsigned char *images, unsigned int idx, float image[96][32][32])
{
	unsigned int offset = idx*96*32*32;
	for (int c = 0; c < 96; c ++) {
		for (int row = 0; row < 32; row ++) {
			for (int col = 0; col < 32; col ++) {
				image[c][row][col] = images[offset + c*32*32 + row*32 + col];
			}
		}
	}
}
*/

//--------------------
//   Fused Function
//--------------------

// Fused bn + relu
void bn_relu(
	int8 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs
	int1 relu_mask[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp

	int8 gamma[CHANNEL_OUT_T],
	int8 beta[CHANNEL_OUT_T],

    int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=bn_inputs complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=relu_mask complete dim=2

#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1

	int N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
#pragma HLS ARRAY_PARTITION variable=var complete dim=1

#pragma HLS DATAFLOW

    // calc mean
	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    mu[c] = mu[c] + bn_inputs[n][c][row][col]/N;
				}
			}
		}
	}
    // calc std variance
	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
                	var[c] = var[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
#pragma HLS PIPELINE
		sigma[c] = hls::sqrtf(var[c]/N);
	}
    // calc affine output
	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
            		out_buf[n][c][row][col] = gamma[c]*(bn_inputs[n][c][row][col]-mu[c])/sigma[c] + beta[c];
					// relu mask
					if (out_buf[n][c][row][col] > 0) {
						relu_mask[n][c][row][col] = 1;
					} else {
						relu_mask[n][c][row][col] = 0;
					}
					out_buf[n][c][row][col] = out_buf[n][c][row][col] * relu_mask[n][c][row][col];
				}
			}
		}
	}			
}

// Fused relu_bp + bn_bp
inline void bn_relu_bp(
	int8 error[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	int8 bn_inputs_fw[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int1 relu_mask[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	int8 gamma[CHANNEL_OUT_T],										// in
	int8 g_gamma[CHANNEL_OUT_T],									// out
	int8 g_beta[CHANNEL_OUT_T],										// out

	int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=error complete dim=2
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=relu_mask complete dim=2

#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_beta complete dim=1

	int N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
#pragma HLS ARRAY_PARTITION variable=var complete dim=1

#pragma HLS DATAFLOW

	// calc mean and relu_bp
	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
					// mean
					mu[c] = mu[c] + bn_inputs_fw[n][c][row][col]/N;
					// relu
					error[n][c][row][col] = relu_mask[n][c][row][col] * error[n][c][row][col];
				}
			}
		}
	}

	// calc std variance
	for (int row = 0; row < H_fmap; row ++){
		for (int col = 0; col < H_fmap; col ++){
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++){
				for (int n = 0; n < BATCH_SIZE; n ++){
                    var[c] = var[c] + (bn_inputs_fw[n][c][row][col]-mu[c]) * (bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
    	sigma[c] = hls::sqrtf(var[c]/N);
    }
	//calc g_gamma and g_beta (also used for error calc)
	for (int row = 0; row < H_fmap; row ++){
		for (int col = 0; col < H_fmap; col ++){
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++){
				for (int n = 0; n < BATCH_SIZE; n ++){
                    g_beta[c] = g_beta[c] + error[n][c][row][col];
                    g_gamma[c] = g_gamma[c] + error[n][c][row][col] * (bn_inputs_fw[n][c][row][col]-mu[c])/sigma[c];
				}
			}
		}
	}
	// calc backprop error
	for (int row = 0; row < H_fmap; row ++){
		for (int col = 0; col < H_fmap; col ++){
#pragma HLS PIPELINE
			for (int c = 0; c < CHANNEL_OUT_T; c ++){
				for (int n = 0; n < BATCH_SIZE; n ++){
            		out_buf[n][c][row][col] = gamma[c]*error[n][c][row][col]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (bn_inputs_fw[n][c][row][col]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
				}
			}
		}
	}			
}

/*
// Fused bn + relu + shortcut, only for forward (since backward is relu+bn+conv+shortcut)
inline void bn_relu_shortcut(
	int8 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs
	int1 relu_mask[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp
	int8 input_b[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in2 for shortcut

	int8 gamma[CHANNEL_OUT_T],
	int8 beta[CHANNEL_OUT_T],

    int H_fmap
)
{
#pragma HLS ARRAY_PARTITION variable=bn_inputs complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_b complete dim=2
#pragma HLS ARRAY_PARTITION variable=relu_mask complete dim=2

#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1

    int N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
#pragma HLS ARRAY_PARTITION variable=var complete dim=1

    // calc mean
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
			for (int col = 0; col < H_fmap; col ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    mu[c] = mu[c] + bn_inputs[n][c][row][col]/N;
				}
			}
		}
	}
    // calc std variance
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
			for (int col = 0; col < H_fmap; col ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
                	var[c] = var[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
		sigma[c] = hls::sqrtf(var[c]/N);
	}

    // calc affine output
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
			for (int col = 0; col < H_fmap; col ++) {
				for (int n = 0; n < BATCH_SIZE; n ++) {
            		out_buf[n][c][row][col] = gamma[c]*(bn_inputs[n][c][row][col]-mu[c])/sigma[c] + beta[c];
					// relu mask + shortcut
					if (out_buf[n][c][row][col] > 0) {
						relu_mask[n][c][row][col] = 1;
					} else {
						relu_mask[n][c][row][col] = 0;
					}
					out_buf[n][c][row][col] = out_buf[n][c][row][col] * relu_mask[n][c][row][col] + input_b[n][c][row][col];
				}
			}
		}
	}	
}
*/


