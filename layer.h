#ifndef LAYER_H
#define LAYER_H

#include "typedefs.h"
#include "dimension_def.h"
#include <iostream>
//#include <stdlib.h>
//#include <math.h>
//#include <fstream>
#include <hls_math.h>

//--------------------
//   Utils Function
//--------------------

// identity shortcut
void identity_shortcut(
	int8 msb_in[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int8 lsb_out[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int H_fmap_in
)
{
// #pragma HLS ARRAY_PARTITION variable=msb_in complete dim=2

	for (int row = 0; row < H_fmap_in; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                	lsb_out[n][c][row][col] = msb_in[n][c][row][col];
				}
			}
		}
	}
}

// fc weight loading from DRAM
void load_fc_weights(
	int8 linear_weight_tile_buffer[10][512],
	int8 linear_weight[10][512]
)
{
// #pragma HLS ARRAY_PARTITION variable=linear_weight complete dim=2

	for (int col = 0; col < 512; col ++) {
#pragma HLS PIPELINE
		for (int row = 0; row < 10; row ++) {
			linear_weight_tile_buffer[row][col] = linear_weight[row][col];
		}
	}
}

// conv weight loading from DRAM
void load_conv_3x3_weights(
	int8 weight_3x3_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 conv_3x3_weight_all[CHANNEL_OUT_T][CHANNEL_IN_T][3][3]
)
{
// #pragma HLS ARRAY_PARTITION variable=conv_3x3_weight_all complete dim=1
// #pragma HLS ARRAY_PARTITION variable=conv_3x3_weight_all complete dim=2

	for (int c_out = 0; c_out < CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < CHANNEL_IN_T; c_in ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
#pragma HLS PIPELINE
					weight_3x3_tile_buffer[c_out][c_in][row][col] = conv_3x3_weight_all[c_out][c_in][row][col];
				}
			}
		}
	}
}

void load_conv_1x1_weights(
	int8 weight_1x1_tile_buffer[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 conv_1x1_weight_all[CHANNEL_OUT_T][CHANNEL_IN_T]
)
{
// #pragma HLS ARRAY_PARTITION variable=conv_1x1_weight_all complete dim=1
// #pragma HLS ARRAY_PARTITION variable=conv_1x1_weight_all complete dim=2

	for (int c_out = 0; c_out < CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < CHANNEL_IN_T; c_in ++) {
#pragma HLS PIPELINE
			weight_1x1_tile_buffer[c_out][c_in] = conv_1x1_weight_all[c_out][c_in];
		}
	}
}

// Batch Norm
void bn(
	int8 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out

	int8 gamma[CHANNEL_OUT_T],
	int8 beta[CHANNEL_OUT_T],

    int H_fmap
)
{
// #pragma HLS ARRAY_PARTITION variable=bn_inputs complete dim=2
// #pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=beta complete dim=1

	int8 N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
// #pragma HLS ARRAY_PARTITION variable=mu complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=var complete dim=1

    // calc mean
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    mu[c] = mu[c] + bn_inputs[n][c][row][col]/N;
                    // var[c] = var[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
	// /*
    // calc std variance
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                	var[c] = var[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
	// */
    for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		sigma[c] = hls::sqrt(var[c]/N);
	}
    // calc affine output
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
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
// #pragma HLS ARRAY_PARTITION variable=error complete dim=2
// #pragma HLS ARRAY_PARTITION variable=bn_inputs_fw complete dim=2

// #pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=g_gamma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=g_beta complete dim=1

	int8 N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
// #pragma HLS ARRAY_PARTITION variable=mu complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=var complete dim=1

	// calc mean
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    mu[c] = mu[c] + bn_inputs_fw[n][c][row][col]/N;
                    // var[c] = var[c] + (bn_inputs_fw[n][c][row][col]-mu[c])*(bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
		}
	}
	// /*
	// calc std variance
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
					var[c] = var[c] + (bn_inputs_fw[n][c][row][col]-mu[c])*(bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
		}
	}
	// */
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
    	sigma[c] = hls::sqrt(var[c]/N);
    }
	//calc g_gamma and g_beta (also used for error calc)
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    g_beta[c] = g_beta[c] + error[n][c][row][col];
                    g_gamma[c] = g_gamma[c] + error[n][c][row][col]*(bn_inputs_fw[n][c][row][col]-mu[c])/sigma[c];
				}
			}
		}
	}
	// calc backprop error
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
            		// out_buf[n][c][row][col] = gamma[c]*error[n][c][row][col]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (bn_inputs_fw[n][c][row][col]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
					out_buf[n][c][row][col] = gamma[c]*error[n][c][row][col]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (bn_inputs_fw[n][c][row][col]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
				}
			}
		}
	}
}

// AvgPool
void avgpool(
	int8 avg_inputs[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],	// in
	int8 out_buf[BATCH_SIZE][512],						// out, avg_outputs

	int1 ctrl_avgpool,	// 0 for forward and 1 for backward
	int c_out
)
{
// #pragma HLS ARRAY_PARTITION variable=avg_inputs complete dim=2

	for (int ci = 0; ci < 512; ci ++) {
#pragma HLS PIPELINE
		for (int bi = 0; bi < BATCH_SIZE; bi ++) {
			if (ctrl_avgpool == 0) {
				out_buf[bi][ci] = 0;
			} else {
				for (int row = 0; row < 4; row ++) {
					for (int col = 0; col < 4; col ++) {
						avg_inputs[bi][ci][row][col] = 0;
					}
				}
			}
		}
	}

	// int H_fmap_OUT = H_fmap/stride;
	// int stride2 = stride*stride;
	for (int c = 0; c < 512; c ++) {
		for (int s = 0; s < 4; s ++) {
			for (int ss = 0; ss < 4; ss ++) {
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
					if (ctrl_avgpool == 0) {
						out_buf[n][c + c_out*CHANNEL_IN_T] += avg_inputs[n][c][s][ss]/16;
					} else {
						avg_inputs[n][c][s][ss] = out_buf[n][c + c_out*CHANNEL_IN_T]/16;
					}
				}
			}
		}
	}
}

/*
// AvgPool Back-prop
void avgpool_bp(
	int8 error[BATCH_SIZE][CHANNEL_IN_T],							// in
	int8 out_buf[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],			// out, error_avg

	int  c_out
	// int stride
	// int H_fmap	// #(-1,512,1,1)
)
{
// #pragma HLS ARRAY_PARTITION variable=error complete dim=2

	// int H_fmap_OUT = stride;
	// int stride2 = stride*stride;
	for (int c = 0; c < CHANNEL_IN_T; c ++) {
		for (int s = 0; s < 4; s ++) {
			for (int ss = 0; ss < 4; ss ++) {
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
					out_buf[n][c][s][ss] = error[n][c]/16;
				}
			}
		}
	}
}
*/

// FC (no bias)
void FC(
	int8 inputs[BATCH_SIZE][512],
	int8 linear_weight[10][512],
	int8 outputs[BATCH_SIZE][10],

	int1 ctrl_fc	// 0 for forward and 1 for backward
)
{
// #pragma HLS ARRAY_PARTITION variable=inputs complete dim=2
// #pragma HLS ARRAY_PARTITION variable=linear_weight complete dim=2

	for (int bi = 0; bi < BATCH_SIZE; bi ++) {
		if (ctrl_fc == 0) {
			for (int ii = 0; ii < 10; ii ++) {
#pragma HLS PIPELINE
				outputs[bi][ii] = 0;
			}
		} else {
			for (int co = 0; co < 512; co ++) {
#pragma HLS PIPELINE
				inputs[bi][co] = 0;
			}
		}
	}

	for (int cii = 0; cii < 512; cii++) {
		for (int coo = 0; coo < 10; coo ++) {
#pragma HLS PIPELINE
			for (int bii = 0; bii < BATCH_SIZE; bii++) {
				if (ctrl_fc == 0) {
					outputs[bii][coo] += inputs[bii][cii] * linear_weight[coo][cii];
				} else {
					inputs[bii][coo] += outputs[bii][coo] * linear_weight[coo][cii];
				}
			}
		}
	}
}

/*
// FC Back-prop (no bias)
void FC_bp(
	int8 inputs[BATCH_SIZE][10],
	// int8 linear_weight_transpose[CHANNEL_OUT_T][10],
	int8 linear_weight[10][CHANNEL_OUT_T],
	int8 outputs[BATCH_SIZE][CHANNEL_OUT_T]
)
{
// #pragma HLS ARRAY_PARTITION variable=linear_weight complete dim=2

	for (int cii = 0; cii < CHANNEL_OUT_T; cii ++) {
		for (int coo = 0; coo < 10; coo++) {
#pragma HLS PIPELINE
			for (int bii = 0; bii < BATCH_SIZE; bii++) {
				outputs[bii][coo] += inputs[bii][coo] * linear_weight[coo][cii];
			}
		}
	}
}
*/

// Shortcut- identity branch
void shortcut(
	int8 input_a[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	int8 input_b[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	int8 out_buf_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip

	int H_fmap,
	int1 ctrl_sc	// if ctrl_sc=1, generate and send out_copy into DDR
)
{
// #pragma HLS ARRAY_PARTITION variable=input_a complete dim=2
// #pragma HLS ARRAY_PARTITION variable=input_b complete dim=2

	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
					out_buf[n][c][row][col] = input_a[n][c][row][col] + input_b[n][c][row][col];
					if (ctrl_sc == 1) {
						out_buf_DDR[n][c][row][col] = out_buf[n][c][row][col];			
					}
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
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// activations on-chip for inference
	int8 output_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// activations off-chip for backprop
	
	int stride,
	// int H_fmap_in,
	int H_fmap_out,
	int c_tile	// c_in in forward and c_out in backward
)
{
// #pragma HLS ARRAY_PARTITION variable=input complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=2
	if (c_tile == 0){
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
				for (int col = 0; col < H_fmap_out; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
					for (int bi = 0; bi < BATCH_SIZE; bi ++) {
						output[bi][co][row][col] = 0;
					}
				}
			}
		}
	}

	// input 0-padding(1, 1, 1, 1)
	// conv (weight stationary)
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
					for (int row = 0; row < H_fmap_out; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
						for (int col = 0; col < H_fmap_out; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
							for (int bi = 0; bi < BATCH_SIZE; bi ++) {
								int row_in = row*stride + krow;		
								int col_in = col*stride + kcol;
								output[bi][co][row][col] += input[bi][ci][row_in][col_in] * weight[co][ci][krow + 1][kcol + 1];		// +1 due to 0-padding
								output_DDR[bi][co][row][col] = output[bi][co][row][col];
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
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// activations on-chip for inference
	int8 output_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// activations off-chip for backprop

	int stride,
	// int H_fmap_in,
	int H_fmap_out,
	int c_tile
)
{
// #pragma HLS ARRAY_PARTITION variable=input complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=2

	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int row = 0; row < H_fmap_out; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap_out; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
				for (int bi = 0; bi < BATCH_SIZE; bi ++) {
					if (c_tile == 0){
						output[bi][co][row][col] = 0;
					}
				}
			}
		}
	}

	// conv (weight stationary)
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			for (int row = 0; row < H_fmap_out; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
				for (int col = 0; col < H_fmap_out; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
					for (int bi = 0; bi < BATCH_SIZE; bi++) {
						int row_in = row*stride;	// krow = kcol = 0
						int col_in = col*stride;
						output[bi][co][row][col] += input[bi][ci][row_in][col_in] * weight[co][ci]; //[0][0];
						output_DDR[bi][co][row][col] = output[bi][co][row][col];
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
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],		// error in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// error out on-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_tile,
	int out_channels_after_pack
)
{
// #pragma HLS ARRAY_PARTITION variable=input complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=2

	// input dilation and 0-padding(1, 1+A, 1, 1+A)
	int8 input_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
	int8 weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
// #pragma HLS ARRAY_PARTITION variable=input_dil complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight_rot complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight_rot complete dim=2

	if (c_tile == out_channels_after_pack - 1) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
				for (int col = 0; col < H_fmap_out; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
					for (int bi = 0; bi < BATCH_SIZE; bi ++) {
						output[bi][co][row][col] = 0;
					}
				}
			}
		}
	}

	for (int row = 0; row < H_fmap_in; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
		for (int col = 0; col < H_fmap_in; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int co = 0; co < CHANNEL_OUT_T; co++) {
				// weight rot180
				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
#pragma HLS PIPELINE
					for (int row = 0; row < 3; row ++) {
						for (int col = 0; col < 3; col ++) {
							weight_rot[cin][co][row][col] = weight[co][cin][3-row-1][3-col-1];
						}
					}
				}
				// input dilation and padding
				for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
					input_dil[bi][co][row*stride + 1][col*stride + 1] = input[bi][co][row][col];	// +1 due to 0-padding
				}
			}
		}
	}

	// conv (weight stationary)
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
					for (int row = 0; row < H_fmap_out; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
						for (int col = 0; col < H_fmap_out; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
							for (int bi = 0; bi < BATCH_SIZE; bi ++) {
								int row_in = row + krow;	// stride 1 transposed conv
								int col_in = col + kcol;
								output[bi][co][row][col] += input_dil[bi][ci][row_in][col_in] * weight_rot[co][ci][krow][kcol];
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
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],		// error in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// error out on-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_tile,
	int out_channels_after_pack
)
{
// #pragma HLS ARRAY_PARTITION variable=input complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=2

	// weight rot180 & input dilation
	int8 weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T];
	int8 input_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
// #pragma HLS ARRAY_PARTITION variable=weight_rot complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight_rot complete dim=2
// #pragma HLS ARRAY_PARTITION variable=input_dil complete dim=2

	if (c_tile == out_channels_after_pack - 1) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
				for (int col = 0; col < H_fmap_out; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
					for (int bi = 0; bi < BATCH_SIZE; bi ++) {
						output[bi][co][row][col] = 0;
					}
				}
			}
		}
	}

	for (int row = 0; row < H_fmap_in; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
		for (int col = 0; col < H_fmap_in; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int co = 0; co < CHANNEL_OUT_T; co++) {
				// weight rot180
				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
#pragma HLS PIPELINE
					weight_rot[cin][co] = weight[co][cin];
				}
				// input dilation
				for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
					input_dil[bi][co][row*stride][col*stride] = input[bi][co][row][col];
				}
			}
		}
	}

	// conv (weight stationary)
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
				for (int col = 0; col < H_fmap_out; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
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
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],		// activation from DDR
	int8 weight[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// error on-chip
	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],			// gradient on-chip

	int stride,
	int H_fmap_in
	// int k_row_in	// weight size (error as weight), k_row_in = H_fmap_in
)
{
// #pragma HLS ARRAY_PARTITION variable=input complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=2

	// initialize grad_buf
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
#pragma HLS PIPELINE
					for (int bi = 0; bi < BATCH_SIZE; bi ++) {
						output[co][ci][row][col] = 0;
					}
				}
			}
		}
	}

	// weight dilation, k_dil = 1 + (k-1)*s
	int8 KERNEL_DIL = (H_fmap_in-1)*stride + 1;	// max={32, (16-1)*2+1=31} < WIDTH
	int8 weight_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];	// buffer for dilated weights
	
	for (int krow = 0; krow < H_fmap_in; krow ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
		for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS PIPELINE
				for (int bi = 0; bi < BATCH_SIZE; bi ++) {
					weight_dil[bi][co][krow*stride][kcol*stride] = weight[bi][co][krow][kcol];
				}
			}
		}
	}

	// input 0-padding(1, 1+A, 1, 1+A)
	// conv stride 1 (output stationary)
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					for (int krow = 0; krow < KERNEL_DIL; krow ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
						for (int kcol = 0; kcol < KERNEL_DIL; kcol ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
							for (int bi = 0; bi < BATCH_SIZE; bi ++) {
								int row_in = row + krow;
								int col_in = col + kcol;
								output[co][ci][row][col] += input[bi][ci][row_in][col_in] * weight_dil[bi][co][krow + 1][kcol + 1]; // +1 due to 0-padding
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
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],		// activation from DDR
	int8 weight[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// error on-chip
	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T],				// gradient on-chip
	
	int stride,
	int H_fmap_in
	// int k_row_in	// weight size (error as weight), k_row_in = H_fmap_in
)
{
// #pragma HLS ARRAY_PARTITION variable=input complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=2

	// initialize grad_buf
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
#pragma HLS PIPELINE
			for (int bi = 0; bi < BATCH_SIZE; bi ++) {
				output[co][ci] = 0;
			}
		}
	}

	// weight dilation, k_dil = 1 + (k-1)*s
	int8 KERNEL_DIL = (H_fmap_in-1)*stride + 1;
	int8 weight_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];	// buffer for dilated weights
	
	for (int krow = 0; krow < H_fmap_in; krow ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
		for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS PIPELINE
				for (int bi = 0; bi < BATCH_SIZE; bi ++) {
					weight_dil[bi][co][krow*stride][kcol*stride] = weight[bi][co][krow][kcol];
				}
			}
		}
	}

	// no 0-padding
	// conv stride 1 (output stationary)
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			for (int krow = 0; krow < KERNEL_DIL; krow++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
				for (int kcol = 0; kcol < KERNEL_DIL; kcol++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
					for (int bi = 0; bi < BATCH_SIZE; bi++) {
						output[co][ci] += input[bi][ci][krow][kcol] * weight_dil[bi][co][krow][kcol];
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
// #pragma HLS ARRAY_PARTITION variable=gradient complete dim=1
// #pragma HLS ARRAY_PARTITION variable=gradient complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=2

	int8 temp[3][3];
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			for (int krow = 0; krow < 3; krow++) {
#pragma HLS PIPELINE
				for (int kcol = 0; kcol < 3; kcol++) {
					// weight_WU[co][ci][krow][kcol] = weight[co][ci][krow][kcol] - lr*gradient[co][ci][krow][kcol];
					temp[krow][kcol] = weight[co][ci][krow][kcol] - lr*gradient[co][ci][krow][kcol];
					weight_WU[co][ci][krow][kcol] = temp[krow][kcol];
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
// #pragma HLS ARRAY_PARTITION variable=gradient complete dim=1
// #pragma HLS ARRAY_PARTITION variable=gradient complete dim=2
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight complete dim=2

	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			weight_WU[co][ci] = weight[co][ci] - lr*gradient[co][ci];
		}
	}
}

//--------------------
//   Fused Function
//--------------------

// Fused bn + relu
void bn_relu(
	int8 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs
	int8 out_buf_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip for backprop
	int1 relu_mask[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp

	int8 gamma[CHANNEL_OUT_T],
	int8 beta[CHANNEL_OUT_T],

    int H_fmap
)
{
// #pragma HLS ARRAY_PARTITION variable=bn_inputs complete dim=2
// #pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=beta complete dim=1

	int8 N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
// #pragma HLS ARRAY_PARTITION variable=mu complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=var complete dim=1

    // calc mean
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    mu[c] = mu[c] + bn_inputs[n][c][row][col]/N;
                    // var[c] = var[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
	// /*
    // calc std variance
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                	var[c] = var[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
	// */
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
		sigma[c] = hls::sqrt(var[c]/N);
	}
    // calc affine output
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
            		out_buf[n][c][row][col] = gamma[c]*(bn_inputs[n][c][row][col]-mu[c])/sigma[c] + beta[c];
					// relu mask
					if (out_buf[n][c][row][col] > 0) {
						relu_mask[n][c][row][col] = 1;
					} else {
						relu_mask[n][c][row][col] = 0;
					}
					out_buf[n][c][row][col] = out_buf[n][c][row][col] * relu_mask[n][c][row][col];
					out_buf_DDR[n][c][row][col] = out_buf[n][c][row][col];
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
// #pragma HLS ARRAY_PARTITION variable=error complete dim=2
// #pragma HLS ARRAY_PARTITION variable=bn_inputs_fw complete dim=2

// #pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=g_gamma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=g_beta complete dim=1

	int8 N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 var[CHANNEL_OUT_T];
// #pragma HLS ARRAY_PARTITION variable=mu complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=var complete dim=1

	// calc mean and relu_bp
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
					// mean
					mu[c] = mu[c] + bn_inputs_fw[n][c][row][col]/N;
					// var[c] = var[c] + (bn_inputs_fw[n][c][row][col]-mu[c]) * (bn_inputs_fw[n][c][row][col]-mu[c]);
					// relu
					error[n][c][row][col] = relu_mask[n][c][row][col] * error[n][c][row][col];
				}
			}
		}
	}
	// /*
	// calc std variance
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    var[c] = var[c] + (bn_inputs_fw[n][c][row][col]-mu[c]) * (bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
		}
	}
	// */
    for (int c = 0; c < CHANNEL_OUT_T; c ++) {
    	sigma[c] = hls::sqrt(var[c]/N);
    }
	//calc g_gamma and g_beta (also used for error calc)
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
                    g_beta[c] = g_beta[c] + error[n][c][row][col];
                    g_gamma[c] = g_gamma[c] + error[n][c][row][col] * (bn_inputs_fw[n][c][row][col]-mu[c])/sigma[c];
				}
			}
		}
	}
	// calc backprop error
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
			for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
#pragma HLS PIPELINE
				for (int n = 0; n < BATCH_SIZE; n ++) {
            		out_buf[n][c][row][col] = gamma[c]*error[n][c][row][col]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (bn_inputs_fw[n][c][row][col]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
				}
			}
		}
	}
}

#endif
