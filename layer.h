#ifndef LAYER_H
#define LAYER_H

#include "typedefs.h"
#include "dimension_def.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
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
	int8 tmp[CHANNEL_IN_T];
#pragma HLS ARRAY_PARTITION variable=temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=msb_in dim=1 complete
#pragma HLS ARRAY_PARTITION variable=msb_in dim=2 complete
#pragma HLS ARRAY_PARTITION variable=lsb_out dim=1 complete
#pragma HLS ARRAY_PARTITION variable=lsb_out dim=2 complete

	for (int row = 0; row < H_fmap_in; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_in; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
					tmp[c] = msb_in[n][c][row][col];
				}
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
					lsb_out[n][c][row][col] = tmp[c];
				}
			}
		}
	}
}

// AvgPool
void avgpool(
	int8 avg_inputs[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],	// in
	int8 out_buf[BATCH_SIZE][64],								// out, avg_outputs

	int1 ctrl_avgpool,	// 0 for forward and 1 for backward
	int c_out
)
{
	int8 out_temp[CHANNEL_IN_T] = {0};
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=avg_inputs dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=2 complete

	// forward
	if (ctrl_avgpool == 0) {
		LOOP_POOL_FW:
		for (int n = 0; n < BATCH_SIZE; n ++) {
			for (int s = 0; s < 8; s ++) {
				for (int ss = 0; ss < 8; ss ++) {
#pragma HLS PIPELINE
					for (int c = 0; c < CHANNEL_IN_T; c ++) {
						out_temp[c] += avg_inputs[n][c][s][ss];
					}
				}
			}
		}
		LOOP_WRITE:
		for (int n = 0; n < BATCH_SIZE; n ++) {
			for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE
				out_buf[n][c + c_out*CHANNEL_IN_T] = out_temp[c]/64;
			}
		}
	}
	// backward
	else {
		LOOP_POOL_BW:
		for (int n = 0; n < BATCH_SIZE; n ++) {
			for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE
				out_temp[c] = out_buf[n][c + c_out*CHANNEL_IN_T];
			}
			for (int s = 0; s < 8; s ++) {
				for (int ss = 0; ss < 8; ss ++) {
#pragma HLS PIPELINE
					for (int c = 0; c < CHANNEL_IN_T; c ++) {
						avg_inputs[n][c][s][ss] = out_temp[c]/64;
					}
				}
			}
		}
	}
}

// FC
void FC(
	int8 inputs[BATCH_SIZE][64],
	int8 linear_weight[10][64],
	int8 outputs[BATCH_SIZE][10],

	int1 ctrl_fc	// 0 for forward and 1 for backward
)
{
	int8 out_temp[10] = {0};
	int8 in_tmp[64] = {0};
#pragma HLS ARRAY_PARTITION variable=inputs dim=2 complete
#pragma HLS ARRAY_PARTITION variable=linear_weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=linear_weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=in_tmp dim=1 complete

	if (ctrl_fc == 0) {
		// forward
		LOOP_FC_FW:
		for (int bii = 0; bii < BATCH_SIZE; bii++) {
			for (int cii = 0; cii < 64; cii++) {
				for (int coo = 0; coo < 10; coo ++) {
#pragma HLS PIPELINE
					int8 act = inputs[bii][cii];
					int8 wt = linear_weight[coo][cii];
					out_temp[coo] += act * wt;
				}
			}
			for (int coo = 0; coo < 10; coo ++) {
#pragma HLS PIPELINE
				outputs[bii][coo] = out_temp[coo];
			}
		}
	}
	// backward
	else {
		LOOP_FC_BW:
		for (int bii = 0; bii < BATCH_SIZE; bii++) {
			for (int cii = 0; cii < 64; cii++) {
				for (int coo = 0; coo < 10; coo ++) {
#pragma HLS PIPELINE
					int8 act = outputs[bii][coo];
					int8 wt = linear_weight[coo][cii];
					in_tmp[cii] += act * wt;
				}
			}
			for (int cii = 0; cii < 64; cii++) {
#pragma HLS PIPELINE
				inputs[bii][cii] = in_tmp[cii];
			}
		}
	}
}

// Shortcut- identity branch
void shortcut(
	int8 input_a[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	int8 input_b[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	int8 out_buf_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip

	int H_fmap_in,
	int1 ctrl_sc	// if ctrl_sc=0, generate and send out_copy into DDR
)
{
	int8 in_1[CHANNEL_OUT_T];
	int8 in_2[CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=in_1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_2 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=input_a dim=2 complete
#pragma HLS ARRAY_PARTITION variable=input_b dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR dim=2 complete

	for (int row = 0; row < H_fmap_in; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_in; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			LOOP_SC_ADD:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_1[c] = input_a[n][c][row][col];
					in_2[c] = input_b[n][c][row][col];
					out_temp[n][c] = in_1[c] + in_2[c];
				}
				LOOP_SC_WRITE_OUTPUT:
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					out_buf[n][c][row][col] = out_temp[n][c];
				}
				LOOP_SC_WRITE_DDR:
				if (ctrl_sc == 0) {
					for (int c = 0; c < CHANNEL_OUT_T; c ++) {
						out_buf_DDR[n][c][row][col] = out_temp[n][c];
					}
				}
			}
		}
	}
}

// Batch Norm
void bn(
	int8 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out

	int8 gamma[CHANNEL_OUT_T],
	int8 beta[CHANNEL_OUT_T],
    int H_fmap_in
)
{
	int N = BATCH_SIZE * H_fmap_in * H_fmap_in;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=sigma dim=1 complete

	int8 in_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=bn_inputs dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=2 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete

	for (int row = 0; row < H_fmap_in; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap_in; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
			// buffer init
			LOOP_INIT:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_temp[n][c] = bn_inputs[n][c][row][col];
				}
			}
			// calc mean and std_var
			LOOP_MEAN:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
                    mu[c] += in_temp[n][c]/N;
                    sigma[c] += (in_temp[n][c]-mu[c])/hls::sqrt(N);
				}
			}
			// bn
			LOOP_BN:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					//out_temp[n][c] = gamma[c]*(in_temp[n][c]-mu[c])/sigma[c] + beta[c];
					out_temp[n][c] = gamma[c]*(in_temp[n][c]-mu[c]) + beta[c];
				}
			}
			// write out
			LOOP_WRITE:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					out_buf[n][c][row][col] = out_temp[n][c];
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
	int H_fmap_in
)
{
	int N = BATCH_SIZE * H_fmap_in * H_fmap_in;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=sigma dim=1 complete

	int8 error_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 in_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=error_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=error dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=2 complete

	for (int row = 0; row < H_fmap_in; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap_in; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
			// buffer init
			LOOP_INIT:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_temp[n][c] = bn_inputs_fw[n][c][row][col];
					error_temp[n][c] = error[n][c][row][col];
				}
			}
			// calc mean and std_var
			LOOP_MEAN:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					mu[c] += in_temp[n][c]/N;
					sigma[c] += (in_temp[n][c]-mu[c])/hls::sqrt(N);
				}
			}
			// calc grad for bn params
			LOOP_GRAD:
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
                    g_beta[c] += error_temp[n][c];
                    //g_gamma[c] += error_temp[n][c] * (in_temp[n][c]-mu[c])/sigma[c];
                    g_gamma[c] += error_temp[n][c] * (in_temp[n][c]-mu[c]);
				}
			}
			// calc backprop error
			LOOP_BN_BP:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					//out_temp[n][c] = gamma[c]*error_temp[n][c]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (in_temp[n][c]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
					out_temp[n][c] = gamma[c]*error_temp[n][c] - gamma[c]*g_beta[c]/N - (in_temp[n][c]-mu[c])*g_gamma[c]/N;
				}
			}
			// write out
			LOOP_WRITE:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					out_buf[n][c][row][col] = out_temp[n][c];
				}
			}
		}
	}
}

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
	int N = BATCH_SIZE * H_fmap * H_fmap;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=sigma dim=1 complete

	int1 relu_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 in_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=bn_inputs dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR dim=2 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=2 complete
#pragma HLS ARRAY_PARTITION variable=relu_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete

	for (int row = 0; row < H_fmap; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
			// buffer init
			LOOP_INIT:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_temp[n][c] = bn_inputs[n][c][row][col];
				}
			}
			// calc mean and std_var
			LOOP_MEAN:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
                    mu[c] += in_temp[n][c]/N;
                    sigma[c] += (in_temp[n][c]-mu[c])/hls::sqrt(N);
				}
			}
			// bn + relu
			LOOP_BN_RELU:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					//out_temp[n][c] = gamma[c]*(in_temp[n][c]-mu[c])/sigma[c] + beta[c];
					out_temp[n][c] = gamma[c]*(in_temp[n][c]-mu[c]) + beta[c];
					relu_temp[n][c] = (out_temp[n][c] < 0) ? 0 : 1;
				}
			}
			// write out
			LOOP_WRITE:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					relu_mask[n][c][row][col] = relu_temp[n][c];
					out_buf[n][c][row][col] = out_temp[n][c];
					out_buf_DDR[n][c][row][col] = out_temp[n][c];
				}
			}
		}
	}
}

// Fused relu_bp + bn_bp
void bn_relu_bp(
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
	int N = BATCH_SIZE * WIDTH * WIDTH;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=sigma dim=1 complete

	int8 error_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 in_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=error_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=error dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw dim=2 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=2 complete

	for (int row = 0; row < H_fmap; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			// buffer init
			LOOP_INIT:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_temp[n][c] = bn_inputs_fw[n][c][row][col];
					// relu
					error_temp[n][c] = relu_mask[n][c][row][col] * error[n][c][row][col];
				}
			}
			// calc mean and std_var
			LOOP_MEAN:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					mu[c] += in_temp[n][c]/N;
					sigma[c] += (in_temp[n][c]-mu[c])/hls::sqrt(N);
				}
			}
			// calc grad for bn params
			LOOP_GRAD:
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE
					g_beta[c] += error_temp[n][c];
					g_gamma[c] += error_temp[n][c] * (bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
			// calc backprop error
			LOOP_BN_RELU_BP:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					//out_temp[n][c] = gamma[c]*error_temp[n][c]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (in_temp[n][c]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
					out_temp[n][c] = gamma[c]*error_temp[n][c] - gamma[c]*g_beta[c]/N - (in_temp[n][c]-mu[c])*g_gamma[c]/N;
				}
			}
			// write out
			LOOP_WRITE:
			for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					out_buf[n][c][row][col] = out_temp[n][c];
				}
			}
		}
	}
}

// ============
// Conv forward
// ============

// conv_3x3, padding=1
void conv_3x3
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out on-chip
	int8 output_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int8 act[CHANNEL_OUT_T];
	int8 wt[CHANNEL_OUT_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=3 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=4 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output_DDR dim=2 complete

	for (int row = 0; row < H_fmap_out; row++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
				// buffer initiation
				LOOP_INIT:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					if (c_in > 0) {
						out_temp[co] = output[bi][co][row][col];
					}
					else {
						out_temp[co] = 0;
					}
				}
				// conv 3x3
				LOOP_CONV:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
#pragma HLS UNROLL
					accum[co] = 0;
					for (int krow = 0; krow < 3; krow ++) {
						for (int kcol = 0; kcol < 3; kcol ++) {
							for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
								int row_in = row*stride + krow - 1;
								int col_in = col*stride + kcol - 1;	// 0-padding
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									act[co] = input[bi][cin][row_in][col_in];
									wt[co] = weight[co][cin][krow][kcol];
								}
							}
						}
					}
				}
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					accum[co] += act[co] * wt[co];
					out_temp[co] += accum[co];
				}
				// write out
				LOOP_WRITE:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					output[bi][co][row][col] = out_temp[co];
					output_DDR[bi][co][row][col] = out_temp[co];
				}
			}
		}
	}
}

// conv_1x1, padding=0
void conv_1x1
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out on-chip
	int8 output_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int8 act[CHANNEL_OUT_T];
	int8 wt[CHANNEL_OUT_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output_DDR dim=2 complete

	for (int row = 0; row < H_fmap_out; row++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
				// buffer initiation
				LOOP_INIT:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					if (c_in > 0) {
						out_temp[co] = output[bi][co][row][col];
					}
					else {
						out_temp[co] = 0;
					}
				}
				// conv 1x1
				LOOP_CONV:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
#pragma HLS UNROLL
					accum[co] = 0;
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						int row_in = row*stride;
						int col_in = col*stride;
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							act[co] = input[bi][cin][row_in][col_in];
							wt[co] = weight[co][cin];
						}
					}
				}
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					accum[co] += act[co] * wt[co];
					out_temp[co] += accum[co];
				}
				// write out
				LOOP_WRITE:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					output[bi][co][row][col] = out_temp[co];
					output_DDR[bi][co][row][col] = out_temp[co];
				}
			}
		}
	}
}

// =============
// Conv backward
// =============

// conv_3x3, padding=1
void conv_3x3_rot_bp
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],		// error in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// error out on-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int row_in;
	int col_in;
	int8 act[CHANNEL_OUT_T];
	int8 wt[CHANNEL_OUT_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=3 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=4 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete

	for (int row = 0; row < H_fmap_out; row++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
				// buffer initiation
				LOOP_INIT:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					if (c_in > 0) {
						out_temp[co] = output[bi][co][row][col];
					}
					else {
						out_temp[co] = 0;
					}
				}
				// conv 3x3
				LOOP_DECONV:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
#pragma HLS UNROLL
					accum[co] = 0;
					for (int krow = 0; krow < 3; krow ++) {
						for (int kcol = 0; kcol < 3; kcol ++) {
							for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
								if (stride == 1) {
									row_in = row + krow - 1;
									col_in = col + kcol - 1;
								}
								else {
									row_in = ((row + krow + 1) % 2 == 1) ? (row + krow - 1)/2 : -1;	// row_in = ((row + krow + pad) % stride == pad) ? (krow + row - pad)/stride : -1;
									col_in = ((col + kcol + 1) % 2 == 1) ? (col + kcol - 1)/2 : -1;
								}
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									act[co] = input[bi][cin][row_in][col_in];
									wt[co] = weight[cin][co][kcol][krow];	// weight rot180
								}
							}
						}
					}
				}
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					accum[co] += act[co] * wt[co];
					out_temp[co] += accum[co];
				}
				// write out
				LOOP_WRITE:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					output[bi][co][row][col] = out_temp[co];
				}
			}
		}
	}
}

// conv_1x1_rot_bp, padding=0
void conv_1x1_rot_bp
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],			// error in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// error out on-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int row_in;
	int col_in;
	int8 act[CHANNEL_OUT_T];
	int8 wt[CHANNEL_OUT_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete

	for (int row = 0; row < H_fmap_out; row++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
				// buffer initiation
				LOOP_INIT:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					if (c_in > 0) {
						out_temp[co] = output[bi][co][row][col];
					}
					else {
						out_temp[co] = 0;
					}
				}
				// conv 3x3
				LOOP_DECONV:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
#pragma HLS UNROLL
					accum[co] = 0;
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						if (stride == 1) {
							row_in = row - 1;
							col_in = col - 1;
						}
						else {
							row_in = (row % 2 == 0) ? row/2 : -1;	// row_in = ((row + krow + pad) % stride == pad) ? (krow + row - pad)/stride : -1;
							col_in = (col % 2 == 0) ? col/2 : -1;
						}
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							act[co] = input[bi][cin][row_in][col_in];
							wt[co] = weight[cin][co];	// weight rot180
						}
					}
				}
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					accum[co] += act[co] * wt[co];
					out_temp[co] += accum[co];
				}
				// write out
				LOOP_WRITE:
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					output[bi][co][row][col] = out_temp[co];
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
)
{
	int8 act[CHANNEL_OUT_T];
	int8 wt[CHANNEL_OUT_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=3 complete
#pragma HLS ARRAY_PARTITION variable=output dim=4 complete

	for (int krow = 0; krow < H_fmap_in; krow ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int bi = 0; bi < BATCH_SIZE; bi ++) {
#pragma HLS PIPELINE
				// dilated conv
				LOOP_DILATED_CONV:
				for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS UNROLL
					accum[co] = 0;
					for (int row = 0; row < 3; row ++) {
						for (int col = 0; col < 3; col ++) {
							for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
								// int row_in = (stride == 1) ? (row + krow - 1) : (row + krow*2 - 1);		// row_in = (krow*stride) % stride == 0) ? row + krow*stride - pad : -1;
								// int col_in = (stride == 1) ? (col + kcol - 1) : (col + kcol*2 - 1);		// 0-padding
								int row_in = row + krow*stride - 1;
								int col_in = col + kcol*stride - 1;
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									wt[co] = weight[bi][co][krow][kcol];
									act[co] = input[bi][ci][row_in][col_in];
								}
							}
						}
					}
				}
				for (int co = 0; co < CHANNEL_OUT_T; co ++) {
					accum[co] += act[co] * wt[co];
					out_temp[co] = lr * accum[co];
				}
				// write out
				LOOP_WRITE:
				for (int co = 0; co < CHANNEL_OUT_T; co ++) {
					for (int row = 0; row < 3; row ++) {
						for (int col = 0; col < 3; col ++) {
							for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
								output[co][ci][row][col] = output[co][ci][row][col] - out_temp[co];
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
)
{
	int8 act[CHANNEL_OUT_T];
	int8 wt[CHANNEL_OUT_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete

	// dilated conv
	for (int krow = 0; krow < H_fmap_in; krow ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int bi = 0; bi < BATCH_SIZE; bi ++) {
#pragma HLS PIPELINE
				// dilated conv
				LOOP_DILATED_CONV:
				for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS UNROLL
					accum[co] = 0;
					for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
						// int row_in = (stride == 1) ? krow : krow*2;	// row_in = (krow*stride) % stride == 0) ? row + krow*stride - pad : -1;
						// int col_in = (stride == 1) ? kcol : kcol*2;
						int row_in = krow*stride;
						int col_in = kcol*stride;
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							wt[co] = weight[bi][co][krow][kcol];
							act[co] = input[bi][ci][row_in][col_in];
						}
					}
				}
				for (int co = 0; co < CHANNEL_OUT_T; co ++) {
					accum[co] += act[co] * wt[co];
					out_temp[co] = lr * accum[co];
				}
				// write out
				LOOP_WRITE:
				for (int co = 0; co < CHANNEL_OUT_T; co ++) {
					for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
						output[co][ci] = output[co][ci] - out_temp[co];
					}
				}
			}
		}
	}
}

#endif
