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
	int8 temp;
#pragma HLS ARRAY_PARTITION variable=msb_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=msb_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=lsb_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=lsb_out complete dim=2

	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
// #pragma HLS LATENCY MAX = 1
					lsb_out[n][c][row][col] = msb_in[n][c][row][col];
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
	int8 out_temp;
#pragma HLS ARRAY_PARTITION variable=avg_inputs complete dim=2

	for (int c = 0; c < CHANNEL_IN_T; c ++) {
		for (int n = 0; n < BATCH_SIZE; n ++) {
#pragma HLS PIPELINE
			LOOP_POOL:
			// forward
			if (ctrl_avgpool == 0) {
				out_temp = 0;
				for (int s = 0; s < 8; s ++) {
					for (int ss = 0; ss < 8; ss ++) {
// #pragma HLS LATENCY MAX = 1
						out_temp += avg_inputs[n][c][s][ss]/64;
					}
				}
				out_buf[n][c + c_out*CHANNEL_IN_T] = out_temp;
			}
			// backward
			else {
				out_temp = out_buf[n][c + c_out*CHANNEL_IN_T]/64;
				for (int s = 0; s < 8; s ++) {
					for (int ss = 0; ss < 8; ss ++) {
// #pragma HLS LATENCY MAX = 1
						avg_inputs[n][c][s][ss] = out_temp;
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
	int8 out_temp[10];
	int8 in_temp[64];
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=2
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1

	for (int bii = 0; bii < BATCH_SIZE; bii++) {
#pragma HLS PIPELINE
		LOOP_FC:
		// forward
		if (ctrl_fc == 0) {
			for (int coo = 0; coo < 10; coo ++) {
				for (int cii = 0; cii < 64; cii++) {
// #pragma HLS LATENCY MAX = 1
#pragma HLS ARRAY_PARTITION variable=linear_weight complete dim=1
					int8 act = inputs[bii][cii];
					int8 wt = linear_weight[coo][cii];
					out_temp[coo] += act * wt;
				}
			}
			for (int coo = 0; coo < 10; coo ++) {
				outputs[bii][coo] = out_temp[coo];
			}
		}
		// backward
		else {
			for (int cii = 0; cii < 64; cii++) {
				for (int coo = 0; coo < 10; coo ++) {
// #pragma HLS LATENCY MAX = 1
#pragma HLS ARRAY_PARTITION variable=linear_weight complete dim=1
					int8 act = outputs[bii][coo];
					int8 wt = linear_weight[coo][cii];
					in_temp[cii] += act * wt;
				}
			}
			for (int cii = 0; cii < 64; cii++) {
				inputs[bii][cii] = in_temp[cii];
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
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=input_a complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_b complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=2

	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE
			LOOP_SC_ADD:
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
					int8 in_1 = input_a[n][c][row][col];
					int8 in_2 = input_b[n][c][row][col];
					out_temp[n][c] = in_1 + in_2;
				}
			}
			LOOP_SC_WRITE_OUTPUT:
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
					out_buf[n][c][row][col] = out_temp[n][c];
					if (ctrl_sc == 0) {
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
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1

	int8 in_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=bn_inputs complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=in_temp complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=2

	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_temp[n][c] = bn_inputs[n][c][row][col];
				}
			}
			// calc mean and std_var
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
                    mu[c] += in_temp[n][c]/N;
                    sigma[c] += (in_temp[n][c]-mu[c])/hls::sqrt(N);
				}
			}
			// bn
			LOOP_BN:
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
					//out_temp[n][c] = gamma[c]*(in_temp[n][c]-mu[c])/sigma[c] + beta[c];
					out_temp[n][c] = gamma[c]*(in_temp[n][c]-mu[c]) + beta[c];
				}
			}
			// write out
			for (int n = 0; n < BATCH_SIZE; n ++) {
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
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1

	int8 error_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 in_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=error_temp complete dim=2
#pragma HLS ARRAY_PARTITION variable=in_temp complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=2

	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_temp[n][c] = bn_inputs_fw[n][c][row][col];
					error_temp[n][c] = error[n][c][row][col];
				}
			}
			// calc mean and std_var
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
					mu[c] += in_temp[n][c]/N;
					sigma[c] += (in_temp[n][c]-mu[c])/hls::sqrt(N);
				}
			}
			// calc grad for bn params
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
                    g_beta[c] += error_temp[n][c];
                    //g_gamma[c] += error_temp[n][c] * (in_temp[n][c]-mu[c])/sigma[c];
                    g_gamma[c] += error_temp[n][c] * (in_temp[n][c]-mu[c]);
				}
			}
			// calc backprop error
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
					//out_temp[n][c] = gamma[c]*error_temp[n][c]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (in_temp[n][c]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
					out_temp[n][c] = gamma[c]*error_temp[n][c] - gamma[c]*g_beta[c]/N - (in_temp[n][c]-mu[c])*g_gamma[c]/N;
				}
			}
			// write out
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
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
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1

	int1 relu_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 in_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=relu_temp complete dim=2
#pragma HLS ARRAY_PARTITION variable=relu_mask complete dim=2
#pragma HLS ARRAY_PARTITION variable=in_temp complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=2

	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_temp[n][c] = bn_inputs[n][c][row][col];
				}
			}
			// calc mean and std_var
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
                    mu[c] += in_temp[n][c]/N;
                    sigma[c] += (in_temp[n][c]-mu[c])/hls::sqrt(N);
				}
			}
			// bn
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
					//out_temp[n][c] = gamma[c]*(in_temp[n][c]-mu[c])/sigma[c] + beta[c];
					out_temp[n][c] = gamma[c]*(in_temp[n][c]-mu[c]) + beta[c];
					// relu mask
					if (out_temp[n][c] > 0) {
						relu_temp[n][c] = 1;
					} else {
						relu_temp[n][c] = 0;
					}
				}
			}
			// write out
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
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
#pragma HLS ARRAY_PARTITION variable=mu complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma complete dim=1

	int8 error_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 in_temp[BATCH_SIZE][CHANNEL_OUT_T];
	int8 out_temp[BATCH_SIZE][CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw complete dim=2
#pragma HLS ARRAY_PARTITION variable=relu_mask complete dim=2
#pragma HLS ARRAY_PARTITION variable=error_temp complete dim=2
#pragma HLS ARRAY_PARTITION variable=in_temp complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=2

	for (int row = 0; row < H_fmap; row ++) {
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS PIPELINE
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					in_temp[n][c] = bn_inputs_fw[n][c][row][col];
					// relu
					error_temp[n][c] = relu_mask[n][c][row][col] * error[n][c][row][col];
				}
			}
			// calc mean and std_var
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					// mean
					mu[c] += in_temp[n][c]/N;
					sigma[c] += (in_temp[n][c]-mu[c])/hls::sqrt(N);
				}
			}
			// calc grad for bn params
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					g_beta[c] += error_temp[n][c];
					g_gamma[c] += error_temp[n][c] * (bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
			// calc backprop error
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					//out_temp[n][c] = gamma[c]*error_temp[n][c]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (in_temp[n][c]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
					out_temp[n][c] = gamma[c]*error_temp[n][c] - gamma[c]*g_beta[c]/N - (in_temp[n][c]-mu[c])*g_gamma[c]/N;
				}
			}
			// write out
			LOOP_WRITE_OUT:
			for (int n = 0; n < BATCH_SIZE; n ++) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS LATENCY MAX = 1
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
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],			// error in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// error out on-chip
	int8 output_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// error out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int8 out_temp[CHANNEL_OUT_T];
// #pragma HLS DEPENDENCE variable=output array inter false
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1

	LOOP_TILE:
	for (int row = 0; row < H_fmap_out; row++) {
		for (int col = 0; col < H_fmap_out; col++) {
			for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
				// buffer initiation
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					if (c_in > 0){
						out_temp[co] = output[bi][co][row][col];
					}
					else{
						out_temp[co] = 0;
					}
				}
				// conv 3x3
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					int8 accum = 0;
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
// #pragma HLS LATENCY MAX = 1
								int row_in = row*stride + krow - 1;
								int col_in = col*stride + kcol - 1;	// 0-padding
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									int8 act = input[bi][cin][row_in][col_in];
									int8 wt = weight[co][cin][krow][kcol];
									accum += act * wt;
								}
							}
						}
					}
					out_temp[co] += accum;
				}
				// write out
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					output[bi][co][row][col] = out_temp[co];
					output_DDR[bi][co][row][col] = out_temp[co];
				}
			}
		}
	}
}

// conv_1x1_rot_bp, padding=0
void conv_1x1
(
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],			// error in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// error out on-chip
	int8 output_DDR[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// error out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int8 out_temp[CHANNEL_OUT_T];
// #pragma HLS DEPENDENCE variable=output array inter false
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1

	for (int row = 0; row < H_fmap_out; row++) {
		for (int col = 0; col < H_fmap_out; col++) {
			for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
				// buffer initiation
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					if (c_in > 0){
						out_temp[co] = output[bi][co][row][col];
					}
					else{
						out_temp[co] = 0;
					}
				}
				// conv 1x1
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					int8 accum = 0;
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
// #pragma HLS LATENCY MAX = 1
						int row_in = row*stride;
						int col_in = col*stride;
						int8 act = input[bi][cin][row_in][col_in];
						int8 wt = weight[co][cin];
						accum += act * wt;
					}
					out_temp[co] += accum;
				}
				// write out
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

// conv_3x3_rot_bp, padding=1
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
	int8 out_temp[CHANNEL_OUT_T];
// #pragma HLS DEPENDENCE variable=output array inter false
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1

	// tranposed conv
	LOOP_TRANSPOSED_CONV:
	for (int row = 0; row < H_fmap_out; row++) {
		for (int col = 0; col < H_fmap_out; col++) {
			for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE
				// buffer initialization
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					if (c_in > 0){
						out_temp[co] = output[bi][co][row][col];
					}
					else{
						out_temp[co] = 0;
					}
				}
				// conv3x3
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					int8 accum = 0;
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
// #pragma HLS LATENCY MAX = 1
								int row_in = ((row + krow)%stride == 0) ? (row+krow)/stride-1 : -1;	// stride 1 transposed conv
								int col_in = ((row + krow)%stride == 0) ? (col+kcol)/stride-1 : -1;	// 0-padding(1, 1+A, 1, 1+A)
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									int8 act = input[bi][cin][row_in][col_in];
									int8 wt = weight[cin][co][2-krow][2-kcol];
									accum += act * wt;
								}
							}
						}
					}
					out_temp[co] += accum;
				}
				// write out
				LOOP_WRITE_OUTPUT:
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
	int8 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],		// error in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// error out on-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int8 out_temp[CHANNEL_OUT_T];
// #pragma HLS DEPENDENCE variable=output array inter false
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1

	// tranposed conv
	LOOP_TRANSPOSED_CONV:
	for (int row = 0; row < H_fmap_out; row++) {
		for (int col = 0; col < H_fmap_out; col++) {
			for (int bi = 0; bi < BATCH_SIZE; bi++) {
#pragma HLS PIPELINE

				// buffer initialization
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					if (c_in > 0){
						out_temp[co] = output[bi][co][row][col];
					}
					else{
						out_temp[co] = 0;
					}
				}

				// conv 1x1
				for (int co = 0; co < CHANNEL_OUT_T; co++) {
					int8 accum = 0;
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
// #pragma HLS LATENCY MAX = 1
						int row_in = (row%stride == 0) ? row/stride : -1;	// stride 1 transposed conv
						int col_in = (col%stride == 0) ? col/stride : -1;
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							int8 act = input[bi][cin][row_in][col_in];
							int8 wt = weight[cin][co];
							accum += act * wt;
						}
					}
					out_temp[co] += accum;
				}

				// write out
				LOOP_WRITE_OUTPUT:
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
	int8 out_temp[CHANNEL_OUT_T];
// #pragma HLS DEPENDENCE variable=output array inter false
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1

	// dilated conv
	LOOP_DILATED_CONV:
	for (int krow = 0; krow < H_fmap_in; krow ++) {
		for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
#pragma HLS PIPELINE
					// dilated conv
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						int8 accum = 0;
						for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
							for (int bi = 0; bi < BATCH_SIZE; bi ++) {
// #pragma HLS LATENCY MAX = 1
								int row_in = (krow%stride == 0) ? row + krow/stride - 1 : -1;
								int col_in = (kcol%stride == 0) ? col + kcol/stride - 1 : -1;	// 0-padding
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									int8 wt = weight[bi][co][krow][kcol];
									int8 act = input[bi][ci][row_in][col_in];
									accum += act * wt;
								}
							}
						}
						out_temp[co] = accum;
					}
					// write out
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
							output[co][ci][row][col] = out_temp[co];
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
	int8 out_temp[CHANNEL_OUT_T];
// #pragma HLS DEPENDENCE variable=output array inter false
#pragma HLS ARRAY_PARTITION variable=input complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1

	// dilated conv
	LOOP_DILATED_CONV:
	for (int krow = 0; krow < H_fmap_in; krow ++) {
		for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
#pragma HLS PIPELINE
			// dilated conv
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				int8 accum = 0;
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					for (int bi = 0; bi < BATCH_SIZE; bi ++) {
// #pragma HLS LATENCY MAX = 1
						int row_in = (krow%stride == 0) ? krow/stride : -1;
						int col_in = (kcol%stride == 0) ? kcol/stride : -1;
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							int8 wt = weight[bi][co][krow][kcol];
							int8 act = input[bi][ci][row_in][col_in];
							accum += act * wt;
						}
					}
				}
				out_temp[co] = accum;
			}
			// write out
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					output[co][ci] = out_temp[co];
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
	int8 weight_temp[3][3];
#pragma HLS ARRAY_PARTITION variable=gradient complete dim=2
#pragma HLS ARRAY_PARTITION variable=gradient complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=0

	LOOP_TILE:
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
#pragma HLS PIPELINE
// #pragma HLS LATENCY MAX = 1
			LOOP_WEIGHT_UPDATE:
			for (int krow = 0; krow < 3; krow++) {
				for (int kcol = 0; kcol < 3; kcol++) {
					weight_temp[krow][kcol] = weight[co][ci][krow][kcol] - lr*gradient[co][ci][krow][kcol];
				}
			}
			LOOP_WRITE_OUTPUT:
			for (int krow = 0; krow < 3; krow++) {
				for (int kcol = 0; kcol < 3; kcol++) {
					weight_WU[co][ci][krow][kcol] = weight_temp[krow][kcol];
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
	int8 weight_temp[CHANNEL_OUT_T][CHANNEL_IN_T];
#pragma HLS ARRAY_PARTITION variable=gradient complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=2

	LOOP_TILE:
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
#pragma HLS PIPELINE
// #pragma HLS LATENCY MAX = 1
		LOOP_WEIGHT_UPDATE:
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			weight_temp[co][ci] = weight[co][ci] - lr*gradient[co][ci];
		}
		LOOP_WRITE_OUTPUT:
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			weight_WU[co][ci] = weight_temp[co][ci];
		}
	}
}

#endif
