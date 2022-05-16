#ifndef LAYER_H
#define LAYER_H

#include "typedefs.h"
#include "dimension_def.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <hls_math.h>

/*
 * FOR BATCH 1 ONLY
*/

//--------------------
//   Utils Function
//--------------------

// identity shortcut
void identity_shortcut(
	int8 msb_in[CHANNEL_IN_T][WIDTH][WIDTH],
	int8 lsb_out[CHANNEL_IN_T][WIDTH][WIDTH],
	int H_fmap_in
)
{
	int8 tmp[CHANNEL_IN_T];
#pragma HLS ARRAY_PARTITION variable=tmp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=msb_in dim=1 complete
#pragma HLS ARRAY_PARTITION variable=lsb_out dim=1 complete

	for (int row = 0; row < H_fmap_in; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_IN_T; c ++) {
				tmp[c] = msb_in[c][row][col];
			}
			for (int c = 0; c < CHANNEL_IN_T; c ++) {
				lsb_out[c][row][col] = tmp[c];
			}
		}
	}
}

// AvgPool
void avgpool(
	int8 avg_inputs[CHANNEL_IN_T][WIDTH][WIDTH],	// in
	int8 out_buf[64],								// out, avg_outputs

	int1 ctrl_avgpool,	// 0 for forward and 1 for backward
	int c_out
)
{
	int8 out_temp[CHANNEL_IN_T] = {0};
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=avg_inputs dim=1 complete

	// forward
	if (ctrl_avgpool == 0) {
		for (int s = 0; s < 8; s ++) {
			for (int ss = 0; ss < 8; ss ++) {
#pragma HLS PIPELINE II=1
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
					out_temp[c] += avg_inputs[c][s][ss];
				}
			}
		}
		for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE II=1
			out_buf[c + c_out*CHANNEL_IN_T] = out_temp[c]/64;
		}
	}
	// backward
	else {
		for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE II=1
			out_temp[c] = out_buf[c + c_out*CHANNEL_IN_T];
		}
		for (int s = 0; s < 8; s ++) {
			for (int ss = 0; ss < 8; ss ++) {
#pragma HLS PIPELINE II=1
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
					avg_inputs[c][s][ss] = out_temp[c]/64;
				}
			}
		}
	}
}

// FC
void FC(
	int8 inputs[64],
	int8 linear_weight[10][64],
	int8 outputs[10],

	int1 ctrl_fc	// 0 for forward and 1 for backward
)
{
	int8 out_temp[10] = {0};
	int8 in_tmp[64] = {0};

#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_tmp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=inputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=outputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=linear_weight dim=1 complete

	// forward
	if (ctrl_fc == 0) {
		for (int cii = 0; cii < 64; cii++) {
#pragma HLS PIPELINE II=1
			for (int coo = 0; coo < 10; coo ++) {
				int8 act = inputs[cii];
				int8 wt = linear_weight[coo][cii];
				out_temp[coo] += act * wt;
			}
		}
		for (int coo = 0; coo < 10; coo ++) {
#pragma HLS PIPELINE II=1
			outputs[coo] = out_temp[coo];
		}
	}
	// backward
	else {
		for (int cii = 0; cii < 64; cii++) {
#pragma HLS PIPELINE II=1
			for (int coo = 0; coo < 10; coo ++) {
				int8 act = outputs[coo];
				int8 wt = linear_weight[coo][cii];
				in_tmp[cii] += act * wt;
			}
		}
		for (int cii = 0; cii < 64; cii++) {
#pragma HLS PIPELINE II=1
			inputs[cii] = in_tmp[cii];
		}
	}
}

// Shortcut- identity branch
void shortcut(
	int8 input_a[CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	int8 input_b[CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	int8 out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip

	int H_fmap_in,
	int1 ctrl_sc	// if ctrl_sc=0, generate and send out_copy into DDR
)
{
	int8 in_1[CHANNEL_OUT_T];
	int8 in_2[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=in_1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_2 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input_a dim=1 complete
#pragma HLS ARRAY_PARTITION variable=input_b dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR dim=1 complete

	for (int row = 0; row < H_fmap_in; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_1[c] = input_a[c][row][col];
				in_2[c] = input_b[c][row][col];
				out_temp[c] = in_1[c] + in_2[c];
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
			}
			if (ctrl_sc == 0) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					out_buf_DDR[c][row][col] = out_temp[c];
				}
			}
		}
	}
}

// Batch Norm
void bn(
	int8 bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out

	int8 gamma[CHANNEL_OUT_T],
	int8 beta[CHANNEL_OUT_T],
    int H_fmap
)
{
	int N = H_fmap * H_fmap;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=gamma dim=1 complete
#pragma HLS ARRAY_PARTITION variable=beta dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=sigma dim=1 complete

	int8 in_temp[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=bn_inputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete

	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs[c][row][col];
			}
			// calc mean and std_var
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] += in_temp[c]/N;
				sigma[c] += (in_temp[c]-mu[c])/hls::sqrt(N);
			}
		}
	}
	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1
			// bn
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				//out_temp[c] = gamma[c]*(in_temp[c]-mu[c])/sigma[c] + beta[c];
				out_temp[c] = gamma[c]*(in_temp[c]-mu[c]) + beta[c];
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
			}
		}
	}
}

// Batch Norm Back-prop
void bn_bp(
	int8 error[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	int8 bn_inputs_fw[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	int8 gamma[CHANNEL_OUT_T],							// in
	int H_fmap
)
{
	int N = H_fmap * H_fmap;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 g_gamma[CHANNEL_OUT_T];						// out
	int8 g_beta[CHANNEL_OUT_T];							// out
#pragma HLS ARRAY_PARTITION variable=gamma dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=sigma dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_gamma dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_beta dim=1 complete

	int8 error_temp[CHANNEL_OUT_T];
	int8 in_temp[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=error_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=error dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete

	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
				error_temp[c] = error[c][row][col];
			}
			// calc mean and std_var
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] += in_temp[c]/N;
				sigma[c] += (in_temp[c]-mu[c])/hls::sqrt(N);
			}
			// calc grad for bn params
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				g_beta[c] += error_temp[c];
				//g_gamma[c] += error_temp[c] * (in_temp[c]-mu[c])/sigma[c];
				g_gamma[c] = error_temp[c] * (in_temp[c]-mu[c]);
			}
		}
	}
	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1
			// calc backprop error
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				//out_temp[c] = gamma[c]*error_temp[c]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (in_temp[c]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
				out_temp[c] = gamma[c]*error_temp[c] - gamma[c]*g_beta[c]/N - (in_temp[c]-mu[c])*g_gamma[c]/N;
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
			}
		}
	}
}

// Fused bn + relu
void bn_relu(
	int8 bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs
	int8 out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip for backprop
	int1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp

	int8 gamma[CHANNEL_OUT_T],
	int8 beta[CHANNEL_OUT_T],
    int H_fmap
)
{
	int N = H_fmap * H_fmap;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=gamma dim=1 complete
#pragma HLS ARRAY_PARTITION variable=beta dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=sigma dim=1 complete

	int1 relu_temp[CHANNEL_OUT_T];
	int8 in_temp[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=relu_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=bn_inputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR dim=1 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=1 complete

	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs[c][row][col];
			}
			// calc mean and std_var
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] += in_temp[c]/N;
				sigma[c] += (in_temp[c]-mu[c])/hls::sqrt(N);
			}
		}
	}
	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1
			// bn + relu
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				//out_temp[c] = gamma[c]*(in_temp[c]-mu[c])/sigma[c] + beta[c];
				out_temp[c] = gamma[c]*(in_temp[c]-mu[c]) + beta[c];
				relu_temp[c] = (out_temp[c] < 0) ? 0 : 1;
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				relu_mask[c][row][col] = relu_temp[c];
				out_buf[c][row][col] = out_temp[c];
				out_buf_DDR[c][row][col] = out_temp[c];
			}
		}
	}
}

// Fused relu_bp + bn_bp
void bn_relu_bp(
	int8 error[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	int8 bn_inputs_fw[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	int8 gamma[CHANNEL_OUT_T],							// in
	int H_fmap
)
{
	int N = H_fmap * H_fmap;
	int8 mu[CHANNEL_OUT_T];
	int8 sigma[CHANNEL_OUT_T];
	int8 g_gamma[CHANNEL_OUT_T];						// out
	int8 g_beta[CHANNEL_OUT_T];							// out
#pragma HLS ARRAY_PARTITION variable=gamma dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=sigma dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_gamma dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_beta dim=1 complete

	int8 error_temp[CHANNEL_OUT_T];
	int8 in_temp[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=error_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=error dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw dim=1 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete

	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
				// relu
				error_temp[c] = relu_mask[c][row][col] * error[c][row][col];
			}
			// calc mean and std_var
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] += in_temp[c]/N;
				sigma[c] += (in_temp[c]-mu[c])/hls::sqrt(N);
			}
			// calc grad for bn params
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				g_beta[c] += error_temp[c];
				g_gamma[c] += error_temp[c] * (bn_inputs_fw[c][row][col]-mu[c]);
			}
		}
	}
	for (int row = 0; row < H_fmap; row ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1
			// calc backprop error
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				//out_temp[c] = gamma[c]*error_temp[c]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (in_temp[c]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
				out_temp[c] = gamma[c]*error_temp[c] - gamma[c]*g_beta[c]/N - (in_temp[c]-mu[c])*g_gamma[c]/N;
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
			}
		}
	}
}

/*
// ============
// Conv forward
// ============

// conv_3x3, padding=1
void conv_3x3
(
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[CHANNEL_OUT_T][WIDTH][WIDTH],			// out on-chip
	int8 output_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int row_in;
	int col_in;
	int8 act[CHANNEL_IN_T];
	int8 wt[CHANNEL_OUT_T][CHANNEL_IN_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=2 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output_DDR dim=1 complete

	for (int row = 0; row < H_fmap_out; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
#pragma HLS PIPELINE II=1
					// buffer initiation
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						if (c_in > 0) {
							out_temp[co] = output[co][row][col];
						}
						else {
							out_temp[co] = 0;
						}
					}
					// load activation
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						row_in = (stride == 1) ? row + krow - 1 : row*2 + krow - 1;
						col_in = (stride == 1) ? col + kcol - 1 : col*2 + kcol - 1;	// 0-padding
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							act[cin] = input[cin][row_in][col_in];
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							wt[co][cin] = weight[co][cin][krow][kcol];
						}
					}
					// conv 3x3
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						accum[co] = 0;
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum[co] += act[cin] * wt[co][cin];
						}
						out_temp[co] += accum[co];
					}
					// write out
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						output[co][row][col] = out_temp[co];
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						output_DDR[co][row][col] = out_temp[co];
					}
				}
			}
		}
	}
}


// conv_1x1, padding=0
void conv_1x1
(
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 output[CHANNEL_OUT_T][WIDTH][WIDTH],		// out on-chip
	int8 output_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],	// out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];

#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output_DDR dim=1 complete

	for (int row = 0; row < H_fmap_out; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE
			// buffer initiation
			LOOP_INIT:
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				if (c_in > 0) {
					out_temp[co] = output[co][row][col];
				}
				else {
					out_temp[co] = 0;
				}
			}
			// conv 1x1
			LOOP_CONV:
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				accum[co] = 0;
				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
					int row_in = row*2;
					int col_in = col*2;
					if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
						int8 act = input[cin][row_in][col_in];
						int8 wt = weight[co][cin];
						accum[co] += act * wt;
					}
				}
				out_temp[co] += accum[co];
			}
			// write out
			LOOP_WRITE:
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				output[co][row][col] = out_temp[co];
				output_DDR[co][row][col] = out_temp[co];
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
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[CHANNEL_OUT_T][WIDTH][WIDTH],			// out on-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int row_in;
	int col_in;
	int8 act[CHANNEL_IN_T];
	int8 wt[CHANNEL_OUT_T][CHANNEL_IN_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=2 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete

	for (int row = 0; row < H_fmap_out; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
#pragma HLS PIPELINE II=1
					// buffer initiation
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						if (c_in > 0) {
							out_temp[co] = output[co][row][col];
						}
						else {
							out_temp[co] = 0;
						}
					}
					// load activation
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						row_in = (stride == 1) ? row + krow - 1 : (((row + krow - 1) % 2 == 0) ? (row + krow - 1)/2 : -1);	// row_in = ((row + krow + pad) % stride == pad) ? (krow + row - pad)/stride : -1;
						col_in = (stride == 1) ? col + kcol - 1 : (((col + kcol - 1) % 2 == 0) ? (col + kcol - 1)/2 : -1);
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							act[cin] = input[cin][row_in][col_in];
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							wt[co][cin] = weight[cin][co][kcol][krow];
						}
					}
					// conv 3x3
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						accum[co] = 0;
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum[co] += act[cin] * wt[co][cin];
						}
						out_temp[co] += accum[co];
					}
					// write out
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						output[co][row][col] = out_temp[co];
					}
				}
			}
		}
	}
}

// conv_1x1_rot_bp, padding=0
void conv_1x1_rot_bp
(
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],			// error in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 output[CHANNEL_OUT_T][WIDTH][WIDTH],		// error out on-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
)
{
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];

#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete

	for (int row = 0; row < H_fmap_out; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			// buffer initiation
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				if (c_in > 0) {
					out_temp[co] = output[co][row][col];
				}
				else {
					out_temp[co] = 0;
				}
			}
			// conv 1x1
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				accum[co] = 0;
				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
					int row_in = (row % 2 == 0) ? row/2 : -1;	// row_in = ((row + krow + pad) % stride == pad) ? (krow + row - pad)/stride : -1;
					int	col_in = (col % 2 == 0) ? col/2 : -1;
					if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
						int8 act = input[cin][row_in][col_in];
						int8 wt = weight[cin][co];	// weight rot180
						accum[co] += act * wt;
					}
				}
				out_temp[co] += accum[co];
			}
			// write out
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				output[co][row][col] = out_temp[co];
			}
		}
	}
}
*/
// =============
// Conv unified
// =============

// conv_3x3, padding=1
void conv_3x3_uni
(
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[CHANNEL_OUT_T][WIDTH][WIDTH],			// out on-chip
	int8 output_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in,
	int1 ctrl_conv
)
{
	int row_in;
	int col_in;
	int8 act[CHANNEL_IN_T];
	int8 wt[CHANNEL_OUT_T][CHANNEL_IN_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=2 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output_DDR dim=1 complete

	for (int row = 0; row < H_fmap_out; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
#pragma HLS PIPELINE II=1
					// buffer initiation
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						if (c_in > 0) {
							out_temp[co] = output[co][row][col];
						}
						else {
							out_temp[co] = 0;
						}
					}
					// load activation
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						if (ctrl_conv == 0) {
							row_in = (stride == 1) ? row + krow - 1 : row*2 + krow - 1;
							col_in = (stride == 1) ? col + kcol - 1 : col*2 + kcol - 1;	// 0-padding
						}
						else {
							row_in = (stride == 1) ? row + krow - 1 : (((row + krow - 1) % 2 == 0) ? (row + krow - 1)/2 : -1);	// row_in = ((row + krow + pad) % stride == pad) ? (krow + row - pad)/stride : -1;
							col_in = (stride == 1) ? col + kcol - 1 : (((col + kcol - 1) % 2 == 0) ? (col + kcol - 1)/2 : -1);
						}
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							act[cin] = input[cin][row_in][col_in];
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							if (ctrl_conv == 0) {
								wt[co][cin] = weight[co][cin][krow][kcol];
							}
							else {
								wt[co][cin] = weight[cin][co][kcol][krow];
							}
						}
					}
					// conv 3x3
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						accum[co] = 0;
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum[co] += act[cin] * wt[co][cin];
						}
						out_temp[co] += accum[co];
					}
					// write out
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						output[co][row][col] = out_temp[co];
					}
					if (ctrl_conv == 0) {
						for (int co = 0; co < CHANNEL_OUT_T; co ++) {
							output_DDR[co][row][col] = out_temp[co];
						}
					}
				}
			}
		}
	}
}

void conv_1x1_uni
(
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int8 output[CHANNEL_OUT_T][WIDTH][WIDTH],		// out on-chip
	int8 output_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],	// out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in,
	int1 ctrl_conv
)
{
	int row_in;
	int col_in;
	int8 act[CHANNEL_IN_T];
	int8 wt[CHANNEL_OUT_T][CHANNEL_IN_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=2 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output_DDR dim=1 complete

	for (int row = 0; row < H_fmap_out; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			// buffer initiation
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				if (c_in > 0) {
					out_temp[co] = output[co][row][col];
				}
				else {
					out_temp[co] = 0;
				}
			}
			// load activation
			for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
				if (ctrl_conv == 0) {
					row_in = row*2;
					col_in = col*2;
				}
				else {
					row_in = (row % 2 == 0) ? row/2 : -1;	// row_in = ((row + krow + pad) % stride == pad) ? (krow + row - pad)/stride : -1;
					col_in = (col % 2 == 0) ? col/2 : -1;
				}
				if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
					act[cin] = input[cin][row_in][col_in];
				}
			}
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
					if (ctrl_conv == 0) {
						wt[co][cin] = weight[co][cin];
					}
					else {
						wt[co][cin] = weight[cin][co];
					}
				}
			}
			// conv 3x3
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				accum[co] = 0;
				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
					accum[co] += act[cin] * wt[co][cin];
				}
				out_temp[co] += accum[co];
			}
			// write out
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				output[co][row][col] = out_temp[co];
			}
			if (ctrl_conv == 0) {
				for (int co = 0; co < CHANNEL_OUT_T; co ++) {
					output_DDR[co][row][col] = out_temp[co];
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
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// activation from DDR
	int8 weight[CHANNEL_OUT_T][WIDTH][WIDTH],			// error on-chip
	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],		// gradient on-chip

	int stride,
	int H_fmap_in
)
{
	int8 act[CHANNEL_IN_T];
	int8 wt[CHANNEL_OUT_T];
	int8 accum[CHANNEL_OUT_T][CHANNEL_IN_T];

#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=2 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete

	for (int krow = 0; krow < H_fmap_in; krow ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
#pragma HLS PIPELINE II=1
					// dilated conv
					for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
						int row_in = (stride == 1) ? row + krow - 1 : row + krow*2 - 1;		// row_in = (krow*stride) % stride == 0) ? row + krow*stride - pad : -1;
						int col_in = (stride == 1) ? col + kcol - 1 : col + kcol*2 - 1;		// 0-padding
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							act[ci] = input[ci][row_in][col_in];
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						wt[co] = weight[co][krow][kcol];
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
							accum[co][ci] += act[ci] * wt[co];
						}
					}
					// weight update
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
							output[co][ci][row][col] = output[co][ci][row][col] - lr * accum[co][ci];
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
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],		// activation from DDR
	int8 weight[CHANNEL_OUT_T][WIDTH][WIDTH],	// error on-chip
	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T],	// gradient on-chip

	int stride,
	int H_fmap_in
)
{
	int8 act[CHANNEL_IN_T];
	int8 wt[CHANNEL_OUT_T];
	int8 accum[CHANNEL_OUT_T][CHANNEL_IN_T];
#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=2 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete

	for (int krow = 0; krow < H_fmap_in; krow ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			// dilated conv
			for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
				int row_in = krow*2;	// row_in = (krow*stride) % stride == 0) ? row + krow*stride - pad : -1;
				int col_in = kcol*2;
				if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
					act[ci] = input[ci][row_in][col_in];
				}
			}
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				wt[co] = weight[co][krow][kcol];
			}
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					accum[co][ci] += act[ci] * wt[co];
				}
			}
			// weight update
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					output[co][ci] = output[co][ci] - lr * accum[co][ci];
				}
			}
		}
	}
}

#endif
