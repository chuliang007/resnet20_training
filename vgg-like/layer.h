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

using namespace std;
const uint2 exp_bias = 2;	// ieee-754 shared_exp_bias
float lr = 1e-2;
float eps = 1e-10;
//float ee = 2.718281828459;


//--------------------------------
// block minifloat qantisation
//--------------------------------

void quant_wt
(
	float input[CHANNEL_OUT_T][CHANNEL_IN_T][3][3]
)
{
	float input_tmp[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];

	// float exp = 2;
	// float man = 5;
	float emax = 1.0;  // 2**(exp)-1 - 2**(exp-1);
	float emin = -2.0; // -2**(exp-1);

	float max_exponent[CHANNEL_OUT_T] = {0};
	float offset[CHANNEL_OUT_T] = {0};
	float shift[CHANNEL_OUT_T] = {0};

	float max_number = 2*(2-1/16); // 2**(emax)*(2-2**(-man));
	float abs_max[CHANNEL_OUT_T] = {0};

	float esbn = 0.5; 	// 2**(emin+1)
	float lsbn = 2.0;   // 2**(emax)
	float mval = 32.0;  // 2**(man)

	float i;
	float ie;
	float me;
	float f;
	float clipped;
	float k;
	float out[CHANNEL_OUT_T][CHANNEL_IN_T][WIDTH][WIDTH];

	float e;
	float sgn;

	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		abs_max[co] = 0;
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
					if(abs_max[co] < abs(input[co][ci][krow][kcol])) {
						abs_max[co] = input[co][ci][krow][kcol];
					}
				}
			}
		}
	}

	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		max_exponent[co] = float(int(log2(abs_max[co])));

		// clamp
		if (max_exponent[co] < -128.0) {
			max_exponent[co] = -128.0;
		}
		else if (max_exponent[co] > 127.0) {
			max_exponent[co] = 127.0;
		}

		offset[co] = max_exponent[co] - emax;
//		printf("offset[%d] = %f \n", ci, offset[ci]);

		// shared exponent shifting
		shift[co] = pow(2, -offset[co]);
//		printf("shift[%d] = %f \n", ci, shift[ci]);
	}

	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {

					// handle subnormal and normal quantization
					input_tmp[co][ci][krow][kcol] = input[co][ci][krow][kcol] * shift[co];
					i = input_tmp[co][ci][krow][kcol];

					sgn = (i > 0) ? 1.0 : -1.0;
					i = abs(i);
					e = float(int(log2(i)));
					// clamp the exponent
					// e.clamp_(emin+1, emax); // emin+1 for subnormal region
					if (e < -1) {
						e = -1;
					}
					else if (e > 1) {
						e = 1;
					}

	//				printf("i = %f; e = %f \n", i, e);

					// unpack frac for subnormal and normal region
					ie = i * pow(2, -e);
					me = pow(2, e);
					// f = torch.where(i<esbn, ie, ie-1);
					if (i < esbn) {
						f = ie;
					}
					else {
						f = ie - 1;
					}

	//				printf("ie = %f; f = %f \n", ie, f);

					// rounding on frac
					// f.mul_(mval).round_()
					f = float(int(f * mval));
					// clipped.div_(mval).mul_(me)
					clipped = f/mval * me;

	//				printf("clipped = %f \n", clipped);

					// sign magnitude multiplication for subnormal and normal
					// k = torch.where(i<esbn, clipped, me+clipped)
					if (i < esbn) {
						k = clipped;
					}
					else {
						k = me+clipped;
					}

					// k.clamp_(-max_number, max_number);
					if (k < -max_number) {
						k = -max_number;
					}
					else if (k > max_number) {
						k = max_number;
					}

	//				printf("k = %f \n", k);

					out[co][ci][krow][kcol] = sgn * k * pow(2, offset[co]);

					// convert to fp32 after quantisation
					input[co][ci][krow][kcol] = out[co][ci][krow][kcol];

	//				printf("quant_out[%d][%d][%d] = %f \n", ci, krow, kcol, out[ci][krow][kcol]);
				}
			}
		}
	}

}

void quant_act
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],
	int H_fmap_in
)
{
	float input_tmp[CHANNEL_IN_T][WIDTH][WIDTH];

	// float exp = 2;
	// float man = 5;
	float emax = 1.0;  // 2**(exp)-1 - 2**(exp-1);
	float emin = -2.0; // -2**(exp-1);

	float max_exponent[CHANNEL_IN_T] = {0};
	float offset[CHANNEL_IN_T] = {0};
	float shift[CHANNEL_IN_T] = {0};

	float max_number = 2*(2-1/16); // 2**(emax)*(2-2**(-man));
	float abs_max[CHANNEL_IN_T] = {0};

	float esbn = 0.5; // 2**(emin+1)
	float lsbn = 2.0;   // 2**(emax)
	float mval = 32.0;  // 2**(man)

	float i;
	float ie;
	float me;
	float f;
	float clipped;
	float k;
	float out[CHANNEL_IN_T][WIDTH][WIDTH];

	float e;
	float sgn;

	for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
		abs_max[ci] = 0;
		for (int krow = 0; krow < H_fmap_in; krow ++) {
			for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
				if(abs_max[ci] < abs(input[ci][krow][kcol])) {
					abs_max[ci] = input[ci][krow][kcol];
				}
			}
		}
	}

	for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
		max_exponent[ci] = float(int(log2(abs_max[ci])));

		// clamp
		if (max_exponent[ci] < -128.0) {
			max_exponent[ci] = -128.0;
		}
		else if (max_exponent[ci] > 127.0) {
			max_exponent[ci] = 127.0;
		}

		offset[ci] = max_exponent[ci] - emax;
//		printf("offset[%d] = %f \n", ci, offset[ci]);

		// shared exponent shifting
		shift[ci] = pow(2, -offset[ci]);
//		printf("shift[%d] = %f \n", ci, shift[ci]);
	}

	for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
		for (int krow = 0; krow < H_fmap_in; krow ++) {
			for (int kcol = 0; kcol < H_fmap_in; kcol ++) {

				// handle subnormal and normal quantization
				input_tmp[ci][krow][kcol] = input[ci][krow][kcol] * shift[ci];
				i = input_tmp[ci][krow][kcol];

				sgn = (i > 0) ? 1.0 : -1.0;
				i = abs(i);
				e = float(int(log2(i)));
				// clamp the exponent
				// e.clamp_(emin+1, emax); // emin+1 for subnormal region
				if (e < -1) {
					e = -1;
				}
				else if (e > 1) {
					e = 1;
				}

//				printf("i = %f; e = %f \n", i, e);

				// unpack frac for subnormal and normal region
				ie = i * pow(2, -e);
				me = pow(2, e);
				// f = torch.where(i<esbn, ie, ie-1);
				if (i < esbn) {
					f = ie;
				}
				else {
					f = ie - 1;
				}

//				printf("ie = %f; f = %f \n", ie, f);

				// rounding on frac
				// f.mul_(mval).round_()
				f = float(int(f * mval));
				// clipped.div_(mval).mul_(me)
				clipped = f/mval * me;

//				printf("clipped = %f \n", clipped);

				// sign magnitude multiplication for subnormal and normal
				// k = torch.where(i<esbn, clipped, me+clipped)
				if (i < esbn) {
					k = clipped;
				}
				else {
					k = me+clipped;
				}

				// k.clamp_(-max_number, max_number);
				if (k < -max_number) {
					k = -max_number;
				}
				else if (k > max_number) {
					k = max_number;
				}

//				printf("k = %f \n", k);

				out[ci][krow][kcol] = sgn * k * pow(2, offset[ci]);

				// convert to fp32 after quantisation
				input[ci][krow][kcol] = out[ci][krow][kcol];

//				printf("quant_out[%d][%d][%d] = %f \n", ci, krow, kcol, out[ci][krow][kcol]);
			}
		}
	}
}

//--------------------------------
// floating-point golden reference
//--------------------------------

void rot180_3x3
(
	float weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	float weight_rot[CHANNEL_IN_T][CHANNEL_OUT_T][3][3]
)
{
	float act;
	float wt;
	float weight_tmp[CHANNEL_IN_T][CHANNEL_OUT_T][3][3];

	// transpose
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
					weight_tmp[cin][co][kcol][krow] = weight[co][cin][krow][kcol];
				}
			}
		}
	}
	// diagonal
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
			weight_tmp[cin][co][0][0] = weight[cin][co][2][2];
			weight_tmp[cin][co][2][2] = weight[cin][co][0][0];
		}
	}
	// write back
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
					weight_rot[cin][co][krow][kcol] = weight_tmp[cin][co][krow][kcol];
				}
			}
		}
	}
}

void rot180_1x1
(
	float weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	float weight_rot[CHANNEL_IN_T][CHANNEL_OUT_T]
)
{
	float act;
	float wt;

	// rotate 180
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
			weight_rot[cin][co] = weight[co][cin];
		}
	}
}

void conv_3x3
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	float weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	float output[CHANNEL_OUT_T][WIDTH][WIDTH],			// out on-chip
	float output_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in,
	uint1 ctrl_conv,

	float bias_gap4add[CHANNEL_OUT_T]
)
{
// #pragma HLS DEPENDENCE variable=output inter false

	int row_input;
	int col_input;
	int row_in;
	int col_in;
	float out_temp[CHANNEL_OUT_T];
	float output_tmp[CHANNEL_OUT_T][WIDTH][WIDTH];
//#pragma HLS DEPENDENCE variable=out_temp inter false
//#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output_tmp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=3 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=4 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output_DDR dim=1 complete

//	float max_abs[CHANNEL_OUT_T] = {0};
//#pragma HLS ARRAY_PARTITION variable=max_abs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=shared_exp_bias dim=1 complete

	float line_buffer_act[CHANNEL_IN_T][2][WIDTH] = {0};
	float window_buffer_act[CHANNEL_IN_T][3][3] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=2
#pragma HLS ARRAY_PARTITION variable=window_buffer_act complete dim=0

	for (int ii = 0; ii < stride; ii ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
		for (int row = 0; row < H_fmap_out; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
			for (int jj = 0; jj < stride; jj ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
				for (int col = 0; col < H_fmap_out; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
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

					// update window buffer and line buffer
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						for (int i = 0; i < 3; i ++) {
							window_buffer_act[cin][i][0] = window_buffer_act[cin][i][1];
							window_buffer_act[cin][i][1] = window_buffer_act[cin][i][2];
						}
						// stride-2 transposed Conv
						if (stride > 1 && ctrl_conv == 1) {
							row_input = row/2;
							col_input = col/2;
							window_buffer_act[cin][0][2] = (line_buffer_act[cin][0][col]);
							window_buffer_act[cin][1][2] = (line_buffer_act[cin][0][col] = line_buffer_act[cin][1][col]);
							window_buffer_act[cin][2][2] = (line_buffer_act[cin][1][col] = (row_input % 2 == 0 && col_input % 2 == 0) ? input[cin][row_input][col_input] : float(0));	// dilated with 0-row
						}
						else {
							row_input = row + ii*H_fmap_out;
							col_input = col + jj*H_fmap_out;
							window_buffer_act[cin][0][2] = (line_buffer_act[cin][0][col]);
							window_buffer_act[cin][1][2] = (line_buffer_act[cin][0][col] = line_buffer_act[cin][1][col]);
							window_buffer_act[cin][2][2] = (line_buffer_act[cin][1][col] = input[cin][row_input][col_input]);
						}
					}

					// conv 3x3
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						float accum = 0;
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
								for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
									row_in = (stride > 1 && ctrl_conv == 0) ? (row + krow - 2 + ii*H_fmap_out) : (row + krow - 2);
									col_in = (stride > 1 && ctrl_conv == 0) ? (col + kcol - 2 + jj*H_fmap_out) : (col + kcol - 2);
									if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
										float act = window_buffer_act[cin][krow][kcol];
										float wt = weight[co][cin][krow][kcol];
										accum += act * wt;
										// accum = bm_mac(act, wt, accum);
									}
								}
							}
						}
						out_temp[co] += accum;
//						out_temp[co] = bm_add(accum, out_temp[co], bias_gap4add[co]);
						if (stride > 1 && ctrl_conv == 0) {
							output_tmp[co][row + ii*H_fmap_out][col + jj*H_fmap_out] = out_temp[co];
						}
					}

					// write out
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						if (stride > 1 && ctrl_conv == 0) {
							output[co][row][col] = output_tmp[co][row*2][col*2];
						} else {
							output[co][row][col] = out_temp[co];
						}
					}
					if (ctrl_conv == 0) {
						for (int co = 0; co < CHANNEL_OUT_T; co ++) {
							output_DDR[co][row][col] = out_temp[co];
//							if (out_temp[co].range(1,0) > max_abs[co].range(1,0)) max_abs[co] = out_temp[co];
						}
					}
				}
			}
		}
	}
}

void conv_1x1
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	float weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	float output[CHANNEL_OUT_T][WIDTH][WIDTH],		// out on-chip
	float output_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],	// out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in,
	uint1 ctrl_conv,

//	float shared_exp_bias[CHANNEL_OUT_T],
	float bias_gap4add[CHANNEL_OUT_T]
//	float act_bias_shift[CHANNEL_OUT_T]
)
{
// #pragma HLS DEPENDENCE variable=output inter false

	int row_in;
	int col_in;
	float act[CHANNEL_IN_T];
	float wt[CHANNEL_OUT_T][CHANNEL_IN_T];
	float accum[CHANNEL_OUT_T];
	float out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=2 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

//	float max_abs[CHANNEL_OUT_T] = {0};
//#pragma HLS ARRAY_PARTITION variable=max_abs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=shared_exp_bias dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output_DDR dim=1 complete

	for (int row = 0; row < H_fmap_out; row++) {
// // #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_out; col++) {
// // #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
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
//					accum[co] = bm_mac(act[cin], wt[co][cin], accum[co]);
				}
				out_temp[co] += accum[co];
//				out_temp[co] = bm_add(out_temp[co], accum[co], bias_gap4add[co]);
			}
			// write out
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				output[co][row][col] = out_temp[co];
			}
			if (ctrl_conv == 0) {
				for (int co = 0; co < CHANNEL_OUT_T; co ++) {
					output_DDR[co][row][col] = out_temp[co];
//					if (out_temp[co].range(1,0) > max_abs[co].range(1,0)) max_abs[co] = out_temp[co];
				}
			}
		}
	}
}

void conv_3x3_grad
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],				// in on-chip
	float weight[CHANNEL_OUT_T][WIDTH][WIDTH],
	float output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],		// out on-chip

	int stride,
	int H_fmap_in
)
{
	float accum;
	float weight_dil[CHANNEL_OUT_T][WIDTH][WIDTH];

	// buffer init
	for (int row_in = 0; row_in < WIDTH; row_in ++) {
		for (int col_in = 0; col_in < WIDTH; col_in ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				weight_dil[co][row_in][col_in] = 0;
			}
		}
	}

	// weight dilation
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int krow = 0; krow < H_fmap_in; krow ++) {
			for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
				weight_dil[co][krow*stride][kcol*stride] = weight[co][krow][kcol];
			}
		}
	}

	// conv3x3 grad
	for (int row = 0; row < 3; row ++) {
		for (int col = 0; col < 3; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					accum = 0;
					for (int krow = 0; krow < H_fmap_in*stride; krow ++) {
						for (int kcol = 0; kcol < H_fmap_in*stride; kcol ++) {
							int row_in = row + krow - 1;
							int col_in = col + kcol - 1;
							if (row_in >= 0 && row_in < H_fmap_in*stride && col_in >= 0 && col_in < H_fmap_in*stride) {
								float act = input[ci][row_in][col_in];
								float wt = weight_dil[co][krow][kcol];
								accum += act * wt;
//								if (act != 0 && wt != 0) printf("act: %f, wt: %f \n", act, wt);
							}
						}
					}
					output[co][ci][row][col] += -lr * accum;
//					if (accum != 0) printf("weight grad [%d][%d][%d][%d]: %f \n", co, ci, row, col, accum);
//					if (accum != 0 && abs(output[co][ci][row][col]) > 1) printf("stride: %d, updated weight [%d][%d][%d][%d]: %f \n", stride, co, ci, row, col, output[co][ci][row][col]);
				}
			}
		}
	}
}

void conv_1x1_grad
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	float weight[CHANNEL_OUT_T][WIDTH][WIDTH],
	float output[CHANNEL_OUT_T][CHANNEL_IN_T],			// out on-chip

	int stride,
	int H_fmap_in
)
{
	int row_input;
	int col_input;
	int row_in;
	int col_in;

	uint2 skip_krow = 0;
	uint2 skip_kcol = 0;

	float wt;
	float act;
	float accum;
	static float out_temp[CHANNEL_OUT_T][CHANNEL_IN_T] = {0};
#pragma HLS DEPENDENCE variable=out_temp inter false
#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete

	static float line_buffer_act[CHANNEL_IN_T][3][WIDTH] = {0};
	static float window_buffer_act[CHANNEL_IN_T][4][4] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_act complete dim=0

	static float line_buffer_wt[CHANNEL_IN_T][3][WIDTH] = {0};
	static float window_buffer_wt[CHANNEL_IN_T][4][4] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_wt complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_wt complete dim=0

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete
#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete

	for (int ii = 0; ii < stride; ii ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
		for (int krow = 0; krow < H_fmap_in; krow ++) {
			skip_krow += 1;
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
			for (int jj = 0; jj < stride; jj ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
				for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
					skip_kcol += 1;
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1

					// activation- update window buffer and line buffer
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						for (int i = 0; i < 4; i ++) {
							window_buffer_act[cin][i][0] = window_buffer_act[cin][i][1];
							window_buffer_act[cin][i][1] = window_buffer_act[cin][i][2];
							window_buffer_act[cin][i][2] = window_buffer_act[cin][i][3];
						}

						row_input = krow + ii*H_fmap_in;
						col_input = kcol + jj*H_fmap_in;

						window_buffer_act[cin][0][3] = (line_buffer_act[cin][0][kcol]);
						window_buffer_act[cin][1][3] = (line_buffer_act[cin][0][kcol] = line_buffer_act[cin][1][kcol]);
						window_buffer_act[cin][2][3] = (line_buffer_act[cin][1][kcol] = line_buffer_act[cin][2][kcol]);
						window_buffer_act[cin][3][3] = (line_buffer_act[cin][2][kcol] = input[cin][row_input][col_input]);
//						printf("skip_krow: %d, skip_kcol: %d;    input[%d][%d][%d]: %d \n", skip_krow.to_int(), skip_kcol.to_int(), cin, row_input, col_input, input[cin][row_input][col_input].to_int());
					}

					// dilated weight- update window buffer and line buffer
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int i = 0; i < 4; i ++) {
							window_buffer_wt[co][i][0] = window_buffer_wt[co][i][1];
							window_buffer_wt[co][i][1] = window_buffer_wt[co][i][2];
							window_buffer_wt[co][i][2] = window_buffer_wt[co][i][3];
						}

						window_buffer_wt[co][0][3] = (line_buffer_wt[co][0][kcol]);
						window_buffer_wt[co][1][3] = (line_buffer_wt[co][0][kcol] = line_buffer_wt[co][1][kcol]);
						window_buffer_wt[co][2][3] = (line_buffer_wt[co][1][kcol] = line_buffer_wt[co][2][kcol]);
						window_buffer_wt[co][3][3] = (line_buffer_wt[co][2][kcol]);
						line_buffer_wt[co][2][kcol] = (stride > 1 && krow % 2 > 0 && kcol % 2 > 0) ? float(0) : weight[co][krow][kcol];
//						printf("skip_krow: %d, skip_kcol: %d;    weight[%d][%d][%d]: %d \n", skip_krow.to_int(), skip_kcol.to_int(), co, krow, kcol, weight[co][krow][kcol].to_int());
					}

					// dilated conv
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							// row_in = krow;
							// col_in = kcol;
							accum = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow][pkcol];
									act = window_buffer_act[cin][pkrow][pkcol];
									accum += wt * act;

									// printf("skip_krow = %d, skip_kcol = %d, ", skip_krow.to_int(), skip_kcol.to_int());
									// printf("accum = %d \n", accum.to_int());
								}
							}
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip_krow == 0 && skip_kcol == 0) {
								out_temp[co][cin] += accum;
//								printf("skip_krow = %d, skip_kcol = %d, ", skip_krow.to_int(), skip_kcol.to_int());
								// printf("out_temp[%d][%d] = %d \n", co, cin, out_temp[co][cin].to_int());
							}
						}
					}
				}
			}
		}
	}
	// write out
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS PIPELINE II=2
		for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
			output[co][cin] += -lr * out_temp[co][cin];
		}
	}
}

// AvgPool
void avgpool(
	float avg_inputs[CHANNEL_IN_T][WIDTH][WIDTH],	// in, 64x8x8
	float out_buf[64],								// out, avg_outputs

	uint1 ctrl_avgpool,	// 0 for forward and 1 for backward
	int c_out,

	float out_buf_copy[64],
	float bias_gap4add[CHANNEL_IN_T]
)
{
	// forward
	if (ctrl_avgpool == 0) {
		// buffer init
		for (int c = 0; c < 64; c ++) {
			out_buf[c] = 0;
		}
		for (int c = 0; c < CHANNEL_IN_T; c ++) {
			for (int s = 0; s < 8; s ++) {
				for (int ss = 0; ss < 8; ss ++) {
					 out_buf[c + c_out*CHANNEL_IN_T] += avg_inputs[c][s][ss]/64;
					 out_buf_copy[c + c_out*CHANNEL_IN_T] += avg_inputs[c][s][ss]/64;
				}
			}
//			printf("avgpool out: %f \n", out_buf[c + c_out*CHANNEL_IN_T]);
		}
	}
	// backward
	else {
		for (int s = 0; s < 8; s ++) {
			for (int ss = 0; ss < 8; ss ++) {
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
					avg_inputs[c][s][ss] = out_buf[c + c_out*CHANNEL_IN_T]/64;
				}
			}
		}
	}
}

// FC
void FC(
	float inputs[64],
	float inputs_FW[64],
	float linear_weight[10][64],
	float outputs[10],

	uint1 ctrl_fc	// 0 for forward and 1 for backward
)
{
	float linear_weight_t[64][10];

	// forward
	if (ctrl_fc == 0) {
		// buffer init
		for (int coo = 0; coo < 10; coo ++) {
			outputs[coo] = 0;
		}
		for (int coo = 0; coo < 10; coo ++) {
			for (int cii = 0; cii < 64; cii++) {
				outputs[coo] += inputs[cii] * linear_weight[coo][cii];
			}
//			printf("fc out: %f \n", outputs[coo]);
		}
	}
	// backward
	else {
		// buffer init
		for (int cii = 0; cii < 10; cii ++) {
			inputs[cii] = 0;
		}
		for (int cii = 0; cii < 64; cii++) {
			for (int coo = 0; coo < 10; coo ++) {
				linear_weight_t[cii][coo] = linear_weight[coo][cii];
				inputs[cii] += outputs[coo] * linear_weight_t[cii][coo];
			}
		}
		// weight update
		for (int cii = 0; cii < 64; cii++) {
			for (int coo = 0; coo < 10; coo ++) {
				linear_weight[coo][cii] += -lr * inputs_FW[cii] * outputs[coo];
			}
		}
	}
}

// bn_relu
void bn_relu(
	float bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs

	float out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip for backprop
	uint1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp
//	float out_buf_SC[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, ideneity for shortcut

	float bn_wt[CHANNEL_OUT_T],
	float bn_bias[CHANNEL_OUT_T],

    int H_fmap_in,
//	uint1 ctrl_bn_id,

//	float shared_exp_bias[CHANNEL_OUT_T],
	float bias_gap4add[CHANNEL_OUT_T]
//	float act_bias_shift[CHANNEL_OUT_T]
)
{
	int N = H_fmap_in * H_fmap_in;
	float mu[CHANNEL_OUT_T];
	float std_var[CHANNEL_OUT_T];
	float sum[CHANNEL_OUT_T] = {0};
#pragma HLS ARRAY_PARTITION variable=bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=std_var dim=1 complete

	uint1 relu_temp[CHANNEL_OUT_T] = {0};
	float in_temp[CHANNEL_OUT_T] = {0};
	float out_temp[CHANNEL_OUT_T] = {0};
#pragma HLS ARRAY_PARTITION variable=relu_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

//	float max_abs[CHANNEL_OUT_T] = {0};
//#pragma HLS ARRAY_PARTITION variable=max_abs dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=shared_exp_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete

#pragma HLS ARRAY_PARTITION variable=bn_inputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR dim=1 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=1 complete

	// mean
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs[c][row][col];
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				sum[c] += in_temp[c];
			}
		}
	}
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] = sum[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs[c][row][col];
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (in_temp[c]-mu[c])*(in_temp[c]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE II=1
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// bn_relu
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs[c][row][col];
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_temp[c] = bn_wt[c]*(in_temp[c]-mu[c])/(std_var[c] + eps) + bn_bias[c];
				relu_temp[c] = (out_temp[c] < 0) ? 0 : 1;
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				relu_mask[c][row][col] = relu_temp[c];
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = (relu_temp[c] == 0) ? float(0) : out_temp[c];
				out_buf_DDR[c][row][col] = (relu_temp[c] == 0) ? float(0) : out_temp[c];
			}
		}
	}
}

void bn_relu_bp(
	float error[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	float bn_inputs_fw[CHANNEL_OUT_T][WIDTH][WIDTH],	// in
	uint1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	float bn_wt[CHANNEL_OUT_T],
	float bn_bias[CHANNEL_OUT_T],

	int H_fmap_in,
	float bias_gap4add[CHANNEL_OUT_T]
)
{
	int N = H_fmap_in * H_fmap_in;
	float mu[CHANNEL_OUT_T];
	float std_var[CHANNEL_OUT_T] = {0};
	float sum[CHANNEL_OUT_T] = {0};
	float g_bn_wt[CHANNEL_OUT_T] = {0};					// out
	float g_bn_bias[CHANNEL_OUT_T] = {0};				// out
#pragma HLS ARRAY_PARTITION variable=bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=std_var dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_bn_bias dim=1 complete

	float error_temp[CHANNEL_OUT_T] = {0};
	float in_temp[CHANNEL_OUT_T] = {0};
	float out_temp[CHANNEL_OUT_T] = {0};
#pragma HLS ARRAY_PARTITION variable=error_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=error dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw dim=1 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete

	// temp buffer init
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = 0;
		sum[c] = 0;
		g_bn_bias[c] = 0;
		g_bn_wt[c] = 0;
	}

	// relu_bp and mean
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				sum[c] += in_temp[c];
			}
		}
	}
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] = sum[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (in_temp[c]-mu[c])*(in_temp[c]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE II=1
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// grad of bn params
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
				error_temp[c] = (relu_mask[c][row][col] == 1) ? error[c][row][col] : float(0);
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				g_bn_bias[c] += error_temp[c];
				g_bn_wt[c] += error_temp[c] * (in_temp[c]-mu[c])/(std_var[c]+eps);
			}
		}
	}

	// bn_bp
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
				error_temp[c] = (relu_mask[c][row][col] == 1) ? error[c][row][col] : float(0);
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_temp[c] = bn_wt[c]*error_temp[c]/(std_var[c]+eps) - bn_wt[c]*g_bn_bias[c]/(N*(std_var[c]+eps)) - bn_wt[c]*(in_temp[c]-mu[c])*g_bn_wt[c]/(N*(std_var[c]*std_var[c]+eps));
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
			}
		}
	}
	
	// bn_sw params update
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE II=1
		bn_bias[c] += -lr * g_bn_bias[c];
		bn_wt[c] += -lr * g_bn_wt[c];
	}
}

#endif
