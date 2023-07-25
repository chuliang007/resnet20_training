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
float lr = 0.02; //1e-2;
float eps = 1e-10;
//float ee = 2.718281828459;
float mom = 0.9;

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
		max_exponent[co] = float(int(hls::log2(abs_max[co])));

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
					e = float(int(hls::log2(i)));
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
		max_exponent[ci] = float(int(hls::log2(abs_max[ci])));

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
				e = float(int(hls::log2(i)));
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
	uint1 ctrl_conv
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
									row_in = (stride > 1 && ctrl_conv == 0) ? (row + krow - 1 + ii*H_fmap_out) : (row + krow - 1);
									col_in = (stride > 1 && ctrl_conv == 0) ? (col + kcol - 1 + jj*H_fmap_out) : (col + kcol - 1);
									if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
										float act = window_buffer_act[cin][krow][kcol];
										float wt = weight[co][cin][krow][kcol];
										accum += act * wt;
									}
								}
							}
						}
						out_temp[co] += accum;
						if (stride > 1 && ctrl_conv == 0) {
							output_tmp[co][row + ii*H_fmap_out][col + jj*H_fmap_out] = out_temp[co];
						}
					}

					// write out
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						if (stride > 1 && ctrl_conv == 0) {
							out_temp[co] = output_tmp[co][row*2][col*2];
						}
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

void conv_3x3_grad
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],				// activation from DDR
	float weight[CHANNEL_OUT_T][WIDTH][WIDTH],				// error on-chip
	float output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],		// gradient on-chip
	float vel[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],

	uint1 ctrl_frz,

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
	float out_tmp;
	float out_temp[CHANNEL_OUT_T][CHANNEL_IN_T][3][3] = {0};
#pragma HLS DEPENDENCE variable=out_temp inter false
//#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=3 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=4 complete

	float line_buffer_act[CHANNEL_IN_T][4][WIDTH] = {0};
	float window_buffer_act[CHANNEL_IN_T][5][5] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_act complete dim=0

	float line_buffer_wt[CHANNEL_IN_T][4][WIDTH] = {0};
	float window_buffer_wt[CHANNEL_IN_T][5][5] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_wt complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_wt complete dim=0

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=3 complete
#pragma HLS ARRAY_PARTITION variable=output dim=4 complete

	for (int ii = 0; ii < stride; ii ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
		for (int krow = 0; krow < H_fmap_in + 1; krow ++) {
			skip_krow += 1;
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
			for (int jj = 0; jj < stride; jj ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
				for (int kcol = 0; kcol < H_fmap_in + 1; kcol ++) {
					skip_kcol += 1;
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1

					// dilated weight- update window buffer and line buffer
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int i = 0; i < 5; i ++) {
							window_buffer_wt[co][i][0] = window_buffer_wt[co][i][1];
							window_buffer_wt[co][i][1] = window_buffer_wt[co][i][2];
							window_buffer_wt[co][i][2] = window_buffer_wt[co][i][3];
							window_buffer_wt[co][i][3] = window_buffer_wt[co][i][4];
						}

						window_buffer_wt[co][0][4] = (line_buffer_wt[co][0][kcol]);
						window_buffer_wt[co][1][4] = (line_buffer_wt[co][0][kcol] = line_buffer_wt[co][1][kcol]);
						window_buffer_wt[co][2][4] = (line_buffer_wt[co][1][kcol] = line_buffer_wt[co][2][kcol]);
						window_buffer_wt[co][3][4] = (line_buffer_wt[co][2][kcol] = line_buffer_wt[co][3][kcol]);
						window_buffer_wt[co][4][4] = (line_buffer_wt[co][3][kcol]);
						line_buffer_wt[co][3][kcol] = ((stride > 1 && krow % 2 > 0 && kcol % 2 > 0) || krow >= WIDTH || kcol >= WIDTH) ? float(0) : weight[co][krow][kcol];
					}

					// activation- update window buffer and line buffer (padding = 1)
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						for (int i = 0; i < 5; i ++) {
							window_buffer_act[cin][i][0] = window_buffer_act[cin][i][1];
							window_buffer_act[cin][i][1] = window_buffer_act[cin][i][2];
							window_buffer_act[cin][i][2] = window_buffer_act[cin][i][3];
							window_buffer_act[cin][i][3] = window_buffer_act[cin][i][4];
						}

						row_input = krow + ii*H_fmap_in;
						col_input = kcol + jj*H_fmap_in;

						window_buffer_act[cin][0][4] = (line_buffer_act[cin][0][kcol]);
						window_buffer_act[cin][1][4] = (line_buffer_act[cin][0][kcol] = line_buffer_act[cin][1][kcol]);
						window_buffer_act[cin][2][4] = (line_buffer_act[cin][1][kcol] = line_buffer_act[cin][2][kcol]);
						window_buffer_act[cin][3][4] = (line_buffer_act[cin][2][kcol] = line_buffer_act[cin][3][kcol]);
						window_buffer_act[cin][4][4] = (line_buffer_act[cin][3][kcol]);
						line_buffer_act[cin][3][kcol] = (row_input >= WIDTH || col_input >= WIDTH) ? float(0) : input[cin][row_input][col_input];
					}

					/////////////////
					// dilated conv_0
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow + 1][pkcol + 1];
									act = window_buffer_act[cin][pkrow][pkcol];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(0, 0)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip_krow == 0 && skip_kcol == 2) {
								out_temp[co][cin][0][0] += out_tmp;
							}
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow + 1][pkcol + 1];
									act = window_buffer_act[cin][pkrow][pkcol + 1];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(0, 1)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip_krow == 0 && skip_kcol == 3) {
								out_temp[co][cin][0][1] += out_tmp;
							}
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow + 1][pkcol];
									act = window_buffer_act[cin][pkrow][pkcol + 1];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(0, 2)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip_krow == 0 && skip_kcol == 0) {
								out_temp[co][cin][0][2] += out_tmp;
							}
						}
					}
					/////////////////
					// dilated conv_1
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow + 1][pkcol + 1];
									act = window_buffer_act[cin][pkrow + 1][pkcol];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(1, 0)
							if (krow >= 0 && krow < H_fmap_in + 1 && kcol >= 0 && kcol < H_fmap_in - 1 && skip_krow == 0 && skip_kcol == 2) {
								out_temp[co][cin][1][0] += out_tmp;
							}
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow + 1][pkcol + 1];
									act = window_buffer_act[cin][pkrow + 1][pkcol + 1];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(1, 1)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip_krow == 0 && skip_kcol == 3) {
								out_temp[co][cin][1][1] += out_tmp;
							}
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow + 1][pkcol];
									act = window_buffer_act[cin][pkrow + 1][pkcol + 1];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(1, 2)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip_krow == 0 && skip_kcol == 0) {
								out_temp[co][cin][1][2] += out_tmp;
//								printf("out_temp[%d][%d][1][2] = %d \n", co, cin, out_temp[co][cin][1][2].to_int());
							}
						}
					}
					/////////////////
					// dilated conv_2
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow][pkcol + 1];
									act = window_buffer_act[cin][pkrow + 1][pkcol];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(2, 0)
							if (krow >= 1 && krow < H_fmap_in + 1 && kcol >= 0 && kcol < H_fmap_in - 1 && skip_krow == 0 && skip_kcol == 2) {
								out_temp[co][cin][2][0] += out_tmp;
							}
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow][pkcol + 1];
									act = window_buffer_act[cin][pkrow + 1][pkcol + 1];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(2, 1)
							if (krow >= 1 && krow < H_fmap_in + 1 && kcol >= 0 && kcol < H_fmap_in && skip_krow == 0 && skip_kcol == 3) {
								out_temp[co][cin][2][1] += out_tmp;
							}
						}
					}
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							out_tmp = 0;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									wt = window_buffer_wt[co][pkrow][pkcol];
									act = window_buffer_act[cin][pkrow + 1][pkcol + 1];
									accum = wt * act;
									out_tmp += accum;
								}
							}
							// conv(2, 2)
							if (krow >= 1 && krow < H_fmap_in + 1 && kcol >= 1 && kcol < H_fmap_in + 1 && skip_krow == 0 && skip_kcol == 0) {
								out_temp[co][cin][2][2] += out_tmp;
							}

						}
					}

				}
			}
		}
	}

	// write out
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS PIPELINE II=3
		for (int row = 0; row < 3; row ++) {
			for (int col = 0; col < 3; col ++) {
				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
					if (ctrl_frz != 0) {
//						output[co][cin][row][col] += -lr* out_temp[co][cin][row][col];
						vel[co][cin][row][col] = vel[co][cin][row][col]*mom + lr*out_temp[co][cin][row][col];
						output[co][cin][row][col] -= vel[co][cin][row][col];
					}
				}
			}
		}
	}
}

void conv_3x3_grad_v2
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],				// activation from DDR
	float weight[CHANNEL_OUT_T][WIDTH][WIDTH],				// error on-chip
	float output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],		// gradient on-chip
	float vel[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],

	uint1 ctrl_frz,

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
	uint2 skip = 1;

	float wt[CHANNEL_OUT_T];
	float act[CHANNEL_IN_T];
	float out_tmp[CHANNEL_OUT_T][CHANNEL_IN_T];
	float out_temp[CHANNEL_OUT_T][CHANNEL_IN_T][3][3] = {0};

#pragma HLS DEPENDENCE variable=out_temp inter false
#pragma HLS DEPENDENCE variable=vel inter false

#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=2 complete

#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=act dim=2 complete

#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=2 complete

#pragma HLS ARRAY_PARTITION variable=out_tmp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_tmp dim=2 complete

#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=3 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=4 complete

	float line_buffer_act[CHANNEL_IN_T][4][WIDTH] = {0};
	float window_buffer_act[CHANNEL_IN_T][5][5] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_act complete dim=0

	float line_buffer_wt[CHANNEL_IN_T][4][WIDTH] = {0};
	float window_buffer_wt[CHANNEL_IN_T][5][5] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_wt complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_wt complete dim=0

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete
#pragma HLS ARRAY_PARTITION variable=output dim=3 complete
#pragma HLS ARRAY_PARTITION variable=output dim=4 complete

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

					// dilated weight- update window buffer and line buffer
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int i = 0; i < 5; i ++) {
							window_buffer_wt[co][i][0] = window_buffer_wt[co][i][1];
							window_buffer_wt[co][i][1] = window_buffer_wt[co][i][2];
							window_buffer_wt[co][i][2] = window_buffer_wt[co][i][3];
							window_buffer_wt[co][i][3] = window_buffer_wt[co][i][4];
						}

						window_buffer_wt[co][0][4] = (line_buffer_wt[co][0][kcol]);
						window_buffer_wt[co][1][4] = (line_buffer_wt[co][0][kcol] = line_buffer_wt[co][1][kcol]);
						window_buffer_wt[co][2][4] = (line_buffer_wt[co][1][kcol] = line_buffer_wt[co][2][kcol]);
						window_buffer_wt[co][3][4] = (line_buffer_wt[co][2][kcol] = line_buffer_wt[co][3][kcol]);
						window_buffer_wt[co][4][4] = (line_buffer_wt[co][3][kcol]);
						line_buffer_wt[co][3][kcol] = (stride > 1 && krow % 2 > 0 && kcol % 2 > 0) ? float(0) : weight[co][krow][kcol];
					}

					// activation- update window buffer and line buffer (padding = 1)
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						for (int i = 0; i < 5; i ++) {
							window_buffer_act[cin][i][0] = window_buffer_act[cin][i][1];
							window_buffer_act[cin][i][1] = window_buffer_act[cin][i][2];
							window_buffer_act[cin][i][2] = window_buffer_act[cin][i][3];
							window_buffer_act[cin][i][3] = window_buffer_act[cin][i][4];
						}

						row_input = krow + ii*H_fmap_in;
						col_input = kcol + jj*H_fmap_in;

						window_buffer_act[cin][0][4] = (line_buffer_act[cin][0][kcol]);
						window_buffer_act[cin][1][4] = (line_buffer_act[cin][0][kcol] = line_buffer_act[cin][1][kcol]);
						window_buffer_act[cin][2][4] = (line_buffer_act[cin][1][kcol] = line_buffer_act[cin][2][kcol]);
						window_buffer_act[cin][3][4] = (line_buffer_act[cin][2][kcol] = line_buffer_act[cin][3][kcol]);
						window_buffer_act[cin][4][4] = (line_buffer_act[cin][3][kcol]);
						line_buffer_act[cin][3][kcol] = input[cin][row_input][col_input];
					}

					/////////////////
					// dilated conv_0
					skip = 1;
					if (skip_krow == 0) skip = skip_kcol;

					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {

							out_tmp[co][cin] = 0;

							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {
									// conv(0,0)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
										wt[co] = window_buffer_wt[co][pkrow + 1][pkcol + 1];
										act[cin] = window_buffer_act[cin][pkrow][pkcol];
									}
									// conv(0, 1)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip == 3) {
										wt[co] = window_buffer_wt[co][pkrow + 1][pkcol + 1];
										act[cin] = window_buffer_act[cin][pkrow][pkcol + 1];
									}
									// conv(0,2)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip == 0) {
										wt[co] = window_buffer_wt[co][pkrow + 1][pkcol];
										act[cin] = window_buffer_act[cin][pkrow][pkcol + 1];
									}

									// conv(1, 0)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
										wt[co] = window_buffer_wt[co][pkrow + 1][pkcol + 1];
										act[cin] = window_buffer_act[cin][pkrow + 1][pkcol];
									}
									// conv(1, 1)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip == 3) {
										wt[co] = window_buffer_wt[co][pkrow + 1][pkcol + 1];
										act[cin] = window_buffer_act[cin][pkrow + 1][pkcol + 1];
									}
									// conv(1, 2)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip == 0) {
										wt[co] = window_buffer_wt[co][pkrow + 1][pkcol];
										act[cin] = window_buffer_act[cin][pkrow + 1][pkcol + 1];
									}

									// conv(2, 0)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
										wt[co] = window_buffer_wt[co][pkrow][pkcol + 1];
										act[cin] = window_buffer_act[cin][pkrow + 1][pkcol];
									}
									// conv(2, 1)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip == 3) {
										wt[co] = window_buffer_wt[co][pkrow][pkcol + 1];
										act[cin] = window_buffer_act[cin][pkrow + 1][pkcol + 1];
									}
									// conv(2, 2)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip == 0) {
										wt[co] = window_buffer_wt[co][pkrow][pkcol];
										act[cin] = window_buffer_act[cin][pkrow + 1][pkcol + 1];
									}
								}
							}
							out_tmp[co][cin] += wt[co] * act[cin];
						}
					}

					skip = 1;
					if (skip_krow == 0) skip = skip_kcol;

					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {

							// conv(0,0)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
								out_temp[co][cin][0][0] += out_tmp[co][cin];
							}
							// conv(0, 1)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip == 3) {
								out_temp[co][cin][0][1] += out_tmp[co][cin];
							}
							// conv(0,2)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip == 0) {
								out_temp[co][cin][0][2] += out_tmp[co][cin];
							}

							// conv(1, 0)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
								out_temp[co][cin][1][0] += out_tmp[co][cin];
							}
							// conv(1, 1)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip == 3) {
								out_temp[co][cin][1][1] += out_tmp[co][cin];
							}
							// conv(1, 2)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip == 0) {
								out_temp[co][cin][1][2] += out_tmp[co][cin];
							}

							// conv(2, 0)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
								out_temp[co][cin][2][0] += out_tmp[co][cin];
							}
							// conv(2, 1)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip == 3) {
								out_temp[co][cin][2][1] += out_tmp[co][cin];
							}
							// conv(2, 2)
							if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip == 0) {
								out_temp[co][cin][2][2] += out_tmp[co][cin];
							}

						}
					}

				}
			}
		}
	}

	// write out
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
#pragma HLS PIPELINE II=3
		for (int row = 0; row < 3; row ++) {
			for (int col = 0; col < 3; col ++) {
				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
					if (ctrl_frz != 0) {
//						output[co][cin][row][col] += -lr* out_temp[co][cin][row][col];
						vel[co][cin][row][col] = vel[co][cin][row][col]*mom + lr*out_temp[co][cin][row][col];
						output[co][cin][row][col] -= vel[co][cin][row][col];
					}
				}
			}
		}
	}
}

// dataflow wrapper
void conv_3x3_backward (
	// transposed conv
	float input[CHANNEL_IN_T][WIDTH][WIDTH],			// error as input on-chip
	float input_copy[CHANNEL_IN_T][WIDTH][WIDTH],		// error as input on-chip
	float weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	float output[CHANNEL_OUT_T][WIDTH][WIDTH],			// error as output on-chip
	float output_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// not used in bp

	// dilated conv
	float input_fw[CHANNEL_IN_T][WIDTH][WIDTH],			// activation from DDR
	float wt_out[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],	// gradient on-chip
	float vel[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in
) {
#pragma HLS ALLOCATION function instances=conv_3x3 limit=1
#pragma HLS ALLOCATION function instances=conv_3x3_grad limit=1

#pragma HLS dataflow
	conv_3x3(input, weight, output, output_DDR, stride, H_fmap_out, H_fmap_in, c_in, 1);
	conv_3x3_grad_v2(input_fw, input_copy, wt_out, vel, 1, stride, H_fmap_in);
}

// AvgPool
void avgpool(
	float avg_inputs[CHANNEL_IN_T][WIDTH][WIDTH],	// in, 64x8x8
	float out_buf[64],								// out, avg_outputs

	uint1 ctrl_avgpool,	// 0 for forward and 1 for backward
	int c_out,

//	float out_buf_SC[CHANNEL_IN_T][WIDTH][WIDTH],
	float out_buf_copy[64]
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
		}
	}
	// backward
	else {
		for (int s = 0; s < 8; s ++) {
			for (int ss = 0; ss < 8; ss ++) {
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
					avg_inputs[c][s][ss] = out_buf[c + c_out*CHANNEL_IN_T]/64;
//					out_buf_SC[c][s][ss] = out_buf[c + c_out*CHANNEL_IN_T]/64;
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
	float linear_bias[10],
	float outputs[10],

	uint1 ctrl_fc	// 0 for forward and 1 for backward
)
{
	float linear_weight_t[64][10];

	// forward
	if (ctrl_fc == 0) {
		// buffer init
		for (int coo = 0; coo < 10; coo ++) {
			outputs[coo] = linear_bias[coo];
		}
		for (int coo = 0; coo < 10; coo ++) {
			for (int cii = 0; cii < 64; cii++) {
				outputs[coo] += inputs[cii] * linear_weight[coo][cii];
			}
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
		for (int coo = 0; coo < 10; coo ++) {
			linear_bias[coo] += -lr * outputs[coo];
		}
	}
}

void shortcut(
	float input_a[CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	float input_b[CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	float out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip

	int H_fmap_in,
	uint1 ctrl_sc										// if ctrl_sc=0, generate and send out_copy into DDR
)
{
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {

				out_buf[c][row][col] = input_a[c][row][col] + input_b[c][row][col];
				if (ctrl_sc == 0) {
					out_buf_DDR[c][row][col] = out_buf[c][row][col];
				}
			}
		}
	}
}

void bn_relu(
	float bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs

	float out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip for backprop
	uint1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp

	float bn_wt[CHANNEL_OUT_T],
	float bn_bias[CHANNEL_OUT_T],

    int H_fmap_in
)
{
	int N = H_fmap_in * H_fmap_in;
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

#pragma HLS ARRAY_PARTITION variable=bn_inputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR dim=1 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=1 complete

#pragma HLS DEPENDENCE variable=mu intra false
#pragma HLS DEPENDENCE variable=std_var intra false

	float mu[CHANNEL_OUT_T] = {0};
	float std_var[CHANNEL_OUT_T] = {0};

	// mean
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] += bn_inputs[c][row][col]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (bn_inputs[c][row][col]-mu[c])*(bn_inputs[c][row][col]-mu[c])/N;	// var
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
				out_buf[c][row][col] = bn_wt[c]*(bn_inputs[c][row][col]-mu[c])/(std_var[c] + eps) + bn_bias[c];
				relu_mask[c][row][col] = (out_buf[c][row][col] > 0) ? 1 : 0;
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = (relu_mask[c][row][col] == 1) ? out_buf[c][row][col] : float(0);
				out_buf_DDR[c][row][col] = out_buf[c][row][col];
			}


			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs[c][row][col];
			}
			// bn + relu
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_temp[c] = bn_wt[c]*(in_temp[c]-mu[c])/(std_var[c] + eps) + bn_bias[c];
				relu_temp[c] = (out_temp[c] < 0) ? 0 : 1;
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = (relu_temp[c]==0) ? float(0) : out_temp[c];
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				relu_mask[c][row][col] = relu_temp[c];
				out_buf_DDR[c][row][col] = (relu_temp[c]==0) ? float(0) : out_temp[c];
			}
		}
	}
}

void bn_relu_bp(
	float error[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	float bn_inputs_fw[CHANNEL_OUT_T][WIDTH][WIDTH],	// in
	uint1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn
	float out_buf_copy[CHANNEL_OUT_T][WIDTH][WIDTH],	// out, error_bn for dataaflow

	float bn_wt[CHANNEL_OUT_T],
	float bn_bias[CHANNEL_OUT_T],
	float vel_wt[CHANNEL_OUT_T],
	float vel_bias[CHANNEL_OUT_T],

	uint1 ctrl_frz,
	int H_fmap_in
)
{
	int N = H_fmap_in * H_fmap_in;
	float mu[CHANNEL_OUT_T] = {0};
	float std_var[CHANNEL_OUT_T] = {0};
	float g_bn_wt[CHANNEL_OUT_T] = {0};					// out
	float g_bn_bias[CHANNEL_OUT_T] = {0};				// out

#pragma HLS DEPENDENCE variable=g_bn_wt inter false
#pragma HLS DEPENDENCE variable=g_bn_bias inter false

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
#pragma HLS ARRAY_PARTITION variable=out_buf_copy dim=1 complete

#pragma HLS ARRAY_PARTITION variable=vel_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=vel_bias dim=1 complete

#pragma HLS DEPENDENCE variable=in_temp inter false
#pragma HLS DEPENDENCE variable=g_bn_wt inter false

	// mean
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] += bn_inputs_fw[c][row][col]/N;
			}
		}
	}
	// std_var
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (bn_inputs_fw[c][row][col]-mu[c])*(bn_inputs_fw[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE II=1
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// relu_bp
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
				out_buf[c][row][col] = bn_wt[c]*error_temp[c]/(std_var[c]+eps) - bn_wt[c]*g_bn_bias[c]/(N*(std_var[c]+eps)) - bn_wt[c]*(in_temp[c]-mu[c])*g_bn_wt[c]/(N*(std_var[c]*std_var[c]+eps));
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf_copy[c][row][col] = bn_wt[c]*error_temp[c]/(std_var[c]+eps) - bn_wt[c]*g_bn_bias[c]/(N*(std_var[c]+eps)) - bn_wt[c]*(in_temp[c]-mu[c])*g_bn_wt[c]/(N*(std_var[c]*std_var[c]+eps));
			}
		}
	}

	if (ctrl_frz != 0) {
		// bn_sw params update
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE II=1
			vel_wt[c] = vel_wt[c]*mom + lr * g_bn_wt[c];
			bn_wt[c] -= vel_wt[c];

			vel_bias[c] = vel_bias[c]*mom + lr * g_bn_bias[c];
			bn_bias[c] -= vel_bias[c];
		}
	}
}

#endif

