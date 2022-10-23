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
int8 lr = 1;

// int version
//--------------------
//   bm(2,5) mult
//   sign + man + exp
//--------------------
inline int8 bm_mul(
	int8 in_a,	// in
	int8 in_b
)
{
#pragma HLS LATENCY min=0 max=1

	int7 man_a_norm;
	int7 man_b_norm;
	int9 out_raw;
	int7 man_out;
	int4 exp_out;
	int8 out;

	int7 man_a = in_a.range(7, 2);	// signed mantissa 0.xxxxx
	int7 man_b = in_b.range(7, 2);
	uint2 exp_a = in_a.range(1, 0);	// unsigned biased exponent 0,1,2,3
	uint2 exp_b = in_b.range(1, 0);

	// // printf("man_a = %f \n", man_a.to_int());
	// // printf("man_b = %f \n", man_b.to_int());
	// // printf("exp_a = %f \n", exp_a.to_int());
	// // printf("exp_b = %f \n", exp_b.to_int());

	// zero
	if (in_a == 0 || in_b == 0) {
		out = 0;
	}
	// signed normal & denormal
	else {
		// sigend fraction part: 1-bit sign + 1-bit implicit + 5-bit mantissa
//		man_a_norm = (exp_a==0) ? (man_a) : ((man_a.bit(6)==0) ? (man_a + 32) : (man_a - 32));
//		man_b_norm = (exp_b==0) ? (man_b) : ((man_b.bit(6)==0) ? (man_b + 32) : (man_b - 32));
		if (exp_a == 0) {
			man_a_norm = man_a;
		}
		else if (exp_b == 0) {
			man_b_norm = man_b;
		}
		else {
			man_a_norm = (man_a.bit(6)==0) ? (man_a + 32) : (man_a - 32);
			man_b_norm = (man_b.bit(6)==0) ? (man_b + 32) : (man_b - 32);
		}
	}
	// signed multiplication
	out_raw = (man_a_norm * man_b_norm) >> 5;	// 1.xx * 1.xx = 1.xx or 2.xx or 3.xx < 4.xx

// #pragma HLS RESOURCE variable=out_raw core=DSP48
//#pragma HLS BIND_OP variable=out_raw op=mul impl=fabric latency=1

	// mantissa renormalisation
	exp_out = hls::log2(out_raw >> 1);
	man_out = (out_raw.bit(8)==0) ? (out_raw >> exp_out - 32) : (out_raw >> exp_out + 32);

	out.range(7, 2) = man_out;
	out.range(1, 0) = uint2(exp_out + exp_a + exp_b - exp_bias);

	return out;
}

 //--------------------
 //   bm(2,5) add
 //   sign + man + exp
 //--------------------
inline int8 bm_add(
	int8 in_a,	// in
	int8 in_b,
	int8 bias_gap4add
)
{
#pragma HLS LATENCY min=0 max=1

	int7 man_a_norm;
	int7 man_b_norm;
	int12 out_FIX;
	int7 man_out;
	int4 exp_out;
	int8 out;

	int7 man_a = in_a.range(7, 2);	// signed mantissa 0.xxxxx
	int7 man_b = in_b.range(7, 2);
	uint2 exp_a = in_a.range(1, 0);	// unsigned biased exponent 0,1,2,3
	uint2 exp_b = in_b.range(1, 0);

	// shared exponent gap check
	if (bias_gap4add > 3 || bias_gap4add < -3) {	// bias_gap4add = shared_exp_a - shared_exp_b
		out = (bias_gap4add > 3) ? in_a : in_b;	// in_a for act, in_b for wt
	}
	// signed normal & denormal
	else {
		if (exp_a == 0) {
			man_a_norm = man_a;
		}
		else if (exp_b == 0) {
			man_b_norm = man_b;
		}
		else {
			man_a_norm = (man_a.bit(6)==0) ? (man_a + 32) : (man_a - 32);
			man_b_norm = (man_b.bit(6)==0) ? (man_b + 32) : (man_b - 32);
		}
	}
	// signed Kulisch accumulation
	out_FIX = (man_a_norm << exp_a) + (man_b_norm << (exp_b-bias_gap4add));

#pragma HLS RESOURCE variable=out_FIX core=AddSub
//#pragma HLS BIND_OP variable=out_FIX op=add impl=fabric latency=1

	// mantissa renormalisation
	exp_out = hls::log2(out_FIX >> 6);
	man_out = (out_FIX.bit(11) == 0) ? (out_FIX >> exp_out - 32) : (out_FIX >> exp_out + 32);

	out.range(7, 2) = man_out;
	out.range(1, 0) = uint2(exp_out);

	return out;
}

//--------------------
//   bm(2,5) mac
//   sign + man + exp
//--------------------
inline int8 bm_mac(
	int8 in_a,	// in
	int8 in_b,
	int8 in_c
)
{
#pragma HLS LATENCY min=0 max=1

	int10 man_a_norm;
	int10 man_b_norm;
	int10 man_c_norm;
	int20 out_FIX;

	int7 man_out;
	int4 exp_out;
	int8 out;

	int7 man_a = in_a.range(7, 2);	// signed mantissa 0.xxxxx
	int7 man_b = in_b.range(7, 2);
	int7 man_c = in_c.range(7, 2);
	uint2 exp_a = in_a.range(1, 0);	// unsigned biased exponent 0,1,2,3
	uint2 exp_b = in_b.range(1, 0);
	uint2 exp_c = in_c.range(1, 0);

	// zero
	if (in_a == 0 || in_b == 0) { // a*b+c
		out = in_c;
	}
	// signed normal & denormal
	else {
		// sigend fraction part: 1-bit sign + 1-bit implicit + 5-bit mantissa
//		man_a_norm = (exp_a==0) ? (man_a) : ((man_a.bit(6)==0) ? (man_a + 32) : (man_a - 32));
//		man_b_norm = (exp_b==0) ? (man_b) : ((man_b.bit(6)==0) ? (man_b + 32) : (man_b - 32));
//		man_c_norm = (exp_c==0) ? (man_b) : ((man_c.bit(6)==0) ? (man_c + 32) : (man_c - 32));
		if (exp_a == 0) {
			man_a_norm = man_a;
		}
		else if (exp_b == 0) {
			man_b_norm = man_b;
		}
		else if (exp_c == 0) {
			man_c_norm = man_c;
		}
		else {
			man_a_norm = (man_a.bit(6)==0) ? (man_a + 32) << exp_a : (man_a - 32) << exp_a;
			man_b_norm = (man_b.bit(6)==0) ? (man_b + 32) << exp_b : (man_b - 32) << exp_b;
			man_c_norm = (man_c.bit(6)==0) ? (man_c + 32) << exp_c : (man_c - 32) << exp_c;
		}
	}
	// signed multiplication
	out_FIX = man_a_norm * man_b_norm + man_c_norm;	// 1.xx * 1.xx = 1.xx or 2.xx or 3.xx < 4.xx

 #pragma HLS RESOURCE variable=out_FIX core=AddSub_DSP
//#pragma HLS BIND_OP variable=out_FIX impl=dsp latency=1

	// mantissa renormalisation
	exp_out = hls::log2(out_FIX >> 6);
	man_out = (out_FIX.bit(19) == 0) ? (out_FIX >> exp_out - 32) : (out_FIX >> exp_out + 32);

	out.range(7, 2) = man_out;
	out.range(1, 0) = uint2(exp_out);

	return out;
}

inline int8 bm_mac_no_DSP(
	int8 in_a,	// in
	int8 in_b,
	int8 in_c
)
{
#pragma HLS LATENCY min=0 max=1

	int10 man_a_norm;
	int10 man_b_norm;
	int10 man_c_norm;
	int20 out_FIX;

	int7 man_out;
	int4 exp_out;
	int8 out;

	int7 man_a = in_a.range(7, 2);	// signed mantissa 0.xxxxx
	int7 man_b = in_b.range(7, 2);
	int7 man_c = in_c.range(7, 2);
	uint2 exp_a = in_a.range(1, 0);	// unsigned biased exponent 0,1,2,3
	uint2 exp_b = in_b.range(1, 0);
	uint2 exp_c = in_c.range(1, 0);

	// zero
	if (in_a == 0 || in_b == 0) { // a*b+c
		out = in_c;
	}
	// signed normal & denormal
	else {
		// sigend fraction part: 1-bit sign + 1-bit implicit + 5-bit mantissa
//		man_a_norm = (exp_a==0) ? (man_a) : ((man_a.bit(6)==0) ? (man_a + 32) : (man_a - 32));
//		man_b_norm = (exp_b==0) ? (man_b) : ((man_b.bit(6)==0) ? (man_b + 32) : (man_b - 32));
//		man_c_norm = (exp_c==0) ? (man_b) : ((man_c.bit(6)==0) ? (man_c + 32) : (man_c - 32));
		if (exp_a == 0) {
			man_a_norm = man_a;
		}
		else if (exp_b == 0) {
			man_b_norm = man_b;
		}
		else if (exp_c == 0) {
			man_c_norm = man_c;
		}
		else {
			man_a_norm = (man_a.bit(6)==0) ? (man_a + 32) << exp_a : (man_a - 32) << exp_a;
			man_b_norm = (man_b.bit(6)==0) ? (man_b + 32) << exp_b : (man_b - 32) << exp_b;
			man_c_norm = (man_c.bit(6)==0) ? (man_c + 32) << exp_c : (man_c - 32) << exp_c;
		}
	}
	// signed multiplication
	out_FIX = man_a_norm * man_b_norm + man_c_norm;	// 1.xx * 1.xx = 1.xx or 2.xx or 3.xx < 4.xx

 #pragma HLS RESOURCE variable=out_FIX core=AddSub
//#pragma HLS BIND_OP variable=out_FIX impl=fabric latency=1

	// mantissa renormalisation
	exp_out = hls::log2(out_FIX >> 6);
	man_out = (out_FIX.bit(19) == 0) ? (out_FIX >> exp_out - 32) : (out_FIX >> exp_out + 32);

	out.range(7, 2) = man_out;
	out.range(1, 0) = uint2(exp_out);

	return out;
}

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
// // #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_in; col ++) {
// // #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
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
	int8 avg_inputs[CHANNEL_IN_T][WIDTH][WIDTH],	// in, 64x8x8
	int8 out_buf[64],								// out, avg_outputs

	uint1 ctrl_avgpool,	// 0 for forward and 1 for backward
	int c_out,

	int8 out_buf_SC[CHANNEL_IN_T][WIDTH][WIDTH],
	int8 out_buf_copy[64],
	int8 bias_gap4add[CHANNEL_IN_T]
)
{
	int8 out_temp[CHANNEL_IN_T] = {0};
#pragma HLS DEPENDENCE variable=out_temp inter false

#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=avg_inputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_SC dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete

	// forward
	if (ctrl_avgpool == 0) {
		for (int s = 0; s < 8; s ++) {
			for (int ss = 0; ss < 8; ss ++) {
#pragma HLS PIPELINE II=1
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
					// out_temp[c] += avg_inputs[c][s][ss];
					out_temp[c] = bm_add(out_temp[c], avg_inputs[c][s][ss], bias_gap4add[c]);
				}
			}
		}
		for (int c = 0; c < CHANNEL_IN_T; c ++) {
#pragma HLS PIPELINE II=1
			out_buf[c + c_out*CHANNEL_IN_T] = out_temp[c]/64;
			out_buf_copy[c + c_out*CHANNEL_IN_T] = out_temp[c]/64;
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
					out_buf_SC[c][s][ss] = out_temp[c]/64;
				}
			}
		}
	}
}

// FC
void FC(
	int8 inputs[64],
	int8 inputs_FW[64],
	int8 linear_weight[10][64],
	int8 outputs[10],

	uint1 ctrl_fc	// 0 for forward and 1 for backward
)
{
	int8 out_temp[10] = {0};
	int8 in_tmp[64] = {0};
#pragma HLS DEPENDENCE variable=in_tmp inter false
#pragma HLS DEPENDENCE variable=out_temp inter false

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
//				out_temp[coo] += act * wt;
//				out_temp[coo] = bm_add(out_temp[coo], bm_mul(act, wt, bias_gap4add[coo]), bias_gap4add[coo]);
				out_temp[coo] = bm_mac(act, wt, out_temp[coo]);
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
//				in_tmp[cii] += act * wt;
//				in_tmp[cii] = bm_add(in_tmp[cii], bm_mul(act, wt, bias_gap4add[coo]), bias_gap4add[coo]);
				in_tmp[cii] = bm_mac(act, wt, in_tmp[cii]);
			}
		}
		for (int cii = 0; cii < 64; cii++) {
#pragma HLS PIPELINE II=1
			inputs[cii] = in_tmp[cii];
		}
		// weight update
		for (int cii = 0; cii < 64; cii++) {
			for (int coo = 0; coo < 10; coo ++) {
				linear_weight[coo][cii] += -lr * inputs_FW[cii] * outputs[coo];
			}
		}
	}
}

// Shortcut- identity branch
void shortcut(
	int8 input_a[CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	int8 input_b[CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	int8 out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip
//	int8 out_buf_SC[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, ideneity for shortcut

	int H_fmap_in,
	uint1 ctrl_sc,	// if ctrl_sc=0, generate and send out_copy into DDR
//	uint1 ctrl_sc_id,

//	int8 shared_exp_bias[CHANNEL_OUT_T],
	int8 bias_gap4add[CHANNEL_OUT_T]
//	int8 act_bias_shift[CHANNEL_OUT_T]
)
{
	int8 in_1[CHANNEL_OUT_T];
	int8 in_2[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=in_1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_2 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

//	int8 max_abs[CHANNEL_OUT_T] = {0};
//#pragma HLS ARRAY_PARTITION variable=max_abs dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=shared_exp_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete

#pragma HLS ARRAY_PARTITION variable=input_a dim=1 complete
#pragma HLS ARRAY_PARTITION variable=input_b dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR dim=1 complete

	for (int row = 0; row < H_fmap_in; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap_in; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_1[c] = input_a[c][row][col];
				in_2[c] = input_b[c][row][col];
//				out_temp[c] = in_1[c] + in_2[c];
				out_temp[c] = bm_add(in_1[c], in_2[c], bias_gap4add[c]);
			}
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
//				if (out_temp[c].range(1,0) > max_abs[c].range(1,0)) max_abs[c] = out_temp[c];
			}
			if (ctrl_sc == 0) {
				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
					out_buf_DDR[c][row][col] = out_temp[c];
				}
			}
//			if (ctrl_sc_id == 0) {
//				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
//					out_buf_SC[c][row][col] = out_temp[c];
//				}
//			}
		}
	}
 	// update shared exponent bias
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		act_bias_shift[c] = max_abs[c].range(1,0);
// 	}
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		shared_exp_bias[c] = hls::log2(max_abs[c] >> 3);
// 	}
}

// Batch Norm
void bn(
	int8 bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out

	int8 bn_wt[CHANNEL_OUT_T],
	int8 bn_bias[CHANNEL_OUT_T],
	int8 mu[CHANNEL_OUT_T],
	int8 std_var[CHANNEL_OUT_T],

    int H_fmap,
//	int8 shared_exp_bias[CHANNEL_OUT_T],
	int8 bias_gap4add[CHANNEL_OUT_T]
//	int8 act_bias_shift[CHANNEL_OUT_T]
)
{
#pragma HLS ARRAY_PARTITION variable=bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=std_var dim=1 complete

	int8 in_temp[CHANNEL_OUT_T] = {0};
	int8 out_temp[CHANNEL_OUT_T] = {0};
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

//	int8 max_abs[CHANNEL_OUT_T] = {0};
//#pragma HLS ARRAY_PARTITION variable=max_abs dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=shared_exp_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete

#pragma HLS ARRAY_PARTITION variable=bn_inputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete

	for (int row = 0; row < H_fmap; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs[c][row][col];
			}
			// bn
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
//				out_temp[c] = bn_wt[c]*(in_temp[c]-mu[c])/(std_var[c] + eps) + bn_bias[c];
				// out_temp[c] = bn_wt[c]*(in_temp[c]-mu[c]) + bn_bias[c];
				out_temp[c] = bm_mul(bn_wt[c], bm_add(in_temp[c], ~mu[c], bias_gap4add[c]));
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
//				if (out_temp[c].range(1,0) > max_abs[c].range(1,0)) max_abs[c] = out_temp[c];
			}
		}
	}
// 	// update shared exponent bias
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		act_bias_shift[c] = max_abs[c].range(1,0);
// 	}
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		shared_exp_bias[c] = hls::log2(max_abs[c] >> 3);
// 	}
}

// Batch Norm Back-prop
void bn_bp(
	int8 error[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	int8 bn_inputs_fw[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	int8 bn_wt[CHANNEL_OUT_T],							// in
	int8 bn_bias[CHANNEL_OUT_T],						// in
	int8 mu[CHANNEL_OUT_T],
	int8 std_var[CHANNEL_OUT_T],

	int H_fmap,
	int8 bias_gap4add[CHANNEL_OUT_T]
)
{
	int N = H_fmap * H_fmap;
	int8 g_bn_wt[CHANNEL_OUT_T] = {0};					// out
	int8 g_bn_bias[CHANNEL_OUT_T] = {0};					// out
#pragma HLS ARRAY_PARTITION variable=g_bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_bn_bias dim=1 complete

	int8 error_temp[CHANNEL_OUT_T] = {0};
	int8 in_temp[CHANNEL_OUT_T] = {0};
	int8 out_temp[CHANNEL_OUT_T] = {0};
#pragma HLS ARRAY_PARTITION variable=error_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=std_var dim=1 complete
#pragma HLS ARRAY_PARTITION variable=error dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete

#pragma HLS DEPENDENCE variable=in_temp inter false
#pragma HLS DEPENDENCE variable=g_bn_wt inter false

	for (int row = 0; row < H_fmap; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
				error_temp[c] = error[c][row][col];
			}
			// calc grad for bn params
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				g_bn_bias[c] += error_temp[c];
//				g_bn_wt[c] += error_temp[c] * (in_temp[c]-mu[c])/(std_var[c] + eps);
				// g_bn_wt[c] += error_temp[c] * (in_temp[c]-mu[c]);
				g_bn_wt[c] = bm_mul(bm_mac(error_temp[c], bm_add(in_temp[c], ~mu[c], bias_gap4add[c]), g_bn_wt[c]), std_var[c]);
			}

		}
	}
	for (int row = 0; row < H_fmap; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
				error_temp[c] = error[c][row][col];
			}

			// calc backprop error
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
//				out_temp[c] = bn_wt[c]*error_temp[c]/std_var[c] - bn_wt[c]*g_bn_bias[c]/(N*(std_var[c]+eps)) - (in_temp[c]-mu[c])*g_bn_wt[c]/(N*bn_wt[c]*(std_var[c]+eps)*(std_var[c]+eps));
//				out_temp[c] = bn_wt[c]*(error_temp[c] - g_bn_bias[c]/N) - (in_temp[c]-mu[c])*g_bn_wt[c]/N;
				out_temp[c] = bm_add(bm_mul(bn_wt[c], bm_add(error_temp[c], ~g_bn_bias[c], bias_gap4add[c])), ~bm_mul(g_bn_wt[c], bm_add(in_temp[c], ~mu[c], bias_gap4add[c])), bias_gap4add[c]);
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
			}
		}
	}
	// bn params update
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE II=1
		bn_bias[c] += -lr * g_bn_bias[c];
		bn_wt[c] += -lr * g_bn_wt[c];
	}
}

// Fused bn + relu
void bn_relu(
	int8 bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs

	int8 out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip for backprop
	uint1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp
//	int8 out_buf_SC[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, ideneity for shortcut

	int8 bn_wt[CHANNEL_OUT_T],
	int8 bn_bias[CHANNEL_OUT_T],
	int8 mu[CHANNEL_OUT_T],
	int8 std_var[CHANNEL_OUT_T],

    int H_fmap,
//	uint1 ctrl_bn_id,

//	int8 shared_exp_bias[CHANNEL_OUT_T],
	int8 bias_gap4add[CHANNEL_OUT_T]
//	int8 act_bias_shift[CHANNEL_OUT_T]
)
{
	int N = H_fmap * H_fmap;
#pragma HLS ARRAY_PARTITION variable=bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=std_var dim=1 complete

	uint1 relu_temp[CHANNEL_OUT_T] = {0};
	int8 in_temp[CHANNEL_OUT_T] = {0};
	int8 out_temp[CHANNEL_OUT_T] = {0};
#pragma HLS ARRAY_PARTITION variable=relu_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

//	int8 max_abs[CHANNEL_OUT_T] = {0};
//#pragma HLS ARRAY_PARTITION variable=max_abs dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=shared_exp_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete

#pragma HLS ARRAY_PARTITION variable=bn_inputs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf_DDR dim=1 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=1 complete

	for (int row = 0; row < H_fmap; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs[c][row][col];
			}
			// bn + relu
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				// out_temp[c] = bn_wt[c]*(in_temp[c]-mu[c]) + bn_bias[c];
				out_temp[c] = bm_mul(bn_wt[c], bm_add(in_temp[c], ~mu[c], bias_gap4add[c]));
				relu_temp[c] = (out_temp[c] < 0) ? 0 : 1;
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				relu_mask[c][row][col] = relu_temp[c];
				out_buf[c][row][col] = (relu_temp[c]==0) ? int8(0) : out_temp[c];
				out_buf_DDR[c][row][col] = (relu_temp[c]==0) ? int8(0) : out_temp[c];
//				if (out_temp[c].range(1,0) > max_abs[c].range(1,0)) max_abs[c] = out_temp[c];
			}
//			if (ctrl_bn_id == 0) {
//				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
//					out_buf_SC[c][row][col] = (relu_temp[c]==0) ? int8(0) : out_temp[c];
//				}
//			}
		}
	}
// 	// update shared exponent bias
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		act_bias_shift[c] = max_abs[c].range(1,0);
// 	}
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		shared_exp_bias[c] = hls::log2(max_abs[c] >> 3);
// 	}
}

// Fused relu_bp + bn_bp
void bn_relu_bp(
	int8 error[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	int8 bn_inputs_fw[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	uint1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int8 out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	int8 bn_wt[CHANNEL_OUT_T],							// in
	int8 bn_bias[CHANNEL_OUT_T],						// in
	int8 mu[CHANNEL_OUT_T],
	int8 std_var[CHANNEL_OUT_T],

	int H_fmap,
	int8 bias_gap4add[CHANNEL_OUT_T]
)
{
	int N = H_fmap * H_fmap;
	int8 g_bn_wt[CHANNEL_OUT_T] = {0};						// out
	int8 g_bn_bias[CHANNEL_OUT_T] = {0};					// out
#pragma HLS DEPENDENCE variable=g_bn_wt inter false
#pragma HLS DEPENDENCE variable=g_bn_bias inter false

#pragma HLS ARRAY_PARTITION variable=bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_bias dim=1 complete
#pragma HLS ARRAY_PARTITION variable=mu dim=1 complete
#pragma HLS ARRAY_PARTITION variable=std_var dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_bn_wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=g_bn_bias dim=1 complete

	int8 error_temp[CHANNEL_OUT_T] = {0};
	int8 in_temp[CHANNEL_OUT_T] = {0};
	int8 out_temp[CHANNEL_OUT_T] = {0};
#pragma HLS ARRAY_PARTITION variable=error_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=in_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

#pragma HLS ARRAY_PARTITION variable=error dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bn_inputs_fw dim=1 complete
#pragma HLS ARRAY_PARTITION variable=relu_mask dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete

#pragma HLS DEPENDENCE variable=in_temp inter false
#pragma HLS DEPENDENCE variable=g_bn_wt inter false

	for (int row = 0; row < H_fmap; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
		for (int col = 0; col < H_fmap; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
				// relu
				error_temp[c] = (relu_mask[c][row][col] == 1) ? error[c][row][col] : int8(0);
			}
			// calc grad for bn params
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				// g_bn_bias[c] += error_temp[c];
				// g_bn_wt[c] += error_temp[c] * (in_temp[c]-mu[c]);
//				g_bn_bias[c] = bm_add(g_bn_bias[c], error_temp[c], bias_gap4add[c]);
				g_bn_bias[c] = bm_mac(error_temp[c], 1, g_bn_bias[c]);
				g_bn_wt[c] = bm_mul(bm_mac(error_temp[c], bm_add(in_temp[c], ~mu[c], bias_gap4add[c]), g_bn_wt[c]), std_var[c]);
			}

		}
	}
	for (int row = 0; row < H_fmap; row ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
		for (int col = 0; col < H_fmap; col ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1
			// buffer init
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				in_temp[c] = bn_inputs_fw[c][row][col];
				error_temp[c] = error[c][row][col];
			}

			// calc backprop error
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
//				out_temp[c] = bn_wt[c]*error_temp[c]/(std_var[c]+eps) - bn_wt[c]*g_bn_bias[c]/(N*(std_var[c]+eps)) - (in_temp[c]-mu[c])*g_bn_wt[c]/(N*bn_wt[c]*(std_var[c]+eps)*(std_var[c]+eps));
//				out_temp[c] = bn_wt[c]*(error_temp[c] - g_bn_bias[c]/N) - (in_temp[c]-mu[c])*g_bn_wt[c]/N;
				out_temp[c] = bm_add(bm_mul(bn_wt[c], bm_add(error_temp[c], ~g_bn_bias[c], bias_gap4add[c])), ~bm_mul(g_bn_wt[c], bm_add(in_temp[c], ~mu[c], bias_gap4add[c])), bias_gap4add[c]);
			}
			// write out
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = out_temp[c];
			}
		}
	}
	// bn params update
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
#pragma HLS PIPELINE II=1
		bn_bias[c] += -lr * g_bn_bias[c];
		bn_wt[c] += -lr * g_bn_wt[c];
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
	uint1 ctrl_conv,

//	int8 shared_exp_bias[CHANNEL_OUT_T],
	int8 bias_gap4add[CHANNEL_OUT_T]
//	int8 act_bias_shift[CHANNEL_OUT_T]
)
{
// #pragma HLS DEPENDENCE variable=output inter false

	int row_in;
	int col_in;
	int8 act[CHANNEL_IN_T];
	int8 wt[CHANNEL_OUT_T][CHANNEL_IN_T];
	int8 accum[CHANNEL_OUT_T];
	int8 out_temp[CHANNEL_OUT_T];
#pragma HLS ARRAY_PARTITION variable=act dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wt dim=2 complete
#pragma HLS ARRAY_PARTITION variable=accum dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete

//	int8 max_abs[CHANNEL_OUT_T] = {0};
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
					// accum[co] += bm_mul(act[cin], wt[co][cin], bias_gap4add[co]);
					accum[co] = bm_mac(act[cin], wt[co][cin], accum[co]);
				}
				// out_temp[co] += accum[co];
				out_temp[co] = bm_add(out_temp[co], accum[co], bias_gap4add[co]);
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
// 	// update shared exponent bias
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
//#pragma HLS PIPELINE II=1
// 		act_bias_shift[c] = max_abs[c].range(1,0);
// 	}
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		shared_exp_bias[c] = hls::log2(max_abs[c] >> 3);
// 	}
}

void conv_1x1_grad	// revised if-else conditions
(
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// activation from DDR
	int8 weight[CHANNEL_OUT_T][WIDTH][WIDTH],			// error on-chip
	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T],		// gradient on-chip

	int stride,
	int H_fmap_in
//	int8 act_bias_shift[CHANNEL_OUT_T]
)
{
	int row_input;
	int col_input;

	uint6 skip_k8 = 0;
	uint8 skip_k16 = 0;
	uint10 skip_k32 = 0;

	uint1 skip_k = 1;

	uint2 skip_krow = 0;
	uint2 skip_kcol = 0;

	int8 wt;
	int8 act;
	int8 accum;
	int8 out_temp[CHANNEL_OUT_T][CHANNEL_IN_T] = {0};
#pragma HLS DEPENDENCE variable=out_temp inter false
#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete

	int8 line_buffer_act[CHANNEL_IN_T][3][WIDTH] = {0};
	int8 window_buffer_act[CHANNEL_IN_T][4][4] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_act complete dim=0

	int8 line_buffer_wt[CHANNEL_IN_T][3][WIDTH] = {0};
	int8 window_buffer_wt[CHANNEL_IN_T][4][4] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_wt complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_wt complete dim=0

#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
#pragma HLS ARRAY_PARTITION variable=output dim=2 complete

	for (int ii = 0; ii < stride; ii ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
		for (int krow = 0; krow < H_fmap_in; krow ++) {
//			skip_krow += 1;
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
			for (int jj = 0; jj < stride; jj ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
				for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
//					skip_kcol += 1;
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1

					// activation- update window buffer and line buffer
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						for (int i = 0; i < 4; i ++) {
							window_buffer_act[cin][i][0] = window_buffer_act[cin][i][1];
							window_buffer_act[cin][i][1] = window_buffer_act[cin][i][2];
							window_buffer_act[cin][i][2] = window_buffer_act[cin][i][3];
//							// printf("window_buffer_act[%d][%d][0] = %d \n", cin, i, window_buffer_act[cin][i][0].to_int());
//							// printf("window_buffer_act[%d][%d][1] = %d \n", cin, i, window_buffer_act[cin][i][1].to_int());
//							// printf("window_buffer_act[%d][%d][2] = %d \n", cin, i, window_buffer_act[cin][i][2].to_int());
//							// printf("window_buffer_act[%d][%d][3] = %d \n", cin, i, window_buffer_act[cin][i][3].to_int());
						}

						row_input = krow + ii*H_fmap_in;
						col_input = kcol + jj*H_fmap_in;

						window_buffer_act[cin][0][3] = (line_buffer_act[cin][0][kcol]);
						window_buffer_act[cin][1][3] = (line_buffer_act[cin][0][kcol] = line_buffer_act[cin][1][kcol]);
						window_buffer_act[cin][2][3] = (line_buffer_act[cin][1][kcol] = line_buffer_act[cin][2][kcol]);
						window_buffer_act[cin][3][3] = (line_buffer_act[cin][2][kcol] = input[cin][row_input][col_input]);
//						// printf("skip_krow: %d, skip_kcol: %d;    input[%d][%d][%d]: %d \n", skip_krow.to_int(), skip_kcol.to_int(), cin, row_input, col_input, input[cin][row_input][col_input].to_int());
					}

					// dilated weight- update window buffer and line buffer
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int i = 0; i < 4; i ++) {
							window_buffer_wt[co][i][0] = window_buffer_wt[co][i][1];
							window_buffer_wt[co][i][1] = window_buffer_wt[co][i][2];
							window_buffer_wt[co][i][2] = window_buffer_wt[co][i][3];
//							// printf("window_buffer_wt[%d][%d][0] = %d \n", co, i, window_buffer_wt[co][i][0].to_int());
//							// printf("window_buffer_wt[%d][%d][1] = %d \n", co, i, window_buffer_wt[co][i][1].to_int());
//							// printf("window_buffer_wt[%d][%d][2] = %d \n", co, i, window_buffer_wt[co][i][2].to_int());
//							// printf("window_buffer_wt[%d][%d][3] = %d \n", co, i, window_buffer_wt[co][i][3].to_int());
						}

						window_buffer_wt[co][0][3] = (line_buffer_wt[co][0][kcol]);
						window_buffer_wt[co][1][3] = (line_buffer_wt[co][0][kcol] = line_buffer_wt[co][1][kcol]);
						window_buffer_wt[co][2][3] = (line_buffer_wt[co][1][kcol] = line_buffer_wt[co][2][kcol]);
						window_buffer_wt[co][3][3] = (line_buffer_wt[co][2][kcol]);
						line_buffer_wt[co][2][kcol] = (stride > 1 && krow % 2 > 0 && kcol % 2 > 0) ? int8(0) : weight[co][krow][kcol];
//						// printf("skip_krow: %d, skip_kcol: %d;    weight[%d][%d][%d]: %d \n", skip_krow.to_int(), skip_kcol.to_int(), co, krow, kcol, weight[co][krow][kcol].to_int());
					}

					// dilated conv
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							skip_kcol += 1;
							skip_k32 += 1;
							skip_k16 += 1;
							skip_k8 += 1;
							if ((H_fmap_in==32 && (skip_k32<32 || skip_k32>=992)) || (H_fmap_in==16 && (skip_k16<16 || skip_k16>=240)) || (H_fmap_in==8 && (skip_k8<8 || skip_k8>=56))) {
								skip_k = 0;
							}
							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip_kcol == 1 && skip_k == 0) {
								for (int pkrow = 0; pkrow < 4; pkrow ++) {
									for (int pkcol = 0; pkcol < 4; pkcol ++) {
										wt = window_buffer_wt[co][pkrow][pkcol];
										act = window_buffer_act[cin][pkrow][pkcol];
										accum = bm_mac(act, wt, accum);
									}
								}
							}
							out_temp[co][cin] = accum;
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
			// output[co][cin] = out_temp[co][cin];
			output[co][cin] = bm_mac(lr, out_temp[co][cin], output[co][cin]);
		}
	}
}

void conv_3x3_uni_win
(
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// in on-chip
	int8 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 output[CHANNEL_OUT_T][WIDTH][WIDTH],			// out on-chip
	int8 output_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out off-chip

	int stride,
	int H_fmap_in,
	int H_fmap_out,
	int c_in,
	uint1 ctrl_conv,

//	int8 shared_exp_bias[CHANNEL_OUT_T],
	int8 bias_gap4add[CHANNEL_OUT_T]
//	int8 act_bias_shift[CHANNEL_OUT_T]
)
{
// #pragma HLS DEPENDENCE variable=output inter false

	int row_input;
	int col_input;
	int row_in;
	int col_in;
	int8 out_temp[CHANNEL_OUT_T];
	int8 output_tmp[CHANNEL_OUT_T][WIDTH][WIDTH];
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

//	int8 max_abs[CHANNEL_OUT_T] = {0};
//#pragma HLS ARRAY_PARTITION variable=max_abs dim=1 complete
#pragma HLS ARRAY_PARTITION variable=bias_gap4add dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=shared_exp_bias dim=1 complete

	int8 line_buffer_act[CHANNEL_IN_T][2][WIDTH] = {0};
	int8 window_buffer_act[CHANNEL_IN_T][3][3] = {0};
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
							window_buffer_act[cin][2][2] = (line_buffer_act[cin][1][col] = (row_input % 2 == 0 && col_input % 2 == 0) ? input[cin][row_input][col_input] : int8(0));	// dilated with 0-row
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
						int8 accum = 0;
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
								for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
									row_in = (stride > 1 && ctrl_conv == 0) ? (row + krow - 2 + ii*H_fmap_out) : (row + krow - 2);
									col_in = (stride > 1 && ctrl_conv == 0) ? (col + kcol - 2 + jj*H_fmap_out) : (col + kcol - 2);
									if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
										int8 act = window_buffer_act[cin][krow][kcol];
										int8 wt = (ctrl_conv == 0) ? weight[co][cin][krow][kcol] : weight[cin][co][kcol][krow];
										// accum += act * wt;
										accum = bm_mac(act, wt, accum);
									}
								}
							}
						}
						// out_temp[co] += accum;
						out_temp[co] = bm_add(accum, out_temp[co], bias_gap4add[co]);
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
// 	// update shared exponent bias
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		act_bias_shift[c] = max_abs[c].range(1,0);
// 	}
// 	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
// #pragma HLS PIPELINE II=1
// 		shared_exp_bias[c] = hls::log2(max_abs[c] >> 3);
// 	}
}

//void conv_3x3_grad_win	// revised if-else conditions
//(
//	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// activation from DDR
//	int8 weight[CHANNEL_OUT_T][WIDTH][WIDTH],			// error on-chip
//	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],		// gradient on-chip
//
//	int stride,
//	int H_fmap_in,
//	int8 act_bias_shift[CHANNEL_OUT_T]
//)
//{
//	int row_input;
//	int col_input;
//	int row_in;
//	int col_in;
//
//	uint2 skip_0_0 = 3;
//	uint2 skip_0_1 = 3;
//	uint2 skip_0_2 = 3;
//	uint2 skip_1_0 = 3;
//	uint2 skip_1_1 = 3;
//	uint2 skip_1_2 = 3;
//	uint2 skip_2_0 = 3;
//	uint2 skip_2_1 = 3;
//	uint2 skip_2_2 = 3;
//
//	int8 wt;
//	int8 act;
//	int8 out_temp[CHANNEL_OUT_T][CHANNEL_IN_T][3][3] = {0};
//#pragma HLS DEPENDENCE variable=out_temp inter false
//#pragma HLS DEPENDENCE variable=output inter false
//
//#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete
//#pragma HLS ARRAY_PARTITION variable=out_temp dim=3 complete
//#pragma HLS ARRAY_PARTITION variable=out_temp dim=4 complete
//
//	int8 line_buffer_act[CHANNEL_IN_T][3][WIDTH] = {0};
//	int8 window_buffer_act[CHANNEL_IN_T][4][4] = {0};
//#pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=0
//#pragma HLS ARRAY_PARTITION variable=window_buffer_act complete dim=0
//
//	int8 line_buffer_wt[CHANNEL_IN_T][3][WIDTH] = {0};
//	int8 window_buffer_wt[CHANNEL_IN_T][4][4] = {0};
//#pragma HLS ARRAY_PARTITION variable=line_buffer_wt complete dim=0
//#pragma HLS ARRAY_PARTITION variable=window_buffer_wt complete dim=0
//
//#pragma HLS ARRAY_PARTITION variable=input dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=output dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=output dim=2 complete
//#pragma HLS ARRAY_PARTITION variable=output dim=3 complete
//#pragma HLS ARRAY_PARTITION variable=output dim=4 complete
//#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete
//
//	for (int ii = 0; ii < stride; ii ++) {
//// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
//		for (int krow = 0; krow < H_fmap_in; krow ++) {
//// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
//			for (int jj = 0; jj < stride; jj ++) {
//// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
//				for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
//// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
//#pragma HLS PIPELINE II=1
//
//					// activation- update window buffer and line buffer
//					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//						for (int i = 0; i < 4; i ++) {
//							window_buffer_act[cin][i][0] = window_buffer_act[cin][i][1];
//							window_buffer_act[cin][i][1] = window_buffer_act[cin][i][2];
//							window_buffer_act[cin][i][2] = window_buffer_act[cin][i][3];
//						}
//
//						row_input = krow + ii*H_fmap_in;
//						col_input = kcol + jj*H_fmap_in;
//
//						window_buffer_act[cin][0][3] = (line_buffer_act[cin][0][kcol]);
//						window_buffer_act[cin][1][3] = (line_buffer_act[cin][0][kcol] = line_buffer_act[cin][1][kcol]);
//						window_buffer_act[cin][2][3] = (line_buffer_act[cin][1][kcol] = line_buffer_act[cin][2][kcol]);
//						window_buffer_act[cin][3][3] = (line_buffer_act[cin][2][kcol] = input[cin][row_input][col_input]);
//						//// printf("input[%d][%d][%d]: %d \n", cin, row_input, col_input, input[cin][row_input][col_input].to_int());
//					}
//
//					// dilated weight- update window buffer and line buffer
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int i = 0; i < 4; i ++) {
//							window_buffer_wt[co][i][0] = window_buffer_wt[co][i][1];
//							window_buffer_wt[co][i][1] = window_buffer_wt[co][i][2];
//							window_buffer_wt[co][i][2] = window_buffer_wt[co][i][3];
//						}
//
//						window_buffer_wt[co][0][3] = (line_buffer_wt[co][0][kcol]);
//						window_buffer_wt[co][1][3] = (line_buffer_wt[co][0][kcol] = line_buffer_wt[co][1][kcol]);
//						window_buffer_wt[co][2][3] = (line_buffer_wt[co][1][kcol] = line_buffer_wt[co][2][kcol]);
//						window_buffer_wt[co][3][3] = (line_buffer_wt[co][2][kcol]);
//						line_buffer_wt[co][2][kcol] = (stride > 1 && krow % 2 > 0 && kcol % 2 > 0) ? int8(0) : weight[co][krow][kcol];
//						//// printf("weight[%d][%d][%d]: %d \n", cin, krow, kcol, weight[cin][krow][kcol].to_int());
//					}
//
//					// dilated conv_0
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_0_0 += 1;
//							// row_in = krow - 1;
//							// col_in = kcol - 1;	// 0-padding
//							// conv(0, 0)
//							if (krow >= 1 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in && skip_0_0 == 0) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][0][0] += act * wt;
//										out_temp[co][cin][0][0] = bm_mac(act, wt, out_temp[co][cin][0][0]);
//									}
//								}
//							}
//						}
//					}
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_0_1 += 1;
//							// row_in = krow - 1;
//							// col_in = kcol;	// 0-padding
//							// conv(0, 1)
//							if (krow >= 1 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip_0_1 == 1) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][0][1] += act * wt;
//										out_temp[co][cin][0][1] = bm_mac(act, wt, out_temp[co][cin][0][1]);
//									}
//								}
//							}
//						}
//					}
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_0_2 += 1;
//							// row_in = krow - 1;
//							// col_in = kcol + 1;	// 0-padding
//							// conv(0, 2)
//							if (krow >= 1 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip_0_2 == 2) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][0][2] += act * wt;
//										out_temp[co][cin][0][2] = bm_mac(act, wt, out_temp[co][cin][0][2]);
//									}
//								}
//							}
//						}
//					}
//					// dilated conv_1
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_1_0 += 1;
//							// row_in = krow;
//							// col_in = kcol - 1;	// 0-padding
//							// conv(1, 0)
//							if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in && skip_1_0 == 0) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][1][0] += act * wt;
//										out_temp[co][cin][1][0] = bm_mac(act, wt, out_temp[co][cin][1][0]);
//									}
//								}
//							}
//						}
//					}
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_1_1 += 1;
//							// row_in = krow;
//							// col_in = kcol;	// 0-padding
//							// conv(1, 1)
//							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip_1_1 == 1) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][1][1] += act * wt;
//										out_temp[co][cin][1][1] = bm_mac(act, wt, out_temp[co][cin][1][1]);
//									}
//								}
//							}
//						}
//					}
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_1_2 += 1;
//							// row_in = krow;
//							// col_in = kcol + 1;	// 0-padding
//							// conv(1, 2)
//							if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip_1_2 == 2) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][1][2] += act * wt;
//										out_temp[co][cin][1][2] = bm_mac(act, wt, out_temp[co][cin][1][2]);
//									}
//								}
//							}
//						}
//					}
//					// dilated conv_2
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_2_0 += 1;
//							// row_in = krow + 1;
//							// col_in = kcol - 1;	// 0-padding
//							// conv(2, 0)
//							if (krow >= 0 && krow < H_fmap_in - 1 && kcol >= 1 && kcol < H_fmap_in && skip_2_0 == 0) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][2][0] += act * wt;
//										out_temp[co][cin][2][0] = bm_mac(act, wt, out_temp[co][cin][2][0]);
//									}
//								}
//							}
//						}
//					}
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_2_1 += 1;
//							// row_in = krow + 1;
//							// col_in = kcol;	// 0-padding
//							// conv(2, 1)
//							if (krow >= 0 && krow < H_fmap_in - 1 && kcol >= 0 && kcol < H_fmap_in && skip_2_1 == 1) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][2][1] += act * wt;
//										out_temp[co][cin][2][1] = bm_mac(act, wt, out_temp[co][cin][2][1]);
//									}
//								}
//							}
//						}
//					}
//					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//							skip_2_2 += 1;
//							// row_in = krow + 1;
//							// col_in = kcol + 1;	// 0-padding
//							// conv(2, 2)
//							if (krow >= 0 && krow < H_fmap_in - 1 && kcol >= 0 && kcol < H_fmap_in - 1 && skip_2_2 == 2) {
//								for (int row = 0; row < 4; row ++) {
//									for (int col = 0; col < 4; col ++) {
//										wt = window_buffer_wt[co][row][col];
//										act = window_buffer_act[cin][row][col];
////										out_temp[co][cin][2][2] += act * wt;
//										out_temp[co][cin][2][2] = bm_mac(act, wt, out_temp[co][cin][2][2]);
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//	// write out
//	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
//#pragma HLS PIPELINE II=2
//		for (int row = 0; row < 3; row ++) {
//			for (int col = 0; col < 3; col ++) {
//				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//					// output[co][cin][row][col] += lr * out_temp[co][cin][row][col];
//					output[co][cin][row][col] = bm_mac(lr, out_temp[co][cin][row][col], output[co][cin][row][col]);
//				}
//			}
//		}
//	}
//}

// void conv_3x3_grad_win	// revised if-else conditions
// (
// 	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// activation from DDR
// 	int8 weight[CHANNEL_OUT_T][WIDTH][WIDTH],			// error on-chip
// 	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],		// gradient on-chip

// 	int stride,
// 	int H_fmap_in
// //	int8 act_bias_shift[CHANNEL_OUT_T]
// )
// {
// 	int row_input;
// 	int col_input;
// 	int row_in;
// 	int col_in;

// 	uint2 skip = 3;

// 	int8 wt;
// 	int8 act;
// 	int8 accum;
// 	int8 out_temp[CHANNEL_OUT_T][CHANNEL_IN_T][3][3] = {0};
// #pragma HLS DEPENDENCE variable=out_temp inter false
// //#pragma HLS DEPENDENCE variable=output inter false

// #pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
// #pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete
// #pragma HLS ARRAY_PARTITION variable=out_temp dim=3 complete
// #pragma HLS ARRAY_PARTITION variable=out_temp dim=4 complete

// 	int8 line_buffer_act[CHANNEL_IN_T][3][WIDTH] = {0};
// 	int8 window_buffer_act[CHANNEL_IN_T][4][4] = {0};
// #pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=0
// #pragma HLS ARRAY_PARTITION variable=window_buffer_act complete dim=0

// 	int8 line_buffer_wt[CHANNEL_IN_T][3][WIDTH] = {0};
// 	int8 window_buffer_wt[CHANNEL_IN_T][4][4] = {0};
// #pragma HLS ARRAY_PARTITION variable=line_buffer_wt complete dim=0
// #pragma HLS ARRAY_PARTITION variable=window_buffer_wt complete dim=0

// #pragma HLS ARRAY_PARTITION variable=input dim=1 complete
// #pragma HLS ARRAY_PARTITION variable=weight dim=1 complete
// #pragma HLS ARRAY_PARTITION variable=output dim=1 complete
// #pragma HLS ARRAY_PARTITION variable=output dim=2 complete
// #pragma HLS ARRAY_PARTITION variable=output dim=3 complete
// #pragma HLS ARRAY_PARTITION variable=output dim=4 complete
// //#pragma HLS ARRAY_PARTITION variable=act_bias_shift dim=1 complete

// 	for (int ii = 0; ii < stride; ii ++) {
// // #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
// 		for (int krow = 0; krow < H_fmap_in; krow ++) {
// // #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
// 			for (int jj = 0; jj < stride; jj ++) {
// // #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
// 				for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
// // #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
// #pragma HLS PIPELINE II=1

// 					// activation- update window buffer and line buffer
// 					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
// 						for (int i = 0; i < 4; i ++) {
// 							window_buffer_act[cin][i][0] = window_buffer_act[cin][i][1];
// 							window_buffer_act[cin][i][1] = window_buffer_act[cin][i][2];
// 							window_buffer_act[cin][i][2] = window_buffer_act[cin][i][3];
// 						}

// 						row_input = krow + ii*H_fmap_in;
// 						col_input = kcol + jj*H_fmap_in;

// 						window_buffer_act[cin][0][3] = (line_buffer_act[cin][0][kcol]);
// 						window_buffer_act[cin][1][3] = (line_buffer_act[cin][0][kcol] = line_buffer_act[cin][1][kcol]);
// 						window_buffer_act[cin][2][3] = (line_buffer_act[cin][1][kcol] = line_buffer_act[cin][2][kcol]);
// 						window_buffer_act[cin][3][3] = (line_buffer_act[cin][2][kcol] = input[cin][row_input][col_input]);
// 						//// printf("input[%d][%d][%d]: %d \n", cin, row_input, col_input, input[cin][row_input][col_input].to_int());
// 					}

// 					// dilated weight- update window buffer and line buffer
// 					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
// 						for (int i = 0; i < 4; i ++) {
// 							window_buffer_wt[co][i][0] = window_buffer_wt[co][i][1];
// 							window_buffer_wt[co][i][1] = window_buffer_wt[co][i][2];
// 							window_buffer_wt[co][i][2] = window_buffer_wt[co][i][3];
// 						}

// 						window_buffer_wt[co][0][3] = (line_buffer_wt[co][0][kcol]);
// 						window_buffer_wt[co][1][3] = (line_buffer_wt[co][0][kcol] = line_buffer_wt[co][1][kcol]);
// 						window_buffer_wt[co][2][3] = (line_buffer_wt[co][1][kcol] = line_buffer_wt[co][2][kcol]);
// 						window_buffer_wt[co][3][3] = (line_buffer_wt[co][2][kcol]);
// 						line_buffer_wt[co][2][kcol] = (stride > 1 && krow % 2 > 0 && kcol % 2 > 0) ? int8(0) : weight[co][krow][kcol];
// 						//// printf("weight[%d][%d][%d]: %d \n", cin, krow, kcol, weight[cin][krow][kcol].to_int());
// 					}

// 					// dilated conv
// 					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
// 						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
// 							skip += 1;
// 							for (int row = 0; row < 4; row ++) {
// 								for (int col = 0; col < 4; col ++) {
// 									wt = window_buffer_wt[co][row][col];
// 									act = window_buffer_act[cin][row][col];
// //									accum = bm_mac_no_DSP(wt, act, accum);
// 									accum = bm_mul(wt, act);
// 									// conv(0, 0)
// 									if (krow >= 1 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in && skip == 0) {
// 										out_temp[co][cin][0][0] = bm_add(accum, out_temp[co][cin][0][0], 0);
// //										out_temp[co][cin][0][0] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][0][0]);
// 									}
// 									// conv(0, 1)
// 									if (krow >= 1 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip == 1) {
// 										out_temp[co][cin][0][1] = bm_add(accum, out_temp[co][cin][0][1], 0);
// //										out_temp[co][cin][0][1] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][0][1]);
// 									}
// 									// conv(0, 2)
// 									if (krow >= 1 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
// 										out_temp[co][cin][0][2] = bm_add(accum, out_temp[co][cin][0][2], 0);
// //										out_temp[co][cin][0][2] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][0][2]);
// 									}
// 									// conv(1, 0)
// 									if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in && skip == 0) {
// 										out_temp[co][cin][1][0] = bm_add(accum, out_temp[co][cin][1][0], 0);
// //										out_temp[co][cin][1][0] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][1][0]);
// 									}
// 									// conv(1, 1)
// 									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip == 1) {
// 										out_temp[co][cin][1][1] = bm_add(accum, out_temp[co][cin][1][1], 0);
// //										out_temp[co][cin][1][1] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][1][1]);
// 									}
// 									// conv(1, 2)
// 									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
// 										out_temp[co][cin][1][2] = bm_add(accum, out_temp[co][cin][1][2], 0);
// //										out_temp[co][cin][1][2] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][1][2]);
// 									}
// 									// conv(2, 0)
// 									if (krow >= 0 && krow < H_fmap_in - 1 && kcol >= 1 && kcol < H_fmap_in && skip == 0) {
// 										out_temp[co][cin][2][0] = bm_add(accum, out_temp[co][cin][2][0], 0);
// //										out_temp[co][cin][2][0] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][2][0]);
// 									}
// 									// conv(2, 1)
// 									if (krow >= 0 && krow < H_fmap_in - 1 && kcol >= 0 && kcol < H_fmap_in && skip == 1) {
// 										out_temp[co][cin][2][1] = bm_add(accum, out_temp[co][cin][2][1], 0);
// //										out_temp[co][cin][2][1] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][2][1]);
// 									}
// 									// conv(2, 2)
// 									if (krow >= 0 && krow < H_fmap_in - 1 && kcol >= 0 && kcol < H_fmap_in - 1 && skip == 2) {
// 										out_temp[co][cin][2][2] = bm_add(accum, out_temp[co][cin][2][2], 0);
// //										out_temp[co][cin][2][2] = bm_mac_no_DSP(accum, 1, out_temp[co][cin][2][2]);
// 									}
// 								}
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}

// 	// write out
// 	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
// #pragma HLS PIPELINE II=3
// 		for (int row = 0; row < 3; row ++) {
// 			for (int col = 0; col < 3; col ++) {
// 				for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
// 					// output[co][cin][row][col] += lr * out_temp[co][cin][row][col];
// 					output[co][cin][row][col] = bm_mac_no_DSP(lr, out_temp[co][cin][row][col], output[co][cin][row][col]);
// 				}
// 			}
// 		}
// 	}
// }

void conv_3x3_grad_win	// revised if-else conditions
(
	int8 input[CHANNEL_IN_T][WIDTH][WIDTH],				// activation from DDR
	int8 weight[CHANNEL_OUT_T][WIDTH][WIDTH],			// error on-chip
	int8 output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],		// gradient on-chip

	int stride,
	int H_fmap_in
)
{
	int row_input;
	int col_input;
	int row_in;
	int col_in;

	uint6 skip_k8 = 0;
	uint8 skip_k16 = 0;
	uint10 skip_k32 = 0;

	uint1 skip_k = 1;

	uint2 skip_krow = 0;
	uint2 skip_kcol = 0;

	int8 wt;
	int8 act;
	int8 accum;
	int8 out_temp[CHANNEL_OUT_T][CHANNEL_IN_T][3][3] = {0};
#pragma HLS DEPENDENCE variable=out_temp inter false
//#pragma HLS DEPENDENCE variable=output inter false

#pragma HLS ARRAY_PARTITION variable=out_temp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=3 complete
#pragma HLS ARRAY_PARTITION variable=out_temp dim=4 complete

	int8 line_buffer_act[CHANNEL_IN_T][3][WIDTH] = {0};
	int8 window_buffer_act[CHANNEL_IN_T][4][4] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer_act complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer_act complete dim=0

	int8 line_buffer_wt[CHANNEL_IN_T][3][WIDTH] = {0};
	int8 window_buffer_wt[CHANNEL_IN_T][4][4] = {0};
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
//			skip_krow += 1;
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
			for (int jj = 0; jj < stride; jj ++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2
				for (int kcol = 0; kcol < H_fmap_in; kcol ++) {
//					skip_kcol += 1;
// #pragma HLS LOOP_TRIPCOUNT min = 8 max = 16
#pragma HLS PIPELINE II=1

					// dilated weight- update window buffer and line buffer
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int i = 0; i < 4; i ++) {
							window_buffer_wt[co][i][0] = window_buffer_wt[co][i][1];
							window_buffer_wt[co][i][1] = window_buffer_wt[co][i][2];
							window_buffer_wt[co][i][2] = window_buffer_wt[co][i][3];

//							// printf("window_buffer_wt[%d][%d][0] = %d \n", co, i, window_buffer_wt[co][i][0].to_int());
//							// printf("window_buffer_wt[%d][%d][1] = %d \n", co, i, window_buffer_wt[co][i][1].to_int());
//							// printf("window_buffer_wt[%d][%d][2] = %d \n", co, i, window_buffer_wt[co][i][2].to_int());
//							// printf("window_buffer_wt[%d][%d][3] = %d \n", co, i, window_buffer_wt[co][i][3].to_int());
//							// printf("window_buffer_wt[%d][%d][4] = %d \n", co, i, window_buffer_wt[co][i][4].to_int());
						}

						window_buffer_wt[co][0][3] = (line_buffer_wt[co][0][kcol]);
						window_buffer_wt[co][1][3] = (line_buffer_wt[co][0][kcol] = line_buffer_wt[co][1][kcol]);
						window_buffer_wt[co][2][3] = (line_buffer_wt[co][1][kcol] = line_buffer_wt[co][2][kcol]);
						window_buffer_wt[co][3][3] = (line_buffer_wt[co][2][kcol]);
						line_buffer_wt[co][2][kcol] = (stride > 1 && krow % 2 > 0 && kcol % 2 > 0) ? int8(0) : weight[co][krow][kcol];
					}
					
					// activation- update window buffer and line buffer (padding = 1)
					for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
						for (int i = 0; i < 4; i ++) {
							window_buffer_act[cin][i][0] = window_buffer_act[cin][i][1];
							window_buffer_act[cin][i][1] = window_buffer_act[cin][i][2];
							window_buffer_act[cin][i][2] = window_buffer_act[cin][i][3];

//							// printf("window_buffer_act[%d][%d][0] = %d \n", cin, i, window_buffer_act[cin][i][0].to_int());
//							// printf("window_buffer_act[%d][%d][1] = %d \n", cin, i, window_buffer_act[cin][i][1].to_int());
//							// printf("window_buffer_act[%d][%d][2] = %d \n", cin, i, window_buffer_act[cin][i][2].to_int());
//							// printf("window_buffer_act[%d][%d][3] = %d \n", cin, i, window_buffer_act[cin][i][3].to_int());
//							// printf("window_buffer_act[%d][%d][4] = %d \n", cin, i, window_buffer_act[cin][i][4].to_int());
						}

						row_input = krow + ii*H_fmap_in;
						col_input = kcol + jj*H_fmap_in;

						window_buffer_act[cin][0][3] = (line_buffer_act[cin][0][kcol]);
						window_buffer_act[cin][1][3] = (line_buffer_act[cin][0][kcol] = line_buffer_act[cin][1][kcol]);
						window_buffer_act[cin][2][3] = (line_buffer_act[cin][1][kcol] = line_buffer_act[cin][2][kcol]);
						window_buffer_act[cin][3][3] = (line_buffer_act[cin][2][kcol]);
						line_buffer_act[cin][2][kcol] = input[cin][row_input][col_input];
					}

					// dilated conv
					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							skip_kcol += 1;
							skip_k32 += 1;
							skip_k16 += 1;
							skip_k8 += 1;
							if ((H_fmap_in==32 && (skip_k32<32 || skip_k32>=992)) || (H_fmap_in==16 && (skip_k16<16 || skip_k16>=240)) || (H_fmap_in==8 && (skip_k8<8 || skip_k8>=56))) {
								skip_k = 0;
							}

							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {

									wt = window_buffer_wt[co][pkrow][pkcol];
									act = window_buffer_act[cin][pkrow][pkcol];
									// accum += wt * act;
									accum = bm_mul(wt, act);

									// printf("skip_krow = %d, skip_kcol = %d, ", skip_krow.to_int(), skip_kcol.to_int());
									// printf("accum = %d \n", accum.to_int());

									// conv(0, 0)
									if (krow >= 0 && krow < H_fmap_in - 1 && kcol >= 0 && kcol < H_fmap_in - 1 && skip_kcol == 0 && skip_k == 0) {
										out_temp[co][cin][0][0] = bm_add(accum, out_temp[co][cin][0][0], 0);
										// printf("out_temp[%d][%d][0][0] = %d \n", co, cin, out_temp[co][cin][0][0].to_int());
									}
									// conv(0, 1)
									if (krow >= 0 && krow < H_fmap_in - 1  && kcol >= 0 && kcol < H_fmap_in && skip_kcol == 1 && skip_k == 0) {
										out_temp[co][cin][0][1] = bm_add(accum, out_temp[co][cin][0][1], 0);
										// printf("out_temp[%d][%d][0][1] = %d \n", co, cin, out_temp[co][cin][0][1].to_int());
									}
									// conv(0, 2)
									if (krow >= 0 && krow < H_fmap_in - 1 && kcol >= 1 && kcol < H_fmap_in + 1 && skip_kcol == 2 && skip_k == 0) {
										out_temp[co][cin][0][2] = bm_add(accum, out_temp[co][cin][0][2], 0);
										// printf("out_temp[%d][%d][0][2] = %d \n", co, cin, out_temp[co][cin][0][2].to_int());
									}
								}
							}
						}
					}

					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							skip_kcol += 1;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {

									wt = window_buffer_wt[co][pkrow][pkcol];
									act = window_buffer_act[cin][pkrow][pkcol];
									// accum += wt * act;
									accum = bm_mul(wt, act);

									// printf("skip_krow = %d, skip_kcol = %d, ", skip_krow.to_int(), skip_kcol.to_int());
									// printf("accum = %d \n", accum.to_int());

									// conv(1, 0)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip_kcol == 0 && skip_k == 0) {
										out_temp[co][cin][1][0] = bm_add(accum, out_temp[co][cin][1][0], 0);
										// printf("out_temp[%d][%d][1][0] = %d \n", co, cin, out_temp[co][cin][1][0].to_int());
									}
									// conv(1, 1)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip_kcol == 1 && skip_k == 0) {
										out_temp[co][cin][1][1] = bm_add(accum, out_temp[co][cin][1][1], 0);
										// printf("out_temp[%d][%d][1][1] = %d \n", co, cin, out_temp[co][cin][1][1].to_int());
									}
									// conv(1, 2)
									if (krow >= 0 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip_kcol == 2 && skip_k == 0) {
										out_temp[co][cin][1][2] = bm_add(accum, out_temp[co][cin][1][2], 0);
										// printf("out_temp[%d][%d][1][2] = %d \n", co, cin, out_temp[co][cin][2][2].to_int());
									}
								}
							}
						}
					}

					for (int co = 0; co < CHANNEL_OUT_T; co ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							accum = 0;
							skip_kcol += 1;
							for (int pkrow = 0; pkrow < 4; pkrow ++) {
								for (int pkcol = 0; pkcol < 4; pkcol ++) {

									wt = window_buffer_wt[co][pkrow][pkcol];
									act = window_buffer_act[cin][pkrow][pkcol];
									// accum += wt * act;
									accum = bm_mul(wt, act);

									// printf("skip_krow = %d, skip_kcol = %d, ", skip_krow.to_int(), skip_kcol.to_int());
									// printf("accum = %d \n", accum.to_int());

									// conv(2, 0)
									if (krow >= 1 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in - 1 && skip_kcol == 0 && skip_k == 0) {
										out_temp[co][cin][2][0] = bm_add(accum, out_temp[co][cin][2][0], 0);
										// printf("out_temp[%d][%d][2][0] = %d \n", co, cin, out_temp[co][cin][2][0].to_int());
									}
									// conv(2, 1)
									if (krow >= 1 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in && skip_kcol == 1 && skip_k == 0) {
										out_temp[co][cin][2][1] = bm_add(accum, out_temp[co][cin][2][1], 0);
										// printf("out_temp[%d][%d][2][1] = %d \n", co, cin, out_temp[co][cin][2][1].to_int());
									}
									// conv(2, 2)
									if (krow >= 1 && krow < H_fmap_in && kcol >= 1 && kcol < H_fmap_in + 1 && skip_kcol == 2 && skip_k == 0) {
										out_temp[co][cin][2][2] = bm_add(accum, out_temp[co][cin][2][2], 0);
										// printf("out_temp[%d][%d][2][2] = %d \n", co, cin, out_temp[co][cin][2][2].to_int());
									}
								}
							}
						}
					}
					// printf("\n");
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
					// output[co][cin][row][col] = out_temp[co][cin][row][col];
					output[co][cin][row][col] = bm_mac_no_DSP(lr, out_temp[co][cin][row][col], output[co][cin][row][col]);
				}
			}
		}
	}
}

#endif
