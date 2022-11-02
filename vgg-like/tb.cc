#include "bnn.h"
#include "weights_tb.h"
#include "typedefs.h"
#include "conv_weights_tb.h"

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <hls_math.h>

using namespace std;
//const uint2 exp_bias = 2;	// ieee-754 shared_exp_bias
float lr_sw = 1e-2;
float eps_sw = 1e-10;

#define CHANNEL_IN_T 8
#define CHANNEL_OUT_T 8
#define WIDTH 32

#define NUM_3x3_WT 41
#define NUM_ACT 41
#define ACT_BIAS 64/CHANNEL_IN_T
#define WT_BIAS (NUM_3x3_WT)/CHANNEL_OUT_T      // 8*8*3*3
#define NUM_TILE 64/CHANNEL_OUT_T

//--------------------------------
// floating-point golden reference
//--------------------------------

//float image_sw[3][32][32]];
float sum_sw = 0;
float loss_sw = 0;
float error_sw[10];

///////////////////////////
////////  forward  ////////

// layer1_Conv1
//float layer1_conv1_weight[32][3][3][3];
float layer1_conv1_weight_rot[3][32][3][3];
float layer1_conv1_out[32][32][32];
float layer1_bn_relu1_out[32][32][32];
uint1 layer1_relu1_mask[32][32][32];
//float layer1_bn_relu1_bn_wt[32];
//float layer1_bn_relu1_bn_bias[32];

// layer1_Conv2
//float layer1_conv2_weight[32][32][3][3];
float layer1_conv2_weight_rot[32][32][3][3];
float layer1_conv2_out[32][32][32];
float layer1_bn_relu2_out[32][32][32];
uint1 layer1_relu2_mask[32][32][32];
//float layer1_bn_relu2_bn_wt[32];
//float layer1_bn_relu2_bn_bias[32];

// layer1_ConvSC
//float layer1_convSC_weight[64][32][3][3];
float layer1_convSC_weight_rot[32][64][3][3];
float layer1_convSC_out[64][16][16];
float layer1_bn_reluSC_out[64][16][16];
uint1 layer1_reluSC_mask[64][16][16];
//float layer1_bn_reluSC_bn_wt[64];
//float layer1_bn_reluSC_bn_bias[64];

// layer2_Conv1
//float layer2_conv1_weight[64][64][3][3];
float layer2_conv1_weight_rot[64][64][3][3];
float layer2_conv1_out[64][16][16];
float layer2_bn_relu1_out[64][16][16];
uint1 layer2_relu1_mask[64][16][16];
//float layer2_bn_relu1_bn_wt[16];
//float layer2_bn_relu1_bn_bias[16];

// layer2_Conv2
//float layer2_conv2_weight[64][64][3][3];
float layer2_conv2_weight_rot[64][64][3][3];
float layer2_conv2_out[64][16][16];
float layer2_bn_relu2_out[64][16][16];
uint1 layer2_relu2_mask[64][16][16];
//float layer2_bn_relu2_bn_wt[64];
//float layer2_bn_relu2_bn_bias[64];

// layer2_ConvSC
//float layer2_convSC_weight[64][64][3][3];
float layer2_convSC_weight_rot[64][64][3][3];
float layer2_convSC_out[64][8][8];
float layer2_bn_reluSC_out[64][8][8];
uint1 layer2_reluSC_mask[64][8][8];
//float layer2_bn_reluSC_bn_wt[64];
//float layer2_bn_reluSC_bn_bias[64];

// avgpool_sw & FC_sw
float avg_pool_out_sw[64];
//float linear_weight[10][64];
float linear_out_buf_sw[10];

///////////////////////////
////////  backward  ///////

// avgpool_sw & FC_sw
float fc_bp_out[64];
float avg_pool_bp_out[64][8][8];

// layer2_ConvSC
float layer2_bn_reluSC_bp_out[64][8][8];
float layer2_convSC_bp_out[64][16][16];

// layer2_Conv2
float layer2_bn_relu2_bp_out[64][16][16];
float layer2_conv2_bp_out[64][16][16];

// layer2_Conv1
float layer2_bn_relu1_bp_out[64][16][16];
float layer2_conv1_bp_out[64][16][16];

// layer1_ConvSC
float layer1_bn_reluSC_bp_out[64][16][16];
float layer1_convSC_bp_out[32][32][32];

// layer1_Conv2
float layer1_bn_relu2_bp_out[32][32][32];
float layer1_conv2_bp_out[32][32][32];

// layer1_Conv1
float layer1_bn_relu1_bp_out[32][32][32];
float layer1_conv1_bp_out[3][32][32];

template <int in_channels_hw, int out_channels_hw>
void rot_3x3
(
	float weight[out_channels_hw][in_channels_hw][3][3],
	float weight_rot[in_channels_hw][out_channels_hw][3][3]
)
{
	float act;
	float wt;
	float weight_tmp[in_channels_hw][out_channels_hw][3][3];

	//// rotated 180
	// transpose
	for (int co = 0; co < out_channels_hw; co ++) {
		for (int cin = 0; cin < in_channels_hw; cin ++) {
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
					weight_tmp[cin][co][kcol][krow] = weight[co][cin][krow][kcol];
				}
			}
		}
	}
	// diagonal
	for (int co = 0; co < out_channels_hw; co ++) {
		for (int cin = 0; cin < in_channels_hw; cin ++) {
			weight_tmp[cin][co][0][0] = weight[cin][co][2][2];
			weight_tmp[cin][co][2][2] = weight[cin][co][0][0];
		}
	}
	// write back
	for (int co = 0; co < out_channels_hw; co ++) {
		for (int cin = 0; cin < in_channels_hw; cin ++) {
			for (int krow = 0; krow < 3; krow ++) {
				for (int kcol = 0; kcol < 3; kcol ++) {
					weight_rot[cin][co][krow][kcol] = weight_tmp[cin][co][krow][kcol];
				}
			}
		}
	}
}

template <int in_channels_hw, int out_channels_hw>
void rot_1x1
(
	float weight[out_channels_hw][in_channels_hw],
	float weight_rot[in_channels_hw][out_channels_hw]
)
{
	float act;
	float wt;

	// rotated 180
	for (int co = 0; co < out_channels_hw; co ++) {
		for (int cin = 0; cin < in_channels_hw; cin ++) {
			weight_rot[cin][co] = weight[co][cin];
		}
	}
}

// Conv
template <int in_channels_hw, int out_channels_hw, int H_fmap_in_hw, int H_fmap_out_hw, int stride_hw>
void conv_3x3_sw
(
	float input[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],			// in on-chip
	float weight[out_channels_hw][in_channels_hw][3][3],
	float output[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw]		// out on-chip
)
{
	float act;
	float wt;
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
			for (int co = 0; co < out_channels_hw; co ++) {
				float accum = 0;
				for (int krow = 0; krow < 3; krow ++) {
					for (int kcol = 0; kcol < 3; kcol ++) {
						for (int cin = 0; cin < in_channels_hw; cin ++) {
							int row_in = row*stride_hw + krow - 1;
							int col_in = col*stride_hw + kcol - 1;
							if (row_in >= 0 && row_in < H_fmap_in_hw && col_in >= 0 && col_in < H_fmap_in_hw) {
								act = input[cin][row_in][col_in];
								wt = weight[co][cin][krow][kcol];
								accum += act * wt;
							}
						}
					}
				}
				output[co][row][col] = accum;
			}
		}
	}
}

template <int in_channels_hw, int out_channels_hw, int H_fmap_in_hw, int H_fmap_out_hw, int stride_hw>
void conv_1x1_sw
(
	float input[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],			// in on-chip
	float weight[out_channels_hw][in_channels_hw],
	float output[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw]		// out on-chip
)
{
	float act;
	float wt;
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
			for (int co = 0; co < out_channels_hw; co ++) {
				float accum = 0;
				for (int cin = 0; cin < in_channels_hw; cin ++) {
					int row_in = row*stride_hw;
					int col_in = col*stride_hw;
					if (row_in >= 0 && row_in < H_fmap_in_hw && col_in >= 0 && col_in < H_fmap_in_hw) {
						act = input[cin][row_in][col_in];
						wt = weight[co][cin];
						accum += act * wt;
					}
				}
				output[co][row][col] = accum;
			}
		}
	}
}

// Transposed Conv
template <int in_channels_hw, int out_channels_hw, int H_fmap_in_hw, int H_fmap_out_hw, int stride_hw>
void deconv_3x3_sw
(
	float input[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],			// in on-chip
	float weight[out_channels_hw][in_channels_hw][3][3],
	float output[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw]		// out on-chip
)
{
	float act;
	float wt;
	float input_dil[in_channels_hw][H_fmap_out_hw][H_fmap_out_hw];

	// buffer ini_hwt
	for (int row_in = 0; row_in < H_fmap_out_hw; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_out_hw; col_in ++) {
			for (int cin = 0; cin < in_channels_hw; cin ++) {
				input_dil[cin][row_in][col_in] = 0;
			}
		}
	}

	// input dilation
	for (int row_in = 0; row_in < H_fmap_in_hw; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_in_hw; col_in ++) {
			for (int cin = 0; cin < in_channels_hw; cin ++) {
				input_dil[cin][row_in*stride_hw][col_in*stride_hw] = input[cin][row_in][col_in];
			}
		}
	}

	// deconv_3x3_sw
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
			for (int co = 0; co < out_channels_hw; co ++) {
				float accum = 0;
				for (int krow = 0; krow < 3; krow ++) {
					for (int kcol = 0; kcol < 3; kcol ++) {
						for (int cin = 0; cin < in_channels_hw; cin ++) {
							int row_in = row + krow - 1;
							int col_in = col + kcol - 1;
							if (row_in >= 0 && row_in < H_fmap_in_hw*stride_hw && col_in >= 0 && col_in < H_fmap_in_hw*stride_hw) {
								act = input_dil[cin][row_in][col_in];
								wt = weight[co][cin][krow][kcol];
								accum += act * wt;
							}
						}
					}
				}
				output[co][row][col] = accum;
			}
		}
	}
}

template <int in_channels_hw, int out_channels_hw, int H_fmap_in_hw, int H_fmap_out_hw, int stride_hw>
void deconv_1x1_sw
(
	float input[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],			// in on-chip
	float weight[out_channels_hw][in_channels_hw],
	float output[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw]		// out on-chip
)
{
	float act;
	float wt;
	float input_dil[in_channels_hw][H_fmap_out_hw][H_fmap_out_hw];

	// buffer ini_hwt
	for (int row_in = 0; row_in < H_fmap_out_hw; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_out_hw; col_in ++) {
			for (int cin = 0; cin < in_channels_hw; cin ++) {
				input_dil[cin][row_in][col_in] = 0;
			}
		}
	}

	// input dilation
	for (int row_in = 0; row_in < H_fmap_in_hw; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_in_hw; col_in ++) {
			for (int cin = 0; cin < in_channels_hw; cin ++) {
				input_dil[cin][row_in * stride_hw][col_in * stride_hw] = input[cin][row_in][col_in];
			}
		}
	}
	// deconv_1x1_sw
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
			for (int co = 0; co < out_channels_hw; co ++) {
				float accum = 0;
				for (int cin = 0; cin < in_channels_hw; cin ++) {
					int row_in = row;
					int col_in = col;
					if (row_in >= 0 && row_in < H_fmap_in_hw*stride_hw && col_in >= 0 && col_in < H_fmap_in_hw*stride_hw) {
						act = input_dil[cin][row_in][col_in];
						wt = weight[co][cin];
						accum += act * wt;
					}
				}
				output[co][row][col] = accum;
			}
		}
	}
}

template <int in_channels_hw, int out_channels_hw, int H_fmap_out_hw, int stride_hw>
void conv_3x3_grad_sw
(
	float weight[out_channels_hw][H_fmap_out_hw/stride_hw][H_fmap_out_hw/stride_hw],
	float input[in_channels_hw][H_fmap_out_hw][H_fmap_out_hw],		// in on-chip
	float output[out_channels_hw][in_channels_hw][3][3]			// out on-chip
)
{
	float accum;
	float weight_dil[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw] = {0};

	// buffer ini_hwt
	for (int row_in = 0; row_in < H_fmap_out_hw; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_out_hw; col_in ++) {
			for (int co = 0; co < out_channels_hw; co ++) {
				weight_dil[co][row_in][col_in] = 0;
			}
		}
	}

	// weight dilation
	for (int co = 0; co < out_channels_hw; co ++) {
		for (int krow = 0; krow < H_fmap_out_hw/stride_hw; krow ++) {
			for (int kcol = 0; kcol < H_fmap_out_hw/stride_hw; kcol ++) {
				weight_dil[co][krow * stride_hw][kcol * stride_hw] = weight[co][krow][kcol];
			}
		}
	}

	// conv3x3 grad
	for (int co = 0; co < out_channels_hw; co ++) {
		for (int ci = 0; ci < in_channels_hw; ci ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					accum = 0;
					for (int krow = 0; krow < H_fmap_out_hw; krow ++) {
						for (int kcol = 0; kcol < H_fmap_out_hw; kcol ++) {
							int row_in = row + krow - 1;
							int col_in = col + kcol - 1;
							if (row_in >= 0 && row_in < H_fmap_out_hw && col_in >= 0 && col_in < H_fmap_out_hw) {
								float act = input[ci][row_in][col_in];
								float wt = weight_dil[co][krow][kcol];
								accum += act * wt;
								// if (act != 0 && wt != 0) printf("act = %f, wt: %f \n", act, wt);
							}
						}
					}
					output[co][ci][row][col] += -lr_sw * accum;
//					if (accum != 0) printf("weight grad [%d][%d][%d][%d]: %f \n", co, ci, row, col, accum);
					// if (accum != 0) printf("updated weight [%d][%d][%d][%d]: %f \n", co, ci, row, col, output[co][ci][row][col]);
				}
			}
		}
	}
}

template <int in_channels_hw, int out_channels_hw, int H_fmap_out_hw, int stride_hw>
void conv_1x1_grad_sw
(
	float weight[out_channels_hw][H_fmap_out_hw/stride_hw][H_fmap_out_hw/stride_hw],
	float input[in_channels_hw][H_fmap_out_hw][H_fmap_out_hw],		// in on-chip
	float output[out_channels_hw][in_channels_hw]					// out on-chip
)
{
	float accum;
	float weight_dil[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw] = {0};

	// buffer ini_hwt
	for (int row_in = 0; row_in < H_fmap_out_hw; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_out_hw; col_in ++) {
			for (int co = 0; co < out_channels_hw; co ++) {
				weight_dil[co][row_in][col_in] = 0;
			}
		}
	}

	// weight dilation
	for (int co = 0; co < out_channels_hw; co ++) {
		for (int krow = 0; krow < H_fmap_out_hw/stride_hw; krow ++) {
			for (int kcol = 0; kcol < H_fmap_out_hw/stride_hw; kcol ++) {
				weight_dil[co][krow * stride_hw][kcol * stride_hw] = weight[co][krow][kcol];
			}
		}
	}

	// conv1x1 grad
	for (int co = 0; co < out_channels_hw; co ++) {
		for (int ci = 0; ci < in_channels_hw; ci ++) {
			accum = 0;
			for (int krow = 0; krow < H_fmap_out_hw; krow ++) {
				for (int kcol = 0; kcol < H_fmap_out_hw; kcol ++) {
					int row_in = krow;
					int col_in = kcol;
					if (row_in >= 0 && row_in < H_fmap_out_hw && col_in >= 0 && col_in < H_fmap_out_hw) {
						float act = input[ci][row_in][col_in];
						float wt = weight_dil[co][krow][kcol];
						accum += act * wt;
						// if (act != 0 && wt != 0) printf("act = %f, wt: %f \n", act, wt);
					}
				}
			}
			output[co][ci] += -lr_sw * accum;
			// if (accum != 0) printf("weight grad [%d][%d]: %f \n", co, ci, accum);
			// if (accum != 0) printf("updated weight [%d][%d]: %f \n", co, ci, output[co][ci]);
		}
	}
}

// bn_sw
template <int in_channels_hw, int H_fmap_in_hw>
void bn_sw(
	float bn_inputs[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// in
	float out_buf[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// out

	float bn_wt_hw[in_channels_hw],
	float bn_bias_hw[in_channels_hw]
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[in_channels_hw];
	float std_var[in_channels_hw];
	float sum_sw[in_channels_hw] = {0};
	// mean
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				sum_sw[c] += bn_inputs[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				mu[c] = sum_sw[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				std_var[c] += (bn_inputs[c][row][col]-mu[c])*(bn_inputs[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < in_channels_hw; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// generic_bn_hw
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				out_buf[c][row][col] = bn_wt_hw[c]*(bn_inputs[c][row][col]-mu[c])/(std_var[c] + eps_sw) + bn_bias_hw[c];
			}
		}
	}
}

// bn_relu_sw
template <int in_channels_hw, int H_fmap_in_hw>
void bn_relu_sw(
	float bn_inputs[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// in
	float out_buf[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// out, bn_outputs
	uint1 relu_mask_hw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// out, relu_mask_hw for relu_bp

	float bn_wt_hw[in_channels_hw],
	float bn_bias_hw[in_channels_hw]
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[in_channels_hw];
	float std_var[in_channels_hw];
	float sum_sw[in_channels_hw] = {0};
	// mean
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				sum_sw[c] += bn_inputs[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				mu[c] = sum_sw[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				std_var[c] += (bn_inputs[c][row][col]-mu[c])*(bn_inputs[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < in_channels_hw; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// generic_bn_relu_hw
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				out_buf[c][row][col] = bn_wt_hw[c]*(bn_inputs[c][row][col]-mu[c])/(std_var[c] + eps_sw) + bn_bias_hw[c];
				relu_mask_hw[c][row][col] = (out_buf[c][row][col] > 0) ? 1 : 0;
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				out_buf[c][row][col] = (relu_mask_hw[c][row][col] == 1) ? out_buf[c][row][col] : float(0);
			}
		}
	}
}

// bn_bp_sw
template <int in_channels_hw, int H_fmap_in_hw>
void bn_bp_sw(
	float error_sw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw], 		// in
	float bn_inputs_fw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],	// in
	float out_buf[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// out, error_bn

	float bn_wt_hw[in_channels_hw],								// in
	float bn_bias_hw[in_channels_hw]								// in
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[in_channels_hw];
	float std_var[in_channels_hw];
	float sum_sw[in_channels_hw] = {0};
	float g_bn_wt_hw[in_channels_hw] = {0};						// out
	float g_bn_bias_hw[in_channels_hw] = {0};						// out

	// mean
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				sum_sw[c] += bn_inputs_fw[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				mu[c] = sum_sw[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				std_var[c] += (bn_inputs_fw[c][row][col]-mu[c])*(bn_inputs_fw[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < in_channels_hw; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// grad of generic_bn_hw params
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				g_bn_bias_hw[c] += error_sw[c][row][col];
				g_bn_wt_hw[c] += error_sw[c][row][col] * (bn_inputs_fw[c][row][col]-mu[c])/(std_var[c]+eps_sw);
			}
		}
	}

	// bn_bp_sw
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				out_buf[c][row][col] = bn_wt_hw[c]*error_sw[c][row][col]/(std_var[c]+eps_sw) - bn_wt_hw[c]*g_bn_bias_hw[c]/(N*(std_var[c]+eps_sw)) - bn_wt_hw[c]*(bn_inputs_fw[c][row][col]-mu[c])*g_bn_wt_hw[c]/(N*(std_var[c]*std_var[c]+eps_sw));
			}
		}
	}

	// bn_sw params update
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				bn_bias_hw[c] += -lr_sw * g_bn_bias_hw[c];
				bn_wt_hw[c] += -lr_sw * g_bn_wt_hw[c];
			}
		}
	}
}

// bn_relu_bp_sw
template <int in_channels_hw, int H_fmap_in_hw>
void bn_relu_bp_sw(
	float error_sw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw], 		// in
	float bn_inputs_fw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],	// in
	float out_buf[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// out, error_bn
	uint1 relu_mask_hw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// in

	float bn_wt_hw[in_channels_hw],								// in
	float bn_bias_hw[in_channels_hw]								// in
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[in_channels_hw];
	float std_var[in_channels_hw];
	float sum_sw[in_channels_hw] = {0};
	float g_bn_wt_hw[in_channels_hw] = {0};						// out
	float g_bn_bias_hw[in_channels_hw] = {0};						// out

	// relu_bp
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				error_sw[c][row][col] = (relu_mask_hw[c][row][col] == 1) ? error_sw[c][row][col] : float(0);
			}
		}
	}

	// mean
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				sum_sw[c] += bn_inputs_fw[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				mu[c] = sum_sw[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				std_var[c] += (bn_inputs_fw[c][row][col]-mu[c])*(bn_inputs_fw[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < in_channels_hw; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// grad of generic_bn_hw params
	for (int c = 0; c < in_channels_hw; c ++) {
		for (int row = 0; row < H_fmap_in_hw; row ++) {
			for (int col = 0; col < H_fmap_in_hw; col ++) {
				g_bn_bias_hw[c] += error_sw[c][row][col];
				g_bn_wt_hw[c] += error_sw[c][row][col] * (bn_inputs_fw[c][row][col]-mu[c])/(std_var[c]+eps_sw);
			}
		}
//		printf("g_bn_wt_hw[%d] = %f, g_bn_bias_hw[%d] = %f \n", c, g_bn_wt_hw[c], c, g_bn_bias_hw[c]);
	}

	// generic_bn_bp_hw
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				out_buf[c][row][col] = bn_wt_hw[c]*error_sw[c][row][col]/(std_var[c]+eps_sw) - bn_wt_hw[c]*g_bn_bias_hw[c]/(N*(std_var[c]+eps_sw)) - bn_wt_hw[c]*(bn_inputs_fw[c][row][col]-mu[c])*g_bn_wt_hw[c]/(N*(std_var[c]*std_var[c]+eps_sw));
			}
		}
	}

	// bn_sw params update
	for (int c = 0; c < in_channels_hw; c ++) {
		for (int row = 0; row < H_fmap_in_hw; row ++) {
			for (int col = 0; col < H_fmap_in_hw; col ++) {
				bn_bias_hw[c] += -lr_sw * g_bn_bias_hw[c];
				bn_wt_hw[c] += -lr_sw * g_bn_wt_hw[c];
			}
		}
//		printf("bn_wt_hw[%d] = %f, bn_bias_hw[%d] = %f \n", c, bn_wt_hw[c], c, bn_bias_hw[c]);
	}
}

// avgpool_sw
void avgpool_sw(
	float avg_inputs[64][8][8],		// in, 64x8x8
	float out_buf[64],				// out, avg_outputs

	uint1 ctrl_avgpool_hw
)
{
	// forward
	if (ctrl_avgpool_hw == 0) {
		// buffer ini_hwt
		for (int c = 0; c < 64; c ++) {
			out_buf[c] = 0;
		}
		for (int c = 0; c < 64; c ++) {
			for (int s = 0; s < 8; s ++) {
				for (int ss = 0; ss < 8; ss ++) {
					 out_buf[c] += avg_inputs[c][s][ss]/64;
				}
			}
//			printf("avgpool_sw out: %f \n", out_buf[c + c_out*in_channels_hw]);
		}
	}
	// backward
	else {
		for (int s = 0; s < 8; s ++) {
			for (int ss = 0; ss < 8; ss ++) {
				for (int c = 0; c < 64; c ++) {
					avg_inputs[c][s][ss] = out_buf[c]/64;
				}
			}
		}
	}
}

// FC_sw
void FC_sw(
	float inputs[64],
	float inputs_fw[64],
	float linear_weight[10][64],
	float outputs[10],

	uint1 ctrl_fc_hw	// 0 for forward and 1 for backward
)
{
	float linear_weight_t[64][10];

	// forward
	if (ctrl_fc_hw == 0) {
		// buffer ini_hwt
		for (int coo = 0; coo < 10; coo ++) {
			outputs[coo] = 0;
		}
		for (int coo = 0; coo < 10; coo ++) {
			for (int cii = 0; cii < 64; cii++) {
				outputs[coo] += inputs[cii] * linear_weight[coo][cii];
			}
//			printf("FC_sw out: %f \n", outputs[coo]);
		}
	}
	// backward
	else {
		// buffer ini_hwt
		for (int cii = 0; cii < 64; cii++) {
			inputs[cii] = 0;
		}
		for (int cii = 0; cii < 64; cii++) {
			for (int coo = 0; coo < 10; coo ++) {
				inputs[cii] += outputs[coo] * linear_weight[coo][cii];
			}
		}
		// weight update
		for (int cii = 0; cii < 64; cii++) {
			for (int coo = 0; coo < 10; coo ++) {
				linear_weight[coo][cii] += -lr_sw * inputs_fw[cii] * outputs[coo];
			}
		}
	}
}

// shortcut_sw addition
template <int in_channels_hw, int H_fmap_in_hw>
void shortcut_sw(
	float input_a[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],			// in1
	float input_b[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],			// in2
	float out_buf[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw]			// out
)
{
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < in_channels_hw; c ++) {
				out_buf[c][row][col] = input_a[c][row][col] + input_b[c][row][col];
			}
		}
	}
}



//--------------------------------
// floating-point golden reference
//--------------------------------

float msb_fmap_tile_buffer_0_hw[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error_hw on-chip
float msb_fmap_tile_buffer_1_hw[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error_hw on-chip
float msb_fmap_tile_buffer_s2_hw[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error_hw on-chip

float out_buf_t0_hw[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// conv output
float out_buf_t1_hw[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// generic_bn_hw output
uint1 relu_mask_hw[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// relu mask for backprop

//float conv_3x3_weight_tile_buffer_hw[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
//float conv_1x1_weight_tile_buffer_hw[NUM_1x1_WT][CHANNEL_OUT_T][CHANNEL_IN_T];
float conv_3x3_weight_rot_hw[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];

float pool_out_buf_hw[64];
float pool_out_buf_copy_hw[64];	// generic_avgpool_hw buffer
//float linear_weight_tile_buffer_hw[10][64];
float linear_out_buf_hw[10];											// generic_FC_hw buffer

float sum_hw;
float loss_hw = 0;
float error_hw[10];
float bias_gap4add_hw[CHANNEL_OUT_T];									// shared exponent gap between act & wt in bm addition

//--------------------------------
// floating-point golden reference
//--------------------------------

void rot180_3x3_hw
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

void rot180_1x1_hw
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

void generic_conv_3x3_hw
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
	float act;
	float wt;
	float out_temp[CHANNEL_OUT_T][WIDTH][WIDTH];

	// buffer initiation
	for (int row = 0; row < H_fmap_out; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				if (c_in > 0) {
					out_temp[co][row][col] = output[co][row][col];
				}
				else {
					out_temp[co][row][col] = 0;
				}
			}
		}
	}

	// conv 3x3
	for (int row = 0; row < H_fmap_out; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				float accum = 0;
				for (int krow = 0; krow < 3; krow ++) {
					for (int kcol = 0; kcol < 3; kcol ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							int row_in = row*stride + krow - 1;
							int col_in = col*stride + kcol - 1;
							if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
								act = input[cin][row_in][col_in];
								wt = weight[co][cin][krow][kcol];
								accum += act * wt;
//								if (act != 0 && wt != 0) printf("act: %f, wt: %f \n", act, wt);
							}
						}
					}
				}
				out_temp[co][row][col] += accum;

				output[co][row][col] = out_temp[co][row][col];
				output_DDR[co][row][col] = out_temp[co][row][col];
			}
		}
	}
}

void generic_deconv_3x3_hw
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	float weight[CHANNEL_IN_T][CHANNEL_OUT_T][3][3],
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
	float act;
	float wt;
	float input_dil[CHANNEL_IN_T][WIDTH][WIDTH];
	float out_temp[CHANNEL_OUT_T][WIDTH][WIDTH];

	// buffer init
	for (int row_in = 0; row_in < WIDTH; row_in ++) {
		for (int col_in = 0; col_in < WIDTH; col_in ++) {
			for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
				input_dil[cin][row_in][col_in] = 0;
			}
		}
	}

	// input dilation
	for (int row_in = 0; row_in < H_fmap_in; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_in; col_in ++) {
			for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
				input_dil[cin][row_in*stride][col_in*stride] = input[cin][row_in][col_in];
			}
		}
	}

	// buffer initiation
	for (int row = 0; row < H_fmap_out; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				if (c_in > 0) {
					out_temp[co][row][col] = output[co][row][col];
				}
				else {
					out_temp[co][row][col] = 0;
				}
			}
		}
	}

	// conv 3x3
	for (int row = 0; row < H_fmap_out; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				float accum = 0;
				for (int krow = 0; krow < 3; krow ++) {
					for (int kcol = 0; kcol < 3; kcol ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							int row_in = row + krow - 1;
							int col_in = col + kcol - 1;
							if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
								act = input_dil[cin][row_in][col_in];
								wt = weight[co][cin][krow][kcol];
								accum += act * wt;
							}
						}
					}
				}
				out_temp[co][row][col] += accum;

				output[co][row][col] = out_temp[co][row][col];
//				output_DDR[co][row][col] = out_temp[co][row][col];
			}
		}
	}
}

void generic_conv_1x1_hw
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	float weight[CHANNEL_OUT_T][CHANNEL_IN_T],
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
	float act;
	float wt;
	float out_temp[CHANNEL_OUT_T][WIDTH][WIDTH];

	// buffer initiation
	for (int row = 0; row < H_fmap_out; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				if (c_in > 0) {
					out_temp[co][row][col] = output[co][row][col];
				}
				else {
					out_temp[co][row][col] = 0;
				}
			}
		}
	}
	// conv 1x1
	for (int row = 0; row < H_fmap_out; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				float accum = 0;
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					int row_in = row * stride;
					int col_in = row * stride;
					if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
						act = input[ci][row_in][col_in];
						wt = weight[co][ci];
						accum += act * wt;
					}
				}
				out_temp[co][row][col] += accum;

				output[co][row][col] = out_temp[co][row][col];
				output_DDR[co][row][col] = out_temp[co][row][col];
			}
		}
	}
}

void generic_deconv_1x1_hw
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	float weight[CHANNEL_OUT_T][CHANNEL_IN_T],
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
	float act;
	float wt;
	float input_dil[CHANNEL_IN_T][WIDTH][WIDTH];
	float out_temp[CHANNEL_OUT_T][WIDTH][WIDTH];

	// buffer init
	for (int row_in = 0; row_in < WIDTH; row_in ++) {
		for (int col_in = 0; col_in < WIDTH; col_in ++) {
			for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
				input_dil[cin][row_in][col_in] = 0;
			}
		}
	}

	// input dilation
	for (int row_in = 0; row_in < H_fmap_in; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_in; col_in ++) {
			for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
				input_dil[cin][row_in*stride][col_in*stride] = input[cin][row_in][col_in];
			}
		}
	}

	// buffer initiation
	for (int row = 0; row < H_fmap_out; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				if (c_in > 0) {
					out_temp[co][row][col] = output[co][row][col];
				}
				else {
					out_temp[co][row][col] = 0;
				}
			}
		}
	}
	// conv 1x1
	for (int row = 0; row < H_fmap_out; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				float accum = 0;
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					int row_in = row;
					int col_in = col;
					if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
						act = input_dil[ci][row_in][col_in];
						wt = weight[co][ci];
						accum += act * wt;
					}
				}
				out_temp[co][row][col] += accum;

				output[co][row][col] = out_temp[co][row][col];
//				output_DDR[co][row][col] = out_temp[co][row][col];
			}
		}
	}
}

void generic_conv_3x3_grad_hw
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
					output[co][ci][row][col] += -lr_sw * accum;
				}
			}
		}
	}
}

void generic_conv_1x1_grad_hw
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],			// in on-chip
	float weight[CHANNEL_OUT_T][WIDTH][WIDTH],
	float output[CHANNEL_OUT_T][CHANNEL_IN_T],			// out on-chip

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

	// conv1x1 grad
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			accum = 0;
			for (int krow = 0; krow < H_fmap_in * stride; krow ++) {
				for (int kcol = 0; kcol < H_fmap_in * stride; kcol ++) {
					int row_in = krow;
					int col_in = kcol;
					if (row_in >= 0 && row_in < H_fmap_in*stride && col_in >= 0 && col_in < H_fmap_in*stride) {
						float act = input[ci][row_in][col_in];
						float wt = weight_dil[co][krow][kcol];
						accum += act * wt;
					}
				}
			}
			output[co][ci] += -lr_sw * accum;
		}
	}
}

// AvgPool
void generic_avgpool_hw(
	float avg_inputs[CHANNEL_IN_T][WIDTH][WIDTH],	// in, 64x8x8
	float out_buf[64],								// out, avg_outputs

	uint1 ctrl_avgpool_hw,	// 0 for forward and 1 for backward
	int c_out,

	float out_buf_copy[64],
	float bias_gap4add[CHANNEL_IN_T]
)
{
	// forward
	if (ctrl_avgpool_hw == 0) {
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
void generic_FC_hw(
	float inputs[64],
	float inputs_FW[64],
	float linear_weight[10][64],
	float outputs[10],

	uint1 ctrl_fc_hw	// 0 for forward and 1 for backward
)
{
	float linear_weight_t[64][10];

	// forward
	if (ctrl_fc_hw == 0) {
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
				linear_weight[coo][cii] += -lr_sw * inputs_FW[cii] * outputs[coo];
			}
		}
	}
}

void generic_shortcut_hw(
	float input_a[CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	float input_b[CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	float out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip
//	float out_buf_SC[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, ideneity for generic_shortcut_hw

	int H_fmap_in,
	uint1 ctrl_sc,										// if ctrl_sc=0, generate and send out_copy into DDR
//	uint1 ctrl_sc_id,

//	float shared_exp_bias[CHANNEL_OUT_T],
	float bias_gap4add[CHANNEL_OUT_T]
//	float act_bias_shift[CHANNEL_OUT_T]
)
{
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {

				out_buf[c][row][col] = input_a[c][row][col] + input_b[c][row][col];
				if (ctrl_sc == 0) {
					out_buf_DDR[c][row][col] = out_buf[c][row][col];
				}
//				if (ctrl_sc_id == 0) {
//					out_buf_SC[c][row][col] = out_buf[c][row][col];
//				}
			}
		}
	}
}

// generic_bn_relu_hw
void generic_bn_relu_hw(
	float bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs

	float out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip for backprop
	uint1 relu_mask[CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp
//	float out_buf_SC[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, ideneity for generic_shortcut_hw

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
	// mean
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				sum[c] += bn_inputs[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] = sum[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (bn_inputs[c][row][col]-mu[c])*(bn_inputs[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// generic_bn_relu_hw
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = bn_wt[c]*(bn_inputs[c][row][col]-mu[c])/(std_var[c] + eps_sw) + bn_bias[c];
				relu_mask[c][row][col] = (out_buf[c][row][col] > 0) ? 1 : 0;
			}
		}
	}
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = (relu_mask[c][row][col] == 1) ? out_buf[c][row][col] : float(0);
				out_buf_DDR[c][row][col] = out_buf[c][row][col];
			}
		}
	}
//	if (ctrl_bn_id == 0) {
//		for (int row = 0; row < H_fmap_in; row ++) {
//			for (int col = 0; col < H_fmap_in; col ++) {
//				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
//					out_buf_SC[c][row][col] = out_buf[c][row][col];
//				}
//			}
//		}
//	}
}

void generic_bn_relu_bp_hw(
	float error_hw[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
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

	// temp buffer init
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = 0;
		sum[c] = 0;
		g_bn_bias[c] = 0;
		g_bn_wt[c] = 0;
	}

	// relu_bp
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				error_hw[c][row][col] = (relu_mask[c][row][col] == 1) ? error_hw[c][row][col] : float(0);
			}
		}
	}

	// mean
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				sum[c] += bn_inputs_fw[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] = sum[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (bn_inputs_fw[c][row][col]-mu[c])*(bn_inputs_fw[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// grad of generic_bn_hw params
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				g_bn_bias[c] += error_hw[c][row][col];
				g_bn_wt[c] += error_hw[c][row][col] * (bn_inputs_fw[c][row][col]-mu[c])/(std_var[c]+eps_sw);
			}
		}
	}

	// generic_bn_bp_hw
	for (int row = 0; row < H_fmap_in; row ++) {
		for (int col = 0; col < H_fmap_in; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = bn_wt[c]*error_hw[c][row][col]/(std_var[c]+eps_sw) - bn_wt[c]*g_bn_bias[c]/(N*(std_var[c]+eps_sw)) - bn_wt[c]*(bn_inputs_fw[c][row][col]-mu[c])*g_bn_wt[c]/(N*(std_var[c]*std_var[c]+eps_sw));
			}
		}
	}

	// bn_sw params update
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		bn_bias[c] += -lr_sw * g_bn_bias[c];
		bn_wt[c] += -lr_sw * g_bn_wt[c];
	}
}

void FracNet_sw(float image_sw[3][32][32]) {

	//===========================================================
	//=======================	Forward	  =======================
	//===========================================================

	////////////////////////////////////
	//////////// layer1_Conv1  /////////

	conv_3x3_sw<3, 32, 32, 32, 1>(image_sw, layer1_conv1_weight, layer1_conv1_out);
	bn_relu_sw<32, 32>(layer1_conv1_out, layer1_bn_relu1_out, layer1_relu1_mask,  layer1_bn_relu1_bn_wt, layer1_bn_relu1_bn_bias);

	////////////////////////////////////
	//////////// layer1_Conv2  /////////

	conv_3x3_sw<32, 32, 32, 32, 1>(layer1_bn_relu1_out, layer1_conv2_weight, layer1_conv2_out);
	bn_relu_sw<32, 32>(layer1_conv2_out, layer1_bn_relu2_out, layer1_relu2_mask,  layer1_bn_relu2_bn_wt, layer1_bn_relu2_bn_bias);

	////////////////////////////////////
	//////////// layer1_ConvSC  ////////

	conv_3x3_sw<32, 64, 32, 16, 2>(layer1_bn_relu2_out, layer1_convSC_weight, layer1_convSC_out);
	bn_relu_sw<64, 16>(layer1_convSC_out, layer1_bn_reluSC_out, layer1_reluSC_mask,  layer1_bn_reluSC_bn_wt, layer1_bn_reluSC_bn_bias);


	////////////////////////////////////
	//////////// layer2_Conv1  /////////

	conv_3x3_sw<64, 64, 16, 16, 1>(layer1_bn_reluSC_out, layer2_conv1_weight, layer2_conv1_out);
	bn_relu_sw<64, 16>(layer2_conv1_out, layer2_bn_relu1_out, layer2_relu1_mask,  layer2_bn_relu1_bn_wt, layer2_bn_relu1_bn_bias);

	////////////////////////////////////
	//////////// layer2_Conv2  /////////

	conv_3x3_sw<64, 64, 16, 16, 1>(layer2_bn_relu1_out, layer2_conv2_weight, layer2_conv2_out);
	bn_relu_sw<64, 16>(layer2_conv2_out, layer2_bn_relu2_out, layer2_relu2_mask,  layer2_bn_relu2_bn_wt, layer2_bn_relu2_bn_bias);

	////////////////////////////////////
	//////////// layer2_ConvSC  ////////

	conv_3x3_sw<64, 64, 16, 8, 2>(layer2_bn_relu2_out, layer2_convSC_weight, layer2_convSC_out);
	bn_relu_sw<64, 8>(layer2_convSC_out, layer2_bn_reluSC_out, layer2_reluSC_mask,  layer2_bn_reluSC_bn_wt, layer2_bn_reluSC_bn_bias);

	////////////////////////////////////
	//////////// avgpool & FC //////////

	avgpool_sw(layer2_bn_reluSC_out, avg_pool_out_sw, 0);
	FC_sw(avg_pool_out_sw, avg_pool_out_sw, linear_weight_sw, linear_out_buf_sw, 0);

	////////////////////////////////////
	//////////// CrossEntropy loss_hw /////

	softmax_sum_approx:
	sum_sw = 0;
	for (int j = 0; j < 10; j ++) {
		sum_sw += 1 + linear_out_buf_sw[j] + 0.5*(linear_out_buf_sw[j])*(linear_out_buf_sw[j]) + 0.1667*(linear_out_buf_sw[j])*(linear_out_buf_sw[j])*(linear_out_buf_sw[j]);
	}

	CE_loss:
	loss_sw = 0;
	for (int j = 0; j < 10; j ++) {
		loss_sw += -log2((1 + linear_out_buf_sw[j] + 0.5*(linear_out_buf_sw[j])*(linear_out_buf_sw[j]) + 0.1667*(linear_out_buf_sw[j])*(linear_out_buf_sw[j])*(linear_out_buf_sw[j]))/sum_sw) * (labels_sw[j]);
	}

	CE_error:
	for (int j = 0; j < 10; j ++) {
		error_sw[j] = (1 + linear_out_buf_sw[j] + 0.5*(linear_out_buf_sw[j])*(linear_out_buf_sw[j]) + 0.1667*(linear_out_buf_sw[j])*(linear_out_buf_sw[j])*(linear_out_buf_sw[j]))/sum_sw - labels_sw[j];
	}

	//============================================================
	//=======================	Backward   =======================
	//============================================================

	rot_3x3<3, 32>(layer1_conv1_weight,  layer1_conv1_weight_rot);
	rot_3x3<32, 32>(layer1_conv2_weight,  layer1_conv2_weight_rot);
	rot_3x3<32, 64>(layer1_convSC_weight, layer1_convSC_weight_rot);

	rot_3x3<64, 64>(layer2_conv1_weight,  layer2_conv1_weight_rot);
	rot_3x3<64, 64>(layer2_conv2_weight,  layer2_conv2_weight_rot);
	rot_3x3<64, 64>(layer2_convSC_weight, layer2_convSC_weight_rot);

	////////////////////////////////////
	//////////// avgpool & FC //////////

	FC_sw(fc_bp_out, avg_pool_out_sw, linear_weight_sw, error_sw, 1);
	avgpool_sw(avg_pool_bp_out, fc_bp_out, 1);

	////////////////////////////////////
	//////////// layer2_ConvSC  ////////

	bn_relu_bp_sw<64, 8>(avg_pool_bp_out, layer2_convSC_out, layer2_bn_reluSC_bp_out, layer2_reluSC_mask,  layer2_bn_reluSC_bn_wt, layer2_bn_reluSC_bn_bias);
	deconv_3x3_sw<64, 64, 8, 16, 2>(layer2_bn_reluSC_bp_out, layer2_convSC_weight_rot, layer2_convSC_bp_out);
	conv_3x3_grad_sw<64, 64, 16, 2>(layer2_bn_reluSC_bp_out, layer2_bn_relu2_out, layer2_convSC_weight);

	////////////////////////////////////
	//////////// layer2_Conv2  /////////

	bn_relu_bp_sw<64, 16>(layer2_convSC_bp_out, layer2_conv2_out, layer2_bn_relu2_bp_out, layer2_relu2_mask,  layer2_bn_relu2_bn_wt, layer2_bn_relu2_bn_bias);
	deconv_3x3_sw<64, 64, 16, 16, 1>(layer2_bn_relu2_bp_out, layer2_conv2_weight_rot, layer2_conv2_bp_out);
	conv_3x3_grad_sw<64, 64, 16, 1>(layer2_bn_relu2_bp_out, layer2_bn_relu1_out, layer2_conv2_weight);

	////////////////////////////////////
	//////////// layer2_Conv1  /////////

	bn_relu_bp_sw<64, 16>(layer2_conv2_bp_out, layer2_conv1_out, layer2_bn_relu1_bp_out, layer2_relu1_mask,  layer2_bn_relu1_bn_wt, layer2_bn_relu1_bn_bias);
	deconv_3x3_sw<64, 64, 16, 16, 1>(layer2_bn_relu1_bp_out, layer2_conv1_weight_rot, layer2_conv1_bp_out);
	conv_3x3_grad_sw<64, 64, 16, 1>(layer2_bn_relu1_bp_out, layer1_bn_reluSC_out, layer2_conv1_weight);


	////////////////////////////////////
	//////////// layer1_ConvSC  ////////

	bn_relu_bp_sw<64, 16>(layer2_conv1_bp_out, layer1_convSC_out, layer1_bn_reluSC_bp_out, layer1_reluSC_mask,  layer1_bn_reluSC_bn_wt, layer1_bn_reluSC_bn_bias);
	deconv_3x3_sw<64, 32, 16, 32, 2>(layer1_bn_reluSC_bp_out, layer1_convSC_weight_rot, layer1_convSC_bp_out);
	conv_3x3_grad_sw<32, 64, 32, 2>(layer1_bn_reluSC_bp_out, layer1_bn_relu2_out, layer1_convSC_weight);

	////////////////////////////////////
	//////////// layer1_Conv2  /////////

	bn_relu_bp_sw<32, 32>(layer1_convSC_bp_out, layer1_conv2_out, layer1_bn_relu2_bp_out, layer1_relu2_mask,  layer1_bn_relu2_bn_wt, layer1_bn_relu2_bn_bias);
	deconv_3x3_sw<32, 32, 32, 32, 1>(layer1_bn_relu2_bp_out, layer1_conv2_weight_rot, layer1_conv2_bp_out);
	conv_3x3_grad_sw<32, 32, 32, 1>(layer1_bn_relu2_bp_out, layer1_bn_relu1_out, layer1_conv2_weight);

	////////////////////////////////////
	//////////// layer1_Conv1  /////////

	bn_relu_bp_sw<32, 32>(layer1_conv2_bp_out, layer1_conv1_out, layer1_bn_relu1_bp_out, layer1_relu1_mask,  layer1_bn_relu1_bn_wt, layer1_bn_relu1_bn_bias);
	deconv_3x3_sw<32, 3, 32, 32, 1>(layer1_bn_relu1_bp_out, layer1_conv1_weight_rot, layer1_conv1_bp_out);
	conv_3x3_grad_sw<3, 32, 32, 1>(layer1_bn_relu1_bp_out, image_sw, layer1_conv1_weight);
}

void FracNet_hw(float image_hw[3][32][32]) {

	int H_fmap_in_hw, H_fmap_out_hw, in_channels_hw, out_channels_hw, in_channels_after_pack_hw, out_channels_after_pack_hw;
    int stride_hw, conv_3x3_weight_ptr_hw, conv_1x1_weight_ptr_hw, ini_hw, ini_copy_hw, ini_act_bias_hw, ini_wt_bias_hw;
	uint1 ctrl_conv_hw, ctrl_sc_hw, ctrl_fc_hw, ctrl_avgpool_hw;

	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////		Forward path		//////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	ini_hw = 0;
	ini_act_bias_hw = 0;

	ctrl_conv_hw = 0;

	LOOP_GetImg:
	for (int row = 0; row < 32; row ++) {
		for (int col = 0; col < 32; col ++) {
			for (int c = 0; c < 3; c ++) {
				msb_fmap_tile_buffer_s2_hw[0][c][row][col] = image_hw[c][row][col];
				out_buf_t1_hw[ini_hw][c][row][col] = image_hw[c][row][col];
			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	in_channels_hw = 3;
	out_channels_hw = 32;
	H_fmap_in_hw = 32;
	H_fmap_out_hw = 32;
	stride_hw = 1;

	in_channels_after_pack_hw = 1;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	conv_3x3_weight_ptr_hw = 0;

	////////////////////////////////////////////////
	//////////// layer1 PG1 ////////////////////////

	LOOP_layer1_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 1
		conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);
	}

	in_channels_hw = 32;
	out_channels_hw = 32;
	H_fmap_in_hw = 32;
	H_fmap_out_hw = 32;
	stride_hw = 1;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer1 PG2 ////////////////////////

	LOOP_layer1_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 2
		conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 Downsample ////////////////
	////////////////////////////////////////////////

	in_channels_hw = 32;
	out_channels_hw = 64;
	H_fmap_in_hw = 32;
	H_fmap_out_hw = 16;
	stride_hw = 2;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer1 shortcut ///////////////////

	LOOP_layer1_ConvSC:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 3
		conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_1_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	in_channels_hw = 64;
	out_channels_hw = 64;
	H_fmap_in_hw = 16;
	H_fmap_out_hw = 16;
	stride_hw = 1;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer2 PG1 ////////////////////////

	LOOP_layer2_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 4
		conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_s2_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer2 PG2 ////////////////////////

	LOOP_layer2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 5
		conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
		}
		generic_bn_relu_hw(	// wrong output (beyond H_fmap_out_hw)
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Downsample ////////////////
	////////////////////////////////////////////////

	in_channels_hw = 64;
	out_channels_hw = 64;
	H_fmap_in_hw = 16;
	H_fmap_out_hw = 8;
	stride_hw = 2;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer2 shortcut ///////////////////

	LOOP_layer2_ConvSC:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 6
		conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_1_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);
	}

	////////////////////////////////////////////////
	//////////// AvgPool and FC ////////////////////

	// avgpool
	ctrl_avgpool_hw = 0;
	LOOP_AvgPool:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		generic_avgpool_hw(
			msb_fmap_tile_buffer_0_hw[c_out], pool_out_buf_hw, ctrl_avgpool_hw, c_out, pool_out_buf_copy_hw, bias_gap4add_hw
		);
	}

	// FC
	ctrl_fc_hw = 0;
	LOOP_FC:
	generic_FC_hw(
		pool_out_buf_hw, pool_out_buf_copy_hw, linear_weight_tile_buffer_hw, linear_out_buf_hw, ctrl_fc_hw
	);

	////////////////////////////////////////////////
	/////////////////   FC   ///////////////////////

	sum_hw = 0;
	softmax_sum_approx:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		sum_hw += 1 + linear_out_buf_hw[j] + 0.5*(linear_out_buf_hw[j])*(linear_out_buf_hw[j]) + 0.1667*(linear_out_buf_hw[j])*(linear_out_buf_hw[j])*(linear_out_buf_hw[j]);
	}
//	printf("sum_hw: %f \n", sum_hw);

	loss_hw = 0;
	CE_loss:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		loss_hw += -labels_hw[j] * hls::log2((1 + linear_out_buf_hw[j] + 0.5*(linear_out_buf_hw[j])*(linear_out_buf_hw[j]) + 0.1667*(linear_out_buf_hw[j])*(linear_out_buf_hw[j])*(linear_out_buf_hw[j]))/sum_hw);
	}

	CE_error:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		error_hw[j] = (1 + linear_out_buf_hw[j] + 0.5*(linear_out_buf_hw[j])*(linear_out_buf_hw[j]) + 0.1667*(linear_out_buf_hw[j])*(linear_out_buf_hw[j])*(linear_out_buf_hw[j]))/sum_hw - labels_hw[j];
	}


//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// AvgPool and FC ////////////////////

	// FC_bp
	ctrl_fc_hw = 1;
	LOOP_FC_bp:
	generic_FC_hw(
		pool_out_buf_hw, pool_out_buf_copy_hw, linear_weight_tile_buffer_hw, error_hw, ctrl_fc_hw
	);

	// avgpool_bp
	ctrl_avgpool_hw = 1;
	LOOP_AvgPool_bp:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		generic_avgpool_hw(
			msb_fmap_tile_buffer_1_hw[c_out], pool_out_buf_hw, ctrl_avgpool_hw, c_out, pool_out_buf_copy_hw, bias_gap4add_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Upsample //////////////////
	////////////////////////////////////////////////

	in_channels_hw = 64;
	out_channels_hw = 64;
	H_fmap_in_hw = 8;
	H_fmap_out_hw = 16;
	stride_hw = 2;

	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	ctrl_conv_hw = 1;

	////////////////////////////////////////////////
	//////////// layer2 shortcut ///////////////////

	LOOP_layer2_ConvSC_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 6
		conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	in_channels_hw = 64;
	out_channels_hw = 64;
	H_fmap_in_hw = 16;
	H_fmap_out_hw = 16;
	stride_hw = 1;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer2 PG2 ////////////////////////

	LOOP_layer2_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 5
		conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(	// MODIFIED HERE
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer2 PG1 ////////////////////////

	LOOP_layer2_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 4
		conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_out <= out_channels_after_pack_hw/2 - 1) {
				generic_conv_3x3_grad_hw(
					out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 Upsample //////////////////
	////////////////////////////////////////////////

	in_channels_hw = 64;
	out_channels_hw = 32;
	H_fmap_in_hw = 16;
	H_fmap_out_hw = 32;
	stride_hw = 2;

	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer1 shortcut ///////////////////

	LOOP_layer1_ConvSC_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 3
		conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_in <= in_channels_after_pack_hw/2 - 1) {
				generic_conv_3x3_grad_hw(
					out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	in_channels_hw = 32;
	out_channels_hw = 32;
	H_fmap_in_hw = 32;
	H_fmap_out_hw = 32;
	stride_hw = 1;

	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer1 PG2 ////////////////////////

	LOOP_layer1_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 2
		conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(	// MODIFIED HERE
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_out <= 0) {
				generic_conv_3x3_grad_hw(
					out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	in_channels_hw = 32;
	out_channels_hw = 3;
	H_fmap_in_hw = 32;
	H_fmap_out_hw = 32;
	stride_hw = 1;

	out_channels_after_pack_hw = 1;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer1 PG1 ////////////////////////

	LOOP_layer1_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 1
		conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw, bias_gap4add_hw
		);
		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw, bias_gap4add_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_in <= 0) {
				generic_conv_3x3_grad_hw(
					out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}
}


int main(int argc, char **argv)
{

	// run k epochs
	for (int k = 0; k < 50; k ++) {

		cout << "---------------" << " iteration " << k+1 << " ---------------" << endl;

		////////////////////////////////
		//////// SOFTWARE //////////////
		////////////////////////////////

		float image_sw[3][32][32];

		for(int j = 0; j < 3; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					image_sw[j][row][col] = 80.0;
				}
			}
		}

		FracNet_sw(image_sw);

		cout << "loss_sw: " << loss_sw << endl;
		for (int i = 0; i < 10; i ++) {
			cout << "error_sw: " << error_sw[i] << "  " << endl;
		}

/*
		float layer1_conv1_out_max = 0;
		for(int j = 0; j < 32; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					if (abs(layer1_conv1_out_max) < abs(layer1_bn_relu1_out[j][row][col])) {
						layer1_conv1_out_max = layer1_bn_relu1_out[j][row][col];
//						printf("layer1_bn_relu1_out[%d][%d][%d]: %f \n", j, row, col, layer1_bn_relu1_out[j][row][col]);
					}
				}
			}
		}
		cout << "layer1_conv1_out_max: " << layer1_conv1_out_max << endl;

		float layer1_conv2_out_max = 0;
		for(int j = 0; j < 32; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					if (abs(layer1_conv2_out_max) < abs(layer1_bn_relu2_out[j][row][col])) {
						layer1_conv2_out_max = layer1_bn_relu2_out[j][row][col];
					}
				}
			}
		}
		cout << "layer1_conv2_out_max: " << layer1_conv2_out_max << endl;

		float layer1_convSC_out_max = 0;
		for(int j = 0; j < 64; j ++){
			for(int row = 0; row < 16; row ++){
				for(int col = 0; col < 16; col ++){
					if (abs(layer1_convSC_out_max) < abs(layer1_bn_reluSC_out[j][row][col])) {
						layer1_convSC_out_max = layer1_bn_reluSC_out[j][row][col];
					}
				}
			}
		}
		cout << "layer1_convSC_out_max: " << layer1_convSC_out_max << endl;


		float layer2_conv1_out_max = 0;
		for(int j = 0; j < 64; j ++){
			for(int row = 0; row < 16; row ++){
				for(int col = 0; col < 16; col ++){
					if (abs(layer2_conv1_out_max) < abs(layer2_bn_relu1_out[j][row][col])) {
						layer2_conv1_out_max = layer2_bn_relu1_out[j][row][col];
					}
				}
			}
		}
		cout << "layer2_conv1_out_max: " << layer2_conv1_out_max << endl;

		float layer2_conv2_out_max = 0;
		for(int j = 0; j < 64; j ++){
			for(int row = 0; row < 16; row ++){
				for(int col = 0; col < 16; col ++){
					if (abs(layer2_conv2_out_max) < abs(layer2_bn_relu2_out[j][row][col])) {
						layer2_conv2_out_max = layer2_bn_relu2_out[j][row][col];
					}
				}
			}
		}
		cout << "layer2_conv2_out_max: " << layer2_conv2_out_max << endl;

		float layer2_convSC_out_max = 0;
		for(int j = 0; j < 64; j ++){
			for(int row = 0; row < 8; row ++){
				for(int col = 0; col < 8; col ++){
					if (abs(layer2_convSC_out_max) < abs(layer2_bn_reluSC_out[j][row][col])) {
						layer2_convSC_out_max = layer2_bn_reluSC_out[j][row][col];
					}
				}
			}
		}
		cout << "layer2_convSC_out_max: " << layer2_convSC_out_max << endl;

		// =============================================================   backward  =============================================================

		float layer2_convSC_bp_out_max = 0;
		for(int j = 0; j < 64; j ++){
			for(int row = 0; row < 16; row ++){
				for(int col = 0; col < 16; col ++){
					if (abs(layer2_convSC_bp_out_max) < abs(layer2_convSC_bp_out[j][row][col])) {
						layer2_convSC_bp_out_max = layer2_convSC_bp_out[j][row][col];
					}
				}
			}
		}
		cout << "layer2_convSC_bp_out_max: " << layer2_convSC_bp_out_max << endl;

		float layer2_conv2_bp_out_max = 0;
		for(int j = 0; j < 64; j ++){
			for(int row = 0; row < 16; row ++){
				for(int col = 0; col < 16; col ++){
					if (abs(layer2_conv2_bp_out_max) < abs(layer2_conv2_bp_out[j][row][col])) {
						layer2_conv2_bp_out_max = layer2_conv2_bp_out[j][row][col];
					}
				}
			}
		}
		cout << "layer2_conv2_bp_out_max: " << layer2_conv2_bp_out_max << endl;

		float layer2_conv1_bp_out_max = 0;
		for(int j = 0; j < 64; j ++){
			for(int row = 0; row < 16; row ++){
				for(int col = 0; col < 16; col ++){
					if (abs(layer2_conv1_bp_out_max) < abs(layer2_conv1_bp_out[j][row][col])) {
						layer2_conv1_bp_out_max = layer2_conv1_bp_out[j][row][col];
					}
				}
			}
		}
		cout << "layer2_conv1_bp_out_max: " << layer2_conv1_bp_out_max << endl;


		float layer1_convSC_bp_out_max = 0;
		for(int j = 0; j < 32; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					if (abs(layer1_convSC_bp_out_max) < abs(layer1_convSC_bp_out[j][row][col])) {
						layer1_convSC_bp_out_max = layer1_convSC_bp_out[j][row][col];
					}
				}
			}
		}
		cout << "layer1_convSC_bp_out_max: " << layer1_convSC_bp_out_max << endl;

		float layer1_conv2_bp_out_max = 0;
		for(int j = 0; j < 32; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					if (abs(layer1_conv2_bp_out_max) < abs(layer1_conv2_bp_out[j][row][col])) {
						layer1_conv2_bp_out_max = layer1_conv2_bp_out[j][row][col];
					}
				}
			}
		}
		cout << "layer1_conv2_bp_out_max: " << layer1_conv2_bp_out_max << endl;

		float layer1_conv1_bp_out_max = 0;
		for(int j = 0; j < 3; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					if (abs(layer1_conv1_bp_out_max) < abs(layer1_conv1_bp_out[j][row][col])) {
						layer1_conv1_bp_out_max = layer1_conv1_bp_out[j][row][col];
					}
				}
			}
		}
		cout << "layer1_conv1_bp_out_max: " << layer1_conv1_bp_out_max << endl;

*/

		cout << " "<< endl;

		////////////////////////////////
		//////// HARDWARE //////////////
		////////////////////////////////

		FracNet_hw(image_sw);

		cout << "loss_hw: " << loss_hw << endl;
		for (int i = 0; i < 10; i ++) {
			cout << "error_hw: " << error_hw[i] << "  " << endl;
		}

		cout << " "<< endl;

		////////////////////////////////
		//////// bm(2, 5) //////////////
		////////////////////////////////

		float error_bm[10];
		float loss_bm;

		FracNet_T(image_sw, loss_bm, error_bm);

		cout << "loss_bm: " << loss_bm << endl;
		for (int i = 0; i < 10; i ++) {
			cout << "error_bm: " << error_bm[i] << "  " << endl;
		}

		cout << " "<< endl;
	}
	return 0;
}

