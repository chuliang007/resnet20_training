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

#define EPOCH 90
unsigned char images[EPOCH*3*32*32];
unsigned char labels[EPOCH];

const uint2 exp_bias = 2;
float lr_sw = 0.005;
float eps_sw = 1e-10;
float mom_sw = 0.9;

#define CHANNEL_IN_T 8
#define CHANNEL_OUT_T 8
#define WIDTH 33

//--------------------------------
// floating-point golden reference
//--------------------------------

//float image_sw[3][32][32]];
float sum_sw = 0;
float loss_sw = 0;
float error_sw[10];

///////////////////////////
////////  forward  ////////

// conv 1
//float conv1_weight[16][3][3][3];
float conv1_weight_rot[3][16][3][3];
float conv1_out[16][32][32];
float bn_relu_1_out[16][32][32];
uint1 conv1_relu_mask[16][32][32];
//float conv1_bn_wt[16];
//float conv1_bn_bias[16];

// layer1_0
//float layer1_0_conv1_weight[16][16][3][3];
float layer1_0_conv1_weight_rot[16][16][3][3];
float layer1_0_conv1_out[16][32][32];
float layer1_0_bn_relu1_out[16][32][32];
uint1 layer1_0_relu1_mask[16][32][32];
//float layer1_0_bn_relu1_bn_wt[16];
//float layer1_0_bn_relu1_bn_bias[16];

//float layer1_0_conv2_weight[16][16][3][3];
float layer1_0_conv2_weight_rot[16][16][3][3];
float layer1_0_conv2_out[16][32][32];
float layer1_0_bn_relu2_out[16][32][32];
uint1 layer1_0_relu2_mask[16][32][32];
//float layer1_0_bn_relu2_bn_wt[16];
//float layer1_0_bn_relu2_bn_bias[16];

float layer1_0_sc_out[16][32][32];

// layer1_1
//float layer1_1_conv1_weight[16][16][3][3];
float layer1_1_conv1_weight_rot[16][16][3][3];
float layer1_1_conv1_out[16][32][32];
float layer1_1_bn_relu1_out[16][32][32];
uint1 layer1_1_relu1_mask[16][32][32];
//float layer1_1_bn_relu1_bn_wt[16];
//float layer1_1_bn_relu1_bn_bias[16];

//float layer1_1_conv2_weight[16][16][3][3];
float layer1_1_conv2_weight_rot[16][16][3][3];
float layer1_1_conv2_out[16][32][32];
float layer1_1_bn_relu2_out[16][32][32];
uint1 layer1_1_relu2_mask[16][32][32];
//float layer1_1_bn_relu2_bn_wt[16];
//float layer1_1_bn_relu2_bn_bias[16];

float layer1_1_sc_out[16][32][32];

// layer1_2
//float layer1_2_conv1_weight[16][16][3][3];
float layer1_2_conv1_weight_rot[16][16][3][3];
float layer1_2_conv1_out[16][32][32];
float layer1_2_bn_relu1_out[16][32][32];
uint1 layer1_2_relu1_mask[16][32][32];
//float layer1_2_bn_relu1_bn_wt[16];
//float layer1_2_bn_relu1_bn_bias[16];

//float layer1_2_conv2_weight[16][16][3][3];
float layer1_2_conv2_weight_rot[16][16][3][3] ;
float layer1_2_conv2_out[16][32][32];
float layer1_2_bn_relu2_out[16][32][32];
uint1 layer1_2_relu2_mask[16][32][32];
//float layer1_2_bn_relu2_bn_wt[16];
//float layer1_2_bn_relu2_bn_bias[16];

float layer1_2_sc_out[16][32][32];

// layer2_sc
//float layer2_sc_conv_weight[32][16];
float layer2_sc_conv_weight_rot[16][32];
float layer2_sc_conv_out[32][16][16];
float layer2_sc_bn_out[32][16][16];
//float layer2_sc_bn_wt[32];
//float layer2_sc_bn_bias[32];
// layer2_0
//float layer2_0_conv1_weight[32][16][3][3];
float layer2_0_conv1_weight_rot[16][32][3][3];
float layer2_0_conv1_out[32][16][16];
float layer2_0_bn_relu1_out[32][16][16];
uint1 layer2_0_relu1_mask[32][16][16];
//float layer2_0_bn_relu1_bn_wt[32];
//float layer2_0_bn_relu1_bn_bias[32];

//float layer2_0_conv2_weight[32][32][3][3];
float layer2_0_conv2_weight_rot[32][32][3][3];
float layer2_0_conv2_out[32][16][16];
float layer2_0_bn_relu2_out[32][16][16];
uint1 layer2_0_relu2_mask[32][16][16];
//float layer2_0_bn_relu2_bn_wt[32];
//float layer2_0_bn_relu2_bn_bias[32];

float layer2_0_sc_out[32][16][16];

// layer2_1
//float layer2_1_conv1_weight[32][32][3][3];
float layer2_1_conv1_weight_rot[32][32][3][3];
float layer2_1_conv1_out[32][16][16];
float layer2_1_bn_relu1_out[32][16][16];
uint1 layer2_1_relu1_mask[32][16][16];
//float layer2_1_bn_relu1_bn_wt[32];
//float layer2_1_bn_relu1_bn_bias[32];

//float layer2_1_conv2_weight[32][32][3][3];
float layer2_1_conv2_weight_rot[32][32][3][3];
float layer2_1_conv2_out[32][16][16];
float layer2_1_bn_relu2_out[32][16][16];
uint1 layer2_1_relu2_mask[32][16][16];
//float layer2_1_bn_relu2_bn_wt[32];
//float layer2_1_bn_relu2_bn_bias[32];

float layer2_1_sc_out[32][16][16];

// layer2_2
//float layer2_2_conv1_weight[32][32][3][3];
float layer2_2_conv1_weight_rot[32][32][3][3];
float layer2_2_conv1_out[32][16][16];
float layer2_2_bn_relu1_out[32][16][16];
uint1 layer2_2_relu1_mask[32][16][16];
//float layer2_2_bn_relu1_bn_wt[32];
//float layer2_2_bn_relu1_bn_bias[32];

//float layer2_2_conv2_weight[32][32][3][3];
float layer2_2_conv2_weight_rot[32][32][3][3];
float layer2_2_conv2_out[32][16][16];
float layer2_2_bn_relu2_out[32][16][16];
uint1 layer2_2_relu2_mask[32][16][16];
//float layer2_2_bn_relu2_bn_wt[32];
//float layer2_2_bn_relu2_bn_bias[32];

float layer2_2_sc_out[32][16][16];

// layer3_sc
//float layer3_sc_conv_weight[64][32];
float layer3_sc_conv_weight_rot[32][64];
float layer3_sc_conv_out[64][8][8];
float layer3_sc_bn_out[64][8][8];
//float layer3_sc_bn_wt[64];
//float layer3_sc_bn_bias[64];
// layer3_0
//float layer3_0_conv1_weight[64][32][3][3];
float layer3_0_conv1_weight_rot[32][64][3][3];
float layer3_0_conv1_out[64][8][8];
float layer3_0_bn_relu1_out[64][8][8];
uint1 layer3_0_relu1_mask[64][8][8];
//float layer3_0_bn_relu1_bn_wt[64];
//float layer3_0_bn_relu1_bn_bias[64];

//float layer3_0_conv2_weight[64][64][3][3];
float layer3_0_conv2_weight_rot[64][64][3][3];
float layer3_0_conv2_out[64][8][8];
float layer3_0_bn_relu2_out[64][8][8];
uint1 layer3_0_relu2_mask[64][8][8];
//float layer3_0_bn_relu2_bn_wt[64];
//float layer3_0_bn_relu2_bn_bias[64];

float layer3_0_sc_out[64][8][8];

// layer3_1
//float layer3_1_conv1_weight[64][64][3][3];
float layer3_1_conv1_weight_rot[64][64][3][3];
float layer3_1_conv1_out[64][8][8];
float layer3_1_bn_relu1_out[64][8][8];
uint1 layer3_1_relu1_mask[64][8][8];
//float layer3_1_bn_relu1_bn_wt[64];
//float layer3_1_bn_relu1_bn_bias[64];

//float layer3_1_conv2_weight[64][64][3][3];
float layer3_1_conv2_weight_rot[64][64][3][3];
float layer3_1_conv2_out[64][8][8];
float layer3_1_bn_relu2_out[64][8][8];
uint1 layer3_1_relu2_mask[64][8][8];
//float layer3_1_bn_relu2_bn_wt[64];
//float layer3_1_bn_relu2_bn_bias[64];

float layer3_1_sc_out[64][8][8];

// layer3_2
//float layer3_2_conv1_weight[64][64][3][3];
float layer3_2_conv1_weight_rot[64][64][3][3];
float layer3_2_conv1_out[64][8][8];
float layer3_2_bn_relu1_out[64][8][8];
uint1 layer3_2_relu1_mask[64][8][8];
//float layer3_2_bn_relu1_bn_wt[64];
//float layer3_2_bn_relu1_bn_bias[64];

//float layer3_2_conv2_weight[64][64][3][3];
float layer3_2_conv2_weight_rot[64][64][3][3];
float layer3_2_conv2_out[64][8][8];
float layer3_2_bn_relu2_out[64][8][8];
uint1 layer3_2_relu2_mask[64][8][8];
//float layer3_2_bn_relu2_bn_wt[64];
//float layer3_2_bn_relu2_bn_bias[64];

float layer3_2_sc_out[64][8][8];

// avgpool_sw & FC_sw
float avg_pool_out_sw[64];
//float linear_weight[10][64];
float linear_out_buf_sw[10];

///////////////////////////
////////  backward  ///////

// avgpool_sw & FC_sw
float fc_bp_out[64];
float avg_pool_bp_out[64][8][8];

// layer3_2
float layer3_2_bn_relu2_bp_out[64][8][8];
float layer3_2_conv2_bp_out[64][8][8];

float layer3_2_bn_relu1_bp_out[64][8][8];
float layer3_2_conv1_bp_out[64][8][8];

float layer3_2_sc_bp_out[64][8][8];

// layer3_1
float layer3_1_bn_relu2_bp_out[64][8][8];
float layer3_1_conv2_bp_out[64][8][8];

float layer3_1_bn_relu1_bp_out[64][8][8];
float layer3_1_conv1_bp_out[64][8][8];

float layer3_1_sc_bp_out[64][8][8];

// layer3_0
float layer3_0_bn_relu2_bp_out[64][8][8];
float layer3_0_conv2_bp_out[64][8][8];

float layer3_0_bn_relu1_bp_out[64][8][8];
float layer3_0_conv1_bp_out[32][16][16];
// layer3_sc
float layer3_sc_bn_bp_out[64][8][8];
float layer3_sc_conv_bp_out[32][16][16];

float layer3_0_sc_bp_out[32][16][16];

// layer2_2
float layer2_2_bn_relu2_bp_out[32][16][16];
float layer2_2_conv2_bp_out[32][16][16];

float layer2_2_bn_relu1_bp_out[32][16][16];
float layer2_2_conv1_bp_out[32][16][16];

float layer2_2_sc_bp_out[32][16][16];

// layer2_1
float layer2_1_bn_relu2_bp_out[32][16][16];
float layer2_1_conv2_bp_out[32][16][16];

float layer2_1_bn_relu1_bp_out[32][16][16];
float layer2_1_conv1_bp_out[32][16][16];

float layer2_1_sc_bp_out[32][16][16];

// layer2_0
float layer2_0_bn_relu2_bp_out[32][16][16];
float layer2_0_conv2_bp_out[32][16][16];

float layer2_0_bn_relu1_bp_out[32][16][16];
float layer2_0_conv1_bp_out[16][32][32];
// layer2_sc
float layer2_sc_bn_bp_out[32][16][16];
float layer2_sc_conv_bp_out[16][32][32];

float layer2_0_sc_bp_out[16][32][32];

// layer1_2
float layer1_2_bn_relu2_bp_out[16][32][32];
float layer1_2_conv2_bp_out[16][32][32];

float layer1_2_bn_relu1_bp_out[16][32][32];
float layer1_2_conv1_bp_out[16][32][32];

float layer1_2_sc_bp_out[16][32][32];

// layer1_1
float layer1_1_bn_relu2_bp_out[16][32][32];
float layer1_1_conv2_bp_out[16][32][32];

float layer1_1_bn_relu1_bp_out[16][32][32];
float layer1_1_conv1_bp_out[16][32][32];

float layer1_1_sc_bp_out[16][32][32];

// layer1_0
float layer1_0_bn_relu2_bp_out[16][32][32];
float layer1_0_conv2_bp_out[16][32][32];

float layer1_0_bn_relu1_bp_out[16][32][32];
float layer1_0_conv1_bp_out[16][32][32];

float layer1_0_sc_bp_out[16][32][32];

// conv1
float bn_relu_1_bp_out[16][32][32];

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
	float input[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// in on-chip
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
	float input[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// in on-chip
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
	float input[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// in on-chip
	float weight[out_channels_hw][in_channels_hw][3][3],
	float output[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw]		// out on-chip
)
{
	float act;
	float wt;
	float input_dil[in_channels_hw][H_fmap_out_hw][H_fmap_out_hw];

	// buffer init
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

	// buffer init
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
	float output[out_channels_hw][in_channels_hw][3][3],			// out on-chip
	float vel[out_channels_hw][in_channels_hw][3][3]
)
{
	float accum;
	float weight_dil[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw] = {0};

	// buffer init
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
							}
						}
					}
//					output[co][ci][row][col] += -lr_sw * accum;
					vel[co][ci][row][col] = vel[co][ci][row][col]*mom_sw + lr_sw*accum;
					output[co][ci][row][col] -= vel[co][ci][row][col];
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
	float output[out_channels_hw][in_channels_hw],					// out on-chip
	float vel[out_channels_hw][in_channels_hw]
)
{
	float accum;
	float weight_dil[out_channels_hw][H_fmap_out_hw][H_fmap_out_hw] = {0};

	// buffer init
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
					}
				}
			}
//			output[co][ci] += -lr_sw * accum;
		    vel[co][ci] = vel[co][ci]*mom_sw + lr_sw*accum;
		    output[co][ci] -= vel[co][ci];
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
	float mu[in_channels_hw] = {0};
	float std_var[in_channels_hw] = {0};
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
	float out_buf[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],			// out, bn_outputs
	uint1 relu_mask_hw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// out, relu_mask_hw for relu_bp

	float bn_wt_hw[in_channels_hw],
	float bn_bias_hw[in_channels_hw]
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[in_channels_hw] = {0};
	float std_var[in_channels_hw] = {0};
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
	float error_sw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw], 	// in
	float bn_inputs_fw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],	// in
	float out_buf[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// out, error_bn

	float bn_wt_hw[in_channels_hw],									// in
	float bn_bias_hw[in_channels_hw],								// in
	float vel_wt[in_channels_hw],
	float vel_bias[in_channels_hw]
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[in_channels_hw] = {0};
	float std_var[in_channels_hw] = {0};
	float sum_sw[in_channels_hw] = {0};
	float g_bn_wt_hw[in_channels_hw] = {0};							// out
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
//				bn_bias_hw[c] += -lr_sw * g_bn_bias_hw[c];
//				bn_wt_hw[c] += -lr_sw * g_bn_wt_hw[c];
				vel_wt[c] = vel_wt[c]*mom_sw + lr_sw * g_bn_wt_hw[c];
				bn_wt_hw[c] -= vel_wt[c];
				vel_bias[c] = vel_bias[c]*mom_sw + lr_sw * g_bn_bias_hw[c];
				bn_bias_hw[c] -= vel_bias[c];
			}
		}
	}
}

// bn_relu_bp_sw
template <int in_channels_hw, int H_fmap_in_hw>
void bn_relu_bp_sw(
	float error_sw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw], 	// in
	float bn_inputs_fw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],	// in
	float out_buf[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],		// out, error_bn
	uint1 relu_mask_hw[in_channels_hw][H_fmap_in_hw][H_fmap_in_hw],	// in

	float bn_wt_hw[in_channels_hw],									// in
	float bn_bias_hw[in_channels_hw],								// in
	float vel_wt[in_channels_hw],
	float vel_bias[in_channels_hw]
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[in_channels_hw] = {0};
	float std_var[in_channels_hw] = {0};
	float sum_sw[in_channels_hw] = {0};
	float g_bn_wt_hw[in_channels_hw] = {0};							// out
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
//				bn_bias_hw[c] += -lr_sw * g_bn_bias_hw[c];
//				bn_wt_hw[c] += -lr_sw * g_bn_wt_hw[c];
				vel_wt[c] = vel_wt[c]*mom_sw + lr_sw * g_bn_wt_hw[c];
				bn_wt_hw[c] -= vel_wt[c];
				vel_bias[c] = vel_bias[c]*mom_sw + lr_sw * g_bn_bias_hw[c];
				bn_bias_hw[c] -= vel_bias[c];
			}
		}
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
		// buffer init
		for (int c = 0; c < 64; c ++) {
			out_buf[c] = 0;
		}
		for (int c = 0; c < 64; c ++) {
			for (int s = 0; s < 8; s ++) {
				for (int ss = 0; ss < 8; ss ++) {
					 out_buf[c] += avg_inputs[c][s][ss]/64;
				}
			}
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
	float linear_bias[10],
	float outputs[10],

	uint1 ctrl_fc_hw	// 0 for forward and 1 for backward
)
{
	// forward
	if (ctrl_fc_hw == 0) {
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
		// bias update
		for (int coo = 0; coo < 10; coo ++) {
			linear_bias[coo] += -lr_sw * outputs[coo];
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

float msb_fmap_tile_buffer_0_hw[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
float msb_fmap_tile_buffer_1_hw[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// activation/error on-chip
float msb_fmap_tile_buffer_s2_hw[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH]; // activation/error on-chip
float lsb_fmap_tile_buffer_hw[NUM_TILE][CHANNEL_IN_T][WIDTH][WIDTH];	// shortcut activation/error on-chip

float out_buf_t0_hw[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// conv output
float out_buf_t1_hw[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// generic_bn_hw output
uint1 relu_mask_hw[NUM_ACT][CHANNEL_OUT_T][WIDTH][WIDTH];				// relu mask for backprop

float conv_3x3_weight_rot_hw[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
float conv_1x1_weight_rot_hw[CHANNEL_OUT_T][CHANNEL_IN_T];

float pool_out_buf_hw[64];
float pool_out_buf_copy_hw[64];
float linear_out_buf_hw[10];											// generic_FC_hw buffer

float sum_hw;
float loss_hw = 0;
float error_hw[10];
float bias_gap4add_hw[CHANNEL_OUT_T];									// shared exponent gap between act & wt in bm addition

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

	int stride_hw,
	int H_fmap_in_hw,
	int H_fmap_out_hw,
	int c_in,
	uint1 ctrl_conv_hw
)
{
	float act;
	float wt;
	float out_temp[CHANNEL_OUT_T][WIDTH][WIDTH];

	// buffer initiation
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
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
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				float accum = 0;
				for (int krow = 0; krow < 3; krow ++) {
					for (int kcol = 0; kcol < 3; kcol ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
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

	int stride_hw,
	int H_fmap_in_hw,
	int H_fmap_out_hw,
	int c_in,
	uint1 ctrl_conv_hw
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
	for (int row_in = 0; row_in < H_fmap_in_hw; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_in_hw; col_in ++) {
			for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
				input_dil[cin][row_in*stride_hw][col_in*stride_hw] = input[cin][row_in][col_in];
			}
		}
	}

	// buffer initiation
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
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
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				float accum = 0;
				for (int krow = 0; krow < 3; krow ++) {
					for (int kcol = 0; kcol < 3; kcol ++) {
						for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
							int row_in = row + krow - 1;
							int col_in = col + kcol - 1;
							if (row_in >= 0 && row_in < H_fmap_in_hw && col_in >= 0 && col_in < H_fmap_in_hw) {
								act = input_dil[cin][row_in][col_in];
								wt = weight[co][cin][krow][kcol];
								accum += act * wt;
							}
						}
					}
				}
				out_temp[co][row][col] += accum;
				output[co][row][col] = out_temp[co][row][col];
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

	int stride_hw,
	int H_fmap_in_hw,
	int H_fmap_out_hw,
	int c_in,
	uint1 ctrl_conv_hw
)
{
	float act;
	float wt;
	float out_temp[CHANNEL_OUT_T][WIDTH][WIDTH];

	// buffer initiation
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
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
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				float accum = 0;
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					int row_in = row * stride_hw;
					int col_in = row * stride_hw;
					if (row_in >= 0 && row_in < H_fmap_in_hw && col_in >= 0 && col_in < H_fmap_in_hw) {
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

	int stride_hw,
	int H_fmap_in_hw,
	int H_fmap_out_hw,
	int c_in,
	uint1 ctrl_conv_hw
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
	for (int row_in = 0; row_in < H_fmap_in_hw; row_in ++) {
		for (int col_in = 0; col_in < H_fmap_in_hw; col_in ++) {
			for (int cin = 0; cin < CHANNEL_IN_T; cin ++) {
				input_dil[cin][row_in*stride_hw][col_in*stride_hw] = input[cin][row_in][col_in];
			}
		}
	}

	// buffer initiation
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
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
	for (int row = 0; row < H_fmap_out_hw; row ++) {
		for (int col = 0; col < H_fmap_out_hw; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				float accum = 0;
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					int row_in = row;
					int col_in = col;
					if (row_in >= 0 && row_in < H_fmap_in_hw && col_in >= 0 && col_in < H_fmap_in_hw) {
						act = input_dil[ci][row_in][col_in];
						wt = weight[co][ci];
						accum += act * wt;
					}
				}
				out_temp[co][row][col] += accum;

				output[co][row][col] = out_temp[co][row][col];
			}
		}
	}
}

void generic_conv_3x3_grad_hw
(
	float input[CHANNEL_IN_T][WIDTH][WIDTH],				// in on-chip
	float weight[CHANNEL_OUT_T][WIDTH][WIDTH],
	float output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],		// out on-chip

	int stride_hw,
	int H_fmap_in_hw
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
		for (int krow = 0; krow < H_fmap_in_hw; krow ++) {
			for (int kcol = 0; kcol < H_fmap_in_hw; kcol ++) {
				weight_dil[co][krow*stride_hw][kcol*stride_hw] = weight[co][krow][kcol];
			}
		}
	}

	// conv3x3 grad
	for (int row = 0; row < 3; row ++) {
		for (int col = 0; col < 3; col ++) {
			for (int co = 0; co < CHANNEL_OUT_T; co ++) {
				for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
					accum = 0;
					for (int krow = 0; krow < H_fmap_in_hw*stride_hw; krow ++) {
						for (int kcol = 0; kcol < H_fmap_in_hw*stride_hw; kcol ++) {
							int row_in = row + krow - 1;
							int col_in = col + kcol - 1;
							if (row_in >= 0 && row_in < H_fmap_in_hw*stride_hw && col_in >= 0 && col_in < H_fmap_in_hw*stride_hw) {
								float act = input[ci][row_in][col_in];
								float wt = weight_dil[co][krow][kcol];
								accum += act * wt;
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

	int stride_hw,
	int H_fmap_in_hw
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
		for (int krow = 0; krow < H_fmap_in_hw; krow ++) {
			for (int kcol = 0; kcol < H_fmap_in_hw; kcol ++) {
				weight_dil[co][krow*stride_hw][kcol*stride_hw] = weight[co][krow][kcol];
			}
		}
	}

	// conv1x1 grad
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			accum = 0;
			for (int krow = 0; krow < H_fmap_in_hw * stride_hw; krow ++) {
				for (int kcol = 0; kcol < H_fmap_in_hw * stride_hw; kcol ++) {
					int row_in = krow;
					int col_in = kcol;
					if (row_in >= 0 && row_in < H_fmap_in_hw*stride_hw && col_in >= 0 && col_in < H_fmap_in_hw*stride_hw) {
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

	float out_buf_SC[CHANNEL_IN_T][WIDTH][WIDTH],
	float out_buf_copy[64]
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
		}
	}
	// backward
	else {
		for (int s = 0; s < 8; s ++) {
			for (int ss = 0; ss < 8; ss ++) {
				for (int c = 0; c < CHANNEL_IN_T; c ++) {
					avg_inputs[c][s][ss] = out_buf[c + c_out*CHANNEL_IN_T]/64;
					out_buf_SC[c][s][ss] = out_buf[c + c_out*CHANNEL_IN_T]/64;
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
	float linear_bias[10],
	float outputs[10],

	uint1 ctrl_fc_hw	// 0 for forward and 1 for backward
)
{
	float linear_weight_t[64][10];

	// forward
	if (ctrl_fc_hw == 0) {
		// buffer init
		for (int coo = 0; coo < 10; coo ++) {
			outputs[coo] = linear_bias[coo];
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
		for (int coo = 0; coo < 10; coo ++) {
			linear_bias[coo] += -lr_sw * outputs[coo];
		}
	}
}

void generic_shortcut_hw(
	float input_a[CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	float input_b[CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	float out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip
//	float out_buf_SC[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, ideneity for generic_shortcut_hw

	int H_fmap_in_hw,
	uint1 ctrl_sc_hw									// if ctrl_sc_hw=0, generate and send out_copy into DDR

)
{
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {

				out_buf[c][row][col] = input_a[c][row][col] + input_b[c][row][col];
				if (ctrl_sc_hw == 0) {
					out_buf_DDR[c][row][col] = out_buf[c][row][col];
				}

			}
		}
	}
}

// generic_bn_hw
void generic_bn_hw(
	float bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out

	float out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip for backprop
	float bn_wt_hw[CHANNEL_OUT_T],
	float bn_bias_hw[CHANNEL_OUT_T],

    int H_fmap_in_hw
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[CHANNEL_OUT_T];
	float std_var[CHANNEL_OUT_T];
	float sum_hw[CHANNEL_OUT_T] = {0};
	// mean
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				sum_hw[c] += bn_inputs[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] = sum_hw[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (bn_inputs[c][row][col]-mu[c])*(bn_inputs[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// generic_bn_hw
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = bn_wt_hw[c]*(bn_inputs[c][row][col]-mu[c])/(std_var[c] + eps_sw) + bn_bias_hw[c];
				out_buf_DDR[c][row][col] = bn_wt_hw[c]*(bn_inputs[c][row][col]-mu[c])/(std_var[c] + eps_sw) + bn_bias_hw[c];
			}
		}
	}
}

// generic_bn_bp_hw
void generic_bn_bp_hw(
	float error[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	float bn_inputs_fw[CHANNEL_OUT_T][WIDTH][WIDTH],	// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	float bn_wt_hw[CHANNEL_OUT_T],							// in
	float bn_bias_hw[CHANNEL_OUT_T],

	int H_fmap_in_hw
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[CHANNEL_OUT_T];
	float std_var[CHANNEL_OUT_T] = {0};
	float sum_hw[CHANNEL_OUT_T] = {0};
	float g_bn_wt[CHANNEL_OUT_T] = {0};					// out
	float g_bn_bias[CHANNEL_OUT_T] = {0};				// out

	// temp buffer init
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = 0;
		sum_hw[c] = 0;
		g_bn_bias[c] = 0;
		g_bn_wt[c] = 0;
	}

	// mean
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				sum_hw[c] += bn_inputs_fw[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] = sum_hw[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (bn_inputs_fw[c][row][col]-mu[c])*(bn_inputs_fw[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// grad of generic_bn_hw params
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				g_bn_bias[c] += error[c][row][col];
				g_bn_wt[c] += error[c][row][col] * (bn_inputs_fw[c][row][col]-mu[c])/(std_var[c]+eps_sw);
			}
		}
	}

	// bn_bp_sw
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = bn_wt_hw[c]*error[c][row][col]/(std_var[c]+eps_sw) - bn_wt_hw[c]*g_bn_bias[c]/(N*(std_var[c]+eps_sw)) - bn_wt_hw[c]*(bn_inputs_fw[c][row][col]-mu[c])*g_bn_wt[c]/(N*(std_var[c]*std_var[c]+eps_sw));
			}
		}
	}

	// bn_sw params update
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		bn_bias_hw[c] += -lr_sw * g_bn_bias[c];
		bn_wt_hw[c] += -lr_sw * g_bn_wt[c];
	}
}

// generic_bn_relu_hw
void generic_bn_relu_hw(
	float bn_inputs[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs

	float out_buf_DDR[CHANNEL_OUT_T][WIDTH][WIDTH],		// out_copy, off-chip for backprop
	uint1 relu_mask_hw[CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask_hw for relu_bp

	float bn_wt_hw[CHANNEL_OUT_T],
	float bn_bias_hw[CHANNEL_OUT_T],

    int H_fmap_in_hw
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[CHANNEL_OUT_T];
	float std_var[CHANNEL_OUT_T];
	float sum_hw[CHANNEL_OUT_T] = {0};
	// mean
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				sum_hw[c] += bn_inputs[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] = sum_hw[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (bn_inputs[c][row][col]-mu[c])*(bn_inputs[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// generic_bn_relu_hw
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = bn_wt_hw[c]*(bn_inputs[c][row][col]-mu[c])/(std_var[c] + eps_sw) + bn_bias_hw[c];
				relu_mask_hw[c][row][col] = (out_buf[c][row][col] > 0) ? 1 : 0;
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = (relu_mask_hw[c][row][col] == 1) ? out_buf[c][row][col] : float(0);
				out_buf_DDR[c][row][col] = out_buf[c][row][col];
			}
		}
	}
//	if (ctrl_bn_id == 0) {
//		for (int row = 0; row < H_fmap_in_hw; row ++) {
//			for (int col = 0; col < H_fmap_in_hw; col ++) {
//				for (int c = 0; c < CHANNEL_OUT_T; c ++) {
//					out_buf_SC[c][row][col] = out_buf[c][row][col];
//				}
//			}
//		}
//	}
}

void generic_bn_relu_bp_hw(
	float error[CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	float bn_inputs_fw[CHANNEL_OUT_T][WIDTH][WIDTH],	// in
	uint1 relu_mask_hw[CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	float out_buf[CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	float bn_wt_hw[CHANNEL_OUT_T],
	float bn_bias_hw[CHANNEL_OUT_T],

	int H_fmap_in_hw
)
{
	int N = H_fmap_in_hw * H_fmap_in_hw;
	float mu[CHANNEL_OUT_T];
	float std_var[CHANNEL_OUT_T] = {0};
	float sum_hw[CHANNEL_OUT_T] = {0};
	float g_bn_wt[CHANNEL_OUT_T] = {0};					// out
	float g_bn_bias[CHANNEL_OUT_T] = {0};				// out

	// temp buffer init
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = 0;
		sum_hw[c] = 0;
		g_bn_bias[c] = 0;
		g_bn_wt[c] = 0;
	}

	// relu_bp
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				error[c][row][col] = (relu_mask_hw[c][row][col] == 1) ? error[c][row][col] : float(0);
			}
		}
	}

	// mean
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				sum_hw[c] += bn_inputs_fw[c][row][col];
			}
		}
	}
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				mu[c] = sum_hw[c]/N;
			}
		}
	}

	// std_var
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				std_var[c] += (bn_inputs_fw[c][row][col]-mu[c])*(bn_inputs_fw[c][row][col]-mu[c])/N;	// var
			}
		}
	}
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		std_var[c] = hls::sqrt(std_var[c]);
	}

	// grad of generic_bn_hw params
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				g_bn_bias[c] += error[c][row][col];
				g_bn_wt[c] += error[c][row][col] * (bn_inputs_fw[c][row][col]-mu[c])/(std_var[c]+eps_sw);
			}
		}
	}

	// generic_bn_bp_hw
	for (int row = 0; row < H_fmap_in_hw; row ++) {
		for (int col = 0; col < H_fmap_in_hw; col ++) {
			for (int c = 0; c < CHANNEL_OUT_T; c ++) {
				out_buf[c][row][col] = bn_wt_hw[c]*error[c][row][col]/(std_var[c]+eps_sw) - bn_wt_hw[c]*g_bn_bias[c]/(N*(std_var[c]+eps_sw)) - bn_wt_hw[c]*(bn_inputs_fw[c][row][col]-mu[c])*g_bn_wt[c]/(N*(std_var[c]*std_var[c]+eps_sw));
			}
		}
	}

	// bn_sw params update
	for (int c = 0; c < CHANNEL_OUT_T; c ++) {
		bn_bias_hw[c] += -lr_sw * g_bn_bias[c];
		bn_wt_hw[c] += -lr_sw * g_bn_wt[c];
	}
}

void FracNet_sw(float image_sw[3][32][32]) {

	//===========================================================
	//=======================	Forward	  =======================
	//===========================================================

	////////////////////////////////////
	//////////// Conv 1  ///////////////

	conv_3x3_sw<3, 16, 32, 32, 1>(image_sw, conv1_weight, conv1_out);
	bn_relu_sw<16, 32>(conv1_out, bn_relu_1_out, conv1_relu_mask,  conv1_bn_wt, conv1_bn_bias);

	////////////////////////////////////
	//////////// layer1_0  /////////////

	conv_3x3_sw<16, 16, 32, 32, 1>(bn_relu_1_out, layer1_0_conv1_weight, layer1_0_conv1_out);
	bn_relu_sw<16, 32>(layer1_0_conv1_out, layer1_0_bn_relu1_out, layer1_0_relu1_mask,  layer1_0_bn_relu1_bn_wt, layer1_0_bn_relu1_bn_bias);

	conv_3x3_sw<16, 16, 32, 32, 1>(layer1_0_bn_relu1_out, layer1_0_conv2_weight, layer1_0_conv2_out);
	bn_relu_sw<16, 32>(layer1_0_conv2_out, layer1_0_bn_relu2_out, layer1_0_relu2_mask,  layer1_0_bn_relu2_bn_wt, layer1_0_bn_relu2_bn_bias);

	shortcut_sw<16, 32>(bn_relu_1_out, layer1_0_bn_relu2_out, layer1_0_sc_out);

	////////////////////////////////////
	//////////// layer1_1  /////////////

	conv_3x3_sw<16, 16, 32, 32, 1>(layer1_0_sc_out, layer1_1_conv1_weight, layer1_1_conv1_out);
	bn_relu_sw<16, 32>(layer1_1_conv1_out, layer1_1_bn_relu1_out, layer1_1_relu1_mask,  layer1_1_bn_relu1_bn_wt, layer1_1_bn_relu1_bn_bias);

	conv_3x3_sw<16, 16, 32, 32, 1>(layer1_1_bn_relu1_out, layer1_1_conv2_weight, layer1_1_conv2_out);
	bn_relu_sw<16, 32>(layer1_1_conv2_out, layer1_1_bn_relu2_out, layer1_1_relu2_mask,  layer1_1_bn_relu2_bn_wt, layer1_1_bn_relu2_bn_bias);

	shortcut_sw<16, 32>(layer1_0_sc_out, layer1_1_bn_relu2_out, layer1_1_sc_out);

	////////////////////////////////////
	//////////// layer1_2  /////////////

	conv_3x3_sw<16, 16, 32, 32, 1>(layer1_1_sc_out, layer1_2_conv1_weight, layer1_2_conv1_out);
	bn_relu_sw<16, 32>(layer1_2_conv1_out, layer1_2_bn_relu1_out, layer1_2_relu1_mask,  layer1_2_bn_relu1_bn_wt, layer1_2_bn_relu1_bn_bias);

	conv_3x3_sw<16, 16, 32, 32, 1>(layer1_2_bn_relu1_out, layer1_2_conv2_weight, layer1_2_conv2_out);
	bn_relu_sw<16, 32>(layer1_2_conv2_out, layer1_2_bn_relu2_out, layer1_2_relu2_mask,  layer1_2_bn_relu2_bn_wt, layer1_2_bn_relu2_bn_bias);

	shortcut_sw<16, 32>(layer1_1_sc_out, layer1_2_bn_relu2_out, layer1_2_sc_out);


	////////////////////////////////////
	//////////// layer2_sc  ////////////

	conv_1x1_sw<16, 32, 32, 16, 2>(layer1_2_sc_out, layer2_sc_conv_weight, layer2_sc_conv_out);
	bn_sw<32, 16>(layer2_sc_conv_out, layer2_sc_bn_out,  layer2_sc_bn_wt, layer2_sc_bn_bias);

	////////////////////////////////////
	//////////// layer2_0  /////////////

	conv_3x3_sw<16, 32, 32, 16, 2>(layer1_2_sc_out, layer2_0_conv1_weight, layer2_0_conv1_out);
	bn_relu_sw<32, 16>(layer2_0_conv1_out, layer2_0_bn_relu1_out, layer2_0_relu1_mask,  layer2_0_bn_relu1_bn_wt, layer2_0_bn_relu1_bn_bias);

	conv_3x3_sw<32, 32, 16, 16, 1>(layer2_0_bn_relu1_out, layer2_0_conv2_weight, layer2_0_conv2_out);
	bn_relu_sw<32, 16>(layer2_0_conv2_out, layer2_0_bn_relu2_out, layer2_0_relu2_mask,  layer2_0_bn_relu2_bn_wt, layer2_0_bn_relu2_bn_bias);

	shortcut_sw<32, 16>(layer2_sc_bn_out, layer2_0_bn_relu2_out, layer2_0_sc_out);

	////////////////////////////////////
	//////////// layer2_1  /////////////

	conv_3x3_sw<32, 32, 16, 16, 1>(layer2_0_sc_out, layer2_1_conv1_weight, layer2_1_conv1_out);
	bn_relu_sw<32, 16>(layer2_1_conv1_out, layer2_1_bn_relu1_out, layer2_1_relu1_mask,  layer2_1_bn_relu1_bn_wt, layer2_1_bn_relu1_bn_bias);

	conv_3x3_sw<32, 32, 16, 16, 1>(layer2_1_bn_relu1_out, layer2_1_conv2_weight, layer2_1_conv2_out);
	bn_relu_sw<32, 16>(layer2_1_conv2_out, layer2_1_bn_relu2_out, layer2_1_relu2_mask,  layer2_1_bn_relu2_bn_wt, layer2_1_bn_relu2_bn_bias);

	shortcut_sw<32, 16>(layer2_0_sc_out, layer2_1_bn_relu2_out, layer2_1_sc_out);

	////////////////////////////////////
	//////////// layer2_2  /////////////

	conv_3x3_sw<32, 32, 16, 16, 1>(layer2_1_sc_out, layer2_2_conv1_weight, layer2_2_conv1_out);
	bn_relu_sw<32, 16>(layer2_2_conv1_out, layer2_2_bn_relu1_out, layer2_2_relu1_mask,  layer2_2_bn_relu1_bn_wt, layer2_2_bn_relu1_bn_bias);

	conv_3x3_sw<32, 32, 16, 16, 1>(layer2_2_bn_relu1_out, layer2_2_conv2_weight, layer2_2_conv2_out);
	bn_relu_sw<32, 16>(layer2_2_conv2_out, layer2_2_bn_relu2_out, layer2_2_relu2_mask,  layer2_2_bn_relu2_bn_wt, layer2_2_bn_relu2_bn_bias);

	shortcut_sw<32, 16>(layer2_1_sc_out, layer2_2_bn_relu2_out, layer2_2_sc_out);



	////////////////////////////////////
	//////////// layer3_sc  ////////////

	conv_1x1_sw<32, 64, 16, 8, 2>(layer2_2_sc_out, layer3_sc_conv_weight, layer3_sc_conv_out);
	bn_sw<64, 8>(layer3_sc_conv_out, layer3_sc_bn_out,  layer3_sc_bn_wt, layer3_sc_bn_bias);

	////////////////////////////////////
	//////////// layer3_0  /////////////

	conv_3x3_sw<32, 64, 16, 8, 2>(layer2_2_sc_out, layer3_0_conv1_weight, layer3_0_conv1_out);
	bn_relu_sw<64, 8>(layer3_0_conv1_out, layer3_0_bn_relu1_out, layer3_0_relu1_mask,  layer3_0_bn_relu1_bn_wt, layer3_0_bn_relu1_bn_bias);

	conv_3x3_sw<64, 64, 8, 8, 1>(layer3_0_bn_relu1_out, layer3_0_conv2_weight, layer3_0_conv2_out);
	bn_relu_sw<64, 8>(layer3_0_conv2_out, layer3_0_bn_relu2_out, layer3_0_relu2_mask,  layer3_0_bn_relu2_bn_wt, layer3_0_bn_relu2_bn_bias);

	shortcut_sw<64, 8>(layer3_sc_bn_out, layer3_0_bn_relu2_out, layer3_0_sc_out);

	////////////////////////////////////
	//////////// layer3_1  /////////////

	conv_3x3_sw<64, 64, 8, 8, 1>(layer3_0_sc_out, layer3_1_conv1_weight, layer3_1_conv1_out);
	bn_relu_sw<64, 8>(layer3_1_conv1_out, layer3_1_bn_relu1_out, layer3_1_relu1_mask,  layer3_1_bn_relu1_bn_wt, layer3_1_bn_relu1_bn_bias);

	conv_3x3_sw<64, 64, 8, 8, 1>(layer3_1_bn_relu1_out, layer3_1_conv2_weight, layer3_1_conv2_out);
	bn_relu_sw<64, 8>(layer3_1_conv2_out, layer3_1_bn_relu2_out, layer3_1_relu2_mask,  layer3_1_bn_relu2_bn_wt, layer3_1_bn_relu2_bn_bias);

	shortcut_sw<64, 8>(layer3_0_sc_out, layer3_1_bn_relu2_out, layer3_1_sc_out);

	////////////////////////////////////
	//////////// layer3_2  /////////////

	conv_3x3_sw<64, 64, 8, 8, 1>(layer3_1_sc_out, layer3_2_conv1_weight, layer3_2_conv1_out);
	bn_relu_sw<64, 8>(layer3_2_conv1_out, layer3_2_bn_relu1_out, layer3_2_relu1_mask,  layer3_2_bn_relu1_bn_wt, layer3_2_bn_relu1_bn_bias);

	conv_3x3_sw<64, 64, 8, 8, 1>(layer3_2_bn_relu1_out, layer3_2_conv2_weight, layer3_2_conv2_out);
	bn_relu_sw<64, 8>(layer3_2_conv2_out, layer3_2_bn_relu2_out, layer3_2_relu2_mask,  layer3_2_bn_relu2_bn_wt, layer3_2_bn_relu2_bn_bias);

	shortcut_sw<64, 8>(layer3_1_sc_out, layer3_2_bn_relu2_out, layer3_2_sc_out);

	////////////////////////////////////
	//////////// avgpool & FC //////////

	avgpool_sw(layer3_2_sc_out, avg_pool_out_sw, 0);
	FC_sw(avg_pool_out_sw, avg_pool_out_sw, linear_weight_sw, linear_bias_sw, linear_out_buf_sw, 0);

	////////////////////////////////////
	//////////// CrossEntropy loss /////

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

	rot_3x3<3, 16>(conv1_weight, conv1_weight_rot);
	rot_3x3<16, 16>(layer1_0_conv1_weight, layer1_0_conv1_weight_rot);
	rot_3x3<16, 16>(layer1_0_conv2_weight, layer1_0_conv2_weight_rot);
	rot_3x3<16, 16>(layer1_1_conv1_weight, layer1_1_conv1_weight_rot);
	rot_3x3<16, 16>(layer1_1_conv2_weight, layer1_1_conv2_weight_rot);
	rot_3x3<16, 16>(layer1_2_conv1_weight, layer1_2_conv1_weight_rot);
	rot_3x3<16, 16>(layer1_2_conv2_weight, layer1_2_conv2_weight_rot);

	rot_1x1<16, 32>(layer2_sc_conv_weight, layer2_sc_conv_weight_rot);
	rot_3x3<16, 32>(layer2_0_conv1_weight, layer2_0_conv1_weight_rot);
	rot_3x3<32, 32>(layer2_0_conv2_weight, layer2_0_conv2_weight_rot);
	rot_3x3<32, 32>(layer2_1_conv1_weight, layer2_1_conv1_weight_rot);
	rot_3x3<32, 32>(layer2_1_conv2_weight, layer2_1_conv2_weight_rot);
	rot_3x3<32, 32>(layer2_2_conv1_weight, layer2_2_conv1_weight_rot);
	rot_3x3<32, 32>(layer2_2_conv2_weight, layer2_2_conv2_weight_rot);

	rot_1x1<32, 64>(layer3_sc_conv_weight, layer3_sc_conv_weight_rot);
	rot_3x3<32, 64>(layer3_0_conv1_weight, layer3_0_conv1_weight_rot);
	rot_3x3<64, 64>(layer3_0_conv2_weight, layer3_0_conv2_weight_rot);
	rot_3x3<64, 64>(layer3_1_conv1_weight, layer3_1_conv1_weight_rot);
	rot_3x3<64, 64>(layer3_1_conv2_weight, layer3_1_conv2_weight_rot);
	rot_3x3<64, 64>(layer3_2_conv1_weight, layer3_2_conv1_weight_rot);
	rot_3x3<64, 64>(layer3_2_conv2_weight, layer3_2_conv2_weight_rot);

	////////////////////////////////////
	//////////// avgpool & FC //////////

	FC_sw(fc_bp_out, avg_pool_out_sw, linear_weight_sw, linear_bias_sw, error_sw, 1);
	avgpool_sw(avg_pool_bp_out, fc_bp_out, 1);

	////////////////////////////////////
	//////////// layer3_2  /////////////

	bn_relu_bp_sw<64, 8>(avg_pool_bp_out, layer3_2_conv2_out, layer3_2_bn_relu2_bp_out, layer3_2_relu2_mask,  layer3_2_bn_relu2_bn_wt, layer3_2_bn_relu2_bn_bias, layer3_2_bn_relu2_bn_wt_vel, layer3_2_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<64, 64, 8, 8, 1>(layer3_2_bn_relu2_bp_out, layer3_2_conv2_weight_rot, layer3_2_conv2_bp_out);
	conv_3x3_grad_sw<64, 64, 8, 1>(layer3_2_bn_relu2_bp_out, layer3_2_bn_relu1_out, layer3_2_conv2_weight, layer3_2_conv2_weight_vel);

	bn_relu_bp_sw<64, 8>(layer3_2_conv2_bp_out, layer3_2_conv1_out, layer3_2_bn_relu1_bp_out, layer3_2_relu1_mask,  layer3_2_bn_relu1_bn_wt, layer3_2_bn_relu1_bn_bias, layer3_2_bn_relu1_bn_wt_vel, layer3_2_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<64, 64, 8, 8, 1>(layer3_2_bn_relu1_bp_out, layer3_2_conv1_weight_rot, layer3_2_conv1_bp_out);
	conv_3x3_grad_sw<64, 64, 8, 1>(layer3_2_bn_relu1_bp_out, layer3_1_sc_out, layer3_2_conv1_weight, layer3_2_conv1_weight_vel);

	shortcut_sw<64, 8>(avg_pool_bp_out, layer3_2_conv1_bp_out, layer3_2_sc_bp_out);

	////////////////////////////////////
	//////////// layer3_1  /////////////

	bn_relu_bp_sw<64, 8>(layer3_2_sc_bp_out, layer3_1_conv2_out, layer3_1_bn_relu2_bp_out, layer3_1_relu2_mask,  layer3_1_bn_relu2_bn_wt, layer3_1_bn_relu2_bn_bias, layer3_1_bn_relu2_bn_wt_vel, layer3_1_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<64, 64, 8, 8, 1>(layer3_1_bn_relu2_bp_out, layer3_1_conv2_weight_rot, layer3_1_conv2_bp_out);
	conv_3x3_grad_sw<64, 64, 8, 1>(layer3_1_bn_relu2_bp_out, layer3_1_bn_relu1_out, layer3_1_conv2_weight, layer3_1_conv2_weight_vel);

	bn_relu_bp_sw<64, 8>(layer3_1_conv2_bp_out, layer3_1_conv1_out, layer3_1_bn_relu1_bp_out, layer3_1_relu1_mask,  layer3_1_bn_relu1_bn_wt, layer3_1_bn_relu1_bn_bias, layer3_1_bn_relu1_bn_wt_vel, layer3_1_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<64, 64, 8, 8, 1>(layer3_1_bn_relu1_bp_out, layer3_1_conv1_weight_rot, layer3_1_conv1_bp_out);
	conv_3x3_grad_sw<64, 64, 8, 1>(layer3_1_bn_relu1_bp_out, layer3_0_sc_out, layer3_1_conv1_weight, layer3_1_conv1_weight_vel);

	shortcut_sw<64, 8>(layer3_2_sc_bp_out, layer3_1_conv1_bp_out, layer3_1_sc_bp_out);

	////////////////////////////////////
	//////////// layer3_0  /////////////

	bn_relu_bp_sw<64, 8>(layer3_1_sc_bp_out, layer3_0_conv2_out, layer3_0_bn_relu2_bp_out, layer3_0_relu2_mask,  layer3_0_bn_relu2_bn_wt, layer3_0_bn_relu2_bn_bias, layer3_0_bn_relu2_bn_wt_vel, layer3_0_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<64, 64, 8, 8, 1>(layer3_0_bn_relu2_bp_out, layer3_0_conv2_weight_rot, layer3_0_conv2_bp_out);
	conv_3x3_grad_sw<64, 64, 8, 1>(layer3_0_bn_relu2_bp_out, layer3_0_bn_relu1_out, layer3_0_conv2_weight, layer3_0_conv2_weight_vel);

	bn_relu_bp_sw<64, 8>(layer3_0_conv2_bp_out, layer3_0_conv1_out, layer3_0_bn_relu1_bp_out, layer3_0_relu1_mask,  layer3_0_bn_relu1_bn_wt, layer3_0_bn_relu1_bn_bias, layer3_0_bn_relu1_bn_wt_vel, layer3_0_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<64, 32, 8, 16, 2>(layer3_0_bn_relu1_bp_out, layer3_0_conv1_weight_rot, layer3_0_conv1_bp_out);
	conv_3x3_grad_sw<32, 64, 16, 2>(layer3_0_bn_relu1_bp_out, layer2_2_sc_out, layer3_0_conv1_weight, layer3_0_conv1_weight_vel);

	////////////////////////////////////
	//////////// layer3_sc  ////////////

	bn_bp_sw<64, 8>(layer3_1_sc_bp_out, layer3_sc_conv_out, layer3_sc_bn_bp_out,  layer3_sc_bn_wt, layer3_sc_bn_bias, layer3_sc_bn_wt_vel, layer3_sc_bn_bias_vel);
	deconv_1x1_sw<64, 32, 8, 16, 2>(layer3_sc_bn_bp_out, layer3_sc_conv_weight_rot, layer3_sc_conv_bp_out);
	conv_1x1_grad_sw<32, 64, 16, 2>(layer3_sc_bn_bp_out, layer2_2_sc_out, layer3_sc_conv_weight, layer3_sc_conv_weight_vel);

	shortcut_sw<32, 16>(layer3_sc_conv_bp_out, layer3_0_conv1_bp_out, layer3_0_sc_bp_out);



	////////////////////////////////////
	//////////// layer2_2  /////////////

	bn_relu_bp_sw<32, 16>(layer3_0_sc_bp_out, layer2_2_conv2_out, layer2_2_bn_relu2_bp_out, layer2_2_relu2_mask,  layer2_2_bn_relu2_bn_wt, layer2_2_bn_relu2_bn_bias, layer2_2_bn_relu2_bn_wt_vel, layer2_2_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<32, 32, 16, 16, 1>(layer2_2_bn_relu2_bp_out, layer2_2_conv2_weight_rot, layer2_2_conv2_bp_out);	// layer2_2_conv2_bp_out looks wrong
	conv_3x3_grad_sw<32, 32, 16, 1>(layer2_2_bn_relu2_bp_out, layer2_2_bn_relu1_out, layer2_2_conv2_weight, layer2_2_conv2_weight_vel);

	bn_relu_bp_sw<32, 16>(layer2_2_conv2_bp_out, layer2_2_conv1_out, layer2_2_bn_relu1_bp_out, layer2_2_relu1_mask,  layer2_2_bn_relu1_bn_wt, layer2_2_bn_relu1_bn_bias, layer2_2_bn_relu1_bn_wt_vel, layer2_2_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<32, 32, 16, 16, 1>(layer2_2_bn_relu1_bp_out, layer2_2_conv1_weight_rot, layer2_2_conv1_bp_out);
	conv_3x3_grad_sw<32, 32, 16, 1>(layer2_2_bn_relu1_bp_out, layer2_1_sc_out, layer2_2_conv1_weight, layer2_2_conv1_weight_vel);

	shortcut_sw<32, 16>(layer3_0_sc_bp_out, layer2_2_conv1_bp_out, layer2_2_sc_bp_out);

	////////////////////////////////////
	//////////// layer2_1  /////////////

	bn_relu_bp_sw<32, 16>(layer2_2_sc_bp_out, layer2_1_conv2_out, layer2_1_bn_relu2_bp_out, layer2_1_relu2_mask,  layer2_1_bn_relu2_bn_wt, layer2_1_bn_relu2_bn_bias, layer2_1_bn_relu2_bn_wt_vel, layer2_1_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<32, 32, 16, 16, 1>(layer2_1_bn_relu2_bp_out, layer2_1_conv2_weight_rot, layer2_1_conv2_bp_out);
	conv_3x3_grad_sw<32, 32, 16, 1>(layer2_1_bn_relu2_bp_out, layer2_1_bn_relu1_out, layer2_1_conv2_weight, layer2_1_conv2_weight_vel);

	bn_relu_bp_sw<32, 16>(layer2_1_conv2_bp_out, layer2_1_conv1_out, layer2_1_bn_relu1_bp_out, layer2_1_relu1_mask,  layer2_1_bn_relu1_bn_wt, layer2_1_bn_relu1_bn_bias, layer2_1_bn_relu1_bn_wt_vel, layer2_1_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<32, 32, 16, 16, 1>(layer2_1_bn_relu1_bp_out, layer2_1_conv1_weight_rot, layer2_1_conv1_bp_out);
	conv_3x3_grad_sw<32, 32, 16, 1>(layer2_1_bn_relu1_bp_out, layer2_0_sc_out, layer2_1_conv1_weight, layer2_1_conv1_weight_vel);

	shortcut_sw<32, 16>(layer2_2_sc_bp_out, layer2_1_conv1_bp_out, layer2_1_sc_bp_out);

	////////////////////////////////////
	//////////// layer2_0  /////////////

	bn_relu_bp_sw<32, 16>(layer2_1_sc_bp_out, layer2_0_conv2_out, layer2_0_bn_relu2_bp_out, layer2_0_relu2_mask,  layer2_0_bn_relu2_bn_wt, layer2_0_bn_relu2_bn_bias, layer2_0_bn_relu2_bn_wt_vel, layer2_0_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<32, 32, 16, 16, 1>(layer2_0_bn_relu2_bp_out, layer2_0_conv2_weight_rot, layer2_0_conv2_bp_out);
	conv_3x3_grad_sw<32, 32, 16, 1>(layer2_0_bn_relu2_bp_out, layer2_0_bn_relu1_out, layer2_0_conv2_weight, layer2_0_conv2_weight_vel);

	bn_relu_bp_sw<32, 16>(layer2_0_conv2_bp_out, layer2_0_conv1_out, layer2_0_bn_relu1_bp_out, layer2_0_relu1_mask,  layer2_0_bn_relu1_bn_wt, layer2_0_bn_relu1_bn_bias, layer2_0_bn_relu1_bn_wt_vel, layer2_0_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<32, 16, 16, 32, 2>(layer2_0_bn_relu1_bp_out, layer2_0_conv1_weight_rot, layer2_0_conv1_bp_out);
	conv_3x3_grad_sw<16, 32, 32, 2>(layer2_0_bn_relu1_bp_out, layer1_2_sc_out, layer2_0_conv1_weight, layer2_0_conv1_weight_vel);

	////////////////////////////////////
	//////////// layer2_sc  ////////////

	bn_bp_sw<32, 16>(layer2_1_sc_bp_out, layer2_sc_conv_out, layer2_sc_bn_bp_out,  layer2_sc_bn_wt, layer2_sc_bn_bias, layer2_sc_bn_wt_vel, layer2_sc_bn_bias_vel);
	deconv_1x1_sw<32, 16, 16, 32, 2>(layer2_sc_bn_bp_out, layer2_sc_conv_weight_rot, layer2_sc_conv_bp_out);
	conv_1x1_grad_sw<16, 32, 32, 2>(layer2_sc_bn_bp_out, layer1_2_sc_out, layer2_sc_conv_weight, layer2_sc_conv_weight_vel);

	shortcut_sw<16, 32>(layer2_sc_conv_bp_out, layer2_0_conv1_bp_out, layer2_0_sc_bp_out);



	////////////////////////////////////
	//////////// layer1_2  /////////////

	bn_relu_bp_sw<16, 32>(layer2_0_sc_bp_out, layer1_2_conv2_out, layer1_2_bn_relu2_bp_out, layer1_2_relu2_mask,  layer1_2_bn_relu2_bn_wt, layer1_2_bn_relu2_bn_bias, layer1_2_bn_relu2_bn_wt_vel, layer1_2_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<16, 16, 32, 32, 1>(layer1_2_bn_relu2_bp_out, layer1_2_conv2_weight_rot, layer1_2_conv2_bp_out);
	conv_3x3_grad_sw<16, 16, 32, 1>(layer1_2_bn_relu2_bp_out, layer1_2_bn_relu1_out, layer1_2_conv2_weight, layer1_2_conv2_weight_vel);

	bn_relu_bp_sw<16, 32>(layer1_2_conv2_bp_out, layer1_2_conv1_out, layer1_2_bn_relu1_bp_out, layer1_2_relu1_mask,  layer1_2_bn_relu1_bn_wt, layer1_2_bn_relu1_bn_bias, layer1_2_bn_relu1_bn_wt_vel, layer1_2_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<16, 16, 32, 32, 1>(layer1_2_bn_relu1_bp_out, layer1_2_conv1_weight_rot, layer1_2_conv1_bp_out);
	conv_3x3_grad_sw<16, 16, 32, 1>(layer1_2_bn_relu1_bp_out, layer1_1_sc_out, layer1_2_conv1_weight, layer1_2_conv1_weight_vel);

	shortcut_sw<16, 32>(layer2_0_sc_bp_out, layer1_2_conv1_bp_out, layer1_2_sc_bp_out);

	////////////////////////////////////
	//////////// layer1_1  /////////////

	bn_relu_bp_sw<16, 32>(layer1_2_sc_bp_out, layer1_1_conv2_out, layer1_1_bn_relu2_bp_out, layer1_1_relu2_mask,  layer1_1_bn_relu2_bn_wt, layer1_1_bn_relu2_bn_bias, layer1_1_bn_relu2_bn_wt_vel, layer1_1_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<16, 16, 32, 32, 1>(layer1_1_bn_relu2_bp_out, layer1_1_conv2_weight_rot, layer1_1_conv2_bp_out);
	conv_3x3_grad_sw<16, 16, 32, 1>(layer1_1_bn_relu2_bp_out, layer1_1_bn_relu1_out, layer1_1_conv2_weight, layer1_1_conv2_weight_vel);

	bn_relu_bp_sw<16, 32>(layer1_1_conv2_bp_out, layer1_1_conv1_out, layer1_1_bn_relu1_bp_out, layer1_1_relu1_mask,  layer1_1_bn_relu1_bn_wt, layer1_1_bn_relu1_bn_bias, layer1_1_bn_relu1_bn_wt_vel, layer1_1_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<16, 16, 32, 32, 1>(layer1_1_bn_relu1_bp_out, layer1_1_conv1_weight_rot, layer1_1_conv1_bp_out);
	conv_3x3_grad_sw<16, 16, 32, 1>(layer1_1_bn_relu1_bp_out, layer1_0_sc_out, layer1_1_conv1_weight, layer1_1_conv1_weight_vel);

	shortcut_sw<16, 32>(layer1_2_sc_bp_out, layer1_1_conv1_bp_out, layer1_1_sc_bp_out);

	////////////////////////////////////
	//////////// layer1_0  /////////////

	bn_relu_bp_sw<16, 32>(layer1_1_sc_bp_out, layer1_0_conv2_out, layer1_0_bn_relu2_bp_out, layer1_0_relu2_mask,  layer1_0_bn_relu2_bn_wt, layer1_0_bn_relu2_bn_bias, layer1_0_bn_relu2_bn_wt_vel, layer1_0_bn_relu2_bn_bias_vel);
	deconv_3x3_sw<16, 16, 32, 32, 1>(layer1_0_bn_relu2_bp_out, layer1_0_conv2_weight_rot, layer1_0_conv2_bp_out);
	conv_3x3_grad_sw<16, 16, 32, 1>(layer1_0_bn_relu2_bp_out, layer1_0_bn_relu1_out, layer1_0_conv2_weight, layer1_0_conv2_weight_vel);

	bn_relu_bp_sw<16, 32>(layer1_0_conv2_bp_out, layer1_0_conv1_out, layer1_0_bn_relu1_bp_out, layer1_0_relu1_mask,  layer1_0_bn_relu1_bn_wt, layer1_0_bn_relu1_bn_bias, layer1_0_bn_relu1_bn_wt_vel, layer1_0_bn_relu1_bn_bias_vel);
	deconv_3x3_sw<16, 16, 32, 32, 1>(layer1_0_bn_relu1_bp_out, layer1_0_conv1_weight_rot, layer1_0_conv1_bp_out);
	conv_3x3_grad_sw<16, 16, 32, 1>(layer1_0_bn_relu1_bp_out, bn_relu_1_out, layer1_0_conv1_weight, layer1_0_conv1_weight_vel);

	shortcut_sw<16, 32>(layer1_1_sc_bp_out, layer1_0_conv1_bp_out, layer1_0_sc_bp_out);

	////////////////////////////////////
	//////////// Conv 1  ///////////////

	bn_relu_bp_sw<16, 32>(layer1_0_sc_bp_out, conv1_out, bn_relu_1_bp_out, conv1_relu_mask,  conv1_bn_wt, conv1_bn_bias, conv1_bn_wt_vel, conv1_bn_bias_vel);
	conv_3x3_grad_sw<3, 16, 32, 1>(bn_relu_1_bp_out, image_sw, conv1_weight, conv1_weight_vel);
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
	ini_copy_hw = 0;
	ini_act_bias_hw = 0;

	ctrl_sc_hw = 0; // if ctrl_sc_hw=0, generate and send out_copy into DDR
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
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels_hw = 3;
	out_channels_hw = 16;
	H_fmap_in_hw = 32;
	H_fmap_out_hw = 32;
	stride_hw = 1;

	in_channels_after_pack_hw = 1;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	conv_3x3_weight_ptr_hw = 0;

	LOOP_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 2
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 32;
	H_fmap_out_hw = 32;
	in_channels_hw = 16;
	out_channels_hw = 16;
	stride_hw = 1;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////

	LOOP_layer1_0_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 2
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////

	LOOP_layer1_0_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 3
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(	// MODIFIED HERE!
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t1_hw[ini_copy_hw],
			H_fmap_out_hw, ctrl_sc_hw	// generate and send out_copy into DDR
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////

	LOOP_layer1_1_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 4
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_1_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////

	LOOP_layer1_1_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 5
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(	// MODIFIED HERE!
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t1_hw[ini_copy_hw],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_2 PG1 //////////////////////

	LOOP_layer1_2_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 6
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_2 PG2 //////////////////////

	LOOP_layer1_2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 7
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(	// MODIFIED HERE!
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t1_hw[ini_copy_hw],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 32;
	H_fmap_out_hw = 16;
	in_channels_hw = 16;
	out_channels_hw = 32;
	stride_hw = 2;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	conv_1x1_weight_ptr_hw = 0;

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
	LOOP_layer2_0_ConvSC:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 8
		ini_copy_hw += 1;
		// conv_1x1_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_1x1_weight_ptr_hw += 1;
			generic_conv_1x1_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_1x1_weight_tile_buffer_hw[conv_1x1_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], out_buf_t1_hw[ini_hw], // conv+bn shortcut
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////

	LOOP_layer2_0_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 9
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_1_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 16;
	H_fmap_out_hw = 16;
	in_channels_hw = 32;
	out_channels_hw = 32;
	stride_hw = 1;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////

	LOOP_layer2_0_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 10
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_1_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t1_hw[ini_copy_hw],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////

	LOOP_layer2_1_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 11
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////

	LOOP_layer2_1_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 12
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(	// MODIFIED HERE
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t1_hw[ini_copy_hw],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_2 PG1 //////////////////////

	LOOP_layer2_2_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 13
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_1_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_2 PG2 //////////////////////

	LOOP_layer2_2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 14
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(	// MODIFIED HERE
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t1_hw[ini_copy_hw],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 16;
	H_fmap_out_hw = 8;
	in_channels_hw = 32;
	out_channels_hw = 64;
	stride_hw = 2;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
	LOOP_layer3_0_ConvSC:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 15
		ini_copy_hw += 1;
		// conv_1x1_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_1x1_weight_ptr_hw += 1;
			generic_conv_1x1_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_1x1_weight_tile_buffer_hw[conv_1x1_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_hw(
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out],	 out_buf_t1_hw[ini_hw], // conv+bn shortcut
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////

	LOOP_layer3_0_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 16
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 8;
	H_fmap_out_hw = 8;
	in_channels_hw = 64;
	out_channels_hw = 64;
	stride_hw = 1;

	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;
	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////

	LOOP_layer3_0_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 17
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t1_hw[ini_copy_hw],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////

	LOOP_layer3_1_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 18
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_1_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////

	LOOP_layer3_1_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 19
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(	// MODIFIED HERE
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t1_hw[ini_hw],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_2 PG1 //////////////////////

	LOOP_layer3_2_Conv1:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 20
		ini_copy_hw += 1;
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(
			msb_fmap_tile_buffer_0_hw[c_out], msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_2 PG2 //////////////////////

	LOOP_layer3_2_Conv2:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_hw += 1;	// ini_hw = 21
		// conv_3x3_weight_ptr_hw += 1;
		for (int c_in = 0; c_in < in_channels_after_pack_hw; c_in ++) {

			conv_3x3_weight_ptr_hw += 1;
			generic_conv_3x3_hw(
				msb_fmap_tile_buffer_s2_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
		}
		generic_bn_relu_hw(	// MODIFIED HERE
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], out_buf_t1_hw[ini_hw], relu_mask_hw[ini_hw],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_out_hw
		);
	}
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		ini_copy_hw += 1;
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t1_hw[ini_copy_hw],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// AvgPool and FC ////////////////////

	// avgpool
	ctrl_avgpool_hw = 0;
	LOOP_AvgPool:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		generic_avgpool_hw(
			msb_fmap_tile_buffer_0_hw[c_out], pool_out_buf_hw, ctrl_avgpool_hw, c_out, lsb_fmap_tile_buffer_hw[c_out], pool_out_buf_copy_hw
		);
	}

	// FC
	ctrl_fc_hw = 0;
	LOOP_FC:
	generic_FC_hw(
		pool_out_buf_hw, pool_out_buf_copy_hw, linear_weight_tile_buffer_hw, linear_bias_hw, linear_out_buf_hw, ctrl_fc_hw
	);

	////////////////////////////////////////////////
	//////////// CrossEntropy loss /////////////////

	sum_hw = 0;
	softmax_sum_approx:
	for (int j = 0; j < 10; j ++) {
#pragma HLS PIPELINE II=1
		sum_hw += 1 + linear_out_buf_hw[j] + 0.5*(linear_out_buf_hw[j])*(linear_out_buf_hw[j]) + 0.1667*(linear_out_buf_hw[j])*(linear_out_buf_hw[j])*(linear_out_buf_hw[j]);
	}
	// printf("sum_hw: %f \n", sum_hw);

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
		pool_out_buf_hw, pool_out_buf_copy_hw, linear_weight_tile_buffer_hw, linear_bias_hw, error_hw, ctrl_fc_hw
	);

	// avgpool_bp
	ctrl_avgpool_hw = 1;
	LOOP_AvgPool_bp:
	for (int c_out = 0; c_out < out_channels_after_pack_hw; c_out ++) {
		generic_avgpool_hw(
			msb_fmap_tile_buffer_1_hw[c_out], pool_out_buf_hw, ctrl_avgpool_hw, c_out, lsb_fmap_tile_buffer_hw[c_out], pool_out_buf_copy_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 8;
	H_fmap_out_hw = 8;
	in_channels_hw = 64;
	out_channels_hw = 64;
	stride_hw = 1;

	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	ctrl_sc_hw = 1;
	ctrl_conv_hw = 1;

	////////////////////////////////////////////////
	//////////// layer3_2 PG2 //////////////////////

	LOOP_layer3_2_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 21
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	//////////// layer3_2 PG1 //////////////////////

	LOOP_layer3_2_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 20
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(	// MODIFIED HERE
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[0],
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////

	LOOP_layer3_1_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 19
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_0_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);
		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////

	LOOP_layer3_1_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 18
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(	// MODIFIED HERE
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}
	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[0], // DDR not used
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////

	LOOP_layer3_0_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 17
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {	// stride_hw-2 input
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	//////////// LAYER 3 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 8;
	H_fmap_out_hw = 16;
	in_channels_hw = 64;
	out_channels_hw = 32;
	stride_hw = 2;

	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////

	LOOP_layer3_0_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {	// stride_hw-2 input
		ini_hw -= 1;	// ini_hw = 16
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			lsb_fmap_tile_buffer_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_in <= in_channels_after_pack_hw/2 - 1) {
				generic_conv_3x3_grad_hw(
					out_buf_t1_hw[ini_hw - 3*out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////

	LOOP_layer3_0_ConvSC_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {	// stride_hw-2 input
		ini_hw -= 1;	// ini_hw = 15
		// conv_1x1_weight_ptr_hw -= 1;

		generic_bn_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],	// conv+bn shortcut
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_1x1_weight_ptr_hw -= 1;
			rot180_1x1_hw(conv_1x1_weight_tile_buffer_hw[conv_1x1_weight_ptr_hw], conv_1x1_weight_rot_hw);
			generic_deconv_1x1_hw(	// note the index of shortcut input
				msb_fmap_tile_buffer_0_hw[c_in], conv_1x1_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_1x1_weight_grad_cal
			if (c_in <= in_channels_after_pack_hw/2 - 1) {
				generic_conv_1x1_grad_hw(
					out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0_hw[c_in], conv_1x1_weight_tile_buffer_hw[conv_1x1_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_s2_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[0], // DDR not used
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 16;
	H_fmap_out_hw = 16;
	in_channels_hw = 32;
	out_channels_hw = 32;
	stride_hw = 1;

	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer2_2 PG2 //////////////////////
	LOOP_layer2_2_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 14
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_0_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_2 PG1 //////////////////////

	LOOP_layer2_2_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 13
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}
	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[0], // DDR not used
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	LOOP_layer2_1_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 12
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	//////////// layer2_1 PG1 //////////////////////

	LOOP_layer2_1_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 11
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[0],  // DDR not used
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////

	LOOP_layer2_0_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {	// stride_hw-2 input
		ini_hw -= 1;	// ini_hw = 10
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_0_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_out <= out_channels_after_pack_hw/2 - 1) {
				generic_conv_3x3_grad_hw(
					out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 16;
	H_fmap_out_hw = 32;
	in_channels_hw = 32;
	out_channels_hw = 16;
	stride_hw = 2;

	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 9
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			lsb_fmap_tile_buffer_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {	// stride_hw-2 input
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			if (c_in <= in_channels_after_pack_hw/2 - 1) {
				generic_conv_3x3_grad_hw(
					out_buf_t1_hw[ini_hw - 3*out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////

	LOOP_layer2_0_ConvSC_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 8
		// conv_1x1_weight_ptr_hw -= 1;

		generic_bn_bp_hw(
			msb_fmap_tile_buffer_0_hw[c_in], out_buf_t0_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],	// conv+generic_bn_hw shortcut
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {	// stride_hw-2 input
			conv_1x1_weight_ptr_hw -= 1;
			rot180_1x1_hw(conv_1x1_weight_tile_buffer_hw[conv_1x1_weight_ptr_hw], conv_1x1_weight_rot_hw);
			generic_deconv_1x1_hw(	// note the index of shortcut input
				msb_fmap_tile_buffer_1_hw[c_in], conv_1x1_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_1x1_weight_grad_cal
			if (c_in <= in_channels_after_pack_hw/2 - 1) {
				generic_conv_1x1_grad_hw(
					out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_1x1_weight_tile_buffer_hw[conv_1x1_weight_ptr_hw],
					stride_hw, H_fmap_in_hw
				);
			}
			// end gradient calculation
			///////////////////////////
		}
	}

	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_s2_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[0], // DDR not used
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in_hw = 32;
	H_fmap_out_hw = 32;
	in_channels_hw = 16;
	out_channels_hw = 16;
	stride_hw = 1;

	out_channels_after_pack_hw = out_channels_hw/CHANNEL_OUT_T;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	////////////////////////////////////////////////
	//////////// layer1_2 PG2 //////////////////////

	LOOP_layer1_2_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 7
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	//////////// layer1_2 PG1 //////////////////////

	LOOP_layer1_2_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 6
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[0], // DDR not used
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////

	LOOP_layer1_1_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 5
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_0_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////

	LOOP_layer1_1_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 4
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}
	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_0_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_1_hw[c_out], out_buf_t0_hw[0], // DDR not used
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////

	LOOP_layer1_0_Conv2_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 3
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_1_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	//////////// layer1_0 PG1 //////////////////////

	LOOP_layer1_0_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 2
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_s2_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_0_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {	// input from conv1
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_0_hw[c_in], conv_3x3_weight_rot_hw, lsb_fmap_tile_buffer_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
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
	for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
		generic_shortcut_hw(
			msb_fmap_tile_buffer_1_hw[c_out], lsb_fmap_tile_buffer_hw[c_out], msb_fmap_tile_buffer_0_hw[c_out], out_buf_t0_hw[0], // DDR not used
			H_fmap_out_hw, ctrl_sc_hw
		);
	}

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels_hw = 16;
	out_channels_hw = 3;
	H_fmap_in_hw = 32;
	H_fmap_out_hw = 32;
	stride_hw = 1;

	out_channels_after_pack_hw = 1;
	in_channels_after_pack_hw = in_channels_hw/CHANNEL_IN_T;

	LOOP_Conv1_bp:
	for (int c_in = in_channels_after_pack_hw - 1; c_in >= 0; c_in --) {
		ini_hw -= 1;	// ini_hw = 1
		// conv_3x3_weight_ptr_hw -= 1;

		generic_bn_relu_bp_hw(
			msb_fmap_tile_buffer_0_hw[c_in], out_buf_t0_hw[ini_hw], relu_mask_hw[ini_hw], msb_fmap_tile_buffer_1_hw[c_in],
			bn_wt_hw[ini_hw], bn_bias_hw[ini_hw],
			H_fmap_in_hw
		);

		for (int c_out = out_channels_after_pack_hw - 1; c_out >= 0; c_out --) {
			conv_3x3_weight_ptr_hw -= 1;
			rot180_3x3_hw(conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw], conv_3x3_weight_rot_hw);
			generic_deconv_3x3_hw(
				msb_fmap_tile_buffer_1_hw[c_in], conv_3x3_weight_rot_hw, msb_fmap_tile_buffer_s2_hw[c_out], out_buf_t0_hw[ini_hw],
				stride_hw, H_fmap_in_hw, H_fmap_out_hw, c_in, ctrl_conv_hw
			);
			///////////////////////////
			// conv_3x3_weight_grad_cal
			generic_conv_3x3_grad_hw(
				out_buf_t1_hw[ini_hw - out_channels_hw/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1_hw[c_out], conv_3x3_weight_tile_buffer_hw[conv_3x3_weight_ptr_hw],
				stride_hw, H_fmap_in_hw
			);
			// end gradient calculation
			///////////////////////////
		}
	}
}

void load_image_CIFAR10()
{
	std::ifstream ifs_param("data_batch_1.bin", std::ios::in | std::ios::binary);
	ifs_param.read((char*)(images), sizeof(unsigned char)*3* EPOCH *32*32);	// first uint8 is label
	ifs_param.close();
}

void load_image_CIFAR100()
{
	std::ifstream ifs_param("train.bin", std::ios::in | std::ios::binary);
	ifs_param.read((char*)(images), sizeof(unsigned char)*3* EPOCH *32*32);	// first uint8 is label
	ifs_param.close();
}

void get_image_CIFAR10(unsigned char *images, unsigned int idx, float image[3][32][32])
{
	unsigned int offset = idx*3*32*32 + 1;
	for (int c = 0; c < 3; c ++) {
		for (int row = 0; row < 32; row ++) {
			for (int col = 0; col < 32; col ++) {
				image[c][row][col] = images[offset + c*32*32 + row*32 + col];
			}
		}
	}
}

void get_image_CIFAR100(unsigned char *images, unsigned int idx, float image[3][32][32])
{
	unsigned int offset = idx*3*32*32 + 2;
	for (int c = 0; c < 3; c ++) {
		for (int row = 0; row < 32; row ++) {
			for (int col = 0; col < 32; col ++) {
				image[c][row][col] = images[offset + c*32*32 + row*32 + col];
			}
		}
	}
}

int main(int argc, char **argv)
{

	float ctrl_tl;
	float error_bm[10];
	float loss_bm;

	float image_sw[3][32][32];
	load_image_CIFAR100();

	// run k epochs
	for (int k = 0; k < EPOCH; k ++) {

		ctrl_tl = 1;
		cout << "---------------" << " iteration " << k+1 << " ---------------" << endl;

		////////////////////////////////
		//////// SOFTWARE //////////////
		////////////////////////////////

		get_image_CIFAR100(images, 0, image_sw);

		FracNet_sw(image_sw);

		cout << "loss_sw: " << loss_sw << endl;
		cout << "error_sw: " << error_sw[1] << "  " << endl;
		cout << "" << endl;

		//////////////////////////////
		////// HARDWARE //////////////
		//////////////////////////////

		FracNet_hw(image_sw);

		cout << "loss_hw: " << loss_hw << endl;
		cout << "error_hw: " << error_hw[1] << "  " << endl;
		cout << "" << endl;

		////////////////////////////////
		//////// bm(2, 5) //////////////
		////////////////////////////////

		FracNet_T(image_sw, ctrl_tl, loss_bm, error_bm);

		cout << "loss_bm: " << loss_bm << endl;
		cout << "error_bm: " << error_bm[1] << "  " << endl;
	}

	// run k epochs fune-tuning
	load_image_CIFAR10();
	for (int k = 0; k < EPOCH; k ++) {

		ctrl_tl = 0;
		cout << "---------------" << " iteration_tl " << k+1 << " ---------------" << endl;

		get_image_CIFAR10(images, 10, image_sw);

		FracNet_T(image_sw, ctrl_tl, loss_bm, error_bm);

		cout << "tl_loss_bm: " << loss_bm << endl;
		cout << "tl_error_bm: " << error_bm[1] << "  " << endl;
	}
	
	return 0;
}

