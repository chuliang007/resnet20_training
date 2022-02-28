#include "typedefs.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "bnn.h"
#include <fstream>
#include <hls_math.h>

using namespace std;

#define NUM_WT_3x3 16
#define NUM_WT_1x1 3
#define BATCH_SIZE 4
#define CHANNEL_IN_T 32
#define CHANNEL_OUT_T 32

float lr = 0.01;

int16 image[BATCH_SIZE][3][32][32];
int16 output[BATCH_SIZE][10];
int16 weight_3x3_updated[NUM_WT_3x3][CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
int16 weight_1x1_updated[NUM_WT_3x3][CHANNEL_OUT_T][CHANNEL_IN_T];

// conv + bn_sw + relu_sw
/* forward */
float conv1_weight[64][3][3][3];
float conv1_out[BATCH_SIZE][64][32][32];
float bn1_weight[64];
float bn1_bias[64];
float bn1_out[BATCH_SIZE][64][32][32];
float relu1_out[BATCH_SIZE][64][32][32];
/* backward */
float relu1_bp_out[BATCH_SIZE][64][32][32];
float bn1_bp_out[BATCH_SIZE][64][32][32];
float bn1_weight_act[64];
float bn1_bias_act[64];
// float conv1_bp_out[BATCH_SIZE][64][32][32];
/* gradient */
float conv1_grad[64][3][3][3];

///////// LAYER 1 /////////
// output size[BATCH_SIZE][64][32][32]
// layer 1_0
/* forward */
float layer1_0_conv1_weight[64][64][3][3];
float layer1_0_conv1_out[BATCH_SIZE][64][32][32];
float layer1_0_bn1_weight[64];
float layer1_0_bn1_bias[64];
float layer1_0_bn1_out[BATCH_SIZE][64][32][32];
float layer1_0_relu1_out[BATCH_SIZE][64][32][32];

float layer1_0_conv2_weight[64][64][3][3];
float layer1_0_conv2_out[BATCH_SIZE][64][32][32];
float layer1_0_bn2_weight[64];
float layer1_0_bn2_bias[64];
float layer1_0_bn2_out[BATCH_SIZE][64][32][32];
float layer1_0_relu2_out[BATCH_SIZE][64][32][32];
float layer1_0_shortcut_out[BATCH_SIZE][64][32][32];
/* backward */
float layer1_0_relu2_bp_out[BATCH_SIZE][64][32][32];
float layer1_0_bn2_bp_out[BATCH_SIZE][64][32][32];
float layer1_0_bn2_weight_act[64];
float layer1_0_bn2_bias_act[64];
float layer1_0_conv2_bp_out[BATCH_SIZE][64][32][32];

float layer1_0_relu1_bp_out[BATCH_SIZE][64][32][32];
float layer1_0_bn1_bp_out[BATCH_SIZE][64][32][32];
float layer1_0_bn1_weight_act[64];
float layer1_0_bn1_bias_act[64];
float layer1_0_conv1_bp_out[BATCH_SIZE][64][32][32];
float layer1_0_shortcut_bp_out[BATCH_SIZE][64][32][32];
/* gradient */ 
float layer1_0_conv1_grad[64][64][3][3];
float layer1_0_conv2_grad[64][64][3][3];

// layer 1_1
/* forward */
float layer1_1_conv1_weight[64][64][3][3];
float layer1_1_conv1_out[BATCH_SIZE][64][32][32];
float layer1_1_bn1_weight[64];
float layer1_1_bn1_bias[64];
float layer1_1_bn1_out[BATCH_SIZE][64][32][32];
float layer1_1_relu1_out[BATCH_SIZE][64][32][32];

float layer1_1_conv2_weight[64][64][3][3];
float layer1_1_conv2_out[BATCH_SIZE][64][32][32];
float layer1_1_bn2_weight[64];
float layer1_1_bn2_bias[64];
float layer1_1_bn2_out[BATCH_SIZE][64][32][32];
float layer1_1_relu2_out[BATCH_SIZE][64][32][32];
float layer1_1_shortcut_out[BATCH_SIZE][64][32][32];
/* backward */
float layer1_1_relu2_bp_out[BATCH_SIZE][64][32][32];
float layer1_1_bn2_bp_out[BATCH_SIZE][64][32][32];
float layer1_1_bn2_weight_act[64];
float layer1_1_bn2_bias_act[64];
float layer1_1_conv2_bp_out[BATCH_SIZE][64][32][32];

float layer1_1_relu1_bp_out[BATCH_SIZE][64][32][32];
float layer1_1_bn1_bp_out[BATCH_SIZE][64][32][32];
float layer1_1_bn1_weight_act[64];
float layer1_1_bn1_bias_act[64];
float layer1_1_conv1_bp_out[BATCH_SIZE][64][32][32];
float layer1_1_shortcut_bp_out[BATCH_SIZE][64][32][32];
/* gradient */
float layer1_1_conv1_grad[64][64][3][3];
float layer1_1_conv2_grad[64][64][3][3];

///////// LAYER 2 /////////
// output size[BATCH_SIZE][128][16][16]
// layer 2_0
/* forward */
float layer2_0_conv1_weight[128][64][3][3];
float layer2_0_conv1_out[BATCH_SIZE][128][16][16];
float layer2_0_bn1_weight[128];
float layer2_0_bn1_bias[128];
float layer2_0_bn1_out[BATCH_SIZE][128][16][16];
float layer2_0_relu1_out[BATCH_SIZE][128][16][16];

float layer2_0_conv2_weight[128][128][3][3];
float layer2_0_conv2_out[BATCH_SIZE][128][16][16];
float layer2_0_bn2_weight[128];
float layer2_0_bn2_bias[128];
float layer2_0_bn2_out[BATCH_SIZE][128][16][16];
float layer2_0_relu2_out[BATCH_SIZE][128][16][16];
	// layer 2_0 downsample
float layer2_0_conv_sc_weight[128][64][1][1];
float layer2_0_conv_sc_out[BATCH_SIZE][128][16][16];
float layer2_0_bn_sc_weight[128];
float layer2_0_bn_sc_bias[128];
float layer2_0_bn_sc_out[BATCH_SIZE][128][16][16];
float layer2_0_shortcut_out[BATCH_SIZE][128][16][16];
/* backward */
float layer2_0_relu2_bp_out[BATCH_SIZE][128][16][16];
float layer2_0_bn2_bp_out[BATCH_SIZE][128][16][16];
float layer2_0_bn2_weight_act[128];
float layer2_0_bn2_bias_act[128];
float layer2_0_conv2_bp_out[BATCH_SIZE][128][16][16];

float layer2_0_relu1_bp_out[BATCH_SIZE][128][16][16];
float layer2_0_bn1_bp_out[BATCH_SIZE][128][16][16];
float layer2_0_bn1_weight_act[128];
float layer2_0_bn1_bias_act[128];
float layer2_0_conv1_bp_out[BATCH_SIZE][64][32][32];
	// layer 2_0 upsample
float layer2_0_bn_sw_sc_bp_out[BATCH_SIZE][128][16][16];
float layer2_0_bn_sc_weight_act[128];
float layer2_0_bn_sc_bias_act[128];
float layer2_0_conv_sc_bp_out[BATCH_SIZE][64][32][32];
float layer2_0_shortcut_bp_out[BATCH_SIZE][64][32][32];
/* gradient */
float layer2_0_conv_sc_grad[128][64][1][1];
float layer2_0_conv1_grad[128][64][3][3];
float layer2_0_conv2_grad[128][128][3][3];

// layer 2_1
/* forward */
float layer2_1_conv1_weight[128][128][3][3];
float layer2_1_conv1_out[BATCH_SIZE][128][16][16];
float layer2_1_bn1_weight[128];
float layer2_1_bn1_bias[128];
float layer2_1_bn1_out[BATCH_SIZE][128][16][16];
float layer2_1_relu1_out[BATCH_SIZE][128][16][16];

float layer2_1_conv2_weight[128][128][3][3];
float layer2_1_conv2_out[BATCH_SIZE][128][16][16];
float layer2_1_bn2_weight[128];
float layer2_1_bn2_bias[128];
float layer2_1_bn2_out[BATCH_SIZE][128][16][16];
float layer2_1_relu2_out[BATCH_SIZE][128][16][16];
float layer2_1_shortcut_out[BATCH_SIZE][128][16][16];
/* backward */
float layer2_1_relu2_bp_out[BATCH_SIZE][128][16][16];
float layer2_1_bn2_bp_out[BATCH_SIZE][128][16][16];
float layer2_1_bn2_weight_act[128];
float layer2_1_bn2_bias_act[128];
float layer2_1_conv2_bp_out[BATCH_SIZE][128][16][16];

float layer2_1_relu1_bp_out[BATCH_SIZE][128][16][16];
float layer2_1_bn1_bp_out[BATCH_SIZE][128][16][16];
float layer2_1_bn1_weight_act[128];
float layer2_1_bn1_bias_act[128];
float layer2_1_conv1_bp_out[BATCH_SIZE][128][16][16];
float layer2_1_shortcut_bp_out[BATCH_SIZE][128][16][16];
/* gradient */
float layer2_1_conv1_grad[128][128][3][3];
float layer2_1_conv2_grad[128][128][3][3];

///////// LAYER 3 /////////
// output size[BATCH_SIZE][256][8][8]
// layer 3_0
/* forward */
float layer3_0_conv1_weight[256][128][3][3];
float layer3_0_conv1_out[BATCH_SIZE][256][8][8];
float layer3_0_bn1_weight[256];
float layer3_0_bn1_bias[256];
float layer3_0_bn1_out[BATCH_SIZE][256][8][8];
float layer3_0_relu1_out[BATCH_SIZE][256][8][8];

float layer3_0_conv2_weight[256][256][3][3];
float layer3_0_conv2_out[BATCH_SIZE][256][8][8];
float layer3_0_bn2_weight[256];
float layer3_0_bn2_bias[256];
float layer3_0_bn2_out[BATCH_SIZE][256][8][8];
float layer3_0_relu2_out[BATCH_SIZE][256][8][8];
	// layer 3_0 downsample
float layer3_0_conv_sc_weight[256][128][1][1];
float layer3_0_conv_sc_out[BATCH_SIZE][256][8][8];
float layer3_0_bn_sc_weight[256];
float layer3_0_bn_sc_bias[256];
float layer3_0_bn_sc_out[BATCH_SIZE][256][8][8];
float layer3_0_shortcut_out[BATCH_SIZE][256][8][8];
/* backward */
float layer3_0_relu2_bp_out[BATCH_SIZE][256][8][8];
float layer3_0_bn2_bp_out[BATCH_SIZE][256][8][8];
float layer3_0_bn2_weight_act[256];
float layer3_0_bn2_bias_act[256];
float layer3_0_conv2_bp_out[BATCH_SIZE][256][8][8];

float layer3_0_relu1_bp_out[BATCH_SIZE][256][8][8];
float layer3_0_bn1_bp_out[BATCH_SIZE][256][8][8];
float layer3_0_bn1_weight_act[256];
float layer3_0_bn1_bias_act[256];
float layer3_0_conv1_bp_out[BATCH_SIZE][128][16][16];
	// layer 3_0 upsample
float layer3_0_bn_sw_sc_bp_out[BATCH_SIZE][256][8][8];
float layer3_0_bn_sc_weight_act[256];
float layer3_0_bn_sc_bias_act[256];
float layer3_0_conv_sc_bp_out[BATCH_SIZE][128][16][16];
float layer3_0_shortcut_bp_out[BATCH_SIZE][128][16][16];
/* gradient */
float layer3_0_conv_sc_grad[256][128][1][1];
float layer3_0_conv1_grad[256][128][3][3];
float layer3_0_conv2_grad[256][256][3][3];

// layer 3_1
/* forward */
float layer3_1_conv1_weight[256][256][3][3];
float layer3_1_conv1_out[BATCH_SIZE][256][8][8];
float layer3_1_bn1_weight[256];
float layer3_1_bn1_bias[256];
float layer3_1_bn1_out[BATCH_SIZE][256][8][8];
float layer3_1_relu1_out[BATCH_SIZE][256][8][8];

float layer3_1_conv2_weight[256][256][3][3];
float layer3_1_conv2_out[BATCH_SIZE][256][8][8];
float layer3_1_bn2_weight[256];
float layer3_1_bn2_bias[256];
float layer3_1_bn2_out[BATCH_SIZE][256][8][8];
float layer3_1_relu2_out[BATCH_SIZE][256][8][8];
float layer3_1_shortcut_out[BATCH_SIZE][256][8][8];
/* backward */
float layer3_1_relu2_bp_out[BATCH_SIZE][256][8][8];
float layer3_1_bn2_bp_out[BATCH_SIZE][256][8][8];
float layer3_1_bn2_weight_act[256];
float layer3_1_bn2_bias_act[256];
float layer3_1_conv2_bp_out[BATCH_SIZE][256][8][8];

float layer3_1_relu1_bp_out[BATCH_SIZE][256][8][8];
float layer3_1_bn1_bp_out[BATCH_SIZE][256][8][8];
float layer3_1_bn1_weight_act[256];
float layer3_1_bn1_bias_act[256];
float layer3_1_conv1_bp_out[BATCH_SIZE][256][8][8];
float layer3_1_shortcut_bp_out[BATCH_SIZE][256][8][8];
/* gradient */
float layer3_1_conv1_grad[256][256][3][3];
float layer3_1_conv2_grad[256][256][3][3];

///////// LAYER 4 /////////
// output size[BATCH_SIZE][512][4][4]
// layer 4_0
/* forward */
float layer4_0_conv1_weight[512][256][3][3];
float layer4_0_conv1_out[BATCH_SIZE][512][4][4];
float layer4_0_bn1_weight[512];
float layer4_0_bn1_bias[512];
float layer4_0_bn1_out[BATCH_SIZE][512][4][4];
float layer4_0_relu1_out[BATCH_SIZE][512][4][4];

float layer4_0_conv2_weight[512][512][3][3];
float layer4_0_conv2_out[BATCH_SIZE][512][4][4];
float layer4_0_bn2_weight[512];
float layer4_0_bn2_bias[512];
float layer4_0_bn2_out[BATCH_SIZE][512][4][4];
float layer4_0_relu2_out[BATCH_SIZE][512][4][4];
	// layer 4_0 downsample
float layer4_0_conv_sc_weight[512][256][1][1];
float layer4_0_conv_sc_out[BATCH_SIZE][512][4][4];
float layer4_0_bn_sc_weight[512];
float layer4_0_bn_sc_bias[512];
float layer4_0_bn_sc_out[BATCH_SIZE][512][4][4];
float layer4_0_shortcut_out[BATCH_SIZE][512][4][4];
/* backward */
float layer4_0_relu2_bp_out[BATCH_SIZE][512][4][4];
float layer4_0_bn2_bp_out[BATCH_SIZE][512][4][4];
float layer4_0_bn2_weight_act[512];
float layer4_0_bn2_bias_act[512];
float layer4_0_conv2_bp_out[BATCH_SIZE][512][4][4];

float layer4_0_relu1_bp_out[BATCH_SIZE][512][4][4];
float layer4_0_bn1_bp_out[BATCH_SIZE][512][4][4];
float layer4_0_bn1_weight_act[512];
float layer4_0_bn1_bias_act[512];
float layer4_0_conv1_bp_out[BATCH_SIZE][256][8][8];
	// layer 4_0 upsample
float layer4_0_bn_sw_sc_bp_out[BATCH_SIZE][512][4][4];
float layer4_0_bn_sc_weight_act[512];
float layer4_0_bn_sc_bias_act[512];
float layer4_0_conv_sc_bp_out[BATCH_SIZE][256][8][8];
float layer4_0_shortcut_bp_out[BATCH_SIZE][256][8][8];
/* gradient */
float layer4_0_conv_sc_grad[512][256][1][1];
float layer4_0_conv1_grad[512][256][3][3];
float layer4_0_conv2_grad[512][512][3][3];

// layer 4_1
/* forward */
float layer4_1_conv1_weight[512][512][3][3];
float layer4_1_conv1_out[BATCH_SIZE][512][4][4];
float layer4_1_bn1_weight[512];
float layer4_1_bn1_bias[512];
float layer4_1_bn1_out[BATCH_SIZE][512][4][4];
float layer4_1_relu1_out[BATCH_SIZE][512][4][4];

float layer4_1_conv2_weight[512][512][3][3];
float layer4_1_conv2_out[BATCH_SIZE][512][4][4];
float layer4_1_bn2_weight[512];
float layer4_1_bn2_bias[512];
float layer4_1_bn2_out[BATCH_SIZE][512][4][4];
float layer4_1_relu2_out[BATCH_SIZE][512][4][4];
float layer4_1_shortcut_out[BATCH_SIZE][512][4][4];
/* backward */
float layer4_1_relu2_bp_out[BATCH_SIZE][512][4][4];
float layer4_1_bn2_bp_out[BATCH_SIZE][512][4][4];
float layer4_1_bn2_weight_act[512];
float layer4_1_bn2_bias_act[512];
float layer4_1_conv2_bp_out[BATCH_SIZE][512][4][4];

float layer4_1_relu1_bp_out[BATCH_SIZE][512][4][4];
float layer4_1_bn1_bp_out[BATCH_SIZE][512][4][4];
float layer4_1_bn1_weight_act[512];
float layer4_1_bn1_bias_act[512];
float layer4_1_conv1_bp_out[BATCH_SIZE][512][4][4];
float layer4_1_shortcut_bp_out[BATCH_SIZE][512][4][4];
/* gradient */
float layer4_1_conv1_grad[512][512][3][3];
float layer4_1_conv2_grad[512][512][3][3];

// output size[BATCH_SIZE][512][1][1]
float avg_pool_out[BATCH_SIZE][512];
float error_avg_out[BATCH_SIZE][512][4][4];

// output size[BATCH_SIZE][10]
float classifier_out[BATCH_SIZE][10];
float error[BATCH_SIZE][10];
float error_fc_out[BATCH_SIZE][512];
float error_fc_out_T[512][BATCH_SIZE];
float fc_weight_grad[512][BATCH_SIZE];
float linear_weight[10][512];
float linear_weight_bw[512][10];

/* 
=======================
Parameterized templates
=======================
*/ 

// rot180_sw for Conv weights
template <int BATCH, int CHANNEL, int HEIGHT, int WIDTH>
void rot180_sw(
	float mat[BATCH][CHANNEL][HEIGHT][WIDTH],			// in
	float mat_rot[BATCH][CHANNEL][HEIGHT][WIDTH]		// out
)
{
	float mat_tmp[BATCH][CHANNEL][HEIGHT][WIDTH];

	for (int n=0; n<BATCH; n++){
        for (int c=0; c<CHANNEL; c++){
			for (int row=0; row<HEIGHT; row++){
				for (int col=0; col<WIDTH; col++){
					mat_tmp[n][c][row][col] = mat[n][c][HEIGHT-row-1][WIDTH-col-1];
					mat_rot[c][n][row][col] = mat_tmp[n][c][row][col];
				}
			}
		}
	}
}

// Batch Norm
template <int BATCH, int CHANNEL, int HEIGHT, int WIDTH>
void bn_sw(
	float bn_sw_inputs[BATCH][CHANNEL][HEIGHT][WIDTH],		// in
	float bn_sw_outputs[BATCH][CHANNEL][HEIGHT][WIDTH],		// out
	float gamma[CHANNEL],
	float beta[CHANNEL]
)
{
    int N = BATCH * HEIGHT * WIDTH;
	float mu[CHANNEL];
	float sigma[CHANNEL];

    for (int n=0; n<BATCH; n++){
        for (int c=0; c<CHANNEL; c++){
            for (int row=0; row<HEIGHT; row++){
                for (int col=0; col<WIDTH; col++){
                    mu[c] = mu[c] + bn_sw_inputs[n][c][row][col]/N;
				}
			}
		}
	}
    for (int n=0; n<BATCH; n++){
        for (int c=0; c<CHANNEL; c++){
            for (int row=0; row<HEIGHT; row++){
                for (int col=0; col<WIDTH; col++){
                    sigma[c] = sigma[c] + pow(bn_sw_inputs[n][c][row][col]-mu[c],2);
				}
			}
		}
	}
	sigma = sqrt(sigma/N);
	for (int n=0; n<BATCH; n++){
		for (int c=0; c<CHANNEL; c++){
			for (int row=0; row<HEIGHT; row++){
				for (int col=0; col<WIDTH; col++){
            		bn_sw_outputs[n][c][row][col] = gamma[c]*(bn_sw_inputs[n][c][row][col]-mu[c])/sigma[c] + beta[c];
				}
			}
		}
	}			
}

// Batch Norm Back-prop
template <int BATCH, int CHANNEL, int HEIGHT, int WIDTH>
void bn_bp_sw(
	float error[BATCH][CHANNEL][HEIGHT][WIDTH], 			// in
	float bn_sw_inputs_fw[BATCH][CHANNEL][HEIGHT][WIDTH],	// in
	float error_bn_sw[BATCH][CHANNEL][HEIGHT][WIDTH],		// out

	float gamma[CHANNEL],									// in
	float g_gamma[CHANNEL],									// out
	float g_beta[CHANNEL]									// out
)
{
    int N = BATCH * HEIGHT * WIDTH;
	float mu[CHANNEL];
	float sigma[CHANNEL];

    for (int n=0; n<BATCH; n++){
        for (int c=0; c<CHANNEL; c++){
            for (int row=0; row<HEIGHT; row++){
                for (int col=0; col<WIDTH; col++){
                    mu[c] = mu[c] + bn_sw_inputs_fw[n][c][row][col]/N;
				}
			}
		}
	}
    for (int n=0; n<BATCH; n++){
        for (int c=0; c<CHANNEL; c++){
            for (int row=0; row<HEIGHT; row++){
                for (int col=0; col<WIDTH; col++){
                    sigma[c] = sigma[c] + pow(bn_sw_inputs_fw[n][c][row][col]-mu[c],2);
				}
			}
		}
	}
	sigma = sqrt(sigma/N);
    for (int n=0; n<BATCH; n++){
        for (int c=0; c<CHANNEL; c++){
            for (int row=0; row<HEIGHT; row++){
                for (int col=0; col<WIDTH; col++){
                    g_beta[c] = g_beta[c] + error[n][c][row][col];
                    g_gamma[c] = g_gamma[c] + error[n][c][row][col]*(bn_sw_inputs_fw[n][c][row][col]-mu[c])/sigma[c];
				}
			}
		}
	}
    for (int n=0; n<BATCH; n++){
        for (int c=0; c<CHANNEL; c++){
            for (int row=0; row<HEIGHT; row++){
                for (int col=0; col<WIDTH; col++){
            		error_bn_sw[n][c][row][col] = gamma[c]*error[n][c][row][col]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (bn_sw_inputs_fw[n][c][row][col]-mu[c])*g_gamma[c]/(N*gamma[c]*(pow(sigma[c],2)));
				}
			}
		}
	}			
}

// relu_sw
template <int BATCH, int CHANNEL, int HEIGHT, int WIDTH>
void relu_sw(
	float input[BATCH][CHANNEL][HEIGHT][WIDTH],   // in
	float output[BATCH][CHANNEL][HEIGHT][WIDTH]   // out
)
{
	for (int n=0; n<BATCH; n++) {
		for (int c=0; c<CHANNEL; c++) {
			for (int row=0; row<HEIGHT; row++) {
				for (int col=0; col<WIDTH; col++) {
					if (input[n][c][row][col] > 0) {
						output[n][c][row][col] = input[n][c][row][col];
					} else {
						output[n][c][row][col] = 0;
					}
				}
			}
		}
	}
}

// relu_sw Back-prop
template <int BATCH, int CHANNEL, int HEIGHT, int WIDTH>
void relu_bp_sw(
	float error[BATCH][CHANNEL][HEIGHT][WIDTH],  		// error in
	float input_fw[BATCH][CHANNEL][HEIGHT][WIDTH], 		// activation in forward
	float error_relu_sw[BATCH][CHANNEL][HEIGHT][WIDTH]     // error out
)
{
	for (int n=0; n<BATCH; n++) {
		for (int c=0; c<CHANNEL; c++) {
			for (int row=0; row<HEIGHT; row++) {
				for (int col=0; col<WIDTH; col++) {
					if (input_fw[n][c][row][col] > 0) {
						error_relu_sw[n][c][row][col] = error[n][c][row][col];
					} else {
						error_relu_sw[n][c][row][col] = 0;
					}
				}
			}
		}
	}
}

// AvgPool
template <int BATCH, int CHANNEL, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void avgpool(
	float avg_inputs[BATCH][CHANNEL][HEIGHT_IN][WIDTH_IN],
	float avg_outputs[BATCH][CHANNEL][HEIGHT_OUT][WIDTH_OUT],
	int stride
)
{
	// int HEIGHT_OUT = HEIGHT_IN/stride;
	// int WIDTH_OUT = WIDTH_IN/stride;
	int stride2= pow(stride,2);

	for (int n=0; n<BATCH; n++) {
		for (int c=0; c<CHANNEL; c++) {
			for (int row=0; row<HEIGHT_OUT; row++) {
				for (int col=0; col<WIDTH_OUT; col++) {
					for (int s=0; s<stride; s++) {
						for (int ss=0; ss<stride; s++) {
							avg_outputs[n][c][row][col] += avg_inputs[n][c][stride*row+s][stride*col+ss]/stride2;
						}
					}
				}
			}
		}
	}
}

// AvgPool Backprop
template <int BATCH, int CHANNEL, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void avgpool_bp(
	float error[BATCH][CHANNEL][HEIGHT_IN][WIDTH_IN],		 // in
	float error_avg[BATCH][CHANNEL][HEIGHT_OUT][WIDTH_OUT],	 // out
	int stride
)
{
	// int HEIGHT_OUT = HEIGHT_IN*stride;
	// int WIDTH_OUT = WIDTH_IN*stride;
	int stride2 = pow(stride,2);

	for (int n=0; n<BATCH; n++) {
		for (int c=0; c<CHANNEL; c++) {
			for (int row=0; row<HEIGHT_OUT; row++) {
				for (int col=0; col<WIDTH_OUT; col++) {
					for (int s=0; s<stride; s++) {
						for (int ss=0; ss<stride; ss++) {
							error_avg[n][c][stride*row+s][stride*col+ss] = error[n][c][row][col]/stride2;
						}
					}
				}
			}
		}
	}
}

// shortcut_sw- identity branch
template <int BATCH, int CHANNEL, int HEIGHT, int WIDTH>
void shortcut_sw(
	float input_a[BATCH][CHANNEL][HEIGHT][WIDTH],
	float input_b[BATCH][CHANNEL][HEIGHT][WIDTH],
	float output[BATCH][CHANNEL][HEIGHT][WIDTH]
)
{
	for (int n = 0; n < BATCH; n++) {
		for (int c = 0; c < CHANNEL; c++) {
			for (int row = 0; row < HEIGHT; row++) {
				for (int col = 0; col < WIDTH; col++) {
					float x = input_a[n][c][row][col];
					float y = input_b[n][c][row][col];
					output[n][c][row][col] = x + y;
				}
			}
		}
	}
}

// ============
// Conv forward
// ============

// conv_3x3_sw, padding=1
template <int BATCH_IN, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void conv_3x3_sw
(
	float input[BATCH_IN][CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
	float weight[CHANNEL_OUT][CHANNEL_IN][3][3],
	float output[BATCH_IN][CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT],
	int stride
)
{
	// input 0-padding(1, 1, 1, 1)
	float input_pad[BATCH_IN][CHANNEL_IN][HEIGHT_IN+2][WIDTH_IN+2];
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < HEIGHT_IN; row++) {
				for (int col = 0; col < WIDTH_IN; col++) {
					input_pad[bi][co][row+1][col+1] = input[bi][co][row][col];
				}
			}
		}
	}

	// conv
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < HEIGHT_OUT; row++) {
				for (int col = 0; col < WIDTH_OUT; col++) {
					float accum = 0;
					for (int ci = 0; ci < CHANNEL_IN; ci++) {
						for (int krow = 0; krow < 3; krow++) {
							for (int kcol = 0; kcol < 3; kcol++) {
								int row_in = row*stride + krow;
								int col_in = col*stride + kcol;
								accum += input_pad[bi][ci][row_in][col_in] * weight[co][ci][krow][kcol];
							}
						}
					}
					output[bi][co][row][col] = accum;
				}
			}
		}
	}
}

// Conv_1x1, padding=0
template <int BATCH_IN, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void conv_1x1_sw
(
	float input[BATCH_IN][CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
	float weight[CHANNEL_OUT][CHANNEL_IN][1][1],
	float output[BATCH_IN][CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT],
	int stride
)
{
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < HEIGHT_OUT; row++) {
				for (int col = 0; col < WIDTH_OUT; col++) {
					float accum = 0;
					for (int ci = 0; ci < CHANNEL_IN; ci++) {
						int row_in = row*stride;
						int col_in = col*stride;
						accum += input[bi][ci][row_in][col_in] * weight[co][ci][0][0];
					}
					output[bi][co][row][col] = accum;
				}
			}
		}
	}
}

// =============
// Conv backward
// =============

// conv_3x3_sw, padding=1
template <int BATCH_IN, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void conv_3x3_sw_bp
(
	float input[BATCH_IN][CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
	float weight[CHANNEL_OUT][CHANNEL_IN][3][3],
	float output[BATCH_IN][CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT],
	int stride
)
{
	// input dilation and 0-padding(1, 1+A, 1, 1+A)
	int A = (HEIGHT_IN*stride - 1) % stride;
	float input_pad[BATCH_IN][CHANNEL_IN][stride*(HEIGHT_IN-1)+1 + 2+A][stride*(WIDTH_IN-1)+1 + 2+A];
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < HEIGHT_IN; row++) {
				for (int col = 0; col < WIDTH_IN; col++) {
					input_pad[bi][co][row*stride + 1][col*stride + 1] = input[bi][co][row][col];
				}
			}
		}
	}

	// conv
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < HEIGHT_OUT; row++) {
				for (int col = 0; col < WIDTH_OUT; col++) {
					float accum = 0;
					for (int ci = 0; ci < CHANNEL_IN; ci++) {
						for (int krow = 0; krow < 3; krow++) {
							for (int kcol = 0; kcol < 3; kcol++) {
								int row_in = row + krow;
								int col_in = col + kcol;
								accum += input_pad[bi][ci][row_in][col_in] * weight[co][ci][krow][kcol];
							}
						}
					}
					output[bi][co][row][col] = accum;
				}
			}
		}
	}
}

// Conv_1x1, padding=0
template <int BATCH_IN, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void conv_1x1_sw_bp
(
	float input[BATCH_IN][CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
	float weight[CHANNEL_OUT][CHANNEL_IN][1][1],
	float output[BATCH_IN][CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT],
	int stride
)
{
	// input dilation
	int A = (HEIGHT_IN*stride - 1) % stride;
	float input_pad[BATCH_IN][CHANNEL_IN][stride*(HEIGHT_IN-1)+1][stride*(WIDTH_IN-1)+1];
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < HEIGHT_IN; row++) {
				for (int col = 0; col < WIDTH_IN; col++) {
					input_pad[bi][co][row*stride][col*stride] = input[bi][co][row][col];	// no 0-padding
				}
			}
		}
	}

	// conv
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < HEIGHT_OUT; row++) {
				for (int col = 0; col < WIDTH_OUT; col++) {
					float accum = 0;
					for (int ci = 0; ci < CHANNEL_IN; ci++) {
						int row_in = row;
						int col_in = col;
						accum += input_pad[bi][ci][row_in][col_in] * weight[co][ci][0][0];
					}
					output[bi][co][row][col] = accum;
				}
			}
		}
	}
}

// =============
// Conv gradient
// =============

// conv_3x3_sw_grad, padding=1
template <int BATCH_IN, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT, int WIDTH_KER>
void conv_3x3_sw_grad
(
	float input[BATCH_IN][CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
	float weight[BATCH_IN][CHANNEL_OUT][WIDTH_KER][WIDTH_KER],		// WIDTH_KER not as input parameter yet
	float output[CHANNEL_OUT][CHANNEL_IN][HEIGHT_OUT][WIDTH_OUT],
	int stride
)
{
	// weight dilation, k_dil = 1 + (k-1)*s
	const int KERNEL_DIL = (WIDTH_KER-1)*stride + 1;
	float weight_dil[BATCH_IN][CHANNEL_OUT][KERNEL_DIL][KERNEL_DIL];
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int krow = 0; krow < WIDTH_KER; krow++) {
				for (int kcol = 0; kcol < WIDTH_KER; kcol++) {
					weight_dil[bi][co][krow*stride][kcol*stride] = weight[bi][co][krow][kcol];
				}
			}
		}
	}

	// input 0-padding(1, 1+A, 1, 1+A)
	int A = (HEIGHT_IN*stride + 2 - KERNEL_DIL) % stride;
	float input_pad[BATCH_IN][CHANNEL_IN][HEIGHT_IN + 2+A][WIDTH_IN-1 + 2+A];
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < HEIGHT_IN; row++) {
				for (int col = 0; col < WIDTH_IN; col++) {
					input_pad[bi][co][row + 1][col + 1] = input[bi][co][row][col];
				}
			}
		}
	}

	// conv
	for (int co = 0; co < CHANNEL_OUT; co++) {
		for (int ci = 0; ci < CHANNEL_IN; ci++) {
			for (int row = 0; row < HEIGHT_OUT; row++) {
				for (int col = 0; col < WIDTH_OUT; col++) {
					float accum = 0;
					for (int bi = 0; bi < BATCH_IN; ci++) {
						for (int krow = 0; krow < KERNEL_DIL; krow++) {
							for (int kcol = 0; kcol < KERNEL_DIL; kcol++) {
								int row_in = row*stride + krow;
								int col_in = col*stride + kcol;
								accum += input_pad[bi][ci][row_in][col_in] * weight_dil[bi][co][krow][kcol];
							}
						}
					}
					output[co][ci][row][col] = accum;
				}
			}
		}
	}
}

// Conv_1x1_grad, padding=0
template <int BATCH_IN, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT, int WIDTH_KER>
void conv_1x1_sw_grad
(
	float input[BATCH_IN][CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
	float weight[BATCH_IN][CHANNEL_OUT][WIDTH_KER][WIDTH_KER],		// WIDTH_KER not as input parameter yet
	float output[CHANNEL_OUT][CHANNEL_IN][HEIGHT_OUT][WIDTH_OUT],
	int stride
)
{
	// weight dilation, k_dil = 1 + (k-1)*s
	const int KERNEL_DIL = (WIDTH_KER-1)*stride + 1;
	float weight_dil[BATCH_IN][CHANNEL_OUT][KERNEL_DIL][KERNEL_DIL];
	for (int bi = 0; bi < BATCH_IN; bi++) {
		for (int co = 0; co < CHANNEL_OUT; co++) {
			for (int row = 0; row < WIDTH_KER; row++) {
				for (int col = 0; col < WIDTH_KER; col++) {
					weight_dil[bi][co][row*stride][col*stride] = weight[bi][co][row][col];
				}
			}
		}
	}

	// conv
	for (int co = 0; co < CHANNEL_OUT; co++) {
		for (int ci = 0; ci < CHANNEL_IN; ci++) {
			for (int row = 0; row < HEIGHT_OUT; row++) {
				for (int col = 0; col < WIDTH_OUT; col++) {
					float accum = 0;
					for (int bi = 0; bi < BATCH_IN; ci++) {
						for (int krow = 0; krow < KERNEL_DIL; krow++) {
							for (int kcol = 0; kcol < KERNEL_DIL; kcol++) {
								int row_in = row*stride + krow;
								int col_in = col*stride + kcol;
								accum += input[bi][ci][row_in][col_in] * weight_dil[bi][co][krow][kcol];
							}
						}
					}
					output[co][ci][row][col] = accum;
				}
			}
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


// #define BATCH_SIZE 4
// #define lr 0.01

//--------------------
//	  Forward GeMM
//--------------------

void forward(float image[BATCH_SIZE][3][32][32]) 
{
	// conv + bn_sw + relu_sw
	conv_3x3_sw<BATCH_SIZE, 3, 64, 32, 32, 32, 32>(image, conv1_weight, conv1_out, 1);
	bn_sw<BATCH_SIZE, 64, 32, 32>(conv1_out, bn1_out, bn1_weight, bn1_bias);
	relu_sw<BATCH_SIZE, 64, 32, 32>(bn1_out, relu1_out);

	////////////////////////////////////
	//////////// LAYER 1 ///////////////
	////////////////////////////////////

	// layer 1_0
	conv_3x3_sw<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(relu1_out, layer1_0_conv1_weight, layer1_0_conv1_out, 1);
	bn_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_conv1_out, layer1_0_bn1_out, layer1_0_bn1_weight, layer1_0_bn1_bias);
	relu_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_bn1_out, layer1_0_relu1_out);

	conv_3x3_sw<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(layer1_0_relu1_out, layer1_0_conv2_weight, layer1_0_conv2_out, 1);
	bn_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_conv2_out, layer1_0_bn2_out, layer1_0_bn2_weight, layer1_0_bn2_bias);
	relu_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_bn2_out, layer1_0_relu2_out);
	shortcut_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_relu2_out, relu1_out, layer1_0_shortcut_out);

	// layer 1_1
	conv_3x3_sw<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(layer1_0_shortcut_out, layer1_1_conv1_weight, layer1_1_conv1_out, 1);
	bn_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_conv1_out, layer1_1_bn1_out, layer1_1_bn1_weight, layer1_1_bn1_bias);
	relu_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_bn1_out, layer1_1_relu1_out);
	
	conv_3x3_sw<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(layer1_1_relu1_out, layer1_1_conv2_weight, layer1_1_conv2_out, 1);
	bn_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_conv2_out, layer1_1_bn2_out, layer1_1_bn2_weight, layer1_1_bn2_bias);
	relu_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_bn2_out, layer1_1_relu2_out);
	shortcut_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_relu2_out, layer1_0_shortcut_out, layer1_1_shortcut_out);

	////////////////////////////////////
	//////////// LAYER 2 ///////////////
	////////////////////////////////////

	// layer 2 downsample (conv1 & conv_sc stride=2)
	// layer 2_0
	conv_3x3_sw<BATCH_SIZE, 64, 128, 32, 32, 16, 16>(layer1_1_shortcut_out, layer2_0_conv1_weight, layer2_0_conv1_out, 2);
	bn_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_conv1_out, layer2_0_bn1_out, layer2_0_bn1_weight, layer2_0_bn1_bias);
	relu_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_bn1_out, layer2_0_relu1_out);

	conv_3x3_sw<BATCH_SIZE, 128, 128, 16, 16, 16, 16>(layer2_0_relu1_out, layer2_0_conv2_weight, layer2_0_conv2_out, 1);
	bn_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_conv2_out, layer2_0_bn2_out, layer2_0_bn2_weight, layer2_0_bn2_bias);
	relu_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_bn2_out, layer2_0_relu2_out);
	// shortcut_sw: conv_1x1 + bn_sw
	conv_1x1_sw<BATCH_SIZE, 64, 128, 32, 32, 16, 16>(layer1_1_shortcut_out, layer2_0_conv_sc_weight, layer2_0_conv_sc_out, 2);
	bn_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_conv_sc_out, layer2_0_bn_sc_out, layer2_0_bn_sc_weight, layer2_0_bn_sc_bias);
	shortcut_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_relu2_out, layer2_0_bn_sc_out, layer2_0_shortcut_out);

	// layer 2_1
	conv_3x3_sw<BATCH_SIZE, 128, 128, 16, 16, 16, 16>(layer2_0_shortcut_out, layer2_1_conv1_weight, layer2_1_conv1_out, 1);
	bn_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_conv1_out, layer2_1_bn1_out, layer2_1_bn1_weight, layer2_1_bn1_bias);
	relu_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_bn1_out, layer2_1_relu1_out);
	
	conv_3x3_sw<BATCH_SIZE, 128, 128, 16, 16, 16, 16>(layer2_1_relu1_out, layer2_1_conv2_weight, layer2_1_conv2_out, 1);
	bn_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_conv2_out, layer2_1_bn2_out, layer2_1_bn2_weight, layer2_1_bn2_bias);
	relu_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_bn2_out, layer2_1_relu2_out);
	shortcut_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_relu2_out, layer2_0_shortcut_out, layer2_1_shortcut_out);

	////////////////////////////////////
	//////////// LAYER 3 ///////////////
	////////////////////////////////////

	// layer 3 downsample (conv1 & conv_sc stride=2)
	// layer 3_0
	conv_3x3_sw<BATCH_SIZE, 128, 256, 16, 16, 8, 8>(layer2_1_shortcut_out, layer3_0_conv1_weight, layer3_0_conv1_out, 2);
	bn_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_conv1_out, layer3_0_bn1_out, layer3_0_bn1_weight, layer3_0_bn1_bias);
	relu_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_bn1_out, layer3_0_relu1_out);

	conv_3x3_sw<BATCH_SIZE, 256, 256, 8, 8, 8, 8>(layer3_0_relu1_out, layer3_0_conv2_weight, layer3_0_conv2_out, 1);
	bn_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_conv2_out, layer3_0_bn2_out, layer3_0_bn2_weight, layer3_0_bn2_bias);
	relu_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_bn2_out, layer3_0_relu2_out);
	// shortcut_sw: conv_1x1 + bn_sw
	conv_1x1_sw<BATCH_SIZE, 128, 256, 16, 16, 8, 8>(layer2_1_shortcut_out, layer3_0_conv_sc_weight, layer3_0_conv_sc_out, 2);
	bn_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_conv_sc_out, layer3_0_bn_sc_out, layer3_0_bn_sc_weight, layer3_0_bn_sc_bias);
	shortcut_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_relu2_out, layer3_0_bn_sc_out, layer3_0_shortcut_out);

	// layer 3_1
	conv_3x3_sw<BATCH_SIZE, 256, 256, 8, 8, 8, 8>(layer3_0_shortcut_out, layer3_1_conv1_weight, layer3_1_conv1_out, 1);
	bn_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_conv1_out, layer3_1_bn1_out, layer3_1_bn1_weight, layer3_1_bn1_bias);
	relu_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_bn1_out, layer3_1_relu1_out);
	
	conv_3x3_sw<BATCH_SIZE, 256, 256, 8, 8, 8, 8>(layer3_1_relu1_out, layer3_1_conv2_weight, layer3_1_conv2_out, 1);
	bn_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_conv2_out, layer3_1_bn2_out, layer3_1_bn2_weight, layer3_1_bn2_bias);
	relu_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_bn2_out, layer3_1_relu2_out);
	shortcut_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_relu2_out, layer3_0_shortcut_out, layer3_1_shortcut_out);

	////////////////////////////////////
	//////////// LAYER 4 ///////////////
	////////////////////////////////////

	// layer 4 downsample (conv1 & conv_sc stride=2)
	// layer 4_0
	conv_3x3_sw<BATCH_SIZE, 256, 512, 8, 8, 4, 4>(layer3_1_shortcut_out, layer4_0_conv1_weight, layer4_0_conv1_out, 2);
	bn_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_conv1_out, layer4_0_bn1_out, layer4_0_bn1_weight, layer4_0_bn1_bias);
	relu_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_bn1_out, layer4_0_relu1_out);

	conv_3x3_sw<BATCH_SIZE, 512, 512, 4, 4, 4, 4>(layer4_0_relu1_out, layer4_0_conv2_weight, layer4_0_conv2_out, 1);
	bn_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_conv2_out, layer4_0_bn2_out, layer4_0_bn2_weight, layer4_0_bn2_bias);
	relu_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_bn2_out, layer4_0_relu2_out);
	// shortcut_sw: conv_1x1 + bn_sw
	conv_1x1_sw<BATCH_SIZE, 256, 512, 8, 8, 4, 4>(layer3_1_shortcut_out, layer4_0_conv_sc_weight, layer4_0_conv_sc_out, 2);
	bn_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_conv_sc_out, layer4_0_bn_sc_out, layer4_0_bn_sc_weight, layer4_0_bn_sc_bias);
	shortcut_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_relu2_out, layer4_0_bn_sc_out, layer4_0_shortcut_out);

	// layer 4_1
	conv_3x3_sw<BATCH_SIZE, 512, 512, 4, 4, 4, 4>(layer4_0_shortcut_out, layer4_1_conv1_weight, layer4_1_conv1_out, 1);
	bn_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_conv1_out, layer4_1_bn1_out, layer4_1_bn1_weight, layer4_1_bn1_bias);
	relu_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_bn1_out, layer4_1_relu1_out);
	
	conv_3x3_sw<BATCH_SIZE, 512, 512, 4, 4, 4, 4>(layer4_1_relu1_out, layer4_1_conv2_weight, layer4_1_conv2_out, 1);
	bn_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_conv2_out, layer4_1_bn2_out, layer4_1_bn2_weight, layer4_1_bn2_bias);
	relu_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_bn2_out, layer4_1_relu2_out);
	shortcut_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_relu2_out, layer4_0_shortcut_out, layer4_1_shortcut_out);

	// AvgPool_4x4
	for (int b = 0; b < BATCH_SIZE; b ++){
		for (int c = 0; c < 512; c ++) {
			float m = 0;
			for (int row = 0; row < 4; row ++) {
				for (int col = 0; col < 4; col ++) {
					m += layer4_1_shortcut_out[b][c][row][col];
				}
			}
			avg_pool_out[b][c] = m/16.0;	// output size <BATCH_SIZE, 512, 1, 1>
		}
	}

	// FC
	for (int b = 0; b < BATCH_SIZE; b ++){
		for (int row = 0; row < 10; row ++) {
			float m = 0;
			for (int col = 0; col < 512; col ++) {
				m += avg_pool_out[b][col] * linear_weight[row][col];
			}
			classifier_out[b][row] = m; // + linear_bias[row];	// output size <BATCH_SIZE, 10>
		}
	}
}

//--------------------
//	CrossEntropyLoss
//--------------------

// Note that we shall also calculate nn.CrossEntropyLoss from classifier_out[BATCH_SIZE][10]

//--------------------
//	 Backward GeMM
//--------------------
void backward(float error[BATCH_SIZE][10]) // right size ?
{
	// FC backprop
	// error in size of <BATCH_SIZE, 10>
	/* 
	   e_fc = torch.matmul(e_L, weight)
	   g_fc = torch.matmul(e_L.T, input)
	*/
	for (int b = 0; b < BATCH_SIZE; b ++){
		for (int row = 0; row < 512; row ++) {
			float m = 0;
			for (int col = 0; col < 10; col ++) {
				error_fc_out[b][row] += error[b][col] * linear_weight_bw[row][col];			// linear_weight_bw = nn.transpose(linear_weight)
			}
		}
	}
	/* FC weight update */
	for (int b = 0; b < BATCH_SIZE; b ++){
		for (int row = 0; row < 512; row ++) {
			float m = 0;
			for (int col = 0; col < 10; col ++) {
				fc_weight_grad[col][row] += avg_pool_out[b][row] * error_fc_out_T[row][b];	// linear_weight_bw = nn.transpose(linear_weight)
			}
		}
	}

	// AvgPool_4x4 backprop
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int c = 0; c < 512; c ++) {
			for (int row = 0; row < 4; row++) {
				for (int col = 0; col < 4; col ++) {
					for (int s = 0; s < 4; s ++) {
						for (int ss = 0; ss < 4; ss ++) {
							error_avg_out[b][c][4*row + s][4*col + ss] = error_fc_out[b][c]/16.0;	
							// output size <BATCH_SIZE, 512, 4, 4>
						}
					}
				}
			}
		}
	}

	////////////////////////////////////
	///////// LAYER 4 backprop /////////
	////////////////////////////////////

	// layer 4_1
	relu_bp_sw<BATCH_SIZE, 512, 4, 4>(error_avg_out, layer4_1_bn2_out, layer4_1_relu2_bp_out);
	bn_bp_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_relu2_bp_out, layer4_1_conv2_out, layer4_1_bn2_bp_out, layer4_1_bn2_weight, layer4_1_bn2_weight_act, layer4_1_bn2_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 512, 512, 4, 4, 4, 4>(layer4_1_bn2_bp_out, layer4_1_conv2_weight, layer4_1_conv2_bp_out, 1);

	relu_bp_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_conv2_bp_out, layer4_1_bn1_out, layer4_1_relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_relu1_bp_out, layer4_1_conv1_out, layer4_1_bn1_bp_out, layer4_1_bn1_weight, layer4_1_bn1_weight_act, layer4_1_bn1_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 512, 512, 4, 4, 4, 4>(layer4_1_bn1_bp_out, layer4_1_conv1_weight, layer4_1_conv1_bp_out, 1);
	shortcut_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_conv1_bp_out, error_avg_out, layer4_1_shortcut_bp_out);

	// layer 4 downsample (conv1_bp & conv_sc_bp stride=2)
	// layer 4_0
	relu_bp_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_shortcut_bp_out, layer4_0_bn2_out, layer4_0_relu2_bp_out);
	bn_bp_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_relu2_bp_out, layer4_0_conv2_out, layer4_0_bn2_bp_out, layer4_0_bn2_weight, layer4_0_bn2_weight_act, layer4_0_bn2_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 512, 512, 4, 4, 4, 4>(layer4_0_bn2_bp_out, layer4_0_conv2_weight, layer4_0_conv2_bp_out, 1);

	relu_bp_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_conv2_bp_out, layer4_0_bn1_out, layer4_0_relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 512, 4, 4>(layer4_0_relu1_bp_out, layer4_0_conv1_out, layer4_0_bn1_bp_out, layer4_0_bn1_weight, layer4_0_bn1_weight_act, layer4_0_bn1_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 512, 256, 4, 4, 8, 8>(layer4_0_bn1_bp_out, layer4_0_conv1_weight, layer4_0_conv1_bp_out, 2);
	// shortcut_sw bn_sw + conv_1x1
	bn_bp_sw<BATCH_SIZE, 512, 4, 4>(layer4_1_shortcut_bp_out, layer4_0_conv_sc_out, layer4_0_bn_sw_sc_bp_out, layer4_0_bn_sc_weight, layer4_0_bn_sc_weight_act, layer4_0_bn_sc_bias_act);
	conv_1x1_sw_bp<BATCH_SIZE, 512, 256, 4, 4, 8, 8>(layer4_0_bn_sw_sc_bp_out, layer4_0_conv_sc_weight, layer4_0_conv_sc_bp_out, 2);
	shortcut_sw<BATCH_SIZE, 256, 8, 8>(layer4_0_conv_sc_bp_out, layer4_0_conv1_bp_out, layer4_0_shortcut_bp_out);

	////////////////////////////////////
	///////// LAYER 3 backprop /////////
	////////////////////////////////////

	// layer 3_1
	relu_bp_sw<BATCH_SIZE, 256, 8, 8>(layer4_0_shortcut_bp_out, layer3_1_bn2_out, layer3_1_relu2_bp_out);
	bn_bp_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_relu2_bp_out, layer3_1_conv2_out, layer3_1_bn2_bp_out, layer3_1_bn2_weight, layer3_1_bn2_weight_act, layer3_1_bn2_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 256, 256, 8, 8, 8, 8>(layer3_1_bn2_bp_out, layer3_1_conv2_weight, layer3_1_conv2_bp_out, 1);

	relu_bp_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_conv2_bp_out, layer3_1_bn1_out, layer3_1_relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_relu1_bp_out, layer3_1_conv1_out, layer3_1_bn1_bp_out, layer3_1_bn1_weight, layer3_1_bn1_weight_act, layer3_1_bn1_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 256, 256, 8, 8, 8, 8>(layer3_1_bn1_bp_out, layer3_1_conv1_weight, layer3_1_conv1_bp_out, 1);
	shortcut_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_conv1_bp_out, layer4_0_shortcut_bp_out, layer3_1_shortcut_bp_out);

	// layer 3 downsample (conv1_bp & conv_sc_bp stride=2)
	// layer 3_0
	relu_bp_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_shortcut_bp_out, layer3_0_bn2_out, layer3_0_relu2_bp_out);
	bn_bp_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_relu2_bp_out, layer3_0_conv2_out, layer3_0_bn2_bp_out, layer3_0_bn2_weight, layer3_0_bn2_weight_act, layer3_0_bn2_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 256, 256, 8, 8, 8, 8>(layer3_0_bn2_bp_out, layer3_0_conv2_weight, layer3_0_conv2_bp_out, 1);

	relu_bp_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_conv2_bp_out, layer3_0_bn1_out, layer3_0_relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 256, 8, 8>(layer3_0_relu1_bp_out, layer3_0_conv1_out, layer3_0_bn1_bp_out, layer3_0_bn1_weight, layer3_0_bn1_weight_act, layer3_0_bn1_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 256, 128, 8, 8, 16, 16>(layer3_0_bn1_bp_out, layer3_0_conv1_weight, layer3_0_conv1_bp_out, 2);
	// shortcut_sw bn_sw + conv_1x1
	bn_bp_sw<BATCH_SIZE, 256, 8, 8>(layer3_1_shortcut_bp_out, layer3_0_conv_sc_out, layer3_0_bn_sw_sc_bp_out, layer3_0_bn_sc_weight, layer3_0_bn_sc_weight_act, layer3_0_bn_sc_bias_act);
	conv_1x1_sw_bp<BATCH_SIZE, 256, 128, 8, 8, 16, 16>(layer3_0_bn_sw_sc_bp_out, layer3_0_conv_sc_weight, layer3_0_conv_sc_bp_out, 2);
	shortcut_sw<BATCH_SIZE, 128, 16, 16>(layer3_0_conv_sc_bp_out, layer3_0_conv1_bp_out, layer3_0_shortcut_bp_out);

	////////////////////////////////////
	///////// LAYER 2 backprop /////////
	////////////////////////////////////

	// layer 2_1
	relu_bp_sw<BATCH_SIZE, 128, 16, 16>(layer3_0_shortcut_bp_out, layer2_1_bn2_out, layer2_1_relu2_bp_out);
	bn_bp_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_relu2_bp_out, layer2_1_conv2_out, layer2_1_bn2_bp_out, layer2_1_bn2_weight, layer2_1_bn2_weight_act, layer2_1_bn2_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 128, 128, 16, 16, 16, 16>(layer2_1_bn2_bp_out, layer2_1_conv2_weight, layer2_1_conv2_bp_out, 1);

	relu_bp_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_conv2_bp_out, layer2_1_bn1_out, layer2_1_relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_relu1_bp_out, layer2_1_conv1_out, layer2_1_bn1_bp_out, layer2_1_bn1_weight, layer2_1_bn1_weight_act, layer2_1_bn1_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 128, 128, 16, 16, 16, 16>(layer2_1_bn1_bp_out, layer2_1_conv1_weight, layer2_1_conv1_bp_out, 1);
	shortcut_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_conv1_bp_out, layer3_0_shortcut_bp_out, layer2_1_shortcut_bp_out);

	// layer 2 downsample (conv1_bp & conv_sc_bp stride=2)
	// layer 2_0
	relu_bp_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_shortcut_bp_out, layer2_0_bn2_out, layer2_0_relu2_bp_out);
	bn_bp_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_relu2_bp_out, layer2_0_conv2_out, layer2_0_bn2_bp_out, layer2_0_bn2_weight, layer2_0_bn2_weight_act, layer2_0_bn2_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 128, 128, 16, 16, 16, 16>(layer2_0_bn2_bp_out, layer2_0_conv2_weight, layer2_0_conv2_bp_out, 1);

	relu_bp_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_conv2_bp_out, layer2_0_bn1_out, layer2_0_relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 128, 16, 16>(layer2_0_relu1_bp_out, layer2_0_conv1_out, layer2_0_bn1_bp_out, layer2_0_bn1_weight, layer2_0_bn1_weight_act, layer2_0_bn1_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 128, 64, 16, 16, 32, 32>(layer2_0_bn1_bp_out, layer2_0_conv1_weight, layer2_0_conv1_bp_out, 2);
	// shortcut_sw bn_sw + conv_1x1
	bn_bp_sw<BATCH_SIZE, 128, 16, 16>(layer2_1_shortcut_bp_out, layer2_0_conv_sc_out, layer2_0_bn_sw_sc_bp_out, layer3_0_bn_sc_weight, layer3_0_bn_sc_weight_act, layer3_0_bn_sc_bias_act);
	conv_1x1_sw_bp<BATCH_SIZE, 128, 64, 16, 16, 32, 32>(layer2_0_bn_sw_sc_bp_out, layer2_0_conv_sc_weight, layer2_0_conv_sc_bp_out, 2);
	shortcut_sw<BATCH_SIZE, 64, 32, 32>(layer2_0_conv_sc_bp_out, layer2_0_conv1_bp_out, layer2_0_shortcut_bp_out);

	////////////////////////////////////
	///////// LAYER 1 backprop /////////
	////////////////////////////////////

	// layer 1_1
	relu_bp_sw<BATCH_SIZE, 64, 32, 32>(layer2_0_shortcut_bp_out, layer1_1_bn2_out, layer1_1_relu2_bp_out);
	bn_bp_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_relu2_bp_out, layer1_1_conv2_out, layer1_1_bn2_bp_out, layer1_1_bn2_weight, layer1_1_bn2_weight_act, layer1_1_bn2_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(layer1_1_bn2_bp_out, layer1_1_conv2_weight, layer1_1_conv2_bp_out, 1);

	relu_bp_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_conv2_bp_out, layer1_1_bn1_out, layer1_1_relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_relu1_bp_out, layer1_1_conv1_out, layer1_1_bn1_bp_out, layer1_1_bn1_weight, layer1_1_bn1_weight_act, layer1_1_bn1_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(layer1_1_bn1_bp_out, layer1_1_conv1_weight, layer1_1_conv1_bp_out, 1);
	shortcut_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_conv1_bp_out, layer2_0_shortcut_bp_out, layer1_1_shortcut_bp_out);

	// layer 1_0
	relu_bp_sw<BATCH_SIZE, 64, 32, 32>(layer1_1_shortcut_bp_out, layer1_0_bn2_out, layer1_0_relu2_bp_out);
	bn_bp_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_relu2_bp_out, layer1_0_conv2_out, layer1_0_bn2_bp_out, layer1_0_bn2_weight, layer1_0_bn2_weight_act, layer1_0_bn2_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(layer1_0_bn2_bp_out, layer1_0_conv2_weight, layer1_0_conv2_bp_out, 1);

	relu_bp_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_conv2_bp_out, layer1_0_bn1_out, layer1_0_relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_relu1_bp_out, layer1_0_conv1_out, layer1_0_bn1_bp_out, layer1_0_bn1_weight, layer1_0_bn1_weight_act, layer1_0_bn1_bias_act);
	conv_3x3_sw_bp<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(layer1_0_bn1_bp_out, layer1_0_conv1_weight, layer1_0_conv1_bp_out, 1);
	shortcut_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_conv1_bp_out, layer1_1_shortcut_bp_out, layer1_0_shortcut_bp_out);

	// conv + bn_sw + relu_sw
	relu_bp_sw<BATCH_SIZE, 64, 32, 32>(layer1_0_shortcut_bp_out, bn1_out, relu1_bp_out);
	bn_bp_sw<BATCH_SIZE, 64, 32, 32>(relu1_bp_out, conv1_out, bn1_bp_out, bn1_weight, bn1_weight_act, bn1_bias_act);
	// conv_3x3_sw_bp<BATCH_SIZE, 64, 64, 32, 32, 32, 32>(bn1_bp_out, conv1_weight, conv1_bp_out, 1);
}


//--------------------
//  Gradient GeMM
//--------------------

void gradient(float error[BATCH_SIZE][512][10]) 
{
	////////////////////////////////////
	///////// LAYER 4 gradient /////////
	////////////////////////////////////

	// activation as input, error as weight
	// layer 4_1
	conv_3x3_sw_grad<BATCH_SIZE, 512, 512, 4, 4, 3, 3, 4>(layer4_1_relu1_out, layer4_1_bn2_bp_out, layer4_1_conv2_grad, 1);
	conv_3x3_sw_grad<BATCH_SIZE, 512, 512, 4, 4, 3, 3, 4>(layer4_0_shortcut_out, layer4_1_bn1_bp_out, layer4_1_conv1_grad, 1);
		/* weight update (conv & bn_sw) */
	layer4_1_conv2_weight   = layer4_1_conv2_weight   - lr * layer4_1_conv2_grad;
	// layer4_1_bn2_weight     = layer4_1_bn2_weight     - lr * layer4_1_bn2_weight_act;
	// layer4_1_bn2_bias       = layer4_1_bn2_bias       - lr * layer4_1_bn2_bias_act;
	layer4_1_conv1_weight   = layer4_1_conv1_weight   - lr * layer4_1_conv1_grad;
	// layer4_1_bn1_weight     = layer4_1_bn1_weight     - lr * layer4_1_bn1_weight_act;
	// layer4_1_bn1_bias       = layer4_1_bn1_bias       - lr * layer4_1_bn1_bias_act;

	// layer 4 upsample (conv1_bp & conv_sc_bp stride=2)
	// layer 4_0
	conv_3x3_sw_grad<BATCH_SIZE, 512, 512, 4, 4, 3, 3, 4>(layer4_0_relu1_out, layer4_0_bn2_bp_out, layer4_0_conv2_grad, 1);
	conv_3x3_sw_grad<BATCH_SIZE, 256, 512, 8, 8, 3, 3, 4>(layer3_1_shortcut_out, layer4_0_bn1_bp_out, layer4_0_conv1_grad, 2);
	// shortcut_sw bn_sw + conv_1x1
	conv_1x1_sw_grad<BATCH_SIZE, 256, 512, 8, 8, 1, 1, 4>(layer3_1_shortcut_out, layer4_0_bn_sw_sc_bp_out, layer4_0_conv_sc_grad, 2);
		/* weight update (conv & bn_sw & sc) */
	layer4_0_conv_sc_weight = layer4_0_conv_sc_weight - lr * layer4_0_conv_sc_grad;
	// layer4_0_bn_sc_weight   = layer4_0_bn_sc_weight   - lr * layer4_0_bn_sc_weight;
	// layer4_0_bn_sc_bias		= layer4_0_bn_sc_bias     - lr * layer4_0_bn_sc_bias;
	layer4_0_conv2_weight   = layer4_0_conv2_weight   - lr * layer4_0_conv2_grad;
	// layer4_0_bn2_weight     = layer4_0_bn2_weight     - lr * layer4_0_bn2_weight_act;
	// layer4_0_bn2_bias       = layer4_0_bn2_bias       - lr * layer4_0_bn2_bias_act;
	layer4_0_conv1_weight   = layer4_0_conv1_weight   - lr * layer4_0_conv1_grad;
	// layer4_0_bn1_weight     = layer4_0_bn1_weight     - lr * layer4_0_bn1_weight_act;
	// layer4_0_bn1_bias       = layer4_0_bn1_bias       - lr * layer4_0_bn1_bias_act;

	////////////////////////////////////
	///////// LAYER 3 gradient /////////
	////////////////////////////////////

	// layer 3_1
	conv_3x3_sw_grad<BATCH_SIZE, 256, 256, 8, 8, 3, 3, 8>(layer3_1_relu1_out, layer3_1_bn2_bp_out, layer3_1_conv2_grad, 1);
	conv_3x3_sw_grad<BATCH_SIZE, 256, 256, 8, 8, 3, 3, 8>(layer3_0_shortcut_out, layer3_1_bn1_bp_out, layer3_1_conv1_grad, 1);
		/* weight update (conv & bn_sw) */
	layer3_1_conv2_weight   = layer3_1_conv2_weight   - lr * layer3_1_conv2_grad;
	// layer3_1_bn2_weight     = layer3_1_bn2_weight     - lr * layer3_1_bn2_weight_act;
	// layer3_1_bn2_bias       = layer3_1_bn2_bias       - lr * layer3_1_bn2_bias_act;
	layer3_1_conv1_weight   = layer3_1_conv1_weight   - lr * layer3_1_conv1_grad;
	// layer3_1_bn1_weight     = layer3_1_bn1_weight     - lr * layer3_1_bn1_weight_act;
	// layer3_1_bn1_bias       = layer3_1_bn1_bias       - lr * layer3_1_bn1_bias_act;

	// layer 3 upsample (conv1_bp & conv_sc_bp stride=2)
	// layer 3_0
	conv_3x3_sw_grad<BATCH_SIZE, 256, 256, 8, 8, 3, 3, 8>(layer3_0_relu1_out, layer3_0_bn2_bp_out, layer3_0_conv2_grad, 1);
	conv_3x3_sw_grad<BATCH_SIZE, 128, 256, 16, 16, 3, 3, 8>(layer2_1_shortcut_out, layer3_0_bn1_bp_out, layer3_0_conv1_grad, 2);
	// shortcut_sw bn_sw + conv_1x1
	conv_1x1_sw_grad<BATCH_SIZE, 128, 256, 16, 16, 1, 1, 8>(layer2_1_shortcut_out, layer3_0_bn_sw_sc_bp_out, layer3_0_conv_sc_grad, 2);
		/* weight update (conv & bn_sw & sc) */
	layer3_0_conv_sc_weight = layer4_0_conv_sc_weight - lr * layer4_0_conv_sc_grad;
	// layer3_0_bn_sc_weight   = layer4_0_bn_sc_weight   - lr * layer4_0_bn_sc_weight;
	// layer3_0_bn_sc_bias		= layer4_0_bn_sc_bias     - lr * layer4_0_bn_sc_bias;
	layer3_0_conv2_weight   = layer4_0_conv2_weight   - lr * layer4_0_conv2_grad;
	// layer3_0_bn2_weight     = layer4_0_bn2_weight     - lr * layer4_0_bn2_weight_act;
	// layer3_0_bn2_bias       = layer4_0_bn2_bias       - lr * layer4_0_bn2_bias_act;
	layer3_0_conv1_weight   = layer4_0_conv1_weight   - lr * layer4_0_conv1_grad;
	// layer3_0_bn1_weight     = layer4_0_bn1_weight     - lr * layer4_0_bn1_weight_act;
	// layer3_0_bn1_bias       = layer4_0_bn1_bias       - lr * layer4_0_bn1_bias_act;

	////////////////////////////////////
	///////// LAYER 2 gradient /////////
	////////////////////////////////////

	// layer 2_1
	conv_3x3_sw_grad<BATCH_SIZE, 128, 128, 16, 16, 3, 3, 16>(layer2_1_relu1_out, layer2_1_bn2_bp_out, layer2_1_conv2_grad, 1);
	conv_3x3_sw_grad<BATCH_SIZE, 128, 128, 16, 16, 3, 3, 16>(layer2_0_shortcut_out, layer2_1_bn1_bp_out, layer2_1_conv1_grad, 1);
		/* weight update (conv & bn_sw) */
	layer2_1_conv2_weight   = layer2_1_conv2_weight   - lr * layer2_1_conv2_grad;
	// layer2_1_bn2_weight     = layer2_1_bn2_weight     - lr * layer2_1_bn2_weight_act;
	// layer2_1_bn2_bias       = layer2_1_bn2_bias       - lr * layer2_1_bn2_bias_act;
	layer2_1_conv1_weight   = layer2_1_conv1_weight   - lr * layer2_1_conv1_grad;
	// layer2_1_bn1_weight     = layer2_1_bn1_weight     - lr * layer2_1_bn1_weight_act;
	// layer2_1_bn1_bias       = layer2_1_bn1_bias       - lr * layer2_1_bn1_bias_act;

	// layer 2 upsample (conv1_bp & conv_sc_bp stride=2)
	// layer 2_0
	conv_3x3_sw_grad<BATCH_SIZE, 128, 128, 16, 16, 3, 3, 16>(layer2_0_relu1_out, layer2_0_bn2_bp_out, layer2_0_conv2_grad, 1);
	conv_3x3_sw_grad<BATCH_SIZE, 64, 128, 32, 32, 3, 3, 16>(layer1_1_shortcut_out, layer2_0_bn1_bp_out, layer2_0_conv1_grad, 2);
	// shortcut_sw bn_sw + conv_1x1
	conv_1x1_sw_grad<BATCH_SIZE, 64, 128, 32, 32, 1, 1, 16>(layer1_1_shortcut_out, layer2_0_bn_sw_sc_bp_out, layer2_0_conv_sc_grad, 2);
		/* weight update (conv & bn_sw & sc) */
	layer2_0_conv_sc_weight = layer2_0_conv_sc_weight - lr * layer2_0_conv_sc_grad;
	// layer2_0_bn_sc_weight   = layer2_0_bn_sc_weight   - lr * layer2_0_bn_sc_weight;
	// layer2_0_bn_sc_bias		= layer2_0_bn_sc_bias     - lr * layer2_0_bn_sc_bias;
	layer2_0_conv2_weight   = layer2_0_conv2_weight   - lr * layer2_0_conv2_grad;
	// layer2_0_bn2_weight     = layer2_0_bn2_weight     - lr * layer2_0_bn2_weight_act;
	// layer2_0_bn2_bias       = layer2_0_bn2_bias       - lr * layer2_0_bn2_bias_act;
	layer2_0_conv1_weight   = layer2_0_conv1_weight   - lr * layer2_0_conv1_grad;
	// layer2_0_bn1_weight     = layer2_0_bn1_weight     - lr * layer2_0_bn1_weight_act;
	// layer2_0_bn1_bias       = layer2_0_bn1_bias       - lr * layer2_0_bn1_bias_act;

	////////////////////////////////////
	///////// LAYER 1 gradient /////////
	////////////////////////////////////

	// layer 1_1
	conv_3x3_sw_grad<BATCH_SIZE, 64, 64, 32, 32, 3, 3, 32>(layer1_1_relu1_out, layer1_1_bn2_bp_out, layer1_1_conv2_grad, 1);
	conv_3x3_sw_grad<BATCH_SIZE, 64, 64, 32, 32, 3, 3, 32>(layer1_0_shortcut_out, layer1_1_bn1_bp_out, layer1_1_conv1_grad, 1);
		/* weight update (conv & bn_sw) */
	layer1_1_conv2_weight   = layer1_1_conv2_weight   - lr * layer1_1_conv2_grad;
	// layer1_1_bn2_weight     = layer1_1_bn2_weight     - lr * layer1_1_bn2_weight_act;
	// layer1_1_bn2_bias       = layer1_1_bn2_bias       - lr * layer1_1_bn2_bias_act;
	layer1_1_conv1_weight   = layer1_1_conv1_weight   - lr * layer1_1_conv1_grad;
	// layer1_1_bn1_weight     = layer1_1_bn1_weight     - lr * layer1_1_bn1_weight_act;
	// layer1_1_bn1_bias       = layer1_1_bn1_bias       - lr * layer1_1_bn1_bias_act;

	// layer 1_0
	conv_3x3_sw_grad<BATCH_SIZE, 64, 64, 32, 32, 3, 3, 32>(layer1_0_relu1_out, layer1_0_bn2_bp_out, layer1_0_conv2_grad, 1);
	conv_3x3_sw_grad<BATCH_SIZE, 64, 64, 32, 32, 3, 3, 32>(relu1_out, layer1_0_bn1_bp_out, layer1_0_conv1_grad, 1);
		/* weight update (conv & bn_sw) */
	layer1_0_conv2_weight   = layer1_0_conv2_weight   - lr * layer1_0_conv2_grad;
	// layer1_0_bn2_weight     = layer1_0_bn2_weight     - lr * layer1_0_bn2_weight_act;
	// layer1_0_bn2_bias       = layer1_0_bn2_bias       - lr * layer1_0_bn2_bias_act;
	layer1_0_conv1_weight   = layer1_0_conv1_weight   - lr * layer1_0_conv1_grad;
	// layer1_0_bn1_weight     = layer1_0_bn1_weight     - lr * layer1_0_bn1_weight_act;
	// layer1_0_bn1_bias       = layer1_0_bn1_bias       - lr * layer1_0_bn1_bias_act;

	// conv + bn_sw + relu_sw
	conv_3x3_sw_grad<BATCH_SIZE, 3, 64, 32, 32, 3, 3, 32>(image, bn1_bp_out, conv1_grad, 1);
		/* weight update (conv & bn_sw) */
	conv1_weight            = conv1_weight            - lr * conv1_weight;
	// bn1_weight              = bn1_weight              - lr * bn1_weight;
	// bn1_bias                = bn1_bias                - lr * bn1_bias;
}



///////////////////////////////////////////////////////////////////////////////
////////////////////////  main function for testbench  ////////////////////////
///////////////////////////////////////////////////////////////////////////////

//#define SW_TEST
int main(int argc, char **argv)
{
	int16 image_hw[4][3][32][32] = {0};

		for(int j = 0; j < 3; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					for(int b = 0; b < 4; b ++){
						image_hw[b][j][row][col] = 1;
					}
				}
			}
		}
	FracNet_T(image_hw, output, weight_3x3_updated, weight_1x1_updated);
}
