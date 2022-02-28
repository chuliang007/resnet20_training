#include "typedefs.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "bnn.h"
// #include "weights_tb.h"
#include <fstream>
#include <hls_math.h>
#include "dimension_def.h"
#include "weights_fracnet_64.h"
#include "conv_weights.h"

//--------------------
//   Utils Function
//--------------------

// rot180 for Conv weights
void rot180_3x3(
	int16 mat[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],			// in
	int16 out_buf[CHANNEL_OUT_T][CHANNEL_IN_T][3][3]		// out, mat_rot180
)
{
	for (int n = 0; n < CHANNEL_OUT_T; n ++) {
        for (int c = 0; c < CHANNEL_IN_T; c ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					out_buf[c][n][row][col] = mat[n][c][3-row-1][3-col-1];
				}
			}
		}
	}
}

void rot180_1x1(
	int16 mat[CHANNEL_OUT_T][CHANNEL_IN_T], //[1][1],			// in
	int16 out_buf[CHANNEL_OUT_T][CHANNEL_IN_T]	//[1][1]		// out, mat_rot180
)
{
	for (int n = 0; n < CHANNEL_OUT_T; n ++) {
        for (int c = 0; c < CHANNEL_IN_T; c ++) {
        	out_buf[c][n] = mat[n][c];
		}
	}
}

// Batch Norm
void bn(
	int16 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs

	float gamma[CHANNEL_OUT_T],
	float beta[CHANNEL_OUT_T],

    int H_fmap
)
{
    int N = BATCH_SIZE * WIDTH * WIDTH;
	float mu[CHANNEL_OUT_T];
	float sigma[CHANNEL_OUT_T];
    // calc mean
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
                    mu[c] = mu[c] + bn_inputs[n][c][row][col]/N;
				}
			}
		}
	}
    // calc std variance
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
                	sigma[c] = sigma[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
		sigma[c] = hls::sqrtf(sigma[c]/N);
	}

    // calc affine output
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
            		out_buf[n][c][row][col] = gamma[c]*(bn_inputs[n][c][row][col]-mu[c])/sigma[c] + beta[c];
				}
			}
		}
	}			
}

// Batch Norm Back-prop
void bn_bp(
	int16 error[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	int16 bn_inputs_fw[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// in
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	float gamma[CHANNEL_OUT_T],										// in
	int16 g_gamma[CHANNEL_OUT_T],									// out
	int16 g_beta[CHANNEL_OUT_T],									// out

	int H_fmap
)
{
    int N = BATCH_SIZE * WIDTH * WIDTH;
	float mu[CHANNEL_OUT_T];
	float sigma[CHANNEL_OUT_T];
	// calc mean
    for (int n = 0; n < BATCH_SIZE; n ++){
        for (int c = 0; c < CHANNEL_OUT_T; c ++){
            for (int row = 0; row < H_fmap; row ++){
                for (int col = 0; col < H_fmap; col ++){
                    mu[c] = mu[c] + bn_inputs_fw[n][c][row][col]/N;
				}
			}
		}
	}
	// calc std variance
    for (int n = 0; n < BATCH_SIZE; n ++){
        for (int c = 0; c < CHANNEL_OUT_T; c ++){
            for (int row = 0; row < H_fmap; row ++){
                for (int col = 0; col < H_fmap; col ++){
                    sigma[c] = sigma[c] + (bn_inputs_fw[n][c][row][col]-mu[c])*(bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
    	sigma[c] = hls::sqrtf(sigma[c]/N);
    }

	//calc g_gamma and g_beta (also used for error calc)
    for (int n = 0; n < BATCH_SIZE; n ++){
        for (int c = 0; c < CHANNEL_OUT_T; c ++){
            for (int row = 0; row < H_fmap; row ++){
                for (int col = 0; col < H_fmap; col ++){
                    g_beta[c] = g_beta[c] + error[n][c][row][col];
                    g_gamma[c] = g_gamma[c] + error[n][c][row][col]*(bn_inputs_fw[n][c][row][col]-mu[c])/sigma[c];
				}
			}
		}
	}
	// calc backprop error
    for (int n = 0; n < BATCH_SIZE; n ++){
        for (int c = 0; c < CHANNEL_OUT_T; c ++){
            for (int row = 0; row < H_fmap; row ++){
                for (int col = 0; col < H_fmap; col ++){
            		out_buf[n][c][row][col] = gamma[c]*error[n][c][row][col]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (bn_inputs_fw[n][c][row][col]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
				}
			}
		}
	}			
}

// ReLu
void relu(
	int16 input[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],     // in
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],   // out
	int H_fmap
)
{
	for (int n = 0; n < BATCH_SIZE; n ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
			for (int row = 0; row < H_fmap; row ++) {
				for (int col = 0; col < H_fmap; col ++) {
					if (input[n][c][row][col] > 0) {
						out_buf[n][c][row][col] = input[n][c][row][col];
					} else {
						out_buf[n][c][row][col] = 0;
					}
				}
			}
		}
	}
}

// ReLu Back-prop
void relu_bp(
	int16 error[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],  			// error in
	int16 input_fw[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH], 		// activation in foward
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],    		// error out, error_relu
	int H_fmap
)
{
	for (int n = 0; n < BATCH_SIZE; n ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
			for (int row = 0; row < H_fmap; row ++) {
				for (int col = 0; col < H_fmap; col ++) {
					if (input_fw[n][c][row][col] > 0) {
						out_buf[n][c][row][col] = error[n][c][row][col];
					} else {
						out_buf[n][c][row][col] = 0;
					}
				}
			}
		}
	}
}

// AvgPool
void avgpool(
	int16 avg_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T], 						// out, avg_outputs

	int stride,
	int H_fmap
)
{
	int H_fmap_OUT = H_fmap/stride;
	int stride2 = stride*stride;
	int16 accum;

	for (int n = 0; n < BATCH_SIZE; n ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
			for (int row = 0; row < H_fmap_OUT; row ++) {			// note that H_famp_OUT = 1
				for (int col = 0; col < H_fmap_OUT; col ++) {
					accum = 0;
					for (int s = 0; s < stride; s ++) {
						for (int ss = 0; ss < stride; s ++) {
							accum += avg_inputs[n][c][stride*row+s][stride*col+ss];
						}
					}
					out_buf[n][c] = accum/stride2;
				}
			}
		}
	}
}

// AvgPool Back-prop
void avgpool_bp(
	int16 error[BATCH_SIZE][CHANNEL_OUT_T],							// in
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_avg
	int stride
	// int H_fmap	// #(-1,512,1,1)
)
{
	int H_fmap_OUT = stride;
	int stride2 = stride*stride;

	for (int n = 0; n < BATCH_SIZE; n ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
			for (int s = 0; s < stride; s ++) {
				for (int ss = 0; ss < stride; ss ++) {
					out_buf[n][c][stride+s][stride+ss] = error[n][c]/stride2;
				}
			}
		}
	}
}

// FC (no bias)
void FC(
	int16 inputs[BATCH_SIZE][CHANNEL_OUT_T],
	int16 linear_weight[10][CHANNEL_OUT_T],
	int16 outputs[BATCH_SIZE][10]
)
{
	for (int bii = 0; bii < BATCH_SIZE; bii++) {
		for (int cii = 0; cii < CHANNEL_OUT_T; cii++) {
			for (int coo = 0; coo < 10; coo ++) {
				outputs[bii][coo] += inputs[bii][cii] * linear_weight[coo][cii];
			}
		}
	}
}

// FC Back-prop (no bias)
void FC_bp(
	int16 inputs[BATCH_SIZE][10],
	int16 linear_weight_transpose[CHANNEL_OUT_T][10],
	int16 outputs[BATCH_SIZE][CHANNEL_OUT_T]
)
{
	for (int bii = 0; bii < BATCH_SIZE; bii++) {
		for (int cii = 0; cii < 10; cii++) {
			for (int coo = 0; coo < CHANNEL_OUT_T; coo ++) {
				outputs[bii][coo] += inputs[bii][cii] * linear_weight_transpose[coo][cii];
			}
		}
	}
}

// Shortcut- identity branch
void shortcut(
	int16 input_a[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in1
	int16 input_b[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in2
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out
	int H_fmap
)
{
	// int16 out_feature_a[BATCH_SIZE][CHANNEL_OUT_T];
	// int16 out_feature_b[BATCH_SIZE][CHANNEL_OUT_T];
	for (int n = 0; n < BATCH_SIZE; n ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
			for (int row = 0; row < H_fmap; row ++) {
				for (int col = 0; col < H_fmap; col ++) {
					// out_feature_a[n][c] = input_a[n][c][row][col];
					// out_feature_b[n][c] = input_b[n][c][row][col];
					// out_buf[n][c][row][col] = out_feature_a[n][c] + out_feature_b[n][c];
					out_buf[n][c][row][col] = input_a[n][c][row][col] + input_b[n][c][row][col];
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
	int16 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int16 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int16 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	
	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
	// input 0-padding(1, 1, 1, 1)
	// conv
	for (int bi = 0; bi < BATCH_SIZE; bi ++) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
				for (int col = 0; col < H_fmap_out; col ++) {
					int16 accum = 0;
					for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
								int row_in = row*stride + krow - 1;		// -1 due to 0-padding
								int col_in = col*stride + kcol - 1;
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									accum += input[bi][ci][row_in][col_in] * weight[co][ci][krow][kcol];
								}
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
void conv_1x1
(
	int16 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int16 weight[CHANNEL_OUT_T][CHANNEL_IN_T], //[1][1],
	int16 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
	for (int bi = 0; bi < BATCH_SIZE; bi++) {
		for (int co = 0; co < CHANNEL_OUT_T; co++) {
			for (int row = 0; row < H_fmap_out; row++) {
				for (int col = 0; col < H_fmap_out; col++) {
					int16 accum = 0;
					for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
						int row_in = row*stride;	// krow = kcol = 0
						int col_in = col*stride;
						if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
							accum += input[bi][ci][row_in][col_in] * weight[co][ci]; //[0][0];
						}
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

// Conv_3x3, padding=1
void conv_3x3_bp
(
	int16 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int16 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int16 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
	// input dilation and 0-padding(1, 1+A, 1, 1+A)
	int16 input_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
	for (int bi = 0; bi < BATCH_SIZE; bi++) {
		for (int co = 0; co < CHANNEL_OUT_T; co++) {
			for (int row = 0; row < H_fmap_in; row++) {
				for (int col = 0; col < H_fmap_in; col++) {
					input_dil[bi][co][row*stride + 1][col*stride + 1] = input[bi][co][row][col];	// +1 due to 0-padding
				}
			}
		}
	}

	// conv
	for (int bi = 0; bi < BATCH_SIZE; bi ++) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
				for (int col = 0; col < H_fmap_out; col ++) {
					int16 accum = 0;
					for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
								int row_in = row + krow;	// stride 1 transposed conv
								int col_in = col + kcol;
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									accum += input_dil[bi][ci][row_in][col_in] * weight[co][ci][krow][kcol];
								}
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
void conv_1x1_bp
(
	int16 input[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	int16 weight[CHANNEL_OUT_T][CHANNEL_OUT_T], //[1][1],
	int16 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
	// input dilation
	int16 input_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
	for (int bi = 0; bi < BATCH_SIZE; bi++) {
		for (int co = 0; co < CHANNEL_OUT_T; co++) {
			for (int row = 0; row < H_fmap_in; row++) {
				for (int col = 0; col < H_fmap_in; col++) {
					input_dil[bi][co][row*stride][col*stride] = input[bi][co][row][col];
				}
			}
		}
	}

	// conv
	for (int bi = 0; bi < BATCH_SIZE; bi ++) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
				for (int col = 0; col < H_fmap_out; col ++) {
					int16 accum = 0;
					for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
						int row_in = row;	// stride 1 transposed conv, krow = kcol = 0
						int col_in = col;
						accum += input_dil[bi][ci][row_in][col_in] * weight[co][ci]; //[0][0];
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

// Conv_3x3_grad, padding=1
void conv_3x3_grad
(
	int16 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int16 weight[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	int16 output[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int k_row_in	// weight size (error as weight), k_row_in = H_fmap_in
)
{
	// weight dilation, k_dil = 1 + (k-1)*s
	int KERNEL_DIL = (k_row_in-1)*stride + 1;
	int16 weight_dil[BATCH_SIZE][CHANNEL_OUT_T][2*WIDTH][2*WIDTH];	// larger buffer for dilated weights
	for (int bi = 0; bi < BATCH_SIZE; bi ++) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int krow = 0; krow < k_row_in; krow ++) {
				for (int kcol = 0; kcol < k_row_in; kcol ++) {
					weight_dil[bi][co][krow*stride][kcol*stride] = weight[bi][co][krow][kcol];
				}
			}
		}
	}

	// input 0-padding(1, 1+A, 1, 1+A)
	// conv
	for (int co = 0; co < CHANNEL_OUT_T; co ++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					int16 accum = 0;
					for (int bi = 0; bi < BATCH_SIZE; ci ++) {
						for (int krow = 0; krow < KERNEL_DIL; krow ++) {
							for (int kcol = 0; kcol < KERNEL_DIL; kcol ++) {
								int row_in = row*stride + krow - 1;		// -1 due to 0-padding
								int col_in = col*stride + kcol - 1;
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									accum += input[bi][ci][row_in][col_in] * weight_dil[bi][co][krow][kcol];
								}
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
void conv_1x1_grad
(
	int16 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int16 weight[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	int16 output[CHANNEL_OUT_T][CHANNEL_IN_T],
	
	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int k_row_in	// weight size (error as weight), k_row_in = H_fmap_in
)
{
	// weight dilation, k_dil = 1 + (k-1)*s
	int KERNEL_DIL = (k_row_in-1)*stride + 1;
	int16 weight_dil[BATCH_SIZE][CHANNEL_OUT_T][2*WIDTH][2*WIDTH];	// larger buffer for dilated weights
	for (int bi = 0; bi < BATCH_SIZE; bi ++) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int krow = 0; krow < k_row_in; krow ++) {
				for (int kcol = 0; kcol < k_row_in; kcol ++) {
					weight_dil[bi][co][krow*stride][kcol*stride] = weight[bi][co][krow][kcol];
				}
			}
		}
	}

	// no 0-padding
	// conv
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			int16 accum = 0;
			for (int bi = 0; bi < BATCH_SIZE; ci++) {
				for (int krow = 0; krow < KERNEL_DIL; krow++) {
					for (int kcol = 0; kcol < KERNEL_DIL; kcol++) {
						if (krow >= 0 && krow < H_fmap_in && kcol >= 0 && kcol < H_fmap_in) {
							accum += input[bi][ci][krow][kcol] * weight_dil[bi][co][krow][kcol];
						}
					}
				}
			}
			output[co][ci] = accum;
		}
	}
}

// SGD conv_3x3 weight update
void SGD_WU_3x3
(
	int16 gradient[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int16 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int16 weight_WU[CHANNEL_OUT_T][CHANNEL_IN_T][3][3]
)
{
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			for (int krow = 0; krow < 3; krow++) {
				for (int kcol = 0; kcol < 3; kcol++) {
					weight_WU[co][ci][krow][kcol] = weight[co][ci][krow][kcol] - lr*gradient[co][ci][krow][kcol];
				}
			}
		}
	}
}

// SGD conv_1x1 weight update
void SGD_WU_1x1
(
	int16 gradient[CHANNEL_OUT_T][CHANNEL_IN_T],
	int16 weight[CHANNEL_OUT_T][CHANNEL_IN_T],
	int16 weight_WU[CHANNEL_OUT_T][CHANNEL_IN_T]
)
{
	for (int co = 0; co < CHANNEL_OUT_T; co++) {
		for (int ci = 0; ci < CHANNEL_IN_T; ci++) {
			weight_WU[co][ci] = weight[co][ci] - lr*gradient[co][ci];
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

//--------------------
//   Fused Function
//--------------------

// Fused bn + relu
void bn_relu(
	int16 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs
	int1  relu_mask[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp

	float gamma[CHANNEL_OUT_T],
	float beta[CHANNEL_OUT_T],

    int H_fmap
)
{
    int N = BATCH_SIZE * WIDTH * WIDTH;
	float mu[CHANNEL_OUT_T];
	float sigma[CHANNEL_OUT_T];
    // calc mean
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
                    mu[c] = mu[c] + bn_inputs[n][c][row][col]/N;
				}
			}
		}
	}
    // calc std variance
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
                	sigma[c] = sigma[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
		sigma[c] = hls::sqrtf(sigma[c]/N);
	}

    // calc affine output
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
            		out_buf[n][c][row][col] = gamma[c]*(bn_inputs[n][c][row][col]-mu[c])/sigma[c] + beta[c];
					// relu mask
					if (out_buf[n][c][row][col] > 0) {
						relu_mask[n][c][row][col] = 1;
					} else {
						relu_mask[n][c][row][col] = 0;
					}
					out_buf[n][c][row][col] = out_buf[n][c][row][col] * relu_mask[n][c][row][col];
				}
			}
		}
	}			
}

// Fused relu_bp + bn_bp
void bn_relu_bp(
	int16 error[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH], 			// in
	int16 bn_inputs_fw[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],	// in
	int1  relu_mask[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, error_bn

	float gamma[CHANNEL_OUT_T],										// in
	int16 g_gamma[CHANNEL_OUT_T],									// out
	int16 g_beta[CHANNEL_OUT_T],									// out

	int H_fmap
)
{
	int N = BATCH_SIZE * WIDTH * WIDTH;
	float mu[CHANNEL_OUT_T];
	float sigma[CHANNEL_OUT_T];

	// calc mean and relu_bp
	for (int n = 0; n < BATCH_SIZE; n ++) {
		for (int c = 0; c < CHANNEL_OUT_T; c ++) {
			for (int row = 0; row < H_fmap; row ++) {
				for (int col = 0; col < H_fmap; col ++) {
					// mean
					mu[c] = mu[c] + bn_inputs_fw[n][c][row][col]/N;
					// relu
					error[n][c][row][col] = relu_mask[n][c][row][col] * error[n][c][row][col];
				}
			}
		}
	}

	// calc std variance
    for (int n = 0; n < BATCH_SIZE; n ++){
        for (int c = 0; c < CHANNEL_OUT_T; c ++){
            for (int row = 0; row < H_fmap; row ++){
                for (int col = 0; col < H_fmap; col ++){
                    sigma[c] = sigma[c] + (bn_inputs_fw[n][c][row][col]-mu[c]) * (bn_inputs_fw[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
    	sigma[c] = hls::sqrtf(sigma[c]/N);
    }

	//calc g_gamma and g_beta (also used for error calc)
    for (int n = 0; n < BATCH_SIZE; n ++){
        for (int c = 0; c < CHANNEL_OUT_T; c ++){
            for (int row = 0; row < H_fmap; row ++){
                for (int col = 0; col < H_fmap; col ++){
                    g_beta[c] = g_beta[c] + error[n][c][row][col];
                    g_gamma[c] = g_gamma[c] + error[n][c][row][col] * (bn_inputs_fw[n][c][row][col]-mu[c])/sigma[c];
				}
			}
		}
	}
	// calc backprop error
    for (int n = 0; n < BATCH_SIZE; n ++){
        for (int c = 0; c < CHANNEL_OUT_T; c ++){
            for (int row = 0; row < H_fmap; row ++){
                for (int col = 0; col < H_fmap; col ++){
            		out_buf[n][c][row][col] = gamma[c]*error[n][c][row][col]/sigma[c] - gamma[c]*g_beta[c]/(N*sigma[c]) - (bn_inputs_fw[n][c][row][col]-mu[c])*g_gamma[c]/(N*gamma[c]*sigma[c]*sigma[c]);
				}
			}
		}
	}			
}

// Fused bn + relu + shortcut, only for forward (since backward is relu+bn+conv+shortcut)
void bn_relu_shortcut(
	int16 bn_inputs[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// in
	int16 out_buf[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// out, bn_outputs
	int1  relu_mask[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],		// out, relu_mask for relu_bp
	int16 input_b[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],			// in2 for shortcut

	float gamma[CHANNEL_OUT_T],
	float beta[CHANNEL_OUT_T],

    int H_fmap
)
{
    int N = BATCH_SIZE * WIDTH * WIDTH;
	float mu[CHANNEL_OUT_T];
	float sigma[CHANNEL_OUT_T];
    // calc mean
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
                    mu[c] = mu[c] + bn_inputs[n][c][row][col]/N;
				}
			}
		}
	}
    // calc std variance
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
                	sigma[c] = sigma[c] + (bn_inputs[n][c][row][col]-mu[c])*(bn_inputs[n][c][row][col]-mu[c]);
				}
			}
		}
	}
    for (int c = 0; c < CHANNEL_OUT_T; c ++){
		sigma[c] = hls::sqrtf(sigma[c]/N);
	}

    // calc affine output
    for (int n = 0; n < BATCH_SIZE; n ++) {
        for (int c = 0; c < CHANNEL_OUT_T; c ++) {
            for (int row = 0; row < H_fmap; row ++) {
                for (int col = 0; col < H_fmap; col ++) {
            		out_buf[n][c][row][col] = gamma[c]*(bn_inputs[n][c][row][col]-mu[c])/sigma[c] + beta[c];
					// relu mask + shortcut
					if (out_buf[n][c][row][col] > 0) {
						relu_mask[n][c][row][col] = 1;
					} else {
						relu_mask[n][c][row][col] = 0;
					}
					out_buf[n][c][row][col] = out_buf[n][c][row][col] * relu_mask[n][c][row][col] + input_b[n][c][row][col];
				}
			}
		}
	}	
}

void conv_3x3_rot_bp
(
	int16 input[BATCH_SIZE][CHANNEL_IN_T][WIDTH][WIDTH],
	int16 weight[CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int16 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
	// weight rot180
	int16 weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
	for (int n = 0; n < CHANNEL_OUT_T; n ++) {
        for (int c = 0; c < CHANNEL_IN_T; c ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					weight_rot[c][n][row][col] = weight[n][c][3-row-1][3-col-1];
				}
			}
		}
	}

	// input dilation and 0-padding(1, 1+A, 1, 1+A)
	int16 input_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
	for (int bi = 0; bi < BATCH_SIZE; bi++) {
		for (int co = 0; co < CHANNEL_OUT_T; co++) {
			for (int row = 0; row < H_fmap_in; row++) {
				for (int col = 0; col < H_fmap_in; col++) {
					input_dil[bi][co][row*stride + 1][col*stride + 1] = input[bi][co][row][col];	// +1 due to 0-padding
				}
			}
		}
	}

	// conv
	for (int bi = 0; bi < BATCH_SIZE; bi ++) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
				for (int col = 0; col < H_fmap_out; col ++) {
					int16 accum = 0;
					for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
						for (int krow = 0; krow < 3; krow ++) {
							for (int kcol = 0; kcol < 3; kcol ++) {
								int row_in = row + krow;	// stride 1 transposed conv
								int col_in = col + kcol;
								if (row_in >= 0 && row_in < H_fmap_in && col_in >= 0 && col_in < H_fmap_in) {
									accum += input_dil[bi][ci][row_in][col_in] * weight_rot[co][ci][krow][kcol];
								}
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
void conv_1x1_rot_bp
(
	int16 input[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],
	int16 weight[CHANNEL_OUT_T][CHANNEL_OUT_T],
	int16 output[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],

	int stride,
	int ch_in,
	int ch_out,
	int H_fmap_in,
	int H_fmap_out
)
{
	// weight rot180
	int16 weight_rot[CHANNEL_OUT_T][CHANNEL_IN_T];
	for (int n = 0; n < CHANNEL_OUT_T; n ++) {
        for (int c = 0; c < CHANNEL_IN_T; c ++) {
			weight_rot[c][n] = weight[n][c];
		}
	}

	// input dilation
	int16 input_dil[BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH];
	for (int bi = 0; bi < BATCH_SIZE; bi++) {
		for (int co = 0; co < CHANNEL_OUT_T; co++) {
			for (int row = 0; row < H_fmap_in; row++) {
				for (int col = 0; col < H_fmap_in; col++) {
					input_dil[bi][co][row*stride][col*stride] = input[bi][co][row][col];
				}
			}
		}
	}

	// conv
	for (int bi = 0; bi < BATCH_SIZE; bi ++) {
		for (int co = 0; co < CHANNEL_OUT_T; co ++) {
			for (int row = 0; row < H_fmap_out; row ++) {
				for (int col = 0; col < H_fmap_out; col ++) {
					int16 accum = 0;
					for (int ci = 0; ci < CHANNEL_IN_T; ci ++) {
						int row_in = row;	// stride 1 transposed conv, krow = kcol = 0
						int col_in = col;
						accum += input_dil[bi][ci][row_in][col_in] * weight_rot[co][ci];
					}
					output[bi][co][row][col] = accum;
				}
			}
		}
	}
}
