#include "typedefs.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "bnn.h"
#include <fstream>
#include <hls_math.h>

using namespace std;
/*
#define NUM_TESTS 3
// #define LAYER_TEST

unsigned char images[NUM_TESTS*96*32*32];
unsigned char labels[NUM_TESTS];

int8 avg_pool_out[64];
int8 classifier_out[10];

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

int main(int argc, char **argv)
{

	for (int k = 0; k < 1; k ++) {

		////////////////////////////////
		//////// HARDWARE //////////////
		////////////////////////////////

		int8 image_hw[BATCH_SIZE][3][32][32] = {1};
		int8 accelerator_output[BATCH_SIZE][10];

		int8 conv_3x3_weight_tile_buffer[NUM_3x3_WT][16][16][3][3] = {1};
		int8 conv_1x1_weight_tile_buffer[NUM_1x1_WT][16][16] = {1};

		// int8 out_buf_t0[NUM_ACT][BATCH_SIZE][16][33][33];
		// int8 out_buf_t1[NUM_ACT][BATCH_SIZE][16][33][33];

		int8 *****out_buf_t0;
		out_buf_t0 = (int8 *****)malloc(sizeof(int8 ****) * NUM_ACT);
		for (int i = 0; i<NUM_ACT; i ++) {
			out_buf_t0[i] = (int8 ****)malloc(sizeof(int8 ***) * BATCH_SIZE);
			for (int j = 0; j<BATCH_SIZE; j ++) {
				out_buf_t0[i][j] = (int8 ***)malloc(sizeof(int8 **) * 16);
				for (int k = 0; k<16; k++) {
					out_buf_t0[i][j][k] = (int8 **)malloc(sizeof(int8 *) * 33);
					for (int m = 0; m<33; m ++) {
						out_buf_t0[i][j][k][m] = (int8 *)malloc(sizeof(int8) * 33);
					}
				}
			}
		}

		int8 *****out_buf_t1;
		out_buf_t1 = (int8 *****)malloc(sizeof(int8 ****) * NUM_ACT);
		for (int i = 0; i<NUM_ACT; i ++) {
			out_buf_t1[i] = (int8 ****)malloc(sizeof(int8 ***) * BATCH_SIZE);
			for (int j = 0; j<BATCH_SIZE; j ++) {
				out_buf_t1[i][j] = (int8 ***)malloc(sizeof(int8 **) * 16);
				for (int k = 0; k<16; k++) {
					out_buf_t1[i][j][k] = (int8 **)malloc(sizeof(int8 *) * 33);
					for (int m = 0; m<33; m ++) {
						out_buf_t1[i][j][k][m] = (int8 *)malloc(sizeof(int8) * 33);
					}
				}
			}
		}

		FracNet_T(image_hw, accelerator_output, out_buf_t0, out_buf_t1);

		cout << endl << "accelerator output: "<< endl;

		for (int i = 0; i < 3; i ++){
			// cout << accelerator_output[0][i];
			cout << endl << "FracNet_T out: " << accelerator_output[0][i] << endl;
			// cout << "\n" << endl;
		}
		cout << endl;

	return 0;
	}
}
