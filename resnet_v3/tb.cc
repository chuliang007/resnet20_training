#include "typedefs.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "bnn.h"
#include <fstream>
#include <hls_math.h>

using namespace std;

int main(int argc, char **argv)
{

	for (int k = 0; k < 1; k ++) {

		////////////////////////////////
		//////// HARDWARE //////////////
		////////////////////////////////

		int8 image_hw[3][32][32];
		int8 accelerator_output[10];
		int8 image_output_test[3][32][32];

		int8 conv_3x3_weight_tile_buffer[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T][3][3];
		int8 conv_1x1_weight_tile_buffer[NUM_1x1_WT][CHANNEL_OUT_T][CHANNEL_IN_T];
		int8 linear_weight_tile_buffer[10][64];

		int8 gamma[NUM_ACT][CHANNEL_OUT_T];
		int8 beta[NUM_ACT][CHANNEL_OUT_T];

		// input
		for(int j = 0; j < 3; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					image_hw[j][row][col] = 133;
				}
			}
		}
//		// weight conv 3x3
//		for(int j = 0; j < NUM_3x3_WT; j ++) {
//			for(int cout = 0; cout < CHANNEL_OUT_T; cout ++) {
//				for(int cin = 0; cin < CHANNEL_IN_T; cin ++) {
//					for(int row = 0; row < 3; row ++) {
//						for(int col = 0; col < 3; col ++) {
//							conv_3x3_weight_tile_buffer[j][cout][cin][row][col] = 1;
//						}
//					}
//				}
//			}
//		}
//		// weight conv 1x1
//		for(int j = 0; j < NUM_1x1_WT; j ++){
//			for(int row = 0; row < CHANNEL_OUT_T; row ++){
//				for(int col = 0; col < CHANNEL_IN_T; col ++){
//					conv_1x1_weight_tile_buffer[j][row][col] = 13;
//				}
//			}
//		}
//		// weight FC
//		for(int row = 0; row < 10; row ++){
//			for(int col = 0; col < 64; col ++){
//				linear_weight_tile_buffer[row][col] = 13;
//			}
//		}
//		// weight bn
//		for(int row = 0; row < NUM_ACT; row ++){
//			for(int col = 0; col < CHANNEL_OUT_T; col ++){
//				gamma[row][col] = 13;
//			}
//		}
//		// bias bn
//		for(int row = 0; row < NUM_ACT; row ++){
//			for(int col = 0; col < CHANNEL_OUT_T; col ++){
//				beta[row][col] = 0;
//			}
//		}

		FracNet_T(image_hw, image_output_test);

//		for (int i = 0; i < 10; i ++){
//			// cout << accelerator_output[0][i];
//			cout << endl << "accelerator out: " << accelerator_output[i] << endl;
//			// cout << "\n" << endl;
//		}
//		cout << endl;

		for(int j = 0; j < 3; j ++){
			for(int row = 0; row < 3; row ++){
				for(int col = 0; col < 3; col ++){
					cout << endl << "image_output_test out: " << image_output_test[j][row][col] << endl;
				}
			}
		}
		cout << endl;

	return 0;
	}
}
