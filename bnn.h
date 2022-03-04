#ifndef BNN_H
#define BNN_H

#include "typedefs.h"
/*
void FracNet_T(
    int16 image[4][3][32][32],
    int16 out[4][10],
	int16 weight_3x3_updated[16][32][32][3][3],
	int16 weight_1x1_updated[3][32][32]
);
*/
#define BATCH_SIZE 4

#define NUM_3x3_WT 1382
#define NUM_1x1_WT 168
#define NUM_ACT 1382
#define NUM_SC 168

#define CHANNEL_IN_T 64
#define CHANNEL_OUT_T 64
#define WIDTH_T 33

void FracNet_T(
	int8 image[BATCH_SIZE][3][32][32],
	int8 output[BATCH_SIZE][10],

	int8 conv_3x3_weight_all[NUM_3x3_WT][CHANNEL_OUT_T][CHANNEL_IN_T][3][3],
	int8 conv_1x1_weight_all[NUM_1x1_WT][CHANNEL_OUT_T][CHANNEL_IN_T],

	int8 msb_fmap[NUM_ACT][BATCH_SIZE][CHANNEL_IN_T][WIDTH_T][WIDTH_T],
	int8 lsb_fmap[NUM_SC][BATCH_SIZE][CHANNEL_IN_T][WIDTH_T][WIDTH_T],
	int8 out_buf_t0[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH_T][WIDTH_T],
    int8 out_buf_t1[NUM_SC][BATCH_SIZE][CHANNEL_OUT_T][WIDTH_T][WIDTH_T],
	int1 relu_mask[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH_T][WIDTH_T]
);

#endif
