#ifndef BNN_H
#define BNN_H

#include "typedefs.h"
#include "dimension_def.h"
/*
void FracNet_T(
    int16 image[4][3][32][32],
    int16 out[4][10],
	int16 weight_3x3_updated[16][32][32][3][3],
	int16 weight_1x1_updated[3][32][32]
);
*/

void FracNet_T(
	int8 image[BATCH_SIZE][3][32][32],
	int8 output[BATCH_SIZE][10],

	int8 out_buf_t0[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH],				// BN output activation
	int8 out_buf_t1[NUM_ACT][BATCH_SIZE][CHANNEL_OUT_T][WIDTH][WIDTH]				// Conv output activation
);

#endif
