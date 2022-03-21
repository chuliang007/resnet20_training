#ifndef BNN_H
#define BNN_H

#include "typedefs.h"
// #include "dimension_def.h"
/*
void FracNet_T(
    int16 image[4][3][32][32],
    int16 out[4][10],
	int16 weight_3x3_updated[16][32][32][3][3],
	int16 weight_1x1_updated[3][32][32]
);
*/

void FracNet_T(
	int8 image[4][3][32][32],
	int8 output[4][10],

	int8 conv_3x3_weight_all[1196][64][64][3][3],
	int8 conv_1x1_weight_all[168][64][64],
	int8 linear_weight[8][10][64],

	int8 out_buf_t0[1364][4][64][33][33],
	int8 out_buf_t1[1364][4][64][33][33],
    int8 out_buf_sc[168][4][64][33][33],

	int1 relu_mask[1364][4][64][33][33]
);

#endif
