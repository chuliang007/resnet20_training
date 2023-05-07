/*
This file defines some dimension parameters 
used in the accelerator.
*/

/*
#define CHANNEL_IN_T 16
#define CHANNEL_OUT_T 16
#define WIDTH 32

#define NUM_3x3_WT 44
#define NUM_1x1_WT 7
#define NUM_ACT 50
#define ACT_BIAS 16/CHANNEL_IN_T
#define WT_BIAS (NUM_3x3_WT + NUM_1x1_WT) // 16*16*3*3
#define NUM_TILE 64/CHANNEL_OUT_T
*/


#define CHANNEL_IN_T 8
#define CHANNEL_OUT_T 8
#define WIDTH 33

#define NUM_3x3_WT 467 //87
#define NUM_1x1_WT 41  //13
#define NUM_ACT 99
#define ACT_BIAS 64/CHANNEL_IN_T
#define WT_BIAS NUM_3x3_WT      // 8*8*3*3
#define WT_BIAS_1x1 NUM_1x1_WT  // 8*8*3*3
#define NUM_TILE 64/CHANNEL_OUT_T

/*
ini: 98
conv_3x3_weight_ptr: 466
conv_1x1_weight_ptr: 40
ini_act_bias: 0
ini_wt_bias: 58
 */
