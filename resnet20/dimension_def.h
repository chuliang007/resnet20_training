/*
This file defines some dimension parameters 
used in the accelerator.
*/

//#define CHANNEL_IN_T 8
//#define CHANNEL_OUT_T 8
//#define WIDTH 33
//
//#define NUM_3x3_WT 87
//#define NUM_1x1_WT 13
//#define NUM_ACT 99
//#define ACT_BIAS 64/CHANNEL_IN_T
//#define WT_BIAS (NUM_3x3_WT)/CHANNEL_OUT_T // 8*8*3*3
//#define WT_BIAS_1x1 (NUM_1x1_WT)/CHANNEL_OUT_T // 8*8*3*3
//#define NUM_TILE 64/CHANNEL_OUT_T


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
