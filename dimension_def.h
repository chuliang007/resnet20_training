/*
This file defines some dimension parameters 
used in the accelerator.
*/
#define NUM_ACT 19   // activation index in forward path
#define NUM_WT_3x3 16
#define NUM_WT_1x1 3
#define lr 0.01
#define BATCH_SIZE 4
#define CHANNEL_IN 3
#define CHANNEL_IN_T 8
#define CHANNEL_OUT_T 8
#define WIDTH 33    // (66 for dilation and padding)

// #define OUT_CHANNEL_PARALLELISM 16
// #define BN_CHANNEL_PARALLELISM 16
// #define WIDTH_T 11
// #define TILE_SIZE 16
