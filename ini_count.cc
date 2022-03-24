#include <cstdio>

#define lr 0.01
#define BATCH_SIZE 4
/*
#define NUM_3x3_WT 1196	// 299 * BATCH_SIZE
#define NUM_1x1_WT 168	// 42 * BATCH_SIZE
#define NUM_ACT 1196
#define NUM_SC 168
*/
#define CHANNEL_IN_T 64
#define CHANNEL_OUT_T 64
#define WIDTH 33

//--------------------
//  Top Function 
//--------------------
int main(
	/*int int ini,
	int int ini_sc,
	int conv_3x3_weight_ptr,
	int conv_1x1_weight_ptr
	*/
)
{
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////		Forward path		//////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	int ini = 0;

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	int in_channels = 64;
	int in_channels_after_pack = 1;
	int out_channels = 64;
	int H_fmap_in =32;
	int H_fmap_out = 32;
	int stride = 1;
	int conv_3x3_weight_ptr = 0;
	int conv_1x1_weight_ptr = 0;

	int ini_sc = 0;
	
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1; 	// ini = 1
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= Conv 1 + bn 1 + relu 1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 64;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 2
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer1_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// int ini = 3
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer1_0 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 4
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer1_1 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 5
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer1_1 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 2 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 16;
	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 128;
	stride = 2;
	conv_1x1_weight_ptr = 0;

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 6
				ini_sc += 1;	// ini_sc = 0;
				conv_1x1_weight_ptr += 1;

			}
		}
    }

	printf("======= layer2_0 shortcut ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {
				
				ini += 1;	// ini = 7
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer2_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 128;
	in_channels_after_pack = 1;
	out_channels = 128;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// int ini = 8
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer2_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// int ini = 9
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer2_1 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// int ini = 10
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer2_1 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 3 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 8;
	in_channels = 128;
	in_channels_after_pack = 1;
	out_channels = 256;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 11
				ini_sc += 1; // ini_sc = 1
				conv_1x1_weight_ptr += 1;

			}
		}
    }

	printf("======= layer3_0 shortcut ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 12
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer3_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 256;
	in_channels_after_pack = 1;
	out_channels = 256;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 13
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer3_0 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 14
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer3_1 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 15
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer3_1 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 4 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 4;
	in_channels = 256;
	in_channels_after_pack = 1;
	out_channels = 512;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer4_0 shortcut (conv+bn) ///////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 16
				ini_sc += 1; // ini_sc = 2
				conv_1x1_weight_ptr += 1;

			}
		}
    }

	printf("======= layer4_0 shortcut ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_0 PG1 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 17
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer4_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 4 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 4;
	in_channels = 512;
	in_channels_after_pack = 1;
	out_channels = 512;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer4_0 PG2 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 18
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer4_0 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_1 PG1 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 19
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer4_1 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
    for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) { 
		for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {

				ini += 1;	// ini = 20
				conv_3x3_weight_ptr += 1;

			}
		}
    }

	printf("======= layer4_1 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////

	printf("//////////////////////// Finised forward path //////////////////////// \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);
	printf("//////////////////////// Starting backward path //////////////////////// \n");

	////////////////////////////////////////////////
	//////////// LAYER 4 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 4;
	in_channels = 512;
	in_channels_after_pack = 1;
	out_channels = 512;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
	printf("======= layer4_1 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);
		
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1; // int ini = 20
				conv_3x3_weight_ptr -= 1;	
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_1 PG1 //////////////////////
	printf("======= layer4_1 PG1 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 19
				conv_3x3_weight_ptr -= 1;

			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_0 PG2 //////////////////////
	printf("======= layer4_0 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1; // int ini = 18
				conv_3x3_weight_ptr -= 1;

			}
		}
    }

	////////////////////////////////////////////////
	//////////// LAYER 4 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 8;
	in_channels = 512;
	in_channels_after_pack = 1;
	out_channels = 256;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer4_0 PG1 //////////////////////
	printf("======= layer4_0 PG1 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);
 
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 17
				conv_3x3_weight_ptr -= 1;

			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer4_0 shortcut (conv+bn) ///////
	printf("======= layer4_0 shortcut BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 16
				ini_sc -= 1;// ini_sc = 2
				conv_1x1_weight_ptr -= 1;

			}
		}
    }


	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 256;
	in_channels_after_pack = 1;
	out_channels = 256;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
	printf("======= layer3_1 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 15
				conv_3x3_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	printf("======= layer3_1 PG1 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 14
				conv_3x3_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
	printf("======= layer3_0 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1; // int ini = 13
				conv_3x3_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 16;
	in_channels = 256;
	in_channels_after_pack = 1;
	out_channels = 128;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	printf("======= layer3_0 PG1 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 12
				conv_3x3_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
	printf("======= layer3_0 shortcut BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 11
				ini_sc -= 1;	// int ini_sc = 1
				conv_1x1_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 128;
	in_channels_after_pack = 1;
	out_channels = 128;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	printf("======= layer2_1 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 10
				conv_3x3_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	printf("======= layer2_1 PG1 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 9
				conv_3x3_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
	printf("======= layer2_0 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1; // int ini = 8
				conv_3x3_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 32;
	in_channels = 128;
	in_channels_after_pack = 1;
	out_channels = 64;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	printf("======= layer2_0 PG1 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 7
				conv_3x3_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
	printf("======= layer2_0 shortcut BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 6
				ini_sc -= 1;	// int ini_sc = 0
				conv_1x1_weight_ptr -= 1;

			}
		}
	}

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 64;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
	printf("======= layer1_1 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1; // int ini = 5
				conv_3x3_weight_ptr -= 1;

			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	printf("======= layer1_1 PG1 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 4
				conv_3x3_weight_ptr -= 1;

			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	printf("======= layer1_0 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				// lsb_fmap[int ini] = msb_fmap[int ini];	// identity branch
				ini -= 1;	// int ini = 3
				conv_3x3_weight_ptr -= 1;

			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
	printf("======= layer1_0 PG2 BP ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 2
				conv_3x3_weight_ptr -= 1;

			}
		}
    }

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 64;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;

	printf("======= Conv 1 + bn 1 + relu 1 ======= \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {

				ini -= 1;	// int ini = 1
				conv_3x3_weight_ptr -= 1;

			}
		}
    }

	printf("///////////////// Finised backward path ///////////////// \n");
	printf("ini: %d \n", ini);
	printf("ini_sc: %d \n", ini_sc);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

}	// end FracBNN_T


/*
======= Conv 1 + bn 1 + relu 1 ======= 
ini: 4 
ini_sc: 0 
conv_3x3_weight_ptr: 4 
conv_1x1_weight_ptr: 0 
======= layer1_0 PG1 ======= 
ini: 8 
ini_sc: 0 
conv_3x3_weight_ptr: 8 
conv_1x1_weight_ptr: 0 
======= layer1_0 PG2 ======= 
ini: 12 
ini_sc: 0 
conv_3x3_weight_ptr: 12 
conv_1x1_weight_ptr: 0 
======= layer1_1 PG1 ======= 
ini: 16 
ini_sc: 0 
conv_3x3_weight_ptr: 16 
conv_1x1_weight_ptr: 0 
======= layer1_1 PG2 ======= 
ini: 20 
ini_sc: 0 
conv_3x3_weight_ptr: 20 
conv_1x1_weight_ptr: 0 
======= layer2_0 shortcut ======= 
ini: 28 
ini_sc: 8 
conv_3x3_weight_ptr: 20 
conv_1x1_weight_ptr: 8 
======= layer2_0 PG1 ======= 
ini: 36 
ini_sc: 8 
conv_3x3_weight_ptr: 28 
conv_1x1_weight_ptr: 8 
======= layer2_0 PG1 ======= 
ini: 52 
ini_sc: 8 
conv_3x3_weight_ptr: 44 
conv_1x1_weight_ptr: 8 
======= layer2_1 PG1 ======= 
ini: 68 
ini_sc: 8 
conv_3x3_weight_ptr: 60 
conv_1x1_weight_ptr: 8 
======= layer2_1 PG2 ======= 
ini: 84 
ini_sc: 8 
conv_3x3_weight_ptr: 76 
conv_1x1_weight_ptr: 8 
======= layer3_0 shortcut ======= 
ini: 116 
ini_sc: 40 
conv_3x3_weight_ptr: 76 
conv_1x1_weight_ptr: 40 
======= layer3_0 PG1 ======= 
ini: 148 
ini_sc: 40 
conv_3x3_weight_ptr: 108 
conv_1x1_weight_ptr: 40 
======= layer3_0 PG2 ======= 
ini: 212 
ini_sc: 40 
conv_3x3_weight_ptr: 172 
conv_1x1_weight_ptr: 40 
======= layer3_1 PG1 ======= 
ini: 276 
ini_sc: 40 
conv_3x3_weight_ptr: 236 
conv_1x1_weight_ptr: 40 
======= layer3_1 PG2 ======= 
ini: 340 
ini_sc: 40 
conv_3x3_weight_ptr: 300 
conv_1x1_weight_ptr: 40 
======= layer4_0 shortcut ======= 
ini: 468 
ini_sc: 168 
conv_3x3_weight_ptr: 300 
conv_1x1_weight_ptr: 168 
======= layer4_0 PG1 ======= 
ini: 596 
ini_sc: 168 
conv_3x3_weight_ptr: 428 
conv_1x1_weight_ptr: 168 
======= layer4_0 PG2 ======= 
ini: 852 
ini_sc: 168 
conv_3x3_weight_ptr: 684 
conv_1x1_weight_ptr: 168 
======= layer4_1 PG1 ======= 
ini: 1108 
ini_sc: 168 
conv_3x3_weight_ptr: 940 
conv_1x1_weight_ptr: 168 
======= layer4_1 PG2 ======= 
ini: 1364 
ini_sc: 168 
conv_3x3_weight_ptr: 1196 
conv_1x1_weight_ptr: 168 

//////////////////////// Finised forward path //////////////////////// 
ini: 1364 
ini_sc: 168 
conv_3x3_weight_ptr: 1196 
conv_1x1_weight_ptr: 168 
//////////////////////// Starting backward path //////////////////////// 

======= layer4_1 PG2 BP ======= 
ini: 1364 
ini_sc: 168 
conv_3x3_weight_ptr: 1196 
conv_1x1_weight_ptr: 168 
======= layer4_1 PG1 BP ======= 
ini: 1108 
ini_sc: 168 
conv_3x3_weight_ptr: 940 
conv_1x1_weight_ptr: 168 
======= layer4_0 PG2 BP ======= 
ini: 852 
ini_sc: 168 
conv_3x3_weight_ptr: 684 
conv_1x1_weight_ptr: 168 
======= layer4_0 PG1 BP ======= 
ini: 596 
ini_sc: 168 
conv_3x3_weight_ptr: 428 
conv_1x1_weight_ptr: 168 
======= layer4_0 shortcut BP ======= 
ini: 468 
ini_sc: 168 
conv_3x3_weight_ptr: 300 
conv_1x1_weight_ptr: 168 
======= layer3_1 PG2 BP ======= 
ini: 340 
ini_sc: 40 
conv_3x3_weight_ptr: 300 
conv_1x1_weight_ptr: 40 
======= layer3_1 PG1 BP ======= 
ini: 276 
ini_sc: 40 
conv_3x3_weight_ptr: 236 
conv_1x1_weight_ptr: 40 
======= layer3_0 PG2 BP ======= 
ini: 212 
ini_sc: 40 
conv_3x3_weight_ptr: 172 
conv_1x1_weight_ptr: 40 
======= layer3_0 PG1 BP ======= 
ini: 148 
ini_sc: 40 
conv_3x3_weight_ptr: 108 
conv_1x1_weight_ptr: 40 
======= layer3_0 shortcut BP ======= 
ini: 116 
ini_sc: 40 
conv_3x3_weight_ptr: 76 
conv_1x1_weight_ptr: 40 
======= layer2_1 PG2 BP ======= 
ini: 84 
ini_sc: 8 
conv_3x3_weight_ptr: 76 
conv_1x1_weight_ptr: 8 
======= layer2_1 PG1 BP ======= 
ini: 68 
ini_sc: 8 
conv_3x3_weight_ptr: 60 
conv_1x1_weight_ptr: 8 
======= layer2_0 PG2 BP ======= 
ini: 52 
ini_sc: 8 
conv_3x3_weight_ptr: 44 
conv_1x1_weight_ptr: 8 
======= layer2_0 PG1 BP ======= 
ini: 36 
ini_sc: 8 
conv_3x3_weight_ptr: 28 
conv_1x1_weight_ptr: 8 
======= layer2_0 shortcut BP ======= 
ini: 28 
ini_sc: 8 
conv_3x3_weight_ptr: 20 
conv_1x1_weight_ptr: 8 
======= layer1_1 PG2 BP ======= 
ini: 20 
ini_sc: 0 
conv_3x3_weight_ptr: 20 
conv_1x1_weight_ptr: 0 
======= layer1_1 PG1 BP ======= 
ini: 16 
ini_sc: 0 
conv_3x3_weight_ptr: 16 
conv_1x1_weight_ptr: 0 
======= layer1_0 PG2 BP ======= 
ini: 12 
ini_sc: 0 
conv_3x3_weight_ptr: 12 
conv_1x1_weight_ptr: 0 
======= layer1_0 PG2 BP ======= 
ini: 8 
ini_sc: 0 
conv_3x3_weight_ptr: 8 
conv_1x1_weight_ptr: 0 
======= Conv 1 + bn 1 + relu 1 ======= 
ini: 4 
ini_sc: 0 
conv_3x3_weight_ptr: 4 
conv_1x1_weight_ptr: 0 
///////////////// Finised backward path ///////////////// 
ini: 0 
ini_sc: 0 
conv_3x3_weight_ptr: 0 
conv_1x1_weight_ptr: 0 
*/