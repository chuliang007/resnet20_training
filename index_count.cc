#define BATCH_SIZE 4
#define CHANNEL_IN_T 64
#define CHANNEL_OUT_T 64
#define WIDTH 33

//--------------------
//  Top Function 
//--------------------
int main()
{

	int H_fmap_in, H_fmap_out, in_channels, in_channels_after_pack; 
    int out_channels, out_channel_start, stride, conv_3x3_weight_ptr, conv_1x1_weight_ptr, ini;

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////		Forward path		//////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	ini = 0;

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 3;
	in_channels_after_pack = 1;
	out_channels = 64;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;
	conv_3x3_weight_ptr = 0;

	printf("\n======= Conv 1 + bn 1 + relu 1 ======= \n");
    LOOP_Conv1:
	ini += 1;
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		conv_3x3_weight_ptr += 1;
		for (int b = 0; b < BATCH_SIZE; b ++) {
			/* conv_3x3(
				msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
				stride, H_fmap_out
			);
			bn_relu(
				msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
				gamma, beta,
				H_fmap_out
			); */
			printf("Conv 1 out_buf_t1: %d in batch %d\n", ini, b);
		}
	}

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

	printf("\n======= layer2_0 shortcut ======= \n");
	LOOP_layer2_0_ConvSC:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 6
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_1x1_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {
				/* conv_1x1(
					msb_fmap_tile_buffer_1[c_in], conv_1x1_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn(
					msb_fmap_tile_buffer_0[c_in], lsb_fmap_tile_buffer[c_in],	// conv+bn shortcut
					gamma, beta,
					H_fmap_out
				); */
				printf("layer2_0 shortcut out_buf_t0: %d in batch %d\n", ini, b);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////

	printf("\n======= layer2_0 PG1 ======= \n");
	LOOP_layer2_0_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 7
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {
				/* conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				); */
				printf("layer2_0 PG1 out_buf_t1: %d in batch %d\n", ini, b);
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
	//////////// layer2_0 PG2 //////////////////////

	printf("\n======= layer2_0 PG2 ======= \n");
	LOOP_layer2_0_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 8
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {
				/* conv_3x3(
					msb_fmap_tile_buffer_1[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_0[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_0[c_in], msb_fmap_tile_buffer_1[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				); */
				printf("layer2_0 PG2 out_buf_t1: %d in batch %d\n", ini, b);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////

	printf("\n======= layer2_1 PG1 ======= \n");
	LOOP_layer2_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 9
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {
				/* conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				); */
				printf("layer2_1 PG1 out_buf_t1: %d in batch %d\n", ini, b);
			}
		}
    }

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////

	printf("\n======= layer2_1 PG2 ======= \n");
	LOOP_layer2_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 10
    	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
			for (int b = 0; b < BATCH_SIZE; b ++) {
				/* conv_3x3(
					msb_fmap_tile_buffer_0[c_in], conv_3x3_weight_tile_buffer, msb_fmap_tile_buffer_1[c_in], out_buf_t0[ini],
					stride, H_fmap_out
				);
				bn_relu(
					msb_fmap_tile_buffer_1[c_in], msb_fmap_tile_buffer_0[c_in], out_buf_t1[ini], relu_mask[ini],
					gamma, beta,
					H_fmap_out
				); */
				// printf("layer2_1 PG2 out_buf_t1: %d \n", ini);
			}
		}
    }

	printf("\n//////////////////////// Finised forward path //////////////////////// \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);
	printf("//////////////////////// Starting backward path //////////////////////// \n");

//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////


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

	printf("\n======= layer2_1 PG2 bp ======= \n");
	LOOP_layer2_1_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 10
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {
				///////////////////////////
				// conv_3x3_weight_grad_cal
				/* conv_3x3_grad(
					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				);
				// end gradient calculation */
				///////////////////////////
				printf("layer2_1 PG1 bp out_buf_t1: %d in batch %d\n", ini - out_channels/CHANNEL_OUT_T + 1, b);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////

	printf("\n======= layer2_1 PG1 bp ======= \n");
	LOOP_layer2_1_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 9
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {
				///////////////////////////
				// conv_3x3_weight_grad_cal
				/* conv_3x3_grad(
					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				); */
				// end gradient calculation
				///////////////////////////
				printf("layer2_0 PG2 bp out_buf_t1: %d in batch %d\n", ini - out_channels/CHANNEL_OUT_T + 1, b);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////

	printf("\n======= layer2_0 PG2 bp ======= \n");
	LOOP_layer2_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 8
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {
				///////////////////////////
				// conv_3x3_weight_grad_cal
				/* conv_3x3_grad(
					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				); */
				// end gradient calculation
				///////////////////////////
				printf("layer2_0 PG1 bp out_buf_t1: %d in batch %d\n", ini - out_channels/CHANNEL_OUT_T + 1, b);
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

	printf("\n======= layer2_0 PG1 bp ======= \n");
	LOOP_layer2_0_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 7
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {
				///////////////////////////
				// conv_3x3_weight_grad_cal
				/* conv_3x3_grad(
					out_buf_t1[ini - 2*out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1[c_out], grad_buf_t0,
					stride, H_fmap_in
				);
				SGD_WU_3x3(
					grad_buf_t0, conv_3x3_weight_tile_buffer, conv_3x3_weight_all[conv_3x3_weight_ptr]
				); */
				// end gradient calculation
				///////////////////////////
				printf("Conv 1 bp out_buf_t1: %d in batch %d\n", ini - 2*out_channels/CHANNEL_OUT_T + 1, b);
			}
		}
	}

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////

	printf("\n======= layer2_0 shortcut bp ======= \n");
	LOOP_layer2_0_ConvSC_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 6
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_1x1_weight_ptr -= 1;
			for (int b = BATCH_SIZE - 1; b >= 0; b --) {
				///////////////////////////
				// conv_1x1_weight_grad_cal
				/* conv_1x1_grad(
					out_buf_t1[ini - out_channels/CHANNEL_OUT_T + 1], msb_fmap_tile_buffer_1[c_out], grad_buf_t1,
					stride, H_fmap_in
				);
				SGD_WU_1x1(
					grad_buf_t1, conv_1x1_weight_tile_buffer, conv_1x1_weight_all[conv_1x1_weight_ptr]
				); */
				// end gradient calculation
				///////////////////////////
				printf("Conv 1 shortcut bp out_buf_t1: %d in batch %d\n", ini - out_channels/CHANNEL_OUT_T + 1, b);
			}
		}
	}

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 3;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;

	printf("\n======= Conv 1 bp ======= \n");
    LOOP_Conv1_bp:
	ini -= 1;
	for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
		conv_3x3_weight_ptr -= 1;
		for (int b = BATCH_SIZE - 1; b >= 0; b --) {
			///////////////////////////
			// conv_3x3_weight_grad_cal
			/* conv_3x3_grad(
				out_buf_t1[ini - out_channels/CHANNEL_OUT_T ], msb_fmap_tile_buffer_0[c_out], grad_buf_t0,
				stride, H_fmap_in
			); */
			// end gradient calculation
			///////////////////////////
			printf("Image input bp out_buf_t1: %d in batch %d\n", ini - out_channels/CHANNEL_OUT_T, b);
		}
	}
}	// end FracBNN_T

/*
======= Conv 1 + bn 1 + relu 1 ======= 
Conv 1 out_buf_t1: 1 in batch 0
Conv 1 out_buf_t1: 1 in batch 1
Conv 1 out_buf_t1: 1 in batch 2
Conv 1 out_buf_t1: 1 in batch 3

======= layer2_0 shortcut ======= 
layer2_0 shortcut out_buf_t0: 2 in batch 0
layer2_0 shortcut out_buf_t0: 2 in batch 1
layer2_0 shortcut out_buf_t0: 2 in batch 2
layer2_0 shortcut out_buf_t0: 2 in batch 3
layer2_0 shortcut out_buf_t0: 2 in batch 0
layer2_0 shortcut out_buf_t0: 2 in batch 1
layer2_0 shortcut out_buf_t0: 2 in batch 2
layer2_0 shortcut out_buf_t0: 2 in batch 3

======= layer2_0 PG1 ======= 
layer2_0 PG1 out_buf_t1: 3 in batch 0
layer2_0 PG1 out_buf_t1: 3 in batch 1
layer2_0 PG1 out_buf_t1: 3 in batch 2
layer2_0 PG1 out_buf_t1: 3 in batch 3
layer2_0 PG1 out_buf_t1: 3 in batch 0
layer2_0 PG1 out_buf_t1: 3 in batch 1
layer2_0 PG1 out_buf_t1: 3 in batch 2
layer2_0 PG1 out_buf_t1: 3 in batch 3

======= layer2_0 PG2 ======= 
layer2_0 PG2 out_buf_t1: 4 in batch 0
layer2_0 PG2 out_buf_t1: 4 in batch 1
layer2_0 PG2 out_buf_t1: 4 in batch 2
layer2_0 PG2 out_buf_t1: 4 in batch 3
layer2_0 PG2 out_buf_t1: 4 in batch 0
layer2_0 PG2 out_buf_t1: 4 in batch 1
layer2_0 PG2 out_buf_t1: 4 in batch 2
layer2_0 PG2 out_buf_t1: 4 in batch 3
layer2_0 PG2 out_buf_t1: 5 in batch 0
layer2_0 PG2 out_buf_t1: 5 in batch 1
layer2_0 PG2 out_buf_t1: 5 in batch 2
layer2_0 PG2 out_buf_t1: 5 in batch 3
layer2_0 PG2 out_buf_t1: 5 in batch 0
layer2_0 PG2 out_buf_t1: 5 in batch 1
layer2_0 PG2 out_buf_t1: 5 in batch 2
layer2_0 PG2 out_buf_t1: 5 in batch 3

======= layer2_1 PG1 ======= 
layer2_1 PG1 out_buf_t1: 6 in batch 0
layer2_1 PG1 out_buf_t1: 6 in batch 1
layer2_1 PG1 out_buf_t1: 6 in batch 2
layer2_1 PG1 out_buf_t1: 6 in batch 3
layer2_1 PG1 out_buf_t1: 6 in batch 0
layer2_1 PG1 out_buf_t1: 6 in batch 1
layer2_1 PG1 out_buf_t1: 6 in batch 2
layer2_1 PG1 out_buf_t1: 6 in batch 3
layer2_1 PG1 out_buf_t1: 7 in batch 0
layer2_1 PG1 out_buf_t1: 7 in batch 1
layer2_1 PG1 out_buf_t1: 7 in batch 2
layer2_1 PG1 out_buf_t1: 7 in batch 3
layer2_1 PG1 out_buf_t1: 7 in batch 0
layer2_1 PG1 out_buf_t1: 7 in batch 1
layer2_1 PG1 out_buf_t1: 7 in batch 2
layer2_1 PG1 out_buf_t1: 7 in batch 3

======= layer2_1 PG2 ======= 

//////////////////////// Finised forward path //////////////////////// 
ini: 9 
conv_3x3_weight_ptr: 15 
conv_1x1_weight_ptr: 2 
//////////////////////// Starting backward path //////////////////////// 

======= layer2_1 PG2 bp ======= 
layer2_1 PG1 bp out_buf_t1: 7 in batch 3
layer2_1 PG1 bp out_buf_t1: 7 in batch 2
layer2_1 PG1 bp out_buf_t1: 7 in batch 1
layer2_1 PG1 bp out_buf_t1: 7 in batch 0
layer2_1 PG1 bp out_buf_t1: 7 in batch 3
layer2_1 PG1 bp out_buf_t1: 7 in batch 2
layer2_1 PG1 bp out_buf_t1: 7 in batch 1
layer2_1 PG1 bp out_buf_t1: 7 in batch 0
layer2_1 PG1 bp out_buf_t1: 6 in batch 3
layer2_1 PG1 bp out_buf_t1: 6 in batch 2
layer2_1 PG1 bp out_buf_t1: 6 in batch 1
layer2_1 PG1 bp out_buf_t1: 6 in batch 0
layer2_1 PG1 bp out_buf_t1: 6 in batch 3
layer2_1 PG1 bp out_buf_t1: 6 in batch 2
layer2_1 PG1 bp out_buf_t1: 6 in batch 1
layer2_1 PG1 bp out_buf_t1: 6 in batch 0

======= layer2_1 PG1 bp ======= 
layer2_0 PG2 bp out_buf_t1: 5 in batch 3
layer2_0 PG2 bp out_buf_t1: 5 in batch 2
layer2_0 PG2 bp out_buf_t1: 5 in batch 1
layer2_0 PG2 bp out_buf_t1: 5 in batch 0
layer2_0 PG2 bp out_buf_t1: 5 in batch 3
layer2_0 PG2 bp out_buf_t1: 5 in batch 2
layer2_0 PG2 bp out_buf_t1: 5 in batch 1
layer2_0 PG2 bp out_buf_t1: 5 in batch 0
layer2_0 PG2 bp out_buf_t1: 4 in batch 3
layer2_0 PG2 bp out_buf_t1: 4 in batch 2
layer2_0 PG2 bp out_buf_t1: 4 in batch 1
layer2_0 PG2 bp out_buf_t1: 4 in batch 0
layer2_0 PG2 bp out_buf_t1: 4 in batch 3
layer2_0 PG2 bp out_buf_t1: 4 in batch 2
layer2_0 PG2 bp out_buf_t1: 4 in batch 1
layer2_0 PG2 bp out_buf_t1: 4 in batch 0

======= layer2_0 PG2 bp ======= 
layer2_0 PG1 bp out_buf_t1: 3 in batch 3
layer2_0 PG1 bp out_buf_t1: 3 in batch 2
layer2_0 PG1 bp out_buf_t1: 3 in batch 1
layer2_0 PG1 bp out_buf_t1: 3 in batch 0
layer2_0 PG1 bp out_buf_t1: 3 in batch 3
layer2_0 PG1 bp out_buf_t1: 3 in batch 2
layer2_0 PG1 bp out_buf_t1: 3 in batch 1
layer2_0 PG1 bp out_buf_t1: 3 in batch 0
layer2_0 PG1 bp out_buf_t1: 2 in batch 3
layer2_0 PG1 bp out_buf_t1: 2 in batch 2
layer2_0 PG1 bp out_buf_t1: 2 in batch 1
layer2_0 PG1 bp out_buf_t1: 2 in batch 0
layer2_0 PG1 bp out_buf_t1: 2 in batch 3
layer2_0 PG1 bp out_buf_t1: 2 in batch 2
layer2_0 PG1 bp out_buf_t1: 2 in batch 1
layer2_0 PG1 bp out_buf_t1: 2 in batch 0

======= layer2_0 PG1 bp ======= 
Conv 1 bp out_buf_t1: 1 in batch 3
Conv 1 bp out_buf_t1: 1 in batch 2
Conv 1 bp out_buf_t1: 1 in batch 1
Conv 1 bp out_buf_t1: 1 in batch 0
Conv 1 bp out_buf_t1: 1 in batch 3
Conv 1 bp out_buf_t1: 1 in batch 2
Conv 1 bp out_buf_t1: 1 in batch 1
Conv 1 bp out_buf_t1: 1 in batch 0

======= layer2_0 shortcut bp ======= 
Conv 1 shortcut bp out_buf_t1: 1 in batch 3
Conv 1 shortcut bp out_buf_t1: 1 in batch 2
Conv 1 shortcut bp out_buf_t1: 1 in batch 1
Conv 1 shortcut bp out_buf_t1: 1 in batch 0
Conv 1 shortcut bp out_buf_t1: 1 in batch 3
Conv 1 shortcut bp out_buf_t1: 1 in batch 2
Conv 1 shortcut bp out_buf_t1: 1 in batch 1
Conv 1 shortcut bp out_buf_t1: 1 in batch 0

======= Conv 1 bp ======= 
Image input bp out_buf_t1: 0 in batch 3
Image input bp out_buf_t1: 0 in batch 2
Image input bp out_buf_t1: 0 in batch 1
Image input bp out_buf_t1: 0 in batch 0
*/