#define BATCH_SIZE 4
#define CHANNEL_IN_T 64
#define CHANNEL_OUT_T 64
#define WIDTH 33

int main()
{
	int H_fmap_in, H_fmap_out, in_channels, in_channels_after_pack, out_channels_after_pack;
    int out_channels, out_channel_start, stride, conv_3x3_weight_ptr, conv_1x1_weight_ptr, fc_weight_ptr, ini;
	int ctrl_sc, ctrl_fc, ctrl_avgpool;

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////		Forward path		//////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	ini = 0;
	ctrl_sc = 1; // if ctrl_sc=1, generate and send out_copy into DDR

	printf("======= Get Image ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 3;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels = 64;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;
	conv_3x3_weight_ptr = 0;

    LOOP_Conv1:
	ini += 1;	// ini = 1
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
		conv_3x3_weight_ptr += 1;
	}

	printf("======= Conv1 + bn_relu ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 64;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels = 64;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
	LOOP_layer1_0_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 2
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
    }
	printf("======= layer1_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	LOOP_layer1_0_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 3
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
    }
	printf("======= layer1_0 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	LOOP_layer1_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 4
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
    }
	printf("======= layer1_1 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
	LOOP_layer1_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 5
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
    }
	printf("======= layer1_1 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 2 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 16;
	in_channels = 64;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels = 128;
	stride = 2;
	conv_1x1_weight_ptr = 0;

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
	LOOP_layer2_0_ConvSC:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 6
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_1x1_weight_ptr += 1;
		}
    }
	printf("======= layer2_0 shortcut ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 7
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
    }
	printf("======= layer2_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 128;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels = 128;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
	LOOP_layer2_0_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 8
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer2_0 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	LOOP_layer2_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 9
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer2_1 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	LOOP_layer2_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 10
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer2_0 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 3 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 8;
	in_channels = 128;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels = 256;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
	LOOP_layer3_0_ConvSC:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 11
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_1x1_weight_ptr += 1;
		}
	}
	printf("======= layer3_0 shortcut ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 12
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer3_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 256;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels = 256;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
	LOOP_layer3_0_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 13
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer3_0 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);
	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	LOOP_layer3_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 14
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer3_1 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
	LOOP_layer3_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 15
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer3_1 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 4 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 4;
	in_channels = 256;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels = 512;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer4_0 shortcut (conv+bn) ///////
	LOOP_layer4_0_ConvSC:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 16
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_1x1_weight_ptr += 1;
		}
	}
	printf("======= layer4_0 shortcut ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_0 PG1 //////////////////////
	LOOP_layer4_0_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 17
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer4_0 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 4 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 4;
	in_channels = 512;
	in_channels_after_pack = in_channels/CHANNEL_IN_T;
	out_channels = 512;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer4_0 PG2 //////////////////////
	LOOP_layer4_0_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 18
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer4_0 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_1 PG1 //////////////////////
	LOOP_layer4_1_Conv1:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 19
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer4_1 PG1 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
	LOOP_layer4_1_Conv2:
	for (int c_in = 0; c_in < in_channels/CHANNEL_IN_T; c_in ++) {
		ini += 1;	// ini = 20
		for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
			conv_3x3_weight_ptr += 1;
		}
	}
	printf("======= layer4_1 PG2 ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// AvgPool and FC // /////////////////

	// initialize buffers for pooling and FC
	AvgPool_FC_buf_init:
	for (int b = 0; b < BATCH_SIZE; b ++) {
		for (int i = 0; i < 512; i ++) {
		}
		for (int j = 0; j < 10; j ++) {
		}
	}

	// avgpool
	LOOP_AvgPool:
	ctrl_avgpool = 0;
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
	}

	// FC
	LOOP_FC:
	fc_weight_ptr = 0;
	ctrl_fc = 0;

	write_output:
	for (int j = 0; j < 10; j ++) {
		for (int b = 0; b < BATCH_SIZE; b ++) {
		}
	}

	////////////////////////////////////////////////
	//////////// CrossEntropy loss /////////////////

	error_calc:
	// MSE for simplicity
		for (int j = 0; j < 10; j ++) {
			for (int b = 0; b < BATCH_SIZE; b ++) {
            }
	    }

	printf("\n============== Finished forward ============== \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);
	printf("============== Finished forward ============== \n");
    
//////////////////////////////////////////////////////////////////////////////////////
//////////////		Backward path and Gradient Calc & Weight update		//////////////
//////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////
	//////////// AvgPool and FC // /////////////////

	// initialize buffers for pooling and FC
	AvgPool_buf_init_bp:
	for (int i = 0; i < 512; i ++) {
		for (int b = 0; b < BATCH_SIZE; b ++) {
		}
	}

	// FC_bp
	LOOP_FC_bp:
	ctrl_fc = 1;

	// avgpool_bp
	LOOP_AvgPool_bp:
	ctrl_avgpool = 1;
	for (int c_out = 0; c_out < out_channels/CHANNEL_OUT_T; c_out ++) {
	}

	////////////////////////////////////////////////
	//////////// LAYER 4 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 4;
	in_channels = 512;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	out_channels = 512;
	stride = 1;

	ctrl_sc = 0; // do not generate output_copy of shortcut

	////////////////////////////////////////////////
	//////////// layer4_1 PG2 //////////////////////
	LOOP_layer4_1_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 20
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
    }
	printf("\n======= layer4_1 PG2 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_1 PG1 //////////////////////
	LOOP_layer4_1_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 19
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
    }
	printf("======= layer4_1 PG1 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_0 PG2 //////////////////////
	LOOP_layer4_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 18
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
    }
	printf("======= layer4_0 PG2 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 4 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 4;
	H_fmap_out = 8;
	in_channels = 512;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	out_channels = 256;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer4_0 PG1 //////////////////////
	LOOP_layer4_0_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 17
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
    }
	printf("======= layer4_0 PG1 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer4_0 shortcut (conv+bn) ///////
	LOOP_layer4_0_ConvSC_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 16
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_1x1_weight_ptr -= 1;
		}
    }
	printf("======= layer4_0 shortcut bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 256;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	out_channels = 256;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 //////////////////////
	LOOP_layer3_1_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 15
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
	}
	printf("======= layer3_1 PG2 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 //////////////////////
	LOOP_layer3_1_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 14
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
	}
	printf("======= layer3_1 PG1 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 //////////////////////
	LOOP_layer3_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 13
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
	}
	printf("======= layer3_0 PG2 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 3 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 16;
	in_channels = 256;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	out_channels = 128;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 //////////////////////
	LOOP_layer3_0_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 12
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
	}
	printf("======= layer3_0 PG1 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer3_0 shortcut (conv+bn) ///////
	LOOP_layer3_0_ConvSC_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 11
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_1x1_weight_ptr -= 1;
		}
	}
	printf("======= layer3_0 shortcut bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 128;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	out_channels = 128;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 //////////////////////
	LOOP_layer2_1_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 10
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
	}
	printf("======= layer2_1 PG2 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 //////////////////////
	LOOP_layer2_1_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 9
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
	}
	printf("======= layer2_1 PG1 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 //////////////////////
	LOOP_layer2_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 8
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
	}
	printf("======= layer2_0 PG2 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 2 Upsample //////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 32;
	in_channels = 128;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	out_channels = 64;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 //////////////////////
	LOOP_layer2_0_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 7
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
	}
	printf("======= layer2_0 PG1 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer2_0 shortcut (conv+bn) ///////
	LOOP_layer2_0_ConvSC_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 6
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_1x1_weight_ptr -= 1;
		}
	}
	printf("======= layer2_0 shortcut bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 64;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	out_channels = 64;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 //////////////////////
	LOOP_layer1_1_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 5
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
    }
	printf("======= layer1_1 PG2 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 //////////////////////
	LOOP_layer1_1_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 4
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
    }
	printf("======= layer1_1 PG1 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 //////////////////////
	LOOP_layer1_0_Conv2_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 3
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
    }
	printf("======= layer1_0 PG2 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 //////////////////////
	LOOP_layer1_0_Conv1_bp:
	for (int c_out = out_channels/CHANNEL_OUT_T - 1; c_out >= 0; c_out --) {
		ini -= 1;	// ini = 2
		for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
			conv_3x3_weight_ptr -= 1;
		}
    }
	printf("======= layer1_0 PG1 bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);

	////////////////////////////////////////////////
	/////////// Conv 1 + bn 1 + relu 1 /////////////
	////////////////////////////////////////////////

	in_channels = 64;
	out_channels = 3;
	out_channels_after_pack = out_channels/CHANNEL_OUT_T;
	H_fmap_in =32;
	H_fmap_out = 32;
	stride = 1;

    LOOP_Conv1_bp:
	ini -= 1;	// ini = 1
	for (int c_in = in_channels/CHANNEL_IN_T - 1; c_in >=0; c_in --) {
		conv_3x3_weight_ptr -= 1;
	}
	printf("======= Conv 1 + bn_relu bp ======= \n");
	printf("ini: %d \n", ini);
	printf("conv_3x3_weight_ptr: %d \n", conv_3x3_weight_ptr);
	printf("conv_1x1_weight_ptr: %d \n", conv_1x1_weight_ptr);
}	// end FracBNN_T

/*
======= Get Image ======= 
ini: 0 
conv_3x3_weight_ptr: 32 
conv_1x1_weight_ptr: 0 
======= Conv1 + bn_relu ======= 
ini: 1 
conv_3x3_weight_ptr: 1 
conv_1x1_weight_ptr: 0 
======= layer1_0 PG1 ======= 
ini: 2 
conv_3x3_weight_ptr: 2 
conv_1x1_weight_ptr: 0 
======= layer1_0 PG2 ======= 
ini: 3 
conv_3x3_weight_ptr: 3 
conv_1x1_weight_ptr: 0 
======= layer1_1 PG1 ======= 
ini: 4 
conv_3x3_weight_ptr: 4 
conv_1x1_weight_ptr: 0 
======= layer1_1 PG2 ======= 
ini: 5 
conv_3x3_weight_ptr: 5 
conv_1x1_weight_ptr: 0 
======= layer2_0 shortcut ======= 
ini: 6 
conv_3x3_weight_ptr: 5 
conv_1x1_weight_ptr: 2 
======= layer2_0 PG1 ======= 
ini: 7 
conv_3x3_weight_ptr: 7 
conv_1x1_weight_ptr: 2 
======= layer2_0 PG2 ======= 
ini: 9 
conv_3x3_weight_ptr: 11 
conv_1x1_weight_ptr: 2 
======= layer2_1 PG1 ======= 
ini: 11 
conv_3x3_weight_ptr: 15 
conv_1x1_weight_ptr: 2 
======= layer2_0 PG2 ======= 
ini: 13 
conv_3x3_weight_ptr: 19 
conv_1x1_weight_ptr: 2 
======= layer3_0 shortcut ======= 
ini: 15 
conv_3x3_weight_ptr: 19 
conv_1x1_weight_ptr: 10 
======= layer3_0 PG1 ======= 
ini: 17 
conv_3x3_weight_ptr: 27 
conv_1x1_weight_ptr: 10 
======= layer3_0 PG2 ======= 
ini: 21 
conv_3x3_weight_ptr: 43 
conv_1x1_weight_ptr: 10 
======= layer3_1 PG1 ======= 
ini: 25 
conv_3x3_weight_ptr: 59 
conv_1x1_weight_ptr: 10 
======= layer3_1 PG2 ======= 
ini: 29 
conv_3x3_weight_ptr: 75 
conv_1x1_weight_ptr: 10 
======= layer4_0 shortcut ======= 
ini: 33 
conv_3x3_weight_ptr: 75 
conv_1x1_weight_ptr: 42 
======= layer4_0 PG1 ======= 
ini: 37 
conv_3x3_weight_ptr: 107 
conv_1x1_weight_ptr: 42 
======= layer4_0 PG2 ======= 
ini: 45 
conv_3x3_weight_ptr: 171 
conv_1x1_weight_ptr: 42 
======= layer4_1 PG1 ======= 
ini: 53 
conv_3x3_weight_ptr: 235 
conv_1x1_weight_ptr: 42 
======= layer4_1 PG2 ======= 
ini: 61 
conv_3x3_weight_ptr: 299 
conv_1x1_weight_ptr: 42 

============== Finished forward ============== 
ini: 61 
conv_3x3_weight_ptr: 299 
conv_1x1_weight_ptr: 42 
============== Finished forward ============== 

======= layer4_1 PG2 bp ======= 
ini: 53 
conv_3x3_weight_ptr: 235 
conv_1x1_weight_ptr: 42 
======= layer4_1 PG1 bp ======= 
ini: 45 
conv_3x3_weight_ptr: 171 
conv_1x1_weight_ptr: 42 
======= layer4_0 PG2 bp ======= 
ini: 37 
conv_3x3_weight_ptr: 107 
conv_1x1_weight_ptr: 42 
======= layer4_0 PG1 bp ======= 
ini: 33 
conv_3x3_weight_ptr: 75 
conv_1x1_weight_ptr: 42 
======= layer4_0 shortcut bp ======= 
ini: 29 
conv_3x3_weight_ptr: 75 
conv_1x1_weight_ptr: 10 
======= layer3_1 PG2 bp ======= 
ini: 25 
conv_3x3_weight_ptr: 59 
conv_1x1_weight_ptr: 10 
======= layer3_1 PG1 bp ======= 
ini: 21 
conv_3x3_weight_ptr: 43 
conv_1x1_weight_ptr: 10 
======= layer3_0 PG2 bp ======= 
ini: 17 
conv_3x3_weight_ptr: 27 
conv_1x1_weight_ptr: 10 
======= layer3_0 PG1 bp ======= 
ini: 15 
conv_3x3_weight_ptr: 19 
conv_1x1_weight_ptr: 10 
======= layer3_0 shortcut bp ======= 
ini: 13 
conv_3x3_weight_ptr: 19 
conv_1x1_weight_ptr: 2 
======= layer2_1 PG2 bp ======= 
ini: 11 
conv_3x3_weight_ptr: 15 
conv_1x1_weight_ptr: 2 
======= layer2_1 PG1 bp ======= 
ini: 9 
conv_3x3_weight_ptr: 11 
conv_1x1_weight_ptr: 2 
======= layer2_0 PG2 bp ======= 
ini: 7 
conv_3x3_weight_ptr: 7 
conv_1x1_weight_ptr: 2 
======= layer2_0 PG1 bp ======= 
ini: 6 
conv_3x3_weight_ptr: 5 
conv_1x1_weight_ptr: 2 
======= layer2_0 shortcut bp ======= 
ini: 5 
conv_3x3_weight_ptr: 5 
conv_1x1_weight_ptr: 0 
======= layer1_1 PG2 bp ======= 
ini: 4 
conv_3x3_weight_ptr: 4 
conv_1x1_weight_ptr: 0 
======= layer1_1 PG1 bp ======= 
ini: 3 
conv_3x3_weight_ptr: 3 
conv_1x1_weight_ptr: 0 
======= layer1_0 PG2 bp ======= 
ini: 2 
conv_3x3_weight_ptr: 2 
conv_1x1_weight_ptr: 0 
======= layer1_0 PG1 bp ======= 
ini: 1 
conv_3x3_weight_ptr: 1 
conv_1x1_weight_ptr: 0 
======= Conv 1 + bn_relu bp ======= 
ini: 0 
conv_3x3_weight_ptr: 0 
conv_1x1_weight_ptr: 0 
*/