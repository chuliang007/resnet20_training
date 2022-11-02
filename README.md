# CNN training using block minifloat

The blind link describes an HLS-based CNN training accelerator using the back-propagation algorithm with the SGD optimiser. 

## Structure  

**Pytorch**- Verification of back-propagation derivatives, including Conv, transposed Conv, dilated Conv, BN, ReLU, average pooling, and FC.  

> ```BP_function.py``` <br>
> ```fc_test.py``` <br>


**resnet20** and **vgg-like**- HLS design of the accelerator with both input and output channel tiling.

> design source files    
>> ```bnn.h``` <br>
>> ```conv_weights.h``` <br>
>> ```dimension_def.h``` <br> 
>> ```layer.h``` <br>
>> ```resnet20.cc``` <br>
>> ```typedefs.h``` <br>

> testbench files
>> ```conv_weights_tb.h``` <br>
>> ```tb.cc``` <br>
>> ```weights_tb.h``` <br>


## P.S.

The current open-source code is in floating-point format to give a quick functional verification of the training accelerator design. A full version using the block minifloat will be released afterwards.
