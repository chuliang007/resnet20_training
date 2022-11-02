# CNN training using block minifloat

## Structure  

> Pytorch- Verification of back-propagation derivatives, including Conv, transposed Conv, dilated Conv, BN, ReLU, average pooling, and FC.  

>> BP_function.py  
>> fc_test.py

> resnet20 and vgg-like- HLS design of the accelerator with both input and output channel tiling.

>> design source files    
>>> ```bnn.h``` <br>
>>> ```conv_weights.h``` <br>
>>> ```dimension_def.h``` <br> 
>>> ```layer.h``` <br>
>>> ```resnet20.cc``` <br>
>>> ```typedefs.h``` <br>

>> testbench files
>>> ```conv_weights_tb.h``` <br>
>>> ```tb.cc``` <br>
>>> ```weights_tb.h``` <br>
