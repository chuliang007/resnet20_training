# CNN training using block minifloat

## Structure  

> Pytorch- Verification of back-propagation derivatives, including Conv, transposed Conv, dilated Conv, BN, ReLU, average pooling, and FC.  

>> BP_function.py  
>> fc_test.py

> resnet20 and vgg-like- HLS design of the accelerator with both input and output channel tiling.

>> design source files    
>>> ```c bnn.h ```
>>> conv_weights.h   
>>> dimension_def.h   
>>> layer.h   
>>> resnet20.cc  
>>> typedefs.h  

>> testbench files
>>> conv_weights_tb.h  
>>> tb.cc  
>>> weights_tb.h  
