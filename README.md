# CNN training accelerator

The repo describes an HLS-based CNN training accelerator in floating-point format for a reference design, using the back-propagation algorithm with the SGD optimizer.

## Structure  

**pytorch**- Verification of back-propagation derivatives, including Conv, transposed Conv, dilated Conv, BN, ReLU, average pooling, and FC.  

> ```BP_function.py``` <br>
> ```fc_test.py``` <br>


**resnet20**- HLS design of the accelerator with both input and output channel tiling.

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


>> ```data_batch_1.bin``` <br>
>> ```train.bin``` <br> (image data from http://www.cs.toronto.edu/~kriz/cifar.html)


