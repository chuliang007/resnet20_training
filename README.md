# resnet18_training

1. 'resnet20.cc' defines an FPGA-based ResNet18 training accelerator for CIFAR10.
2. Layer functions used in the accelerator are defined in 'layer.h'.
3. Python codes for PyTorch simulations are included in 'BP_function.py' in the directory of 'PyTorch_BP'.

P.S. Note that HLS C simulations cannot be processed due to no data streaming in.
