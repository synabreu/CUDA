
#include "cuda_runtime.h"
#include <stdio.h>


// __global_- keyword in CUDA C/C++ indicates a function that 
// it run on the devices and it is called from host code.
// This is the device components processed by NVIDIA compiler(nvcc).
__global__ void mykernel(void)
{
}

// Host functions processed by standard host compiler. ex) GCC, VS including Nsight. 
int main()
{
	// Launch kernel from host code to device code for executing a function on the GPU!
	// We'll return to the parameters (1,1) in a moment
	mykernel <<<1, 1 >>>();

	printf("Hello, CUDA!\n");
    return 0;
}

