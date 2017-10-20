
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// In device's code, use pointers for the variables, 
// which means parameters in the addTwoIntegersKernel function. 
// So a, b, and c must point to device memory and then allocate memory on GPU internally.
__global__ void addTwoIntegers(int *a, int *b, int *c)
{
//	int i = threadIdx.x;
	*c = *a + *b;
}

int main(void)
{
	int host_a, host_b, host_c; // host copies of a, b, c
	int *device_a, *device_b, *device_c; // device copies of a, b, c
	int size = sizeof(int);

	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&device_a, size);
	cudaMalloc((void **)&device_b, size);
	cudaMalloc((void **)&device_c, size);

	// Setup input values
	host_a = 10;
	host_b = 19;

	// copy inputs to device
	cudaMemcpy(device_a, &host_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, &host_b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	addTwoIntegers<<<1,1>>>(device_a, device_b, device_c);
	
	// Copy result back to host
	cudaMemcpy(&host_c, device_c, size, cudaMemcpyDeviceToHost);
	
	// debug code
	printf("%d\n", host_a + host_b);
	printf("%d\n", host_c);
	printf("%d\n", device_c);

	// Cleanup
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
    return 0;
}

