
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


#define N_BLOCKS 10

__global__ void addKernel(int *a, int *b, int *c)
{
 	// each parallel invocation of add() is referred to as a block.
	// The set of blocks is referred to as a grid.
	// Each invocation can refer to its block index using blockIdx.x.
	// By using blockIdx.x to index into the array, each block handles a different index.
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int* x, int size)
{
	int i;
	for (i = 0; i<size; i++) {
		x[i] = rand() % 10;
	}
}


int main()
{
	int *host_a, *host_b, *host_c;    // host copies of a, b, c
	int *device_a, *device_b, *device_c; // device copies of a, b, c

	int size = N_BLOCKS * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc(&device_a, size);
	cudaMalloc(&device_b, size);
	cudaMalloc(&device_c, size);

	// Alloc space for host copies of a, b, c and setup input values
	host_a = (int*)malloc(size);
	random_ints(host_a, N_BLOCKS);

	host_b = (int*)malloc(size);
	random_ints(host_b, N_BLOCKS);

	host_c = (int*)malloc(size);

	// Copy input to device
	cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with N blocks
	addKernel << <N_BLOCKS, 1 >> > (device_a, device_b, device_c);

	// Copy result back to host
	cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i<N_BLOCKS; i++) {
		printf("host_a[%d]=%d , host_b[%d]=%d, host_c[%d]=%d\n", i, host_a[i], i, host_b[i], i, host_c[i]);
	}

	// Cleanup
	free(host_a); free(host_b); free(host_c);
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);


    return 0;
}

