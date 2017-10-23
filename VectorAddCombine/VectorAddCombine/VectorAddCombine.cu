
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N (2048*2048)
#define N_THREADS_PER_BLOCK 512

// Adapt vector addition to use both blocks and threads
__global__ void addByCombine(int *a, int *b, int *c)
{
	// use the built-in variable blockDim.x for threads per block
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
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
	int *host_a, *host_b, *host_c;    // host copies of host_a, host_b, host_c
	int *device_a, *device_b, *device_c; // device copies of device_a, device_b, device_c

	int size = N * sizeof(int);

	// Alloc space for device copies of device_a, device_b, device_c
	cudaMalloc(&device_a, size);
	cudaMalloc(&device_b, size);
	cudaMalloc(&device_c, size);

	// Alloc space for host copies of host_a, host_b, host_c 
	// and setup input values
	host_a = (int*)malloc(size);
	random_ints(host_a, N);

	host_b = (int*)malloc(size);
	random_ints(host_b, N);

	host_c = (int*)malloc(size);

	// Copy input to device
	cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

	// Launch addByCombine() kernel on GPU 
	addByCombine << <N/ N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK >> > (device_a, device_b, device_c);

	// Copy result back to host
	cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i<N; i++) {
		printf("host_a[%d]=%d , host_b[%d]=%d, host_c[%d]=%d\n", i, host_a[i], i, host_b[i], i, host_c[i]);
	}

	// Cleanup
	free(host_a); free(host_b); free(host_c);
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);


    return 0;
}

