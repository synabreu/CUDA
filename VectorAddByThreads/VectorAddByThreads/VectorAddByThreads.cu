
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


#define N_THREADS 512

__global__ void addByThreads(int *a, int *b, int *c)
{
	// a block can be split into parallel threads.
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
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

	int size = N_THREADS * sizeof(int);

	// Alloc space for device copies of device_a, device_b, device_c
	cudaMalloc(&device_a, size);
	cudaMalloc(&device_b, size);
	cudaMalloc(&device_c, size);

	// Alloc space for host copies of host_a, host_b, host_c 
	// and setup input values
	host_a = (int*)malloc(size);
	random_ints(host_a, N_THREADS);

	host_b = (int*)malloc(size);
	random_ints(host_b, N_THREADS);

	host_c = (int*)malloc(size);

	// Copy input to device
	cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with N threads
	addByThreads << <1, N_THREADS >> > (device_a, device_b, device_c);

	// Copy result back to host
	cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i<N_THREADS; i++) {
		printf("host_a[%d]=%d , host_b[%d]=%d, host_c[%d]=%d\n", i, host_a[i], i, host_b[i], i, host_c[i]);
	}

	// Cleanup
	free(host_a); free(host_b); free(host_c);
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

    return 0;
}

