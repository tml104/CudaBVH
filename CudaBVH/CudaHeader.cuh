#pragma once 

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CALL(x) {const cudaError_t a = (x); if(a!=cudaSuccess){printf("\nCUDA Error: %s(err_num=%d)\n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);}}

__host__ void cuda_error_check(const char* prefix, const char* postfix)
{
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
		cudaDeviceReset();
		//wait_exit();
		exit(1);
	}
}