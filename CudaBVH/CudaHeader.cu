#include "CudaHeader.cuh"

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