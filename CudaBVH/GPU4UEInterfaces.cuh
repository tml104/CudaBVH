#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <vector>

#include "TriangleCuda.cuh"
#include "RayCuda.cuh"


namespace GPU4UE
{
	void InitBVH(const std::vector<TriangleCuda<float4>>& triangles);

	void ParallelRaysIntersectionWithBVHCuda2(std::vector<RayCuda<float4>>& rays, int* results);

	int Test(int a, int b);
}