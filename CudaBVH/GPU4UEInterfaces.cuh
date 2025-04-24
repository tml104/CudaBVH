#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <vector>

#include "TriangleCuda.cuh"
#include "RayCuda.cuh"


namespace GPU4UE
{
	void ParallelRaysIntersectionWithBVHCuda2(std::vector<RayCuda<float4>>& rays, int* results, const std::vector<TriangleCuda<float4>>& triangles = std::vector<TriangleCuda<float4>>{});


	int Test(int a, int b);
}