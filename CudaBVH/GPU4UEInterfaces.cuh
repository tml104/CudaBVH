#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <vector>

#include "TriangleCuda.cuh"
#include "RayCuda.cuh"
#include "BoundBoxCuda.cuh"

namespace GPU4UE
{
	void InitBVH(const std::vector<TriangleCuda<float4>>& triangles);

	void ParallelRaysIntersectionWithBVHCuda2(std::vector<RayCuda<float4>>& rays, int* results);

	void InitCellBounds(std::vector<BoundBoxCuda>& cells);

	void InitMeshBounds(std::vector<BoundBoxCuda>& meshboxes);

	void InitOutRays(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample);

	void GetOutRaysCuda(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample, int st, int ed);


	int Test(int a, int b);
}