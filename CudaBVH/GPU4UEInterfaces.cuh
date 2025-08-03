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

	/*
		暂时没用上
	*/
	void ParallelRaysIntersectionWithBVHAndRaysCuda2(int* results);

	void InitCellBoundsCuda(std::vector<BoundBoxCuda>& cells);

	void InitMeshBoundsCuda(std::vector<BoundBoxCuda>& meshboxes);

	void InitOutRaysCuda(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample);

	void InitResults(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample);

	void ComputeOutRaysCuda(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample, int st, int ed);

	/*
	* Just debug use
	*/
	std::vector<RayCuda<float4>> GetOutRaysFromCuda();


	/*
		无需提供ray，而是直接使用已经在GPU上生成的光线去做求交，且求交结果会先传回host后放在全局
	*/
	void ParallelRaysIntersectionWithBVHAndRaysCuda3();

	int* GetHostResults();

	void CopyHostResultsToVec();

	std::vector<int> GetHostResultsVec();

	size_t GetDevOutRaysLength();

	int Test(int a, int b);
}