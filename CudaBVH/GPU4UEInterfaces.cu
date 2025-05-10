#include "GPU4UEInterfaces.cuh"

#include "ParallelRaysIntersectionWithCuda.cuh"
#include "aabb.cuh"
#include "BoundBoxCuda.cuh"

namespace GPU4UE
{

	static lbvh::bvh<float, TriangleCuda<float4>, TriangleCudaAABBGetter> bvh;

	// TODO: 改成能分线程不断分配的，现在这么写容易炸显存
	static BoundBoxCuda* dev_cells = nullptr;
	static BoundBoxCuda* dev_meshboxes = nullptr;
	static RayCuda<float4>* dev_out_rays = nullptr;


	void InitBVH(const std::vector<TriangleCuda<float4>>& triangles)
	{
		bvh.assign(triangles.begin(), triangles.end());
	}

	void ParallelRaysIntersectionWithBVHCuda2(std::vector<RayCuda<float4>>& rays, int* results)
	{
		int num_rays = rays.size();

		static const auto bvh_dev = bvh.get_device_repr();

		// 利用gpu bvh求交
		ParallelRaysIntersectionWithBVHCuda(rays.data(), num_rays, bvh_dev, results);
	}

	void InitCellBounds(std::vector<BoundBoxCuda>& cells)
	{
		CUDA_CALL(cudaSetDevice(0));

		if (dev_cells)
		{
			CUDA_CALL(cudaFree(dev_cells));
			dev_cells = nullptr;
		}

		CUDA_CALL(cudaMalloc((void**)&dev_cells, cells.size()*sizeof(BoundBoxCuda)));
		CUDA_CALL(cudaMemcpy(dev_cells, cells.data(), cells.size() * sizeof(BoundBoxCuda), cudaMemcpyHostToDevice));
	}

	void InitMeshBounds(std::vector<BoundBoxCuda>& meshboxes)
	{
		CUDA_CALL(cudaSetDevice(0));

		if (dev_meshboxes)
		{
			CUDA_CALL(cudaFree(dev_meshboxes));
			dev_meshboxes = nullptr;
		}

		CUDA_CALL(cudaMalloc((void**)&dev_meshboxes, meshboxes.size() * sizeof(BoundBoxCuda)));
		CUDA_CALL(cudaMemcpy(dev_meshboxes, meshboxes.data(), meshboxes.size() * sizeof(BoundBoxCuda), cudaMemcpyHostToDevice));
	}

	void InitOutRays(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample)
	{
		CUDA_CALL(cudaSetDevice(0));

		if (dev_out_rays)
		{
			CUDA_CALL(cudaFree(dev_out_rays));
			dev_out_rays = nullptr;
		}

		size_t length = num_cells * num_meshboxes * num_cell_sample * num_meshbox_sample;

		CUDA_CALL(cudaMalloc((void**)&dev_out_rays, length * sizeof(BoundBoxCuda)));
	}

	// 多线程call this
	void GetOutRaysCuda(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample, int st, int ed)
	{		
		CUDA_CALL(cudaSetDevice(0));

		size_t total_thread_num = num_cells * num_meshboxes * num_cell_sample * num_meshbox_sample;

		// TODO: 计算方法改变
		//size_t threads_per_block = 256;
		//size_t blocks_per_grid = (total_thread_num +threads_per_block - 1) / threads_per_block;

		dim3 threads_rect(num_cell_sample, num_meshbox_sample); // x,y
		dim3 blocks_rect(num_cells, num_meshboxes); // x,y

		GetOutRaysKernel << < blocks_rect, threads_rect >> > (dev_cells, dev_meshboxes, dev_out_rays);



		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());
		
	}


	int Test(int a, int b)
	{
		return a + b;
	}
}