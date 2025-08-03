#include "GPU4UEInterfaces.cuh"

#include <curand_kernel.h>
#include <chrono>

#include "ParallelRaysIntersectionWithCuda.cuh"
#include "aabb.cuh"
#include "BoundBoxCuda.cuh"
#include "Geometry.cuh"
#include "ComputeOutRaysWithCuda.cuh"


namespace GPU4UE
{

	static lbvh::bvh<float, TriangleCuda<float4>, TriangleCudaAABBGetter> bvh;

	// TODO: 改成能分线程不断分配的，现在这么写容易炸显存
	static BoundBoxCuda* dev_cells = nullptr;
	static BoundBoxCuda* dev_meshboxes = nullptr;
	static RayCuda<float4>* dev_out_rays = nullptr;
	static size_t dev_out_rays_length = 0;

	static int* host_results = nullptr;
	static std::vector<int> host_results_vec;

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

	void InitCellBoundsCuda(std::vector<BoundBoxCuda>& cells)
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

	void InitMeshBoundsCuda(std::vector<BoundBoxCuda>& meshboxes)
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

	void InitOutRaysCuda(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample)
	{
		CUDA_CALL(cudaSetDevice(0));

		if (dev_out_rays)
		{
			CUDA_CALL(cudaFree(dev_out_rays));
			dev_out_rays = nullptr;
		}

		size_t length = num_cells * num_meshboxes * num_cell_sample * num_meshbox_sample;
		dev_out_rays_length = length;

		CUDA_CALL(cudaMalloc((void**)&dev_out_rays, length * sizeof(RayCuda<float4>)));
	}

	void InitResults(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample)
	{
		CUDA_CALL(cudaSetDevice(0));

		if (host_results)
		{
			//CUDA_CALL(cudaFreeHost(host_results));
			delete[] host_results;
			host_results = nullptr;
		}

		size_t length = num_cells * num_meshboxes * num_cell_sample * num_meshbox_sample;
		dev_out_rays_length = length;

		//CUDA_CALL(cudaMallocHost((void**)&host_results, length * sizeof(int)));
		host_results = new int[length];
	}

	// 多线程call this
	// st, ed参数暂时没用
	void ComputeOutRaysCuda(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample, int st, int ed)
	{		
		CUDA_CALL(cudaSetDevice(0));

		curandState_t* device_states;
		CUDA_CALL(cudaMalloc((void**)&device_states, dev_out_rays_length * sizeof(curandState_t)));
		unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();


		// TODO: 计算方法改变
		//size_t threads_per_block = 256;
		//size_t blocks_per_grid = (dev_out_rays_length +threads_per_block - 1) / threads_per_block;

		//dim3 threads_rect(num_cell_sample, num_meshbox_sample); // x,y
		//dim3 blocks_rect(num_cells, num_meshboxes); // x,y

		// 注意：x是变化最快的，所以可能得反着写
		dim3 threads_rect(num_meshbox_sample, num_cell_sample);
		dim3 blocks_rect(num_meshboxes, num_cells);

		SetupCuRand << < blocks_rect, threads_rect >> > (device_states, seed, dev_out_rays_length);
		CUDA_CALL(cudaGetLastError());

		GetOutRaysKernel << < blocks_rect, threads_rect >> > (dev_cells, dev_meshboxes, dev_out_rays, device_states, num_cells, num_meshboxes, num_cell_sample, num_meshbox_sample, dev_out_rays_length);

		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());

		CUDA_CALL(cudaFree(device_states));
	}

	/*
	*  仅调试查看光线生成结果用，实际使用的时候应该避免从GPU把数据传回来，会很慢
	*/
	std::vector<RayCuda<float4>> GetOutRaysFromCuda()
	{
		std::vector<RayCuda<float4>> result_rays_vec;
		RayCuda<float4>* result_rays_ptr;

		if (dev_out_rays)
		{
			CUDA_CALL(cudaSetDevice(0));

			CUDA_CALL(cudaMallocHost((void**)(&result_rays_ptr), dev_out_rays_length * sizeof(RayCuda<float4>)));

			CUDA_CALL(cudaMemcpy(result_rays_ptr, dev_out_rays, dev_out_rays_length * sizeof(RayCuda<float4>), cudaMemcpyDeviceToHost));

			for (int i = 0; i < dev_out_rays_length; i++)
			{
				result_rays_vec.emplace_back(result_rays_ptr[i]);
			}

			CUDA_CALL(cudaFreeHost(result_rays_ptr));
		}

		return result_rays_vec;
	}

	void ParallelRaysIntersectionWithBVHAndRaysCuda2(int* results)
	{
		size_t num_rays = dev_out_rays_length;

		static const auto bvh_dev = bvh.get_device_repr();

		// 利用gpu bvh求交
		ParallelRaysIntersectionWithBVHAndRaysCuda(dev_out_rays, num_rays, bvh_dev, results);
	}

	void ParallelRaysIntersectionWithBVHAndRaysCuda3()
	{
		size_t num_rays = dev_out_rays_length;

		static const auto bvh_dev = bvh.get_device_repr();

		// 利用gpu bvh求交
		ParallelRaysIntersectionWithBVHAndRaysCuda(dev_out_rays, num_rays, bvh_dev, host_results);
	}

	int* GetHostResults()
	{
		return host_results;
	}

	void CopyHostResultsToVec()
	{
		host_results_vec.clear();
		for (int i = 0; i < dev_out_rays_length; i++)
		{
			host_results_vec.emplace_back(host_results[i]);
		}
	}

	std::vector<int> GetHostResultsVec()
	{
		return host_results_vec;
	}

	size_t GetDevOutRaysLength()
	{
		return dev_out_rays_length;
	}

	int Test(int a, int b)
	{
		return a + b;
	}
}