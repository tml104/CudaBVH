#include "GPU4UEInterfaces.cuh"

#include "ParallelRaysIntersectionWithCuda.cuh"
#include "aabb.cuh"


namespace GPU4UE
{

	static lbvh::bvh<float, TriangleCuda<float4>, TriangleCudaAABBGetter> bvh;

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

	int Test(int a, int b)
	{
		return a + b;
	}
}