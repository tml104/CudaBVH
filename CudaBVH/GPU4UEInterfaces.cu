#include "GPU4UEInterfaces.cuh"

#include "ParallelRaysIntersectionWithCuda.cuh"
#include "aabb.cuh"


namespace GPU4UE
{

	void ParallelRaysIntersectionWithBVHCuda2(std::vector<RayCuda<float4>>& rays, int* results, const std::vector<TriangleCuda<float4>>& triangles)
	{
		int num_rays = rays.size();
		int num_triangles = triangles.size();


		static lbvh::bvh<float, TriangleCuda<float4>, TriangleCudaAABBGetter> bvh(triangles.begin(), triangles.end(), true); // 最后一个参数是query_host_enabled
		static const auto bvh_dev = bvh.get_device_repr();

		// 利用并行bvh求交
		ParallelRaysIntersectionWithBVHCuda(rays.data(), num_rays, bvh_dev, results);

	}

	int Test(int a, int b)
	{
		return a + b;
	}
}