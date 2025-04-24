#pragma once

#include "aabb.cuh"
#include "TriangleCuda.cuh"

namespace GPU4UE
{
	struct TriangleCudaAABBGetter
	{
		__device__ __host__ lbvh::aabb<float> operator()(const TriangleCuda<float4> triangle) const noexcept
		{
			lbvh::aabb<float> retval;
			retval.upper = triangle.vertices[0];
			retval.lower = triangle.vertices[0];

			for (int i = 1; i < 3; i++)
			{
				retval.upper.x = fmax(retval.upper.x, triangle.vertices[i].x);
				retval.upper.y = fmax(retval.upper.y, triangle.vertices[i].y);
				retval.upper.z = fmax(retval.upper.z, triangle.vertices[i].z);

				retval.lower.x = fmin(retval.lower.x, triangle.vertices[i].x);
				retval.lower.y = fmin(retval.lower.y, triangle.vertices[i].y);
				retval.lower.z = fmin(retval.lower.z, triangle.vertices[i].z);
			}

			return retval;
		}
	};
}