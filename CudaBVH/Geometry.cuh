#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "aabb.cuh"
#include "utility.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace GPU4UE
{
	// --- VecType
	template<typename T> struct VecTypeReal;

	template<>
	struct VecTypeReal<float4> { using type = float; };

	template<>
	struct VecTypeReal<double4> { using type = double; };


	// --- TriangleCuda
	template<typename T> struct TriangleCuda;

	template<>
	struct TriangleCuda<float4>
	{
		using VecType = float4;

		typename VecType vertices[3];
	};

	template<>
	struct TriangleCuda<double4>
	{
		using VecType = double4;

		typename VecType vertices[3];
	};



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


	// --- TriangleCuda End

	// --- RayCuda
	template<typename T> struct RayCuda;

	template<>
	struct RayCuda<float4>
	{
		using VecType = float4;

		typename VecType origin;
		typename VecType dir;
		typename VecTypeReal<VecType>::type t;
	};

	template<>
	struct RayCuda<double4>
	{
		using VecType = double4;

		typename VecType origin;
		typename VecType dir;
		typename VecTypeReal<VecType>::type t;
	};

	template<typename VecType, typename Real = VecTypeReal<VecType>::type>
	__device__ __host__ bool RayAABBIntersect(RayCuda<VecType> ray, lbvh::aabb<Real> box)
	{
		const Real EPSILON = 1e-6;
	
		Real t_enter = 0.0f;
		Real t_exit = std::numeric_limits<Real>::max();

		// X轴检测
		if (fabs(ray.dir.x) > EPSILON)
		{
			Real t1 = (box.lower.x - ray.origin.x) / ray.dir.x;
			Real t2 = (box.upper.x - ray.origin.x) / ray.dir.x;

			if (t1 > t2) thrust::swap(t1, t2); // 确保 t1 <= t2

			t_enter = std::fmax(t_enter, t1);
			t_exit = std::fmin(t_exit, t2);

			if (t_exit < t_enter) return false; // 如果 t_exit 小于 t_enter，则不相交
		}
		else
		{
			// ray.dir.x 近似为 0，检查 ray.origin.x 是否在 AABB 的 x 范围内
			if (ray.origin.x < box.lower.x || ray.origin.x > box.upper.x) return false;
		}

		// Y轴检测 (逻辑与 X 轴相同)
		if (fabs(ray.dir.y) > EPSILON)
		{
			Real t1 = (box.lower.y - ray.origin.y) / ray.dir.y;
			Real t2 = (box.upper.y - ray.origin.y) / ray.dir.y;

			if (t1 > t2) thrust::swap(t1, t2);

			t_enter = std::fmax(t_enter, t1);
			t_exit = std::fmin(t_exit, t2);

			if (t_exit < t_enter) return false;
		}
		else
		{
			if (ray.origin.y < box.lower.y || ray.origin.y > box.upper.y) return false;
		}

		// Z轴检测 (逻辑与 X 轴相同)
		if (fabs(ray.dir.z) > EPSILON)
		{
			Real t1 = (box.lower.z - ray.origin.z) / ray.dir.z;
			Real t2 = (box.upper.z - ray.origin.z) / ray.dir.z;

			if (t1 > t2) thrust::swap(t1, t2);

			t_enter = std::fmax(t_enter, t1);
			t_exit = std::fmin(t_exit, t2);

			if (t_exit < t_enter) return false;
		}
		else
		{
			if (ray.origin.z < box.lower.z || ray.origin.z > box.upper.z) return false;
		}

		return (t_exit >= 0.0f && t_enter <= ray.t); // 最终的相交条件

	}


	// --- RayCuda End

	template<typename VecType, typename Real = VecTypeReal<VecType>::type>
	__device__ __host__ VecType subtract(const VecType& a, const VecType& b)
	{
		return { a.x - b.x, a.y - b.y, a.z - b.z, 0.0f };
	}

	template<typename VecType, typename Real = VecTypeReal<VecType>::type>
	__device__ __host__ VecType cross(const VecType& a, const VecType& b)
	{
		return {
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x,
			0.0f
		};
	}

	template<typename VecType, typename Real = VecTypeReal<VecType>::type>
	__device__ __host__ Real dot(const VecType& a, const VecType& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	template<typename VecType, typename Real = VecTypeReal<VecType>::type>
	__device__ __host__ Real length(const VecType& v)
	{
		return sqrtf(dot(v, v));
	}

	template<typename VecType, typename Real = VecTypeReal<VecType>::type>
	__device__ __host__ VecType normalize(const VecType& v)
	{
		Real len = length(v);
		if (len == 0.0f)
		{
			return { 0.0f, 0.0f, 0.0f, 0.0f };
		}

		return { v.x / len, v.y / len, v.z / len, 0.0f };
	}

	template<typename VecType, typename Real = VecTypeReal<VecType>::type>
	__device__ __host__ bool RayTriangleIntersect(RayCuda<VecType> ray, TriangleCuda<VecType> triangle, Real* t)
	{
		const Real EPSILON = 1e-6;
		VecType edge1 = subtract(triangle.vertices[1], triangle.vertices[0]);
		VecType edge2 = subtract(triangle.vertices[2], triangle.vertices[0]);
		VecType pvec = cross(ray.dir, edge2);
		Real det = dot(edge1, pvec);

		if (det > -EPSILON && det < EPSILON)
		{
			return false;
		}

		Real inv_det = 1.0f / det;

		VecType tvec = subtract(ray.origin, triangle.vertices[0]);
		Real u = dot(tvec, pvec) * inv_det;
		if (u < 0.0f || u>1.0f)
		{
			return false;
		}

		VecType qvec = cross(tvec, edge1);
		Real v = dot(ray.dir, qvec) * inv_det;
		if (v < 0.0f || u+v>1.0f)
		{
			return false;
		}

		*t = dot(edge2, qvec) * inv_det;
		return (*t > EPSILON && *t<ray.t);
	}

	template<typename VecType, typename Real = VecTypeReal<VecType>::type>
	__global__ void RaysTrianglesIntersectionKernel(RayCuda<VecType>* rays, int numRays, TriangleCuda<VecType>* triangles, int numTriangle, int* results)
	{
		int ray_index = blockIdx.x * blockDim.x + threadIdx.x;
		if (ray_index < numRays)
		{
			Real intersectionT;
			results[ray_index] = 0;

			for (int i = 0; i < numTriangle; i++)
			{
				if (RayTriangleIntersect(rays[ray_index], triangles[i], &intersectionT)) 
				{
					results[ray_index] = 1;
				}
			}

		}
	}

	template<typename VecType, typename Real, typename Objects, bool IsConst>
	__global__ void RaysTrianglesIntersectionWithBVHKernel(RayCuda<VecType>* rays, int numRays, const lbvh::detail::basic_device_bvh<Real, Objects, IsConst>* bvh, int* results)
	{
		using bvh_type = lbvh::detail::basic_device_bvh<Real, Objects, IsConst>;
		using index_type = typename bvh_type::index_type;
		using aabb_type = typename bvh_type::aabb_type;
		using node_type = typename bvh_type::node_type;

		int ray_index = blockIdx.x * blockDim.x + threadIdx.x;

		if (ray_index < numRays)
		{
			results[ray_index] = 0; // 初始化

			index_type stack[64]; // 注意爆栈风险
			index_type* stack_ptr = stack;
			*stack_ptr++ = 0; // root node is always 0

			do {
				const index_type node_id = *--stack_ptr;
				const index_type left_id = bvh->nodes[node_id].left_idx;
				const index_type right_id = bvh->nodes[node_id].right_idx;

				if (RayAABBIntersect(rays[ray_index], bvh->aabbs[left_id]))
				{
					const auto obj_id = bvh->nodes[left_id].object_idx;
					if (obj_id != 0xFFFFFFFF) // leaf
					{
						// 执行三角形求交
						const auto triangle = bvh->objects[obj_id];
						float t;
						bool intersect_res = RayTriangleIntersect(rays[ray_index], triangle, &t);

						if (intersect_res) 
						{
							// TODO: 改成能容纳多个三角形交的结果
							results[ray_index] = (int)obj_id + 1; // 默认的范围是0~n-1，这里+1一下让结果落在1~n，以和无交情况区分
						}

					}
					else
					{
						*stack_ptr++ = left_id;
					}

				}

				if (RayAABBIntersect(rays[ray_index], bvh->aabbs[right_id]))
				{
					const auto obj_id = bvh->nodes[right_id].object_idx;
					if (obj_id != 0xFFFFFFFF) // leaf
					{
						// 执行三角形求交
						const auto triangle = bvh->objects[obj_id];
						float t;
						bool intersect_res = RayTriangleIntersect(rays[ray_index], triangle, &t);

						if (intersect_res)
						{
							// TODO: 改成能容纳多个三角形交的结果
							results[ray_index] =  (int)obj_id + 1; // 默认的范围是0~n-1，这里+1一下让结果落在1~n，以和无交情况区分
						}

					}
					else
					{
						*stack_ptr++ = right_id;
					}

				}


			} while (stack < stack_ptr);


		}

	}
}