#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace GPU4UE
{
	// --- VecType
	template<typename T> struct VecTypeReal;

	template<>
	struct VecTypeReal<float4> { using type = float; };

	template<>
	struct VecTypeReal<double4> { using type = double; };

	// 下面这一堆放这不知道有没有问题，再看吧

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
}