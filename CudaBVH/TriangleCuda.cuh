#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace GPU4UE
{
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
}