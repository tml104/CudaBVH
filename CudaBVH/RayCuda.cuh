#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "VecType.cuh"

namespace GPU4UE 
{
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

}