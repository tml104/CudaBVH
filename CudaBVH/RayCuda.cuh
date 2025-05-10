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
	struct RayCuda<float4> // 这里只用xyz，w的值请无视
	{
		using VecType = float4;

		typename VecType origin; // 原点坐标
		typename VecType dir;	// 方向，必须是单位方向向量
		typename VecTypeReal<VecType>::type t; // 光线从原点发射出去的长度。如果确定了光线的起点和终点，那么这个值构造的时候需要手动计算一下（就是起点到终点距离）
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