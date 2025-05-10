#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace GPU4UE
{
	struct BoundBoxCuda
	{
		float4 minval; // 这里只用xyz，w的值请无视
		
		float4 maxval;
	};
}