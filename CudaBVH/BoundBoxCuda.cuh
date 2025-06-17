#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <corecrt_math.h>

namespace GPU4UE
{
	struct BoundBoxCuda
	{
		float4 minval; // ����ֻ��xyz��w��ֵ������
		
		float4 maxval;
	};


    __device__ __host__
        inline float4 get_content(const BoundBoxCuda& box) noexcept
    {
		float4 res;
		res.x = box.maxval.x - box.minval.x;
		res.y = box.maxval.y - box.minval.y;
		res.z = box.maxval.z - box.minval.z;
		res.w = box.maxval.w - box.minval.w;
        return res;
    }

	__device__ __host__
		inline float4 get_center(const BoundBoxCuda& box) noexcept
	{
		float4 res;
		res.x = (box.maxval.x + box.minval.x) / 2.0f;
		res.y = (box.maxval.y + box.minval.y) / 2.0f;
		res.z = (box.maxval.z + box.minval.z) / 2.0f;
		res.w = (box.maxval.w + box.minval.w) / 2.0f;
        return res;
	}

	__device__ __host__
		inline float get_size(const BoundBoxCuda& box) noexcept
	{
		float4 content = get_content(box);
		return sqrt(content.x * content.x + content.y * content.y + content.z * content.z);
	}

	__device__  __host__
		inline float dot3(const float4& a, const float4& b)
		{
			return a.x * b.x + a.y * b.y + a.z * b.z;
		}

	struct FAxisAlignedCellFace
	{
	public:

	__device__ __host__
		FAxisAlignedCellFace() {}

	__device__ __host__
		FAxisAlignedCellFace(const float4& InFaceDirection, const float4& InFaceMin, const float4& InFaceExtent) :
			FaceDirection(InFaceDirection),
			FaceMin(InFaceMin),
			FaceExtent(InFaceExtent)
		{}

		float4 FaceDirection;
		float4 FaceMin;
		float4 FaceExtent;
	};

	__device__  __host__ float4 inline sample_pos(const FAxisAlignedCellFace& face, float random_x, float random_y, float random_z)
	{
		float4 extent;
		extent.x = face.FaceExtent.x * random_x;
		extent.y = face.FaceExtent.y * random_y;
		extent.z = face.FaceExtent.z * random_z;

		float4 sample;
		sample.x = face.FaceMin.x + extent.x;
		sample.y = face.FaceMin.y + extent.y;
		sample.z = face.FaceMin.z + extent.z;
		sample.w = 1.0f;
		return sample;
	}
}