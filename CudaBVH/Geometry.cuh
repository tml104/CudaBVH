#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <curand_kernel.h>

#include "bvh.cuh"
#include "aabb.cuh"
#include "utility.cuh"

#include "VecType.cuh"
#include "RayCuda.cuh"
#include "TriangleCuda.cuh"
#include "BoundBoxCuda.cuh"
#include "MonteCarlo.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace GPU4UE
{

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

	
	// TEMP: GetOutRaysKernel暂时写在这

	/*
		dev_cells: Cell数组。数组元素个数理论上为gridDim.y
		dev_meshboxes: Mesh包围盒数组。数组元素个数理论上为gridDim.x
		dev_out_rays：输出光线数组。数组元素个数理论上为 gridDim.y * gridDim.x * blockDim.y * blockDim.x

		BoundBoxCuda和RayCuda可以去对应头文件看
	*/

	__device__  __host__ void BuildCellFacesForOneCell(const BoundBoxCuda& cell, FAxisAlignedCellFace* CellFaces) {
		const float4 CellBoundsSize = get_content(cell);
		CellFaces[0] = FAxisAlignedCellFace(
			make_float4(-1, 0, 0, 1),
			make_float4(cell.minval.x, cell.minval.y, cell.minval.z, 1),
			make_float4(0, CellBoundsSize.y, CellBoundsSize.z, 1)
		);
		CellFaces[1] = FAxisAlignedCellFace(
			make_float4(1, 0, 0, 1),
			make_float4(cell.maxval.x, cell.minval.y, cell.minval.z, 1),
			make_float4(0, CellBoundsSize.y, CellBoundsSize.z, 1)
		);
		CellFaces[2] = FAxisAlignedCellFace(
			make_float4(0, -1, 0, 1),
			make_float4(cell.minval.x, cell.minval.y, cell.minval.z, 1),
			make_float4(CellBoundsSize.x, 0, CellBoundsSize.z, 1)
		);
		CellFaces[3] = FAxisAlignedCellFace(
			make_float4(0, 1, 0, 1),
			make_float4(cell.minval.x, cell.maxval.y, cell.minval.z, 1),
			make_float4(CellBoundsSize.x, 0, CellBoundsSize.z, 1)
		);
		CellFaces[4] = FAxisAlignedCellFace(
			make_float4(0, 0, -1, 1),
			make_float4(cell.minval.x, cell.minval.y, cell.minval.z, 1),
			make_float4(CellBoundsSize.x, CellBoundsSize.y, 0, 1)
		);
		CellFaces[5] = FAxisAlignedCellFace(
			make_float4(0, 0, 1, 1),
			make_float4(cell.minval.x, cell.minval.y, cell.maxval.z, 1),
			make_float4(CellBoundsSize.x, CellBoundsSize.y, 0, 1)
		);
	}

	__device__  __host__ void ComputeVisibleCellFacesCUDA(
		const float4& MeshToCellCenter,
		float Distance,
		const FAxisAlignedCellFace* CellFace, // [6]
		int* VisibleCellFaces,               // [6] 输出
		float* VisibleCellFacePDFs,          // [6] 输出
		int& NumVisible                      // 输出：实际可见面数量
	) {
		NumVisible = 0;
		float4 normal_mesh_to_cell_center;
		normal_mesh_to_cell_center.x = MeshToCellCenter.x / Distance;
		normal_mesh_to_cell_center.y = MeshToCellCenter.y / Distance;
		normal_mesh_to_cell_center.z = MeshToCellCenter.z / Distance;
		normal_mesh_to_cell_center.w = MeshToCellCenter.w / Distance;
		for (int i = 0; i < 6; ++i) {
			float DotProduct = -dot3(normal_mesh_to_cell_center, CellFace[i].FaceDirection);

			if (DotProduct > 0.0f) {
				VisibleCellFaces[NumVisible] = i;
				VisibleCellFacePDFs[NumVisible] = DotProduct;
				++NumVisible;
			}
		}
		// Ensure that some of the faces will be sampled
		if (NumVisible == 0) {
			for (int i = 0; i < 6; ++i) {
				VisibleCellFaces[i] = i;
				VisibleCellFacePDFs[i] = float(i);
			}
			NumVisible = 6;
		}
	}

	__device__  __host__ void ComputeVisibleMeshFacesCUDA(
		const float4& MeshToCellCenter,
		float Distance,
		const FAxisAlignedCellFace* CellFace, // [6]
		int* VisibleCellFaces,               // [6] 输出
		int& NumVisible                      // 输出：实际可见面数量
	) {
		NumVisible = 0;
		for (int i = 0; i < 6; ++i) {
			float4 normal_mesh_to_cell_center;
			normal_mesh_to_cell_center.x = MeshToCellCenter.x / Distance;
			normal_mesh_to_cell_center.y = MeshToCellCenter.y / Distance;
			normal_mesh_to_cell_center.z = MeshToCellCenter.z / Distance;
			normal_mesh_to_cell_center.w = MeshToCellCenter.w / Distance;


			float DotProduct = dot3(normal_mesh_to_cell_center, CellFace[i].FaceDirection);

			if (DotProduct > 0.0f) {
				VisibleCellFaces[NumVisible] = i;
				++NumVisible;
			}
		}
	}

	__global__ void GetOutRaysKernel(BoundBoxCuda* dev_cells, BoundBoxCuda* dev_meshboxes, RayCuda<float4>* dev_out_rays)
	{
		const unsigned int cell_sample_id = threadIdx.x;
		const unsigned int meshbox_sample_id = threadIdx.y;

		const unsigned int cell_id = blockIdx.x;
		const unsigned int meshbox_id = blockIdx.y;

		const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x; // block id
		const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x; // thread id
		
		const unsigned int threads_per_block = blockDim.x * blockDim.y;
		const unsigned int index = bid * threads_per_block + tid;

		const unsigned int num_cell = gridDim.y;
		const unsigned int num_meshbox = gridDim.x;
		const unsigned int num_cell_sample = blockDim.y;
		const unsigned int num_meshbox_sample = blockDim.x;

		// TODO: 把pvs里面的采样逻辑缝过来
		// 注意用下标的时候要有越界判断。关于这点可以参考一下前面的写法
		//dev_cells[cell_id].min_x

		VersatileRandomGenerator rng(index);
		const int N = 6;
		//thrust::device_vector<float> vec(N);
		// todo: 初始化
		FAxisAlignedCellFace cell_faces[N];
		FAxisAlignedCellFace mesh_box_faces[N];

		
		if (cell_id < num_cell)
		{
			float4 cell_center = get_center(dev_cells[cell_id]); 

			BuildCellFacesForOneCell(dev_cells[cell_id], cell_faces);

			if (meshbox_id < num_meshbox)
			{
				float4 mesh_to_cell_center;
				float4 mesh_box_center = get_center(dev_meshboxes[meshbox_id]);
				mesh_to_cell_center.x = cell_center.x - mesh_box_center.x;
				mesh_to_cell_center.y = cell_center.y - mesh_box_center.y;
				mesh_to_cell_center.z = cell_center.z - mesh_box_center.z;
				mesh_to_cell_center.w = cell_center.w - mesh_box_center.w;
				float distance = sqrt(mesh_box_center.x * mesh_box_center.x + mesh_box_center.y * mesh_box_center.y + mesh_box_center.z * mesh_box_center.z);

				BuildCellFacesForOneCell(dev_meshboxes[meshbox_id], mesh_box_faces);

				int visible_cell_faces[N];
				int visible_mesh_faces[N];
				float visible_face_pdf[N];
				float visible_face_cdf[N];
				int visible_cell_num, visible_mesh_num;


				ComputeVisibleCellFacesCUDA(mesh_to_cell_center, distance, cell_faces, visible_cell_faces, visible_face_pdf, visible_cell_num);
				ComputeVisibleMeshFacesCUDA(mesh_to_cell_center, distance, mesh_box_faces, visible_mesh_faces, visible_mesh_num);

				// todo: 构造pdf

				float unnormalized_intergral;

				float random_fraction = rng.get_uniform();
				if (cell_sample_id < num_cell_sample)
				{
					if (meshbox_sample_id < num_meshbox_sample)
					{
						CalculateStep1dCDF(visible_face_pdf, visible_face_cdf, visible_cell_num, &unnormalized_intergral);
						float pdf, sample;
						Sample1dCDF_CUDA(visible_face_pdf, visible_face_cdf, visible_cell_num, unnormalized_intergral, random_fraction, pdf, sample);
						int chosen_cell_face_index = int(sample * visible_cell_num);
						if (chosen_cell_face_index >= visible_cell_num)
						{
							chosen_cell_face_index = visible_cell_num - 1;
						}

						const FAxisAlignedCellFace& chosen_cell_face = cell_faces[visible_cell_faces[chosen_cell_face_index]];
						// todo: 生成采样点
						float4 cell_poistion = sample_pos(chosen_cell_face, rng.get_uniform(), rng.get_uniform(), rng.get_uniform());

						int chosen_mesh_face_index = int(random_fraction * visible_mesh_num);
						if (chosen_mesh_face_index >= visible_mesh_num)
						{
							chosen_mesh_face_index = visible_mesh_num - 1;
						}

						const FAxisAlignedCellFace& chosen_mesh_face = mesh_box_faces[visible_mesh_faces[chosen_mesh_face_index]];
						float4 mesh_poistion = sample_pos(chosen_mesh_face, rng.get_uniform(), rng.get_uniform(), rng.get_uniform());

						float4 dir;
						dir.x = mesh_poistion.x - cell_poistion.x;
						dir.y = mesh_poistion.y - cell_poistion.y;
						dir.z = mesh_poistion.z - cell_poistion.z;
						dir.w = mesh_poistion.x - cell_poistion.x;

						float len = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
						float inv_sqrt = 1.0 / len;
						dir.x *= inv_sqrt;
						dir.y *= inv_sqrt;
						dir.z *= inv_sqrt;
						dir.w *= inv_sqrt;


						RayCuda<float4> ray;

						ray.origin = cell_poistion;
						ray.dir = dir;
						ray.t = len;

						dev_out_rays[index] = ray;
					}
				}

			}
		}

	}


}