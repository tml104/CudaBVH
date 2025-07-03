#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

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
	// �˺�������ʼ�� cuRAND ״̬
	__global__ void SetupCuRand(curandState_t* state, unsigned long long seed, size_t total_length)
	{
		const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x; // block id
		const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x; // thread id

		const unsigned int threads_per_block = blockDim.x * blockDim.y;
		const unsigned int index = bid * threads_per_block + tid;
		// ʹ�� seed ��Ψһ���߳�������ʼ��״̬
		// 0 �������кţ�ͨ��������Ϊ 0

		if (index < total_length)
		{
			curand_init(seed, index, 0, &state[index]);
		}
	}


/*
	dev_cells: Cell���顣����Ԫ�ظ���������ΪgridDim.y
	dev_meshboxes: Mesh��Χ�����顣����Ԫ�ظ���������ΪgridDim.x
	dev_out_rays������������顣����Ԫ�ظ���������Ϊ gridDim.y * gridDim.x * blockDim.y * blockDim.x

	BoundBoxCuda��RayCuda����ȥ��Ӧͷ�ļ���
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
		int* VisibleCellFaces,               // [6] ���
		float* VisibleCellFacePDFs,          // [6] ���
		int& NumVisible                      // �����ʵ�ʿɼ�������
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
		int* VisibleCellFaces,               // [6] ���
		int& NumVisible                      // �����ʵ�ʿɼ�������
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

	__global__ void GetOutRaysKernel(BoundBoxCuda* dev_cells, BoundBoxCuda* dev_meshboxes, RayCuda<float4>* dev_out_rays, curandState_t* dev_states,size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample, size_t dev_out_rays_length)
	{
		const unsigned int cell_sample_id = threadIdx.y;
		const unsigned int meshbox_sample_id = threadIdx.x;

		const unsigned int cell_id = blockIdx.y;
		const unsigned int meshbox_id = blockIdx.x;

		const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x; // block id
		const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x; // thread id

		const unsigned int threads_per_block = blockDim.x * blockDim.y;
		const unsigned int index = bid * threads_per_block + tid;

		// �⼸�������ĳɴ�������
		//const unsigned int num_cell = gridDim.y;
		//const unsigned int num_meshbox = gridDim.x;
		//const unsigned int num_cell_sample = blockDim.y;
		//const unsigned int num_meshbox_sample = blockDim.x;

		// TODO: ��pvs����Ĳ����߼������
		// ע�����±��ʱ��Ҫ��Խ���жϡ����������Բο�һ��ǰ���д��

		if (index < dev_out_rays_length)
		{
			//VersatileRandomGenerator rng(index);
			const int N = 6;
			//thrust::device_vector<float> vec(N);
			FAxisAlignedCellFace cell_faces[N];
			FAxisAlignedCellFace mesh_box_faces[N];

			if (cell_id < num_cells)
			{
				float4 cell_center = get_center(dev_cells[cell_id]);

				BuildCellFacesForOneCell(dev_cells[cell_id], cell_faces);

				if (meshbox_id < num_meshboxes)
				{
					float4 mesh_to_cell_center;
					float4 mesh_box_center = get_center(dev_meshboxes[meshbox_id]);
					mesh_to_cell_center.x = cell_center.x - mesh_box_center.x;
					mesh_to_cell_center.y = cell_center.y - mesh_box_center.y;
					mesh_to_cell_center.z = cell_center.z - mesh_box_center.z;
					mesh_to_cell_center.w = cell_center.w - mesh_box_center.w;
					float distance = sqrt(mesh_to_cell_center.x * mesh_to_cell_center.x + mesh_to_cell_center.y * mesh_to_cell_center.y + mesh_to_cell_center.z * mesh_to_cell_center.z);

					BuildCellFacesForOneCell(dev_meshboxes[meshbox_id], mesh_box_faces);

					int visible_cell_faces[N];
					int visible_mesh_faces[N];
					float visible_face_pdf[N];
					float visible_face_cdf[N];
					int visible_cell_num, visible_mesh_num;

					ComputeVisibleCellFacesCUDA(mesh_to_cell_center, distance, cell_faces, visible_cell_faces, visible_face_pdf, visible_cell_num);
					ComputeVisibleMeshFacesCUDA(mesh_to_cell_center, distance, mesh_box_faces, visible_mesh_faces, visible_mesh_num);

					// todo: ����pdf

					float unnormalized_intergral;

					float random_fraction = curand_uniform(&dev_states[index]);
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
							// todo: ���ɲ�����
							float4 cell_poistion = sample_pos(chosen_cell_face, curand_uniform(&dev_states[index]), curand_uniform(&dev_states[index]), curand_uniform(&dev_states[index]));

							int chosen_mesh_face_index = int(random_fraction * visible_mesh_num);
							if (chosen_mesh_face_index >= visible_mesh_num)
							{
								chosen_mesh_face_index = visible_mesh_num - 1;
							}

							const FAxisAlignedCellFace& chosen_mesh_face = mesh_box_faces[visible_mesh_faces[chosen_mesh_face_index]];
							float4 mesh_poistion = sample_pos(chosen_mesh_face, curand_uniform(&dev_states[index]), curand_uniform(&dev_states[index]), curand_uniform(&dev_states[index]));

							float4 dir;
							dir.x = mesh_poistion.x - cell_poistion.x;
							dir.y = mesh_poistion.y - cell_poistion.y;
							dir.z = mesh_poistion.z - cell_poistion.z;
							dir.w = mesh_poistion.w - cell_poistion.w;

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

}