#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <vector>
#include <iostream>
#include <cstdio>

#include "Geometry.cuh"
#include "CudaHeader.cuh"
#include "TriangleCudaAABBGetter.cuh"

#include "bvh.cuh"
#include "aabb.cuh"

namespace GPU4UE
{
    template<typename VecType>
    void ParallelRaysIntersectionWithCuda(RayCuda<VecType>* rays, int numRays, TriangleCuda<VecType>* triangle, int numTriangle, int* results, std::string* error_message)
    {
        RayCuda<VecType>* dev_rays;
        TriangleCuda<VecType>* dev_triangles;
        int* dev_res;

        CUDA_CALL(cudaSetDevice(0));

        CUDA_CALL(cudaMalloc((void**)&dev_rays, numRays * sizeof(RayCuda<VecType>)));
        CUDA_CALL(cudaMalloc((void**)&dev_triangles, numTriangle * sizeof(TriangleCuda<VecType>)));
        CUDA_CALL(cudaMalloc((void**)&dev_res, numRays * sizeof(int)));

        CUDA_CALL(cudaMemcpy(dev_rays, rays, numRays * sizeof(RayCuda<VecType>), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_triangles, triangle, numTriangle * sizeof(TriangleCuda<VecType>), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;

        RaysTrianglesIntersectionKernel << <blocksPerGrid, threadsPerBlock >> > (dev_rays, numRays, dev_triangles, numTriangle, dev_res);

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy(results, dev_res, numRays * sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaFree(dev_rays));
        CUDA_CALL(cudaFree(dev_triangles));
        CUDA_CALL(cudaFree(dev_res));
    }

    template<typename VecType, typename Real, typename Objects, bool IsConst>
    void ParallelRaysIntersectionWithBVHCuda(
        RayCuda<VecType>* rays, 
        int numRays,
        const lbvh::detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        int* results
    )
    {
        using bvh_type = lbvh::detail::basic_device_bvh<Real, Objects, IsConst>;

        RayCuda<VecType>* dev_rays;
        int* dev_res;
        bvh_type* bvh_dev;

        CUDA_CALL(cudaSetDevice(0));
        CUDA_CALL(cudaMalloc((void**)&dev_res, numRays * sizeof(int)));
        CUDA_CALL(cudaMalloc((void**)&dev_rays, numRays * sizeof(RayCuda<VecType>)));
        CUDA_CALL(cudaMalloc((void**)&bvh_dev, sizeof(bvh_type)));

        CUDA_CALL(cudaMemcpy(dev_rays, rays, numRays * sizeof(RayCuda<VecType>), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(bvh_dev, &bvh, sizeof(bvh_type), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;

        RaysTrianglesIntersectionWithBVHKernel << < blocksPerGrid, threadsPerBlock >> > (dev_rays, numRays, bvh_dev, dev_res);

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(results, dev_res, numRays * sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaFree(dev_res));
        CUDA_CALL(cudaFree(dev_rays));
        CUDA_CALL(cudaFree(bvh_dev));
    }

	void Test1()
	{

		int numRays = 10;
        std::vector<RayCuda<float4>> rays(numRays);
        for (int i = 0; i < numRays; ++i) {
            rays[i].origin = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
            //rays[i].dir = normalize_h({ 1.0f, (float)i / numRays - 0.5f, 0.0f }); // 稍微不同的方向
            rays[i].dir = normalize(float4{ 0.0f, 0.5f*(i- numRays/2), 1.0f, 0.0f}); // 稍微不同的方向
        }

        int numTriangles = 1;
        std::vector<TriangleCuda<float4>> triangles(numTriangles);
        for (int i = 0; i < numTriangles; i++)
        {
            triangles[i] = {
                float4{-1.0f, -1.0f, 2.0f, 0.0f},
                float4{1.0f, -1.0f, 2.0f, 0.0f},
                float4{0.0f, 1.0f, 2.0f, 0.0f}
            };
        }

        std::vector<int> results(numRays, 1);
        std::string error_message;

        ParallelRaysIntersectionWithCuda(rays.data(), numRays, triangles.data(), numTriangles, results.data(), &error_message);

        for (int i = 0; i < numRays; i++)
        {
            std::cout << results[i] << " ";
        }
        std::cout << std::endl;


        //2
        //std::vector<int> results2(numRays, 0);
        
        //for (int i = 0; i < numRays; ++i) {
        //    float t;
        //    bool res2 = GPU4UE::RayTriangleIntersect(rays[i], triangles[0], &t);
        //    std::cout << res2 << " ";
        //}
        //std::cout << std::endl;
	}

    void Test2()
    {
        // rays
        int numRays = 100;
        std::vector<RayCuda<float4>> rays(numRays);
        for (int i = 0; i < numRays; ++i) {
            rays[i].origin = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
            //rays[i].dir = normalize_h({ 1.0f, (float)i / numRays - 0.5f, 0.0f }); // 稍微不同的方向

            float sty = -10.0f, edy = 10.0f;
            float dy = (edy - sty) / numRays;

            rays[i].dir = normalize(float4{ 0.0f, sty + dy * i, 1.0f, 0.0f }); // 稍微不同的方向
            rays[i].t = 9.0f;
        }

        //triangles
        int numTriangles = 3;
        std::vector<TriangleCuda<float4>> triangles(numTriangles);
        for (int i = 0; i < numTriangles; i++)
        {
            if (i == 0)
            {
                triangles[i] = {
                    float4{-1.0f, -1.0f, 2.0f, 0.0f},
                    float4{1.0f, -1.0f, 2.0f, 0.0f},
                    float4{0.0f, 1.0f, 2.0f, 0.0f}
                };
            }
            else if (i == 1)
            {
                triangles[i] = triangles[0];
                triangles[i].vertices[0].y += 7.0f;
                triangles[i].vertices[1].y += 7.0f;
                triangles[i].vertices[2].y += 7.0f;
            }
            else if (i == 2)
            {
                triangles[i] = triangles[0];
                triangles[i].vertices[0].y += -9.0f;
                triangles[i].vertices[1].y += -9.0f;
                triangles[i].vertices[2].y += -9.0f;
            }
        }

        // output bounding box
        TriangleCudaAABBGetter aabb_getter;

        for (int i = 0; i < numTriangles; i++)
        {
            lbvh::aabb<float> aabb = aabb_getter(triangles[i]);
            std::cout << aabb.upper.x << " " << aabb.upper.y << " " << aabb.upper.z << std::endl;
            std::cout << aabb.lower.x <<" "<< aabb.lower.y <<" "<< aabb.lower.z << std::endl;

        }



        lbvh::bvh<float, TriangleCuda<float4>, TriangleCudaAABBGetter> bvh(triangles.begin(), triangles.end(), true); // 最后一个参数是query_host_enabled

        const auto bvh_dev = bvh.get_device_repr();

        //// Test2
        // TODO: 把这个改成并行的
        std::cout << "Testing for host: ray intersection" << std::endl;

        for (int i = 0; i < numRays; ++i) {

            std::vector<unsigned int> stack;
            stack.reserve(64);
            stack.emplace_back(0);

            int num_found = 0;

            do
            {
                const unsigned int node_id = stack.back();
                stack.pop_back();
                const unsigned int left_id = bvh.nodes_host()[node_id].left_idx;
                const unsigned int right_id = bvh.nodes_host()[node_id].right_idx;

                if (RayAABBIntersect(rays[i], bvh.aabbs_host()[left_id]))
                {
                    const auto obj_id = bvh.nodes_host()[left_id].object_idx;
                    if (obj_id != 0xFFFFFFFF) // leaf
                    {
                        // 执行三角形求交
                        const auto triangle = bvh.objects_host()[obj_id];
                        float t;
                        bool intersect_res = RayTriangleIntersect(rays[i], triangle, &t);

                        if (intersect_res)
                        {
                            printf("Rays[%d] intersect with triangle obj_id: %lu, t: %.5lf\n", i, (unsigned long)obj_id, t);
                        }

                    }
                    else // internal
                    {
                        stack.push_back(left_id);
                    }
                }

                if (RayAABBIntersect(rays[i], bvh.aabbs_host()[right_id]))
                {
                    const auto obj_id = bvh.nodes_host()[right_id].object_idx;
                    if (obj_id != 0xFFFFFFFF) // leaf
                    {
                        // 执行三角形求交
                        const auto triangle = bvh.objects_host()[obj_id];
                        float t;
                        bool intersect_res = RayTriangleIntersect(rays[i], triangle, &t);

                        if (intersect_res)
                        {
                            printf("Rays[%d] intersect with triangle obj_id: %lu, t: %.5lf\n", i, (unsigned long)obj_id, t);
                        }

                    }
                    else // internal
                    {
                        stack.push_back(right_id);
                    }
                }


            } while (!stack.empty());


        }


        // 利用并行bvh求交
        std::cout << "Testing for device: ray intersection" << std::endl;

        std::vector<int> results(numRays, 1);
        ParallelRaysIntersectionWithBVHCuda(rays.data(), numRays, bvh_dev, results.data());

        for (int i = 0; i < numRays; i++)
        {
            std::cout << results[i] << " ";
        }
        std::cout << std::endl;
    }




}