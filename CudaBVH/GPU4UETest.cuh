#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <vector>
#include <iostream>
#include <cstdio>

#include <thread>

#include "Geometry.cuh"
#include "TriangleCudaAABBGetter.cuh"
#include "ParallelRaysIntersectionWithCuda.cuh"
#include "BoundBoxCuda.cuh"

#include "bvh.cuh"
#include "aabb.cuh"

#include "GPU4UEInterfaces.cuh"

#include "JsonExport.cuh"

namespace GPU4UE
{
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

    void Test3()
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

        InitBVH(triangles);

        //std::vector<int> results(numRays, 1);
        //ParallelRaysIntersectionWithBVHCuda2(rays, results.data());

        //for (int i = 0; i < numRays; i++)
        //{
        //    std::cout << results[i] << " ";
        //}
        //std::cout << std::endl;

        // 测试多线程
        const int TEST3_THREAD_NUM = 100;

        std::vector<int> results_array[TEST3_THREAD_NUM];



        //for (int i = 0; i < THREAD_NUM; i++)
        //{
        //    results_array[i].resize(numRays);
        //}

        auto task_fun = [&](const int index) {
            results_array[index].resize(numRays);

            ParallelRaysIntersectionWithBVHCuda2(rays, results_array[index].data());
        };

        std::vector<std::thread> threads;

        for (int i = 0; i < TEST3_THREAD_NUM; i++)
        {
            threads.emplace_back(task_fun, i);
        }
        
        for (auto& t : threads)
        {
            t.join();
        }

        for (int h = 0; h < TEST3_THREAD_NUM; h++)
        {
            std::cout << "Thread " << h << " Results:" << std::endl;

            for (int i = 0; i < numRays; i++)
            {
                std::cout << results_array[h][i] << " ";
            }
            std::cout << std::endl;

        }

    }


    /*
        测试GPU上的包围盒上光线采样
        3 3 24 30 0 0
        3 3 5 5 0 0
    */
    void Test4()
    {
        // cells & meshbox
        std::vector<BoundBoxCuda> test_cells, test_meshboxes;

        int s_cells = 3;
        int s_meshboxes = 3;
        std::cin >> s_cells >> s_meshboxes;

        //for (int i = 0; i < s0; i++)
        //{
        //    test_cells.push_back(
        //        {
        //            { 1.0f * i, 0.0f, 0.0f, 0.0f },
        //            { 1.0f * (i+1), 1.0f, 1.0f, 0.0f }
        //        }
        //    );


        //    test_meshboxes.push_back(
        //        {
        //            { 1.0f * i, 5.0f, 5.0f, 0.0f },
        //            { 1.0f * (i + 1), 6.0f, 6.0f, 0.0f }
        //        }
        //    );
        //}


        for (int i = 0; i < s_cells; i++)
        {
            test_cells.push_back(
                {
                    { 1.0f * i, 0.0f, 0.0f, 0.0f },
                    { 1.0f * (i+1), 1.0f, 1.0f, 0.0f }
                }
            );
        }

        for (int i=0; i < s_meshboxes; i++)
        {
            test_meshboxes.push_back(
                {
                    { 1.0f * i, 5.0f, 5.0f, 0.0f },
                    { 1.0f * (i + 1), 6.0f, 6.0f, 0.0f }
                }
            );
        }

        CellToJson(test_cells, "C:\\hqh\\code\\UnrealEngine\\Engine\\Saved\\Swarm\\Engine\\Programs\\UnrealLightmass\\Saved\\Logs\\cell_json.json");
        CellToJson(test_meshboxes, "C:\\hqh\\code\\UnrealEngine\\Engine\\Saved\\Swarm\\Engine\\Programs\\UnrealLightmass\\Saved\\Logs\\meshbox_json.json");

        InitCellBoundsCuda(test_cells);
        std::cout << "InitCellBoundsCuda Done" << std::endl;

        InitMeshBoundsCuda(test_meshboxes);
        std::cout << "InitMeshBoundsCuda Done" << std::endl;

        int s1 = 24;
        int s2 = 32;

        std::cin >> s1 >> s2;

        InitOutRaysCuda(test_cells.size(), test_meshboxes.size(), s1, s2);
        std::cout << "InitOutRaysCuda Done" << std::endl;

        InitResults(test_cells.size(), test_meshboxes.size(), s1, s2);
        std::cout << "InitResults Done" << std::endl;

        ComputeOutRaysCuda(test_cells.size(), test_meshboxes.size(), s1, s2, 0, 0); // 尺寸问题
        std::cout << "ComputeOutRaysCuda Done" << std::endl;

        int s_show_ray = 0;
        int s_show_res = 0;

        std::cin >> s_show_ray >> s_show_res;

        std::vector<RayCuda<float4>> test_out_rays = GetOutRaysFromCuda();
        if (s_show_ray)
        {
            for (int i = 0; i < test_out_rays.size(); i++)
            {
                RayCuda<float4>& ray = test_out_rays[i];
                std::cout << "i: " << i << std::endl;
                std::cout << "(" << ray.origin.x << ", " << ray.origin.y << ", " << ray.origin.z << ")";
                std::cout << "(" << ray.dir.x << ", " << ray.dir.y << ", " << ray.dir.z << ")";

                // 计算一下end
                float4 end_point;
                end_point.x = ray.origin.x + (ray.dir.x * ray.t);
                end_point.y = ray.origin.y + (ray.dir.y * ray.t);
                end_point.z = ray.origin.z + (ray.dir.z * ray.t);


                std::cout << " t: " << ray.t << " ";

                std::cout << "(" << end_point.x << ", " << end_point.y << ", " << end_point.z << ")";

                std::cout << std::endl;
            }
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

        InitBVH(triangles);
        std::cout << "InitBVH Done" << std::endl;

        ParallelRaysIntersectionWithBVHAndRaysCuda3();
        std::cout << "ParallelRaysIntersectionWithBVHCuda3 Done" << std::endl;


        int* my_result = GetHostResults();
        std::cout << "GetHostResults Done" << std::endl;

        size_t dev_out_rays_length = GetDevOutRaysLength();
        std::cout << "GetDevOutRaysLength Done" << std::endl;

        std::cout << "dev_out_rays_length: " << dev_out_rays_length << std::endl;

        RayToJson(test_out_rays, my_result, s_cells, s_meshboxes, s1, s2, "C:\\hqh\\code\\UnrealEngine\\Engine\\Saved\\Swarm\\Engine\\Programs\\UnrealLightmass\\Saved\\Logs\\rays_json.json");


        if (s_show_res)
        {
            for (int i = 0; i < dev_out_rays_length; i++)
            {
                std::cout << my_result[i] << " ";
            }
        }

        std::cout << std::endl;

    }
}