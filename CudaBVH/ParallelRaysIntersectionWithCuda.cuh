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


}