#ifndef MONTECARLO_CUH
#define MONTECARLO_CUH

#include <thrust/random.h>
#include <thrust/functional.h>

#include <random>

#include "CudaHeader.cuh"

namespace GPU4UE
{ 
    struct VersatileRandomGenerator {
        // 主机端实现 (C++ <random>)
        std::mt19937 rng_engine_host; // Mersenne Twister 引擎
        std::uniform_real_distribution<float> uniform_dist_host;

    public: // 显式声明构造函数为public
        // 主机端构造函数，允许指定种子
        __device__ __host__ VersatileRandomGenerator(unsigned int seed)
            : rng_engine_host(seed), uniform_dist_host(0.0f, 1.0f) {}

        // 主机端默认构造函数，使用 std::random_device 获取随机种子
        // 这通常能提供较好的随机性，但如果 std::random_device 不可用，可能回退到确定性种子
        __device__ __host__ VersatileRandomGenerator()
            : rng_engine_host(std::random_device{}()), uniform_dist_host(0.0f, 1.0f) {}

        __device__ __host__ float get_uniform() {
            // 生成 [0.0, 1.0) 范围内的均匀分布随机浮点数
            return uniform_dist_host(rng_engine_host);
        }

    //#endif
    };
    struct normalize_functor
    {
        const float denomiator;

        normalize_functor(float denomiator) : denomiator(denomiator) {}

        __device__ __host__
            float operator()(const float& x) const
        {
            return x / denomiator;
        }
    };

        // In MonteCarlo.cuh
        __device__ __host__ void CalculateStep1dCDF(
            const float* pdf_array,    // device ptr to a small array (e.g., on stack)
            float* cdf_array,          // device ptr for output CDF (e.g., on stack)
            int N_val,                 // actual number of elements in pdf_array (e.g., visible_cell_num)
            float* unnormalized_integral
        ) {
            if (N_val <= 0) {
                if (unnormalized_integral) *unnormalized_integral = 0.0f;
                // Consider what cdf_array should contain. Maybe fill with 0.
                for(int i = 0; i < N_val; ++i) cdf_array[i] = 0.0f;
                return;
            }

            float current_sum = 0.0f;
            // Exclusive scan: CDF[0] is 0. PDF values are for intervals [i, i+1).
            // CDF[i] stores sum of PDF[0]...PDF[i-1]
            cdf_array[0] = 0.0f; 

            for (int i = 0; i < N_val; ++i) {
                if (i > 0) {
                    // cdf_array[i] is sum of pdf_array[0] to pdf_array[i-1]
                    cdf_array[i] = cdf_array[i-1] + pdf_array[i-1]; 
                }
                current_sum += pdf_array[i];
            }
            if (unnormalized_integral) *unnormalized_integral = current_sum;

            for (int i = 1; i < N_val; ++i) { 
                cdf_array[i] /= *unnormalized_integral;
            }
        }

    __device__  __host__ void Sample1dCDF_CUDA(
        const float* PDFArray,         // device ptr, size N
        const float* CDFArray,         // device ptr, size N
        int N,                         // 数组长度
        float UnnormalizedIntegral,    // device变量
        float RandomFraction,          // device端生成的[0,1)随机数
        float& PDF,                    // 输出
        float& Sample                  // 输出
    )
    {
        if (N > 1)
        {
            int GreaterElementIndex = -1;
            // 线性查找（如需更高效可用thrust::upper_bound）
            for (int i = 1; i < N; ++i)
            {
                if (CDFArray[i] >= RandomFraction)
                {
                    GreaterElementIndex = i;
                    break;
                }
            }
            if (GreaterElementIndex >= 0)
            {
                float OffsetAlongCDFSegment = (RandomFraction - CDFArray[GreaterElementIndex - 1]) /
                                            (CDFArray[GreaterElementIndex] - CDFArray[GreaterElementIndex - 1]);
                PDF = PDFArray[GreaterElementIndex - 1] / UnnormalizedIntegral;
                Sample = (GreaterElementIndex - 1 + OffsetAlongCDFSegment) / float(N);
            }
            else
            {
                // 最后一个元素
                float OffsetAlongCDFSegment = (RandomFraction - CDFArray[N-1]) / (1.0f - CDFArray[N-1]);
                PDF = PDFArray[N-1] / UnnormalizedIntegral;
                // Clamp
                float s = (N - 1 + OffsetAlongCDFSegment) / float(N);
                Sample = (s < 0.0f) ? 0.0f : ((s > 1.0f - 1e-6f) ? (1.0f - 1e-6f) : s);
            }
        }
        else
        {
            PDF = 1.0f;
            Sample = 0.0f;
        }
    }
}







#endif

