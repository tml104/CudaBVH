# CudaBVH

本项目用于为 UECuda 项目中的 `PrecomputedVisibility.cpp` 提供 CUDA 静态库接口与 GPU 加速算法。当前主要能力包括：

- 基于三角形数据构建 GPU BVH。
- 批量执行光线与 BVH 的并行求交。
- 根据 Visibility Cell 与 Mesh 包围盒在 GPU 上生成采样光线。
- 将调试用 Cell、MeshBox、Ray 数据导出为 JSON。

## Build说明

配置属性 - 常规属性 - 配置类型 - 静态库（.lib）

随后，将 build 出来的静态库放在 UECuda 的 `Engine\Source\Programs\UnrealLightmass\Private\CudaTest\lib` 中。

将 `BoundBoxCuda.cuh`、`GPU4UEInterfaces.cuh`、`json.hpp`、`JsonExport.cuh`、`RayCuda.cuh`、`TriangleCuda.cuh`、`VecType.cuh` 放到 `Engine\Source\Programs\UnrealLightmass\Private\CudaTest\include` 中。

（注：UnrealLightmass 在解决方案资源管理器的 Programs 层级里，不在 Engine/UE5 里）

## 项目结构

```text
CudaBVH/
├── CudaBVH.sln                  # Visual Studio 解决方案
├── LICENSE.txt
└── CudaBVH/
    ├── CudaBVH.vcxproj          # CUDA/C++ 项目文件，包含 Debug、Release、lib 三个 x64 配置
    ├── main.cu                  # 本地测试入口；当前 main 已注释
    ├── GPU4UEInterfaces.cuh     # 对 UE/UnrealLightmass 暴露的主要接口声明
    ├── GPU4UEInterfaces.cu      # 接口实现与 GPU 侧全局资源管理
    ├── GPU4UETest.cuh           # 本地测试用例
    ├── ParallelRaysIntersectionWithCuda.cuh
    │                            # 光线与三角形/BVH 的 CUDA 并行求交封装
    ├── ComputeOutRaysWithCuda.cuh
    │                            # 在 GPU 上根据 Cell 与 MeshBox 包围盒采样外射光线
    ├── TriangleCuda.cuh         # 三角形数据结构
    ├── RayCuda.cuh              # 光线数据结构
    ├── BoundBoxCuda.cuh         # AABB 包围盒与采样辅助结构
    ├── Geometry.cuh             # 几何求交、向量运算等基础函数
    ├── TriangleCudaAABBGetter.cuh
    │                            # 为 BVH 构建提供三角形 AABB
    ├── bvh.cuh / lbvh.cuh       # LBVH 数据结构与构建/查询支持
    ├── aabb.cuh                 # BVH 使用的 AABB 类型
    ├── morton_code.cuh          # Morton 编码
    ├── utility.cuh              # CUDA/Thrust 工具函数
    ├── MonteCarlo.cuh           # 蒙特卡洛采样工具
    ├── JsonExport.cuh/.cu       # 调试用 JSON 导出
    ├── CudaHeader.cuh/.cu       # CUDA 错误检查宏与辅助函数
    └── json.hpp                 # nlohmann/json 单头文件
```

## 核心流程

项目当前主要服务于 UnrealLightmass 的 Precomputed Visibility 计算，典型调用流程如下：

1. 从 UE 侧准备场景三角形数据，填充为 `std::vector<GPU4UE::TriangleCuda<float4>>`。
2. 调用 `GPU4UE::InitBVH(triangles)` 在 GPU 侧构建并缓存 BVH。
3. 从 UE 侧准备 Visibility Cell 包围盒与 Mesh 包围盒，填充为 `std::vector<GPU4UE::BoundBoxCuda>`。
4. 分别调用 `GPU4UE::InitCellBoundsCuda(cells)` 与 `GPU4UE::InitMeshBoundsCuda(meshboxes)` 上传包围盒。
5. 根据采样数量调用 `GPU4UE::InitOutRaysCuda(...)` 与 `GPU4UE::InitResults(...)` 分配光线与结果缓冲区。
6. 调用 `GPU4UE::ComputeOutRaysCuda(...)` 在 GPU 上生成从 Cell 指向 MeshBox 的采样光线。
7. 调用 `GPU4UE::ParallelRaysIntersectionWithBVHAndRaysCuda3()` 使用已生成的 GPU 光线与 BVH 做并行遮挡求交。
8. 通过 `GPU4UE::GetHostResults()` 或 `GPU4UE::GetHostResultsVec()` 读取结果。

## 主要接口

接口均位于 `GPU4UE` 命名空间，声明在 `GPU4UEInterfaces.cuh`。

### BVH 初始化与求交

- `InitBVH(const std::vector<TriangleCuda<float4>>& triangles)`：根据三角形数组构建 BVH，应在执行 BVH 求交前调用。
- `ParallelRaysIntersectionWithBVHCuda2(std::vector<RayCuda<float4>>& rays, int* results)`：对 host 侧传入的光线数组执行 BVH 求交，`results` 长度应不小于 `rays.size()`。
- `ParallelRaysIntersectionWithBVHAndRaysCuda3()`：使用 `ComputeOutRaysCuda` 已生成在 GPU 上的光线执行 BVH 求交，结果会拷贝回内部 host 缓冲区。

### Cell、MeshBox 与采样光线

- `InitCellBoundsCuda(std::vector<BoundBoxCuda>& cells)`：上传 Cell 包围盒到 GPU。
- `InitMeshBoundsCuda(std::vector<BoundBoxCuda>& meshboxes)`：上传 Mesh 包围盒到 GPU。
- `InitOutRaysCuda(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample)`：根据 `num_cells * num_meshboxes * num_cell_sample * num_meshbox_sample` 分配 GPU 光线缓冲区。
- `InitResults(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample)`：分配 host 侧结果数组。
- `ComputeOutRaysCuda(size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample, int st, int ed)`：对每个 Cell/MeshBox 组合做面采样，并生成指向 MeshBox 的光线。当前 `st`、`ed` 参数暂未实际参与分段计算。
- `GetOutRaysFromCuda()`：调试接口，将 GPU 上生成的光线拷回 host。正式流程中应尽量避免频繁调用。
- `GetDevOutRaysLength()`：返回当前 GPU 光线缓冲区长度。

### 结果读取

- `GetHostResults()`：返回内部 host 结果数组指针。
- `CopyHostResultsToVec()`：将内部结果数组复制到 `std::vector<int>` 缓冲中。
- `GetHostResultsVec()`：返回 `CopyHostResultsToVec()` 生成的结果向量。

## 数据结构约定

### `TriangleCuda<float4>`

```cpp
struct TriangleCuda<float4>
{
    float4 vertices[3];
};
```

`vertices` 保存三角形三个顶点。当前几何计算只使用 `x/y/z`，`w` 通常可置为 `0.0f` 或按调用方约定填充。

### `RayCuda<float4>`

```cpp
struct RayCuda<float4>
{
    float4 origin;
    float4 dir;
    float t;
};
```

- `origin`：光线起点。
- `dir`：单位方向向量。
- `t`：光线长度。如果由起点和终点构造光线，需要调用方提前计算距离。

### `BoundBoxCuda`

```cpp
struct BoundBoxCuda
{
    float4 minval;
    float4 maxval;
};
```

`minval` 与 `maxval` 表示 AABB 的最小/最大坐标。当前计算主要使用 `x/y/z`。

## 构建环境

当前工程文件使用：

- Visual Studio C++ Platform Toolset：`v143`
- Windows SDK：`10.0`
- CUDA Build Customizations：`CUDA 12.8`
- 平台：`x64`

项目配置包含：

- `Debug|x64`：控制台应用，便于本地调试。
- `Release|x64`：控制台应用。
- `lib|x64`：静态库配置，供 UE/UnrealLightmass 链接使用。

链接依赖中包含 CUDA runtime 与 cuRAND：

```text
cudart_static.lib
curand.lib
```

## 生成与集成

1. 使用 Visual Studio 打开 `CudaBVH.sln`。
2. 选择 `lib|x64` 配置。
3. 构建项目，生成 `CudaBVH.lib`。
4. 将静态库复制到 UE 侧：

```text
Engine\Source\Programs\UnrealLightmass\Private\CudaTest\lib
```

5. 将对外接口与必要数据结构头文件复制到 UE 侧：

```text
Engine\Source\Programs\UnrealLightmass\Private\CudaTest\include
```

建议至少包含：

```text
BoundBoxCuda.cuh
GPU4UEInterfaces.cuh
json.hpp
JsonExport.cuh
RayCuda.cuh
TriangleCuda.cuh
VecType.cuh
```

如果 UE 侧直接包含了更底层的实现或模板函数，还需要同步对应依赖头文件，例如 `Geometry.cuh`、`aabb.cuh`、`bvh.cuh`、`lbvh.cuh`、`utility.cuh` 等。

## 调试说明

- `GPU4UETest.cuh` 中提供了若干本地测试函数：
  - `Test1()`：直接光线-三角形求交。
  - `Test2()`：BVH host/device 查询对比。
  - `Test3()`：多线程调用 BVH 求交测试。
  - `Test4()`：Cell/MeshBox 光线采样、BVH 求交与 JSON 导出测试。
- `main.cu` 当前注释掉了 `main()`，如需本地运行测试，可临时恢复入口并调用对应测试函数。
- `JsonExport.cu` 可导出 Cell、MeshBox 与 Ray 信息，便于在 UE 或外部工具中检查采样结果。

## 注意事项

- 当前接口内部使用若干静态全局 GPU/host 缓冲区，例如 `dev_cells`、`dev_meshboxes`、`dev_out_rays` 与 `host_results`。重复初始化会释放旧缓冲并重新分配。
- `ComputeOutRaysCuda` 的光线数量为 `num_cells * num_meshboxes * num_cell_sample * num_meshbox_sample`，采样量过大时会显著增加显存占用。
- `GPU4UEInterfaces.cu` 中已有 TODO 标记：当前 GPU 内存分配方式还未做分线程/分块调度，超大规模输入时可能触发显存不足。
- `GetOutRaysFromCuda()` 主要用于调试，会把全部光线从 GPU 拷回 CPU，正式集成中应谨慎使用。
- 当前默认调用 `cudaSetDevice(0)`，多 GPU 场景需要根据调用方环境调整设备选择策略。
