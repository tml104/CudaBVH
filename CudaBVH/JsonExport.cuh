#pragma once
#include <vector>
#include <string>

#include "BoundBoxCuda.cuh"
#include "RayCuda.cuh"

namespace GPU4UE
{
	void CellToJson(std::vector<BoundBoxCuda>& cells, const std::string export_json_path); // 这个路径可能要直接写死，下同

	void RayToJson(std::vector<RayCuda<float4>>& rays, int* results, size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample, const std::string export_json_path);

}