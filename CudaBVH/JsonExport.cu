#include "JsonExport.cuh"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "BoundBoxCuda.cuh"
#include "RayCuda.cuh"

#include "json.hpp"

namespace GPU4UE
{

	void CellToJson(std::vector<BoundBoxCuda>& cells, const std::string export_json_path)
	{
		using json = nlohmann::json;
		json j;

		json cell_boxes_json;


		for (size_t i = 0; i < cells.size(); i++) {

			json one_cell_json;

			auto&& cell = cells[i];
			
			one_cell_json["min_x"] = cell.minval.x;
			one_cell_json["min_y"] = cell.minval.y;
			one_cell_json["min_z"] = cell.minval.z;

			one_cell_json["max_x"] = cell.maxval.x;
			one_cell_json["max_y"] = cell.maxval.y;
			one_cell_json["max_z"] = cell.maxval.z;

			one_cell_json["id"] = i;

			cell_boxes_json.emplace_back(one_cell_json);

		}


		j["cell_boxes"] = cell_boxes_json;


		// 导出
		std::ofstream f(export_json_path);
		f << j << std::endl;
	}


	void RayToJson(std::vector<RayCuda<float4>>& rays, int* results, size_t num_cells, size_t num_meshboxes, size_t num_cell_sample, size_t num_meshbox_sample, const std::string export_json_path)
	{
		using json = nlohmann::json;
		json j;

		json ray_infos_json;

		for (size_t i = 0; i < rays.size(); i++) {  // rays.size() == results.size()

			json one_ray_json;

			auto&& ray = rays[i];
			int result = results[i];

			size_t ray_cell_meshboxes_id = i / (num_cell_sample * num_meshbox_sample);
			size_t ray_from_cell_id = ray_cell_meshboxes_id / num_meshboxes;
			size_t ray_to_meshbox_id = ray_cell_meshboxes_id % num_meshboxes;

			std::vector<float> start_point_coords, direction_coords;
			start_point_coords.emplace_back(ray.origin.x);
			start_point_coords.emplace_back(ray.origin.y);
			start_point_coords.emplace_back(ray.origin.z);

			direction_coords.emplace_back(ray.dir.x);
			direction_coords.emplace_back(ray.dir.y);
			direction_coords.emplace_back(ray.dir.z);

			one_ray_json["st"] = start_point_coords;
			one_ray_json["d"] = direction_coords;
			one_ray_json["r"] = ray.t;

			one_ray_json["id"] = i;
			one_ray_json["from"] = ray_from_cell_id;
			one_ray_json["to"] = ray_to_meshbox_id;

			one_ray_json["res"] = result;

			ray_infos_json.emplace_back(one_ray_json);
		}

		j["ray_infos"] = ray_infos_json;


		// 导出
		std::ofstream f(export_json_path);
		f << j << std::endl;
	}

}