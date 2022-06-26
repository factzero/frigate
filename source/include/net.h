#pragma once
#include <string>
#include "blob.h"
#include "layer.h"


namespace ACNN
{
	class Net
	{
	public:
		Net() {}
		~Net() {}

		int load_param(const char* param_file);
		int load_model(const char* model_file);
		int forward(const aMat& input_data, aMat& output_data);
		int set_input_name(std::string name);
		int set_output_name(std::string name);

	private:
		int find_blob_index_by_name(const char* name) const;
		void forward_layer_order(int layer_idx);
		void compute_calc_seq();

	public:
		Option opt;

	private:
		std::vector<std::shared_ptr<Layer>> m_layers;
		std::vector<Blob> m_blobs;
		std::vector<bool> m_blobs_flag;
		std::vector<int> m_calc_seq;
		int m_input_idx;
		int m_output_idx;
	};
}