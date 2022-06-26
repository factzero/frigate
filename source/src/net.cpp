#include <iostream>
#include "net.h"
#include "platform.h"
#include "data_reader.h"
#include "param_dict.h"
#include "layer_factory.h"


namespace ACNN
{
	int Net::load_param(const char* param_file)
	{
	#define SCAN_VALUE(fmt, v)                \
		if (dr->scan(fmt, &v) != 1)           \
		{                                     \
			ACNN_LOGE("parse " #v " failed"); \
			return -1;                        \
		}

		DataReader dr = std::make_shared<DataReaderFromFile>(param_file);
		int magic = 0;
		SCAN_VALUE("%d", magic);
		if (magic != 7767517)
		{
			ACNN_LOGE("param is too old, please regenerate");
			return -1;
		}

		int layer_count = 0;
		int blob_count = 0;
		SCAN_VALUE("%d", layer_count);
		SCAN_VALUE("%d", blob_count);
		if (layer_count <= 0 || blob_count <= 0)
		{
			ACNN_LOGE("invalid layer_count or blob_count");
			return -1;
		}

		m_layers.resize(layer_count);
		m_blobs.resize(blob_count);
		m_blobs_flag.resize(blob_count);

		ParamDict pd;
		int blob_index = 0;
		for (int i = 0; i < layer_count; i++)
		{
			char layer_type[256];
			char layer_name[256];
			int bottom_count = 0;
			int top_count = 0;
			SCAN_VALUE("%255s", layer_type);
			SCAN_VALUE("%255s", layer_name);
			SCAN_VALUE("%d", bottom_count);
			SCAN_VALUE("%d", top_count);

			ACNN_LOGE("%d layer type %s, layer name %s", i, layer_type, layer_name);

			LayerParam param;
			param.layer_type = std::string(layer_type);
			param.layer_name = std::string(layer_name);

			std::shared_ptr<Layer> layer = LayerRegistry::CreateLayer(param);
			if (!layer)
			{
				ACNN_LOGE("layer %s not exists or registered", layer_type);
				return -1;
			}

			layer->bottoms.resize(bottom_count);
			for (int j = 0; j < bottom_count; j++)
			{
				char bottom_name[256];
				SCAN_VALUE("%255s", bottom_name);
				int bottom_blob_index = find_blob_index_by_name(bottom_name);
				if (-1 == bottom_blob_index)
				{
					Blob& blob = m_blobs[blob_index];
					bottom_blob_index = blob_index;
					blob.name = std::string(bottom_name);
					blob_index++;
				}

				Blob& blob = m_blobs[bottom_blob_index];
				blob.consumer = i;
				layer->bottoms[j] = bottom_blob_index;
			}

			layer->tops.resize(top_count);
			for (int j = 0; j < top_count; j++)
			{
				char blob_name[256];
				SCAN_VALUE("%255s", blob_name);

				Blob& blob = m_blobs[blob_index];
				blob.name = std::string(blob_name);
				blob.producer = i;
				layer->tops[j] = blob_index;
				blob_index++;
			}

			if (pd.load_param(dr))
			{
				ACNN_LOGE("ParamDict load_param %d %s failed", i, layer_name);
				continue;
			}

			if (layer->load_param(pd))
			{
				ACNN_LOGE("layer load_param %d %s failed", i, layer_name);
				continue;
			}

			m_layers[i] = layer;
		}

		compute_calc_seq();

		for (auto& r : m_calc_seq)
		{
			std::cout << r << " ";
		}
		std::cout << std::endl;

		return 0;
	}

	int Net::load_model(const char* model_file)
	{
		DataReader dr = std::make_shared<DataReaderFromFile>(model_file);
		ModelBinFromDataReader mb(dr);

		for (size_t i = 0; i < m_layers.size(); i++)
		{
			std::shared_ptr<Layer> layer = m_layers[i];
			if (layer->load_model(mb))
			{
				ACNN_LOGE("layer load_model %d %s failed", (int)i, layer->get_layer_name().c_str());
				return -1;
			}
		}

		return 0;
	}

	int Net::forward(const aMat& input_data, aMat& output_data)
	{
		int ret = 0;

		m_blobs[m_input_idx].set_data(input_data);

		for (auto& r : m_calc_seq)
		{
			std::shared_ptr<Layer> layer = m_layers[r];
			std::vector<aMat> bottom_blobs;
			std::vector<aMat> top_blobs;
			for (size_t i = 0; i < layer->bottoms.size(); i++)
			{
				bottom_blobs.push_back(m_blobs[layer->bottoms[i]].get_data());
			}
			for (size_t i = 0; i < layer->tops.size(); i++)
			{
				top_blobs.push_back(m_blobs[layer->tops[i]].get_data());
			}

			ACNN_LOGE("calc %3d, %s , %s", r, layer->get_layer_type().c_str(), layer->get_layer_name().c_str());
			ret = layer->forward(bottom_blobs, top_blobs, opt);
			for (size_t i = 0; i < layer->tops.size(); i++)
			{
				m_blobs[layer->tops[i]].set_data(top_blobs[i]);
				{
					int w = top_blobs[i].m_w;
					int h = top_blobs[i].m_h;
					int channels = top_blobs[i].m_c;
					float sum = 0.f;

					for (int q = 0; q < channels; q++)
					{
						const aMat out_m = top_blobs[i].channel(q);
						for (int j = 0; j < w * h; j++)
						{
							sum += out_m[j];
						}
					}
					ACNN_LOGE("  top[%d] sum: %f", (int)i, sum);
				}
			}
		}

		output_data = m_blobs[m_output_idx].get_data();

		return 0;
	}

	int Net::set_input_name(std::string name)
	{
		int ret = -1;
		for (auto& layer : m_layers)
		{
			if (0 == layer->get_layer_name().compare(name.c_str()))
			{
				m_input_idx = layer->tops[0];
				ACNN_LOGE("Found Input Layer, Blob Idx %d", m_input_idx);
				ret = 0;
				break;
			}
		}

		return ret;
	}

	int Net::set_output_name(std::string name)
	{
		int ret = -1;
		for (auto& layer : m_layers)
		{
			if (0 == layer->get_layer_name().compare(name.c_str()))
			{
				m_output_idx = layer->tops[0];
				ACNN_LOGE("Found Output Layer, Blob Idx %d", m_output_idx);
				ret = 0;
				break;
			}
		}

		return ret;
	}

	int Net::find_blob_index_by_name(const char* name) const
	{
		for (size_t i = 0; i < m_blobs.size(); i++)
		{
			const Blob& blob = m_blobs[i];
			if (blob.name == name)
			{
				return static_cast<int>(i);
			}
		}

		ACNN_LOGE("find_blob_index_by_name %s failed", name);
		return -1;
	}

	void Net::forward_layer_order(int layer_idx)
	{
		std::shared_ptr<Layer> layer = m_layers[layer_idx];
		for (size_t i = 0; i < layer->bottoms.size(); i++)
		{
			int bottom_blob_idx = layer->bottoms[i];
			if (!m_blobs_flag[bottom_blob_idx])
			{
				forward_layer_order(m_blobs[bottom_blob_idx].producer);
			}
		}

		m_calc_seq.push_back(layer_idx);
		for (size_t i = 0; i < layer->bottoms.size(); i++)
		{
			int bottom_blob_idx = layer->bottoms[i];
			m_blobs_flag[bottom_blob_idx] = true;
		}
		for (size_t i = 0; i < layer->tops.size(); i++)
		{
			int top_blob_idx = layer->tops[i];
			m_blobs_flag[top_blob_idx] = true;
		}

		return;
	}

	void Net::compute_calc_seq()
	{
		int output_layer = (int)m_layers.size() - 1;
		forward_layer_order(output_layer);

		return;
	}
}