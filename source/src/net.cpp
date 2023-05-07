#include <iostream>
#include <chrono>
#include "net.h"
#include "data_reader.h"
#include "param_dict.h"
#include "layer/layer_factory.h"
#include "logger.h"

using std::chrono::high_resolution_clock;
using std::chrono::microseconds;


namespace ACNN
{
    int Net::load_param(const char* param_file)
    {
    #define SCAN_VALUE(fmt, v)                          \
        if (dr->scan(fmt, &v) != 1)                     \
        {                                               \
            ConsoleELog << "parse " << #v << " failed"; \
            return -1;                                  \
        }

        DataReader dr = std::make_shared<DataReaderFromFile>(param_file);
        int magic = 0;
        SCAN_VALUE("%d", magic);
        if (magic != 7767517)
        {
            ConsoleLog << "param is too old, please regenerate";
            return -1;
        }

        int layer_count = 0;
        int blob_count = 0;
        SCAN_VALUE("%d", layer_count);
        SCAN_VALUE("%d", blob_count);
        if (layer_count <= 0 || blob_count <= 0)
        {
            ConsoleELog << "invalid layer_count or blob_count";
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

            LayerParam param;
            param.layer_type = std::string(layer_type);
            param.layer_name = std::string(layer_name);

            std::shared_ptr<Layer> layer = LayerRegistry::CreateLayer(param);
            if (!layer)
            {
                ConsoleELog << layer_type << " not exists or registered";
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
                ConsoleELog << "ParamDict load_param " << i << " " << layer_name << " failed";
                continue;
            }

            if (layer->load_param(pd))
            {
                ConsoleELog << "layer load_param " << i << " " << layer_name << " failed";
                continue;
            }

            m_layers[i] = layer;
        }

        compute_calc_seq();

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
                ConsoleELog << "layer load_model " << i << " " << layer->get_layer_name() << " failed";
                return -1;
            }
        }

        for (auto& r : m_calc_seq)
        {
            std::shared_ptr<Layer> layer = m_layers[r];
            layer->create_pipeline(opt);
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
            ConsoleLog << "calc " << r << " , " << layer->get_layer_type() << "  :  " << layer->get_layer_name();
            high_resolution_clock::time_point startTime = high_resolution_clock::now();
            ret = layer->forward(bottom_blobs, top_blobs, opt);
            high_resolution_clock::time_point endTime = high_resolution_clock::now();
            microseconds timeInterval = std::chrono::duration_cast<microseconds>(endTime - startTime);
            std::cout << "squeezenet.forward cost time: " << timeInterval.count() / 1000.f << " ms" << std::endl;
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                m_blobs[layer->tops[i]].set_data(top_blobs[i]);
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
                ConsoleLog << "Found Input Layer, Blob Idx " << m_input_idx;
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
                ConsoleLog << "Found Output Layer, Blob Idx " << m_output_idx;
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

        ConsoleELog << "find_blob_index_by_name failed: " << name;
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