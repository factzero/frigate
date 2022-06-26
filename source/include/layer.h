#pragma once
#include <vector>
#include <string>
#include "param_dict.h"
#include "model_bin.h"
#include "option.h"


namespace ACNN
{
    struct LayerParam
    {
        std::string layer_type;
        std::string layer_name;
    };

    class Layer
    {
    public:
        Layer(const LayerParam& layer_param);
        virtual ~Layer() {}

        virtual int load_param(const ParamDict& pd);
        virtual int load_model(const ModelBin& mb);
        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const;

        std::string get_layer_type() const { return m_layer_type; }
        std::string get_layer_name() const { return m_layer_name; }

    public:
        // blob index which this layer needs as input
        std::vector<int> bottoms;
        // blob index which this layer produces as output
        std::vector<int> tops;

    private:
        std::string m_layer_type;
        std::string m_layer_name;
    };

}