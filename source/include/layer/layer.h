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

        virtual int create_pipeline(const Option& opt);

        virtual int forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const;
        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const;
        virtual int forward_inplace (aMat& bottom_top_blob, const Option& opt) const;
        virtual int forward_inplace(std::vector<aMat>& bottom_top_blobs, const Option& opt) const;

        std::string get_layer_type() const { return m_layer_type; }
        std::string get_layer_name() const { return m_layer_name; }

    public:
        // support inplace inference
        bool support_inplace;
        // accept int8
        bool support_int8_storage;

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