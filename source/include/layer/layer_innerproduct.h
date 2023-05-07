#pragma once
#include "layer/layer.h"


namespace ACNN
{
    class InnerProduct : public Layer
    {
    public:
        InnerProduct(const LayerParam& layer_param);
        virtual ~InnerProduct() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int load_model(const ModelBin& mb) override;

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

    private:
        int forward_int8(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const;
        int forward_fp32(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const;

    private:
        // param
        int num_output;
        int bias_term;
        int weight_data_size;
        int int8_scale_term;
        // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
        int activation_type;
        aMat activation_params;

        // model
        aMat weight_data;
        aMat bias_data;

        aMat weight_data_int8_scales;
        aMat bottom_blob_int8_scales;
    };
}