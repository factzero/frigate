#pragma once
#include "layer/layer.h"


namespace ACNN
{
    class Requantize : public Layer
    {
    public:
        Requantize(const LayerParam& layer_param);
        virtual ~Requantize() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int load_model(const ModelBin& mb) override;

        virtual int forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const override;

    public:
        int scale_in_data_size;
        int scale_out_data_size;
        int bias_data_size;
        int activation_type;           // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
        aMat activation_params;
        aMat scale_in_data;
        aMat scale_out_data;
        aMat bias_data;
    };
}