#pragma once
#include "layer/layer.h"


namespace ACNN
{
    class Dequantize : public Layer
    {
    public:
        Dequantize(const LayerParam& layer_param);
        virtual ~Dequantize() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int load_model(const ModelBin& mb) override;
        virtual int forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const override;

    private:
        int scale_data_size;
        int bias_data_size;

        aMat scale_data;
        aMat bias_data;
    };
}