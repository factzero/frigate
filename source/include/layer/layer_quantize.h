#pragma once
#include "layer/layer.h"


namespace ACNN
{
    class Quantize : public Layer
    {
    public:
        Quantize(const LayerParam& layer_param);
        virtual ~Quantize() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int load_model(const ModelBin& mb) override;

        virtual int forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const override;

    public:
        int scale_data_size;
        aMat scale_data;
    };
}