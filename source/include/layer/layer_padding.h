#pragma once
#include "layer/layer.h"


namespace ACNN
{
    class Padding : public Layer
    {
    public:
        Padding(const LayerParam& layer_param);
        virtual ~Padding() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int load_model(const ModelBin& mb) override;

        virtual int forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const override;

    public:
        int top;
        int bottom;
        int left;
        int right;
        int type;                      // 0=CONSTANT 1=REPLICATE 2=REFLECT
        float value;
        int front;
        int behind;
        int per_channel_pad_data_size; // per channel pad value
        aMat per_channel_pad_data;
    };
}