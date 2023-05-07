#pragma once
#include "layer/layer.h"


namespace ACNN
{
    class Flatten : public Layer
    {
    public:
        Flatten(const LayerParam& layer_param);
        virtual ~Flatten() {}

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;
    };
}