#pragma once
#include "layer.h"


namespace ACNN
{
    class Split : public Layer
    {
    public:
        Split(const LayerParam& layer_param);

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

    };
}