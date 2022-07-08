#pragma once
#include "layer.h"


namespace ACNN
{
    class Softmax : public Layer
    {
    public:
        Softmax(const LayerParam& layer_param);
        virtual ~Softmax() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

    private:
        int axis;
    };
}