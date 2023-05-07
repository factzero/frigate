#pragma once
#include "layer.h"


namespace ACNN
{
    class Relu : public Layer
    {
    public:
        Relu(const LayerParam& layer_param);
        virtual ~Relu() {}

        virtual int load_param(const ParamDict& pd) override;

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;
        virtual int forward_inplace(aMat& bottom_top_blob, const Option& opt) const override;
        virtual int forward_inplace(std::vector<aMat>& bottom_top_blobs, const Option& opt) const override;

    protected:
        float slope;
    };
}