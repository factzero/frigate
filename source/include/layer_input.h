#pragma once
#include "layer.h"


namespace ACNN
{
    class Input : public Layer
    {
    public:
        Input(const LayerParam& layer_param);

        virtual int load_param(const ParamDict& pd) override;
        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

    private:
        int w;
        int h;
        int d;
        int c;
    };
}