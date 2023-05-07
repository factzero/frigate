#pragma once
#if __AVX__
#include "layer/layer_dropout.h"


namespace ACNN
{
    class DropoutX86avx : public Dropout
    {
    public:
        DropoutX86avx(const LayerParam& layer_param);
        virtual ~DropoutX86avx() {}

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;
    };
}

#endif