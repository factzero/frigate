#pragma once
#if __AVX__
#include "layer/layer_softmax.h"


namespace ACNN
{
    class SoftmaxX86avx : public Softmax
    {
    public:
        SoftmaxX86avx(const LayerParam& layer_param);
        virtual ~SoftmaxX86avx() {}

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;
    };
}

#endif