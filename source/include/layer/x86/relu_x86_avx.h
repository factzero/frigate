#pragma once
#if __AVX__
#include "layer/layer_relu.h"

namespace ACNN
{
    class ReluX86avx : public Relu
    {
    public:
        ReluX86avx(const LayerParam& layer_param);
        virtual ~ReluX86avx() {}

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;
    };
}

#endif
