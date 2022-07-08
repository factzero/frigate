#pragma once
#if __AVX__
#include "layer/layer_pooling.h"

namespace ACNN
{
    class PoolingX86avx : public Pooling
    {
    public:
        PoolingX86avx(const LayerParam& layer_param);
        virtual ~PoolingX86avx() {}

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;
    };
}


#endif