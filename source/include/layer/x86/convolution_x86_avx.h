#pragma once
#if __AVX__
#include "layer/layer_convolution.h"

namespace ACNN
{
    class ConvolutionX86avx : public Convolution
    {
    public:
        ConvolutionX86avx(const LayerParam& layer_param);
        virtual ~ConvolutionX86avx() {}

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

    private:
        int forward_sgemm_avx(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs) const;

    };
}

#endif