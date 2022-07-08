#pragma once
#include "layer.h"


namespace ACNN
{
    class Pooling : public Layer
    {
    public:
        Pooling(const LayerParam& layer_param);
        virtual ~Pooling() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

    protected:
        void make_padding(const aMat& bottom_blob, aMat& bottom_blob_bordered) const;

    protected:
        int pooling_type;
        int kernel_w;
        int kernel_h;
        int stride_w;
        int stride_h;
        int pad_left;
        int pad_right;
        int pad_top;
        int pad_bottom;
        int global_pooling;
        int pad_mode;                        // 0=full 1=valid 2=SAME_UPPER 3=SAME_LOWER
        int avgpool_count_include_pad;
        int adaptive_pooling;
        int out_w;
        int out_h;
    };
}