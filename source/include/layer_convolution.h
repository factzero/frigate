#pragma once
#include "layer.h"


namespace ACNN
{
    class Convolution : public Layer
    {
    public:
        Convolution(const LayerParam& layer_param);

        virtual int load_param(const ParamDict& pd) override;
        virtual int load_model(const ModelBin& mb) override;
        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

    private:
        void make_padding(const aMat& bottom_blob, aMat& bottom_blob_bordered) const;
        float activation(float v) const;
        int forward_c(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs) const;
        int forward_sgemm(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs) const;

    private:
        int num_output;
        int kernel_w;
        int kernel_h;
        int dilation_w;
        int dilation_h;
        int stride_w;
        int stride_h;
        int pad_left;                        // -233=SAME_UPPER -234=SAME_LOWER
        int pad_right;
        int pad_top;
        int pad_bottom;
        float pad_value;
        int bias_term;
        int weight_data_size;
        int int8_scale_term;
        int activation_type;                 // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
        aMat activation_params;
        int dynamic_weight;

        aMat weight_data;
        aMat bias_data;
    };
}