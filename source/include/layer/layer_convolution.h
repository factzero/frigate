#pragma once
#include "layer.h"


namespace ACNN
{
    void conv_im2col(const aMat& blob, aMat& blob_tm, const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const int outw, const int outh);

    class Convolution : public Layer
    {
    public:
        Convolution(const LayerParam& layer_param);
        virtual ~Convolution() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int load_model(const ModelBin& mb) override;

        virtual int create_pipeline(const Option& opt) override;

        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

    protected:
        void make_padding(const aMat& bottom_blob, aMat& bottom_blob_bordered, const Option& opt) const;
        float activation(float v) const;

    private:
        void convolution_int8(const aMat& bottom_blob_int8, aMat& top_blob_int32, const aMat& weight_data_int8, 
            int kernel_w, int kernel_h, int dilation_w, int dilation_h,int stride_w, int stride_h, const Option& opt) const;
        int forward_int8(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const;
        int forward_fp32(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const;
        int forward_sgemm(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const;

    private:
        int create_pipeline_int8(const Option& opt);

    protected:
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
        mutable aMat weight_sgemm_data;

        aMat weight_data_int8_scales;
        aMat bottom_blob_int8_scales;
        aMat top_blob_int8_scales;

        aMat scale_in_data;
        std::shared_ptr<Layer> m_activation;
    };
}