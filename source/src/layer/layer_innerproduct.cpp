#include "layer/layer_innerproduct.h"
#include "layer/layer_factory.h"
#include "layer/layer_fused_activation.h"
#include "quantize_tools.h"
#include "logger.h"


namespace ACNN
{
    InnerProduct::InnerProduct(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int InnerProduct::load_param(const ParamDict& pd)
    {
        num_output = pd.get(0, 0);
        bias_term = pd.get(1, 0);
        weight_data_size = pd.get(2, 0);
        int8_scale_term = pd.get(8, 0);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, aMat());

        if (int8_scale_term)
        {
            support_int8_storage = true;
        }

        return 0;
    }

    int InnerProduct::load_model(const ModelBin& mb)
    {
        weight_data = mb.load(weight_data_size, 0);
        if (weight_data.empty())
            return -100;

        if (bias_term)
        {
            bias_data = mb.load(num_output, 1);
            if (bias_data.empty())
                return -100;
        }

        if (int8_scale_term)
        {
            weight_data_int8_scales = mb.load(num_output, 1);
            bottom_blob_int8_scales = mb.load(1, 1);
        }

        return 0;
    }

    int InnerProduct::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (opt.use_int8_inference && weight_data.m_elemsize == (size_t)1u)
        {
            return forward_int8(bottom_blobs, top_blobs, opt);
        }
        else
        {
            return forward_fp32(bottom_blobs, top_blobs, opt);
        }
    }

    int InnerProduct::forward_int8(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        const int w = bottom_blobs[0].m_w;
        const int h = bottom_blobs[0].m_h;
        const int c = bottom_blobs[0].m_c;
        const size_t elemsize = bottom_blobs[0].m_elemsize;
        int size = w * h;

        top_blobs[0].create(num_output, h, elemsize, bottom_blobs[0].m_allocator);
        if (top_blobs[0].empty())
        {
            return -100;
        }

        aMat bottom_blob_int8 = bottom_blobs[0];
        if (elemsize != 1)
        {
            quantize_to_int8(bottom_blobs[0], bottom_blob_int8, bottom_blob_int8_scales, opt);
        }

        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
            {
                sum = bias_data[p];
            }

            for (int q = 0; q < c; q++)
            {
                const signed char* m = bottom_blob_int8.channel(q);
                const signed char* w = (const signed char*)weight_data + size * c * p + size * q;

                for (int i = 0; i < size; i++)
                {
                    sum += m[i] * w[i];
                }
            }

            // dequantize and relu
            float scale_in;
            if (weight_data_int8_scales[p] == 0)
            {
                scale_in = 0;
            }
            else
            {
                scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);
            }

            float sumfp32 = sum * scale_in;

            if (bias_term)
            {
                sumfp32 += bias_data[p];
            }

            top_blobs[0][p] = activation_ss(sumfp32, activation_type, activation_params);
        }

        return 0;
    }

    int InnerProduct::forward_fp32(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        const int w = bottom_blobs[0].m_w;
        const int h = bottom_blobs[0].m_h;
        const int c = bottom_blobs[0].m_c;
        const size_t elemsize = bottom_blobs[0].m_elemsize;
        int size = w * h;

        top_blobs[0].create(num_output, h, elemsize, bottom_blobs[0].m_allocator);
        if (top_blobs[0].empty())
        {
            return -100;
        }

        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
            {
                sum = bias_data[p];
            }

            for (int q = 0; q < c; q++)
            {
                const float *m = bottom_blobs[0].channel(q);
                const float* w = (const float*)weight_data + size * c * p + size * q;

                for (int i = 0; i < size; i++)
                {
                    sum += m[i] * w[i];
                }
            }

            top_blobs[0][p] = activation_ss(sum, activation_type, activation_params);
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(InnerProduct, InnerProduct);
}