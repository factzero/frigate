#include "layer/layer_convolution.h"
#include "layer/layer_factory.h"
#include "layer/layer_fused_activation.h"
#include "quantize_tools.h"
#include "common.h"
#include "logger.h"


namespace ACNN
{
    void conv_im2col(const aMat& blob, aMat& blob_tm, const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const int outw, const int outh)
    {
        const int w = blob.m_w;
        const int h = blob.m_h;
        const int inch = blob.m_c;

        blob_tm.create(outw * outh, kernel_w * kernel_h * inch, blob.m_elemsize, blob.m_allocator);
        const int stride = kernel_w * kernel_h * outw * outh;
        float* postr = blob_tm;
        for (int q = 0; q < inch; q++)
        {
            const float* pistr = blob.channel(q);
            int idx = stride * q;
            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    for (int y = 0; y < outh; y++)
                    {
                        for (int x = 0; x < outw; x++)
                        {
                            int col = v + x * stride_w;
                            int row = u + y * stride_h;
                            postr[idx] = pistr[row * w + col];
                            idx++;
                        }
                    }
                }
            }
        }

        return;
    }

    Convolution::Convolution(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Convolution::load_param(const ParamDict& pd)
    {
        num_output = pd.get(0, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        dilation_w = pd.get(2, 1);
        dilation_h = pd.get(12, dilation_w);
        stride_w = pd.get(3, 1);
        stride_h = pd.get(13, stride_w);
        pad_left = pd.get(4, 0);
        pad_right = pd.get(15, pad_left);
        pad_top = pd.get(14, pad_left);
        pad_bottom = pd.get(16, pad_top);
        pad_value = pd.get(18, 0.f);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);
        int8_scale_term = pd.get(8, 0);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, aMat());
        dynamic_weight = pd.get(19, 0);

        if (int8_scale_term)
        {
            support_int8_storage = true;
        }

        return 0;
    }

    int Convolution::load_model(const ModelBin& mb)
    {
        if (dynamic_weight)
            return 0;

        weight_data = mb.load(weight_data_size, 0);
        if (weight_data.empty())
        {
            return -100;
        }

        if (bias_term)
        {
            bias_data = mb.load(num_output, 1);
            if (bias_data.empty())
            {
                return -100;
            }
        }

        if (int8_scale_term)
        {
            weight_data_int8_scales = mb.load(num_output, 1);
            bottom_blob_int8_scales = mb.load(1, 1);
        }

        if (int8_scale_term > 100)
        {
            top_blob_int8_scales = mb.load(1, 1);
        }

        return 0;
    }

    int Convolution::create_pipeline(const Option& opt)
    {
        m_activation = create_activation_layer(activation_type, activation_params, opt);

        if (opt.use_int8_inference && weight_data.m_elemsize == 1)
        {
            create_pipeline_int8(opt);
        }

        return 0;
    }

    int Convolution::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (bottom_blobs.size() != top_blobs.size())
        {
            ConsoleELog << "Convolution ERROR: bottom_blobs size(" << bottom_blobs.size() << ") != top_blobs size(" << top_blobs.size() << ")";
            return -1;
        }

        if (opt.use_int8_inference && int8_scale_term)
        {
            return forward_int8(bottom_blobs, top_blobs, opt);
        }
        if (opt.use_sgemm_convolution)
        {
            return forward_sgemm(bottom_blobs, top_blobs, opt);
        }
        else
        {
            return forward_fp32(bottom_blobs, top_blobs, opt);
        }

        return 0;
    }

    void Convolution::make_padding(const aMat& bottom_blob, aMat& bottom_blob_bordered, const Option& opt) const
    {
        int w = bottom_blob.m_w;
        int h = bottom_blob.m_h;
        const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

        bottom_blob_bordered = bottom_blob;
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, 0, pad_value, opt);
        }
        else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
        {
            // tensorflow padding=SAME or onnx padding=SAME_UPPER
            int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
            int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
            if (wpad > 0 || hpad > 0)
            {
                copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, 0, pad_value, opt);
            }
        }
        else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
            int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
            if (wpad > 0 || hpad > 0)
            {
                copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, 0, pad_value, opt);
            }
        }

        return;
    }

    float Convolution::activation(float v) const
    {
        switch (activation_type)
        {
            case 1:
            {
                v = (float)fmax(v, 0.f);
                break;
            }
            case 2:
            {
                float slope = activation_params[0];
                v = v > 0.f ? v : v * slope;
                break;
            }
            case 3:
            {
                float min = activation_params[0];
                float max = activation_params[1];
                if (v < min)
                    v = min;
                if (v > max)
                    v = max;
                break;
            }
            case 4:
            {
                v = 1.f / (1.f + exp(-v));
                break;
            }
            case 5:
            {
                const float MISH_THRESHOLD = 20;
                float x = v, y;
                if (x > MISH_THRESHOLD)
                {
                    y = x;
                }
                else if (x < -MISH_THRESHOLD)
                {
                    y = expf(x);
                }
                else
                {
                    y = logf(expf(x) + 1.f);
                }
                v = x * tanh(y);                
                break;
            }
            case 6:
            {
                float alpha = activation_params[0];
                float beta = activation_params[1];
                float lower = -beta / alpha;
                float upper = (1.f / alpha) + lower;
                if (v < lower)
                    v = 0.f;
                else if (v > upper)
                    ;
                else
                    v = v * (v * alpha + beta);
                break;
            }
        }

        return v;
    }

    void Convolution::convolution_int8(const aMat& bottom_blob_int8, aMat& top_blob_int32, const aMat& weight_data_int8, 
        int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt) const
    {
        const int w = bottom_blob_int8.m_w;
        const int h = bottom_blob_int8.m_h;
        const int channels = bottom_blob_int8.m_c;
        const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
        const int outw = (w - kernel_extent_w) / stride_w + 1;
        const int outh = (h - kernel_extent_h) / stride_h + 1;

        for (int o = 0; o < num_output; o++)
        {
            int* outptr = top_blob_int32.channel(o);
            for (int y = 0; y < outh; y++)
            {
                for (int x = 0; x < outw; x++)
                {
                    int sum = 0;
                    const char* kptr = (const char*)weight_data_int8 + kernel_w * kernel_h * channels * o;
                    for (int q = 0; q < channels; q++)
                    {
                        const aMat in_m = bottom_blob_int8.channel(q);
                        const char* sptr = (const char*)in_m + y * stride_h * in_m.m_w + x * stride_w;
                        for (int m = 0; m < kernel_h; m++)
                        {
                            for (int n = 0; n < kernel_w; n++)
                            {
                                int val = sptr[n];
                                int wt = kptr[m * kernel_w + n];
                                sum += val * wt;
                            }
                            sptr += in_m.m_w;
                        }
                        kptr += kernel_w * kernel_h;
                    }
                    outptr[x] = sum;
                }
                outptr += outw;
            }
        }

        return;
    }

    int Convolution::forward_int8(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];

            aMat bottom_blob_int8 = bottom_blob;
            if (bottom_blob.m_elemsize != 1)
            {
                quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt);
            }

            aMat bottom_blob_bordered;
            make_padding(bottom_blob_int8, bottom_blob_bordered, opt);

            const int w = bottom_blob_bordered.m_w;
            const int h = bottom_blob_bordered.m_h;
            const int channels = bottom_blob_bordered.m_c;
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
            const int outw = (w - kernel_extent_w) / stride_w + 1;
            const int outh = (h - kernel_extent_h) / stride_h + 1;

            bool use_int8_requantize = int8_scale_term > 100;
            int out_elemsize = use_int8_requantize ? 1u : 4u;

            top_blob.create(outw, outh, num_output, out_elemsize, bottom_blob_bordered.m_allocator);
            aMat top_blob_int32(outw, outh, num_output, (size_t)4u, bottom_blob_bordered.m_allocator);

            convolution_int8(bottom_blob_bordered, top_blob_int32, weight_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (use_int8_requantize)
            {
                requantize_int32_to_int8(top_blob_int32, top_blob, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt);
            }
            else
            {
                dequantize_int32_to_fp32(top_blob_int32, top_blob, scale_in_data, bias_data, opt);
                if (m_activation)
                {
                    m_activation->forward_inplace(top_blob, opt);
                }
            }
        }

        return 0;
    }

    int Convolution::forward_fp32(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            aMat bottom_blob_bordered;
            make_padding(bottom_blob, bottom_blob_bordered, opt);

            const int w = bottom_blob_bordered.m_w;
            const int h = bottom_blob_bordered.m_h;
            const int channels = bottom_blob_bordered.m_c;
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
            const int outw = (w - kernel_extent_w) / stride_w + 1;
            const int outh = (h - kernel_extent_h) / stride_h + 1;

            top_blob.create(outw, outh, num_output, bottom_blob_bordered.m_elemsize, bottom_blob_bordered.m_allocator);
            for (int o = 0; o < num_output; o++)
            {
                float* outptr = top_blob.channel(o);
                for (int y = 0; y < outh; y++)
                {
                    for (int x = 0; x < outw; x++)
                    {
                        float sum = 0.f;
                        const float* kptr = (const float*)weight_data + kernel_w * kernel_h * channels * o;
                        for (int q = 0; q < channels; q++)
                        {
                            const aMat in_m = bottom_blob_bordered.channel(q);
                            const float* sptr = (const float*)in_m + y * stride_h * in_m.m_w + x * stride_w;
                            for (int m = 0; m < kernel_h; m++)
                            {
                                for (int n = 0; n < kernel_w; n++)
                                {
                                    float val = sptr[n];
                                    float wt = kptr[m * kernel_w + n];
                                    sum += val * wt;
                                }
                                sptr += in_m.m_w;
                            }
                            kptr += kernel_w * kernel_h;
                        }

                        if (bias_term)
                        {
                            sum += bias_data[o];
                        }
                        outptr[x] = activation(sum);
                    }
                    outptr += outw;
                }
            }
        }

        return 0;
    }

    int Convolution::forward_sgemm(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            aMat bottom_blob_bordered;
            make_padding(bottom_blob, bottom_blob_bordered, opt);

            const int w = bottom_blob_bordered.m_w;
            const int h = bottom_blob_bordered.m_h;
            const int channels = bottom_blob_bordered.m_c;
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
            const int outw = (w - kernel_extent_w) / stride_w + 1;
            const int outh = (h - kernel_extent_h) / stride_h + 1;

            aMat bottom_im2col;
            conv_im2col(bottom_blob_bordered, bottom_im2col, kernel_w, kernel_h, stride_w, stride_h, outw, outh);

            // weight2cal
            aMat weight_sgemm_data = weight_data.reshape(kernel_w * kernel_h * channels, num_output, weight_data.m_allocator);

            top_blob.create(outw, outh, num_output, bottom_blob_bordered.m_elemsize, bottom_blob_bordered.m_allocator);
            for (int m = 0; m < num_output; m++)
            {
                const float* pistr = bottom_im2col;
                const float* pwstr = weight_sgemm_data;
                float* postr = top_blob.channel(m);
                for (int n = 0; n < outw * outh; n++)
                {
                    float sum = 0.f;
                    for (int k = 0; k < kernel_w * kernel_h * channels; k++)
                    {
                        float wt = pwstr[m * weight_sgemm_data.m_w + k];
                        float val = pistr[k * bottom_im2col.m_w + n];
                        sum += wt * val;
                    }

                    if (bias_term)
                    {
                        sum += bias_data[m];
                    }
                    postr[n] = activation(sum);
                }
                pwstr += weight_sgemm_data.m_w;
            }
        }

        return 0;
    }

    int Convolution::create_pipeline_int8(const Option& opt)
    {
        scale_in_data.create(num_output);
        for (int p = 0; p < num_output; p++)
        {
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
            scale_in_data[p] = scale_in;
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Convolution, Convolution);
}