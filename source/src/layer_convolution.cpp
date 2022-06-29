#include "layer_convolution.h"
#include "layer_factory.h"
#include "logger.h"


namespace ACNN
{
    void conv_im2col_sgemm_transform_kernel_sse_8x8(const aMat& kernel, aMat& kernel_tm, const int kernel_w, const int kernel_h, const int inch, const int outch)
    {
        const int kernel_size = kernel_w * kernel_h;

        // src = kernel_size-inch-outch
        // dst = 8b-kernel_size-inch-outch/8b
        kernel_tm.create(8 * kernel_size, inch, outch / 8 + outch % 8, kernel.m_elemsize, kernel.m_allocator);

        const float* kernel_data = kernel;
        int q = 0;
        for (; q + 7 < outch; q += 8)
        {
            const float* k0 = kernel_data + inch * kernel_size * (q + 0);
            const float* k1 = kernel_data + inch * kernel_size * (q + 1);
            const float* k2 = kernel_data + inch * kernel_size * (q + 2);
            const float* k3 = kernel_data + inch * kernel_size * (q + 3);
            const float* k4 = kernel_data + inch * kernel_size * (q + 4);
            const float* k5 = kernel_data + inch * kernel_size * (q + 5);
            const float* k6 = kernel_data + inch * kernel_size * (q + 6);
            const float* k7 = kernel_data + inch * kernel_size * (q + 7);
            float* postr = kernel_tm.channel(q / 8);
            for (int i = 0; i < inch * kernel_size; i++)
            {
                postr[0] = k0[i];
                postr[1] = k1[i];
                postr[2] = k2[i];
                postr[3] = k3[i];
                postr[4] = k4[i];
                postr[5] = k5[i];
                postr[6] = k6[i];
                postr[7] = k7[i];
                postr += 8;
            }
        }

        for (; q < outch; q++)
        {
            const float* k0 = kernel_data + inch * kernel_size * q;
            float* postr = kernel_tm.channel(q / 8 + q % 8);
            for (int i = 0; i < inch * kernel_size; i++)
            {
                postr[0] = k0[i];
            }
        }

        return;
    }

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

    void im2col_sgemm_sse(const aMat& bottom_im2col, aMat& top_blob, const aMat& kernel, const aMat& bias, const int kernel_w, const int kernel_h, 
        const int inch, const int outch, const int outw, const int outh)
    {
        const int kernel_size = kernel_w * kernel_h;
        const int out_size = outw * outh;
        // bottom_im2col memory packed 8x8
        aMat bottom_tm(8 * kernel_size, inch, out_size / 8 + out_size % 8, bottom_im2col.m_elemsize, bottom_im2col.m_allocator);
        {
            int i = 0;
            for (; i + 7 < out_size; i += 8)
            {
                const float* img0 = bottom_im2col.channel(0);
                img0 += i;
                float* tmpptr = bottom_tm.channel(i / 8);
                for (int q = 0; q < inch * kernel_size; q++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];
                    tmpptr[2] = img0[2];
                    tmpptr[3] = img0[3];
                    tmpptr[4] = img0[4];
                    tmpptr[5] = img0[5];
                    tmpptr[6] = img0[6];
                    tmpptr[7] = img0[7];
                    tmpptr += 8;
                    img0 += out_size;
                }
            }

            for (; i < out_size; i++)
            {
                const float* img0 = bottom_im2col.channel(0);
                img0 += i;
                float* tmpptr = bottom_tm.channel(i / 8 + i % 8);
                for (int q = 0; q < inch * kernel_size; q++)
                {
                    tmpptr[q] = img0[0];
                    img0 += out_size;
                }
            }
        }

        // sgemm(int M, int N, int L, float* A, float* B, float* C)
        {
            int N = outw * outh;
            int L = kernel_w * kernel_w * inch;
            const float* bias_data = bias;

            int pp = 0;
            for (; pp + 7 < outch; pp += 8)
            {
                float* output0 = top_blob.channel(pp + 0);
                float* output1 = top_blob.channel(pp + 1);
                float* output2 = top_blob.channel(pp + 2);
                float* output3 = top_blob.channel(pp + 3);
                float* output4 = top_blob.channel(pp + 4);
                float* output5 = top_blob.channel(pp + 5);
                float* output6 = top_blob.channel(pp + 6);
                float* output7 = top_blob.channel(pp + 7);

                const float zeros[8] = { 0.f };
                const float* biasptr = bias_data ? bias_data + pp : zeros;
                int j = 0;
                for (; j +7 < N; j += 8)
                {
                    const float* vb = bottom_tm.channel(j / 8);
                    const float* va = kernel.channel(pp / 8);
                    float sum0[8] = { 0.f };
                    float sum1[8] = { 0.f };
                    float sum2[8] = { 0.f };
                    float sum3[8] = { 0.f };
                    float sum4[8] = { 0.f };
                    float sum5[8] = { 0.f };
                    float sum6[8] = { 0.f };
                    float sum7[8] = { 0.f };

                    int k = 0;
                    for (; k + 7 < L; k += 8)
                    {
                        for (int n = 0; n < 8; n++)
                        {
                            sum0[n] += va[0] * vb[n];
                            sum1[n] += va[1] * vb[n];
                            sum2[n] += va[2] * vb[n];
                            sum3[n] += va[3] * vb[n];
                            sum4[n] += va[4] * vb[n];
                            sum5[n] += va[5] * vb[n];
                            sum6[n] += va[6] * vb[n];
                            sum7[n] += va[7] * vb[n];
                            va += 8;

                            sum0[n] += va[0] * vb[n + 8];
                            sum1[n] += va[1] * vb[n + 8];
                            sum2[n] += va[2] * vb[n + 8];
                            sum3[n] += va[3] * vb[n + 8];
                            sum4[n] += va[4] * vb[n + 8];
                            sum5[n] += va[5] * vb[n + 8];
                            sum6[n] += va[6] * vb[n + 8];
                            sum7[n] += va[7] * vb[n + 8];
                            va += 8;

                            sum0[n] += va[0] * vb[n + 16];
                            sum1[n] += va[1] * vb[n + 16];
                            sum2[n] += va[2] * vb[n + 16];
                            sum3[n] += va[3] * vb[n + 16];
                            sum4[n] += va[4] * vb[n + 16];
                            sum5[n] += va[5] * vb[n + 16];
                            sum6[n] += va[6] * vb[n + 16];
                            sum7[n] += va[7] * vb[n + 16];
                            va += 8;

                            sum0[n] += va[0] * vb[n + 24];
                            sum1[n] += va[1] * vb[n + 24];
                            sum2[n] += va[2] * vb[n + 24];
                            sum3[n] += va[3] * vb[n + 24];
                            sum4[n] += va[4] * vb[n + 24];
                            sum5[n] += va[5] * vb[n + 24];
                            sum6[n] += va[6] * vb[n + 24];
                            sum7[n] += va[7] * vb[n + 24];
                            va += 8;

                            sum0[n] += va[0] * vb[n + 32];
                            sum1[n] += va[1] * vb[n + 32];
                            sum2[n] += va[2] * vb[n + 32];
                            sum3[n] += va[3] * vb[n + 32];
                            sum4[n] += va[4] * vb[n + 32];
                            sum5[n] += va[5] * vb[n + 32];
                            sum6[n] += va[6] * vb[n + 32];
                            sum7[n] += va[7] * vb[n + 32];
                            va += 8;

                            sum0[n] += va[0] * vb[n + 40];
                            sum1[n] += va[1] * vb[n + 40];
                            sum2[n] += va[2] * vb[n + 40];
                            sum3[n] += va[3] * vb[n + 40];
                            sum4[n] += va[4] * vb[n + 40];
                            sum5[n] += va[5] * vb[n + 40];
                            sum6[n] += va[6] * vb[n + 40];
                            sum7[n] += va[7] * vb[n + 40];
                            va += 8;

                            sum0[n] += va[0] * vb[n + 48];
                            sum1[n] += va[1] * vb[n + 48];
                            sum2[n] += va[2] * vb[n + 48];
                            sum3[n] += va[3] * vb[n + 48];
                            sum4[n] += va[4] * vb[n + 48];
                            sum5[n] += va[5] * vb[n + 48];
                            sum6[n] += va[6] * vb[n + 48];
                            sum7[n] += va[7] * vb[n + 48];
                            va += 8;

                            sum0[n] += va[0] * vb[n + 56];
                            sum1[n] += va[1] * vb[n + 56];
                            sum2[n] += va[2] * vb[n + 56];
                            sum3[n] += va[3] * vb[n + 56];
                            sum4[n] += va[4] * vb[n + 56];
                            sum5[n] += va[5] * vb[n + 56];
                            sum6[n] += va[6] * vb[n + 56];
                            sum7[n] += va[7] * vb[n + 56];
                            va -= 56;
                        }
                        va += 64;
                        vb += 64;
                    }

                    for (; k < L; k++)
                    {
                        for (int n = 0; n < 8; n++)
                        {
                            sum0[n] += va[0] * vb[n];
                            sum1[n] += va[1] * vb[n];
                            sum2[n] += va[2] * vb[n];
                            sum3[n] += va[3] * vb[n];
                            sum4[n] += va[4] * vb[n];
                            sum5[n] += va[5] * vb[n];
                            sum6[n] += va[6] * vb[n];
                            sum7[n] += va[7] * vb[n];
                        }

                        va += 8;
                        vb += 8;
                    }

                    for (int n = 0; n < 8; n++)
                    {
                        output0[n] = sum0[n] + biasptr[0];
                        output1[n] = sum1[n] + biasptr[1];
                        output2[n] = sum2[n] + biasptr[2];
                        output3[n] = sum3[n] + biasptr[3];
                        output4[n] = sum4[n] + biasptr[4];
                        output5[n] = sum5[n] + biasptr[5];
                        output6[n] = sum6[n] + biasptr[6];
                        output7[n] = sum7[n] + biasptr[7];
                    }

                    output0 += 8;
                    output1 += 8;
                    output2 += 8;
                    output3 += 8;
                    output4 += 8;
                    output5 += 8;
                    output6 += 8;
                    output7 += 8;
                }

                for (; j < N; j++)
                {
                    const float* vb = bottom_tm.channel(j / 8 + j % 8);
                    const float* va = kernel.channel(pp / 8);
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;
                    float sum4 = 0.f;
                    float sum5 = 0.f;
                    float sum6 = 0.f;
                    float sum7 = 0.f;

                    for (int k = 0; k < L; k++)
                    {
                        sum0 += va[0] * vb[0];
                        sum1 += va[1] * vb[0];
                        sum2 += va[2] * vb[0];
                        sum3 += va[3] * vb[0];
                        sum4 += va[4] * vb[0];
                        sum5 += va[5] * vb[0];
                        sum6 += va[6] * vb[0];
                        sum7 += va[7] * vb[0];
                        va += 8;
                        vb += 1;
                    }

                    output0[0] = sum0 + biasptr[0];
                    output1[0] = sum1 + biasptr[1];
                    output2[0] = sum2 + biasptr[2];
                    output3[0] = sum3 + biasptr[3];
                    output4[0] = sum4 + biasptr[4];
                    output5[0] = sum5 + biasptr[5];
                    output6[0] = sum6 + biasptr[6];
                    output7[0] = sum7 + biasptr[7];

                    output0++;
                    output1++;
                    output2++;
                    output3++;
                    output4++;
                    output5++;
                    output6++;
                    output7++;
                }
            }

            for (; pp < outch; pp++)
            {
                float* output = top_blob.channel(pp);
                const float bias0 = bias_data ? bias_data[pp] : 0.f;

                int j = 0;
                for (; j + 7 < N; j += 8)
                {
                    const float* vb = bottom_tm.channel(j / 8);
                    const float* va = kernel.channel(pp / 8 + pp % 8);
                    float sum[8] = { 0.f };

                    int k = 0;
                    for (; k + 7 < L; k+= 8)
                    {
                        for (int n = 0; n < 8; n++)
                        {
                            sum[n] += va[0] * vb[n];
                            sum[n] += va[1] * vb[n + 8];
                            sum[n] += va[2] * vb[n + 16];
                            sum[n] += va[3] * vb[n + 24];
                            sum[n] += va[4] * vb[n + 32];
                            sum[n] += va[5] * vb[n + 40];
                            sum[n] += va[6] * vb[n + 48];
                            sum[n] += va[7] * vb[n + 56];
                        }
                        va += 8;
                        vb += 64;
                    }

                    for (; k < L; k++)
                    {
                        for (int n = 0; n < 8; n++)
                        {
                            sum[n] += va[0] * vb[n];
                        }
                        va += 1;
                        vb += 8;
                    }

                    for (int n = 0; n < 8; n++)
                    {
                        output[n] = sum[n] + bias0;
                    }
                    output += 8;
                }

                for (; j < N; j++)
                {
                    const float* vb = bottom_tm.channel(j / 8 + j % 8);
                    const float* va = kernel.channel(pp / 8 + pp % 8);
                    float sum0 = 0.f;

                    for (int k = 0; k < L; k++)
                    {
                        sum0 += va[k] * vb[k];
                    }
                    output[0] = sum0;
                    output++;
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

        return 0;
    }

    int Convolution::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (bottom_blobs.size() != top_blobs.size())
        {
            ConsoleELog << "Convolution ERROR: bottom_blobs size(" << bottom_blobs.size() << ") != top_blobs size(" << top_blobs.size() << ")";
            return -1;
        }

        int ret = 0;
        if (opt.use_sgemm_convolution)
        {
            ret = forward_sgemm_sse(bottom_blobs, top_blobs);
        }
        else
        {
            ret = forward_c(bottom_blobs, top_blobs);
        }

        return 0;
    }

    void Convolution::make_padding(const aMat& bottom_blob, aMat& bottom_blob_bordered) const
    {
        int w = bottom_blob.m_w;
        int h = bottom_blob.m_h;
        const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

        bottom_blob_bordered = bottom_blob;
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, 0, pad_value);
        }
        else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
        {
            // tensorflow padding=SAME or onnx padding=SAME_UPPER
            int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
            int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
            if (wpad > 0 || hpad > 0)
            {
                copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, 0, pad_value);
            }
        }
        else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
            int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
            if (wpad > 0 || hpad > 0)
            {
                copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, 0, pad_value);
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

    int Convolution::forward_c(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs) const
    {
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            aMat bottom_blob_bordered;
            make_padding(bottom_blob, bottom_blob_bordered);

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

    int Convolution::forward_sgemm(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs) const
    {
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            aMat bottom_blob_bordered;
            make_padding(bottom_blob, bottom_blob_bordered);

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

    int Convolution::forward_sgemm_sse(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs) const
    {
        conv_im2col_sgemm_transform_kernel_sse_8x8(weight_data, weight_sgemm_data, kernel_w, kernel_h, bottom_blobs[0].m_c, num_output);

        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            aMat bottom_blob_bordered;
            make_padding(bottom_blob, bottom_blob_bordered);

            const int w = bottom_blob_bordered.m_w;
            const int h = bottom_blob_bordered.m_h;
            const int inch = bottom_blob_bordered.m_c;
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
            const int outw = (w - kernel_extent_w) / stride_w + 1;
            const int outh = (h - kernel_extent_h) / stride_h + 1;

            aMat bottom_im2col;
            conv_im2col(bottom_blob_bordered, bottom_im2col, kernel_w, kernel_h, stride_w, stride_h, outw, outh);

            top_blob.create(outw, outh, num_output, bottom_blob_bordered.m_elemsize, bottom_blob_bordered.m_allocator);
            im2col_sgemm_sse(bottom_im2col, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, inch, num_output, outw, outh);
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Convolution, Convolution);
}