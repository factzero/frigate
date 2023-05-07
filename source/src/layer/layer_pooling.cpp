#include "layer/layer_pooling.h"
#include "layer/layer_factory.h"
#include "common.h"
#include "logger.h"


namespace ACNN
{
    Pooling::Pooling(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Pooling::load_param(const ParamDict& pd)
    {
        pooling_type = pd.get(0, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        stride_w = pd.get(2, 1);
        stride_h = pd.get(12, stride_w);
        pad_left = pd.get(3, 0);
        pad_right = pd.get(14, pad_left);
        pad_top = pd.get(13, pad_left);
        pad_bottom = pd.get(15, pad_top);
        global_pooling = pd.get(4, 0);
        pad_mode = pd.get(5, 0);
        avgpool_count_include_pad = pd.get(6, 0);
        adaptive_pooling = pd.get(7, 0);
        out_w = pd.get(8, 0);
        out_h = pd.get(18, out_w);

        return 0;
    }

    int Pooling::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (bottom_blobs.size() != top_blobs.size())
        {
            ConsoleELog << "Pooling ERROR: bottom_blobs size(" << bottom_blobs.size() << ") != top_blobs size(" << top_blobs.size() << ")";
            return -1;
        }

        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            int w = bottom_blob.m_w;
            int h = bottom_blob.m_h;
            int channels = bottom_blob.m_c;

            if (global_pooling)
            {
                top_blob.create(channels, bottom_blob.m_elemsize, bottom_blob.m_allocator);
                if (top_blob.empty())
                {
                    return -100;
                }

                int size = w * h;

                if (pooling_type == 0)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_blob.channel(q);
                        float max = ptr[0];
                        for (int i = 0; i < size; i++)
                        {
                            max = std::max(max, ptr[i]);
                        }

                        top_blob[q] = max;
                    }
                }
                else if (pooling_type == 1)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_blob.channel(q);
                        float sum = 0.f;
                        for (int i = 0; i < size; i++)
                        {
                            sum += ptr[i];
                        }

                        top_blob[q] = sum / size;
                    }
                }

                return 0;
            }
            else if (adaptive_pooling)
            {
                ConsoleELog << "Pooling Type is not supported: " << adaptive_pooling << " , " << pooling_type;
                return -1;
            }
            else
            {
                aMat bottom_blob_bordered;
                make_padding(bottom_blob, bottom_blob_bordered, opt);
                if (bottom_blob_bordered.empty())
                {
                    return -100;
                }

                w = bottom_blob_bordered.m_w;
                h = bottom_blob_bordered.m_h;

                int outw = (w - kernel_w) / stride_w + 1;
                int outh = (h - kernel_h) / stride_h + 1;

                top_blob.create(outw, outh, channels, bottom_blob.m_elemsize, bottom_blob.m_allocator);
                if (top_blob.empty())
                {
                    return -100;
                }

                const int maxk = kernel_w * kernel_h;
                // kernel offsets
                std::vector<int> _space_ofs(maxk);
                int* space_ofs = &_space_ofs[0];
                {
                    int p1 = 0;
                    int p2 = 0;
                    int gap = w - kernel_w;
                    for (int i = 0; i < kernel_h; i++)
                    {
                        for (int j = 0; j < kernel_w; j++)
                        {
                            space_ofs[p1] = p2;
                            p1++;
                            p2++;
                        }
                        p2 += gap;
                    }
                }

                if (pooling_type == 0)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const float* inptr = bottom_blob_bordered.channel(q);
                        float* outptr = top_blob.channel(q);

                        for (int y = 0; y < outh; y++)
                        {
                            for (int x = 0; x < outw; x++)
                            {
                                const float* sptr = inptr + y * stride_h * w + x * stride_w;
                                float max = sptr[0];
                                for (int m = 0; m < kernel_h; m++)
                                {
                                    for (int n = 0; n < kernel_w; n++)
                                    {
                                        float val = sptr[n];
                                        max = std::max(max, val);
                                    }
                                    sptr += w;
                                }
                                outptr[x] = max;
                            }
                            outptr += outw;
                        }
                    }
                }
                else
                {
                    ConsoleELog << "Pooling Type is not supported: " << adaptive_pooling << " , " << pooling_type;
                    return -1;                    
                }
            }
            
        }

        return 0;
    }

    void Pooling::make_padding(const aMat& bottom_blob, aMat& bottom_blob_bordered, const Option& opt) const
    {
        int w = bottom_blob.m_w;
        int h = bottom_blob.m_h;

        bottom_blob_bordered = bottom_blob;

        float pad_value = 0.f;
        if (pooling_type == 0)
        {
            pad_value = bottom_blob.m_elemsize == 1 ? -128.f : -FLT_MAX;
        }

        int wtailpad = 0;
        int htailpad = 0;

        if (pad_mode == 0) // full padding
        {
            int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
            int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

            if (wtail != 0)
                wtailpad = stride_w - wtail;
            if (htail != 0)
                htailpad = stride_h - htail;

            copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, 0, pad_value, opt);
        }
        else if (pad_mode == 1) // valid padding
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, 0, pad_value, opt);
        }
        else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
        {
            int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
            int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
            if (wpad > 0 || hpad > 0)
            {
                copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, 0, pad_value, opt);
            }
        }
        else if (pad_mode == 3) // onnx padding=SAME_LOWER
        {
            int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
            int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
            if (wpad > 0 || hpad > 0)
            {
                copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, 0, pad_value, opt);
            }
        }

        return;
    }

    REGISTER_LAYER_CLASS(Pooling, Pooling);
}