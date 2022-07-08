#if __AVX__
#include <immintrin.h>
#include "layer/x86/pooling_x86_avx.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    PoolingX86avx::PoolingX86avx(const LayerParam& layer_param)
        : Pooling(layer_param)
    {}

    int PoolingX86avx::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
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
                        __m256 _max = _mm256_set1_ps(ptr[0]);
                        int i = 0;
                        for (; i + 7 < size; i += 8)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            _max = _mm256_max_ps(_max, _p);
                            ptr += 8;
                        }
                        for (; i < size; i++)
                        {
                            max = std::max(max, ptr[0]);
                            ptr++;
                        }
                        float max_a[8];
                        _mm256_storeu_ps(max_a, _max);
                        max = std::max(max, max_a[0]);
                        max = std::max(max, max_a[1]);
                        max = std::max(max, max_a[2]);
                        max = std::max(max, max_a[3]);
                        max = std::max(max, max_a[4]);
                        max = std::max(max, max_a[5]);
                        max = std::max(max, max_a[6]);
                        max = std::max(max, max_a[7]);

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
                make_padding(bottom_blob, bottom_blob_bordered);
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
                    const int maxk = kernel_w * kernel_h;
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

    REGISTER_LAYER_CLASS(Pooling_x86_avx, PoolingX86avx);
}

#endif