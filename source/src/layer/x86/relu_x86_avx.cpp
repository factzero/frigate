#if __AVX__
#include <immintrin.h>
#include "layer/x86/relu_x86_avx.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    ReluX86avx::ReluX86avx(const LayerParam& layer_param)
        : Relu(layer_param)
    {}

    int ReluX86avx::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (bottom_blobs.size() != top_blobs.size())
        {
            ConsoleELog << "ReluX86avx ERROR: bottom_blobs size(" << bottom_blobs.size() << ") != top_blobs size(" << top_blobs.size() << ")";
            return -1;
        }

        __m256 _zeros = _mm256_setzero_ps();
        __m256 _slope = _mm256_set1_ps(slope);
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            int w = bottom_blob.m_w;
            int h = bottom_blob.m_h;
            int channels = bottom_blob.m_c;
            int out_size = w * h;

            top_blob.create(w, h, channels, bottom_blob.m_elemsize, bottom_blob.m_allocator);
            for (int q = 0; q < channels; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                int j = 0;
                for (; j + 7 < out_size; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(inptr);
                    __m256 _pos = _mm256_max_ps(_zeros, _p);
                    __m256 _neg = _mm256_min_ps(_zeros, _p);
                    _p = _mm256_add_ps(_pos, _mm256_mul_ps(_slope, _neg));
                    _mm256_storeu_ps(outptr, _p);
                    inptr += 8;
                    outptr += 8;
                }

                for (; j < out_size; j++)
                {
                    if (inptr[0] < 0)
                    {
                        outptr[0] = inptr[0] * slope;
                    }
                    else
                    {
                        outptr[0] = inptr[0];
                    }
                    inptr++;
                    outptr++;
                }
            }
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(ReLU_x86_avx, ReluX86avx);
}

#endif