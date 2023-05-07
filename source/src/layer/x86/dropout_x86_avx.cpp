#if __AVX__
#include <immintrin.h>
#include "layer/x86/dropout_x86_avx.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    DropoutX86avx::DropoutX86avx(const LayerParam& layer_param)
        : Dropout(layer_param)
    {}

    int DropoutX86avx::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (bottom_blobs.size() != top_blobs.size())
        {
            ConsoleELog << "DropoutX86avx ERROR: bottom_blobs size(" << bottom_blobs.size() << ") != top_blobs size(" << top_blobs.size() << ")";
            return -1;
        }

        __m256 _scale = _mm256_set1_ps(scale);
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            int inw = bottom_blob.m_w;
            int inh = bottom_blob.m_h;
            int inch = bottom_blob.m_c;
            int out_size = inw * inh;

            top_blob.create(inw, inh, inch, bottom_blob.m_elemsize, bottom_blob.m_allocator);
            for (int q = 0; q < inch; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                int j = 0;
                for (; j + 7 < out_size; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(inptr);
                    _p = _mm256_mul_ps(_p, _scale);
                    _mm256_storeu_ps(outptr, _p);
                    inptr += 8;
                    outptr += 8;
                }

                for (; j < out_size; j++)
                {
                    outptr[0] = inptr[0] * scale;
                    inptr++;
                    outptr++;
                }
            }
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Dropout_x86_avx, DropoutX86avx);
}

#endif