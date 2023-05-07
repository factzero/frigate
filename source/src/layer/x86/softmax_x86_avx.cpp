#if __AVX__
#include "layer/x86/softmax_x86_avx.h"
#include "layer/x86/math_function.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    SoftmaxX86avx::SoftmaxX86avx(const LayerParam& layer_param)
        : Softmax(layer_param)
    {}

    int SoftmaxX86avx::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (bottom_blobs.size() != top_blobs.size())
        {
            ConsoleELog << "SoftmaxX86avx ERROR: bottom_blobs size(" << bottom_blobs.size() << ") != top_blobs size(" << top_blobs.size() << ")";
            return -1;
        }

        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            int dims = bottom_blob.m_dims;

            if (1 == dims)
            {
                int w = bottom_blob.m_w;
                const float* ptr = bottom_blob;
                float max = AVXMax(ptr, w);

                top_blob.create(w, bottom_blob.m_elemsize, bottom_blob.m_allocator);
                float* outptr = top_blob;
                float sum = 0.f;
                for (int j = 0; j < w; j++)
                {
                    outptr[j] = static_cast<float>(exp(ptr[j] - max));
                    sum += outptr[j];
                }

                for (int j = 0; j < w; j++)
                {
                    outptr[j] /= sum;
                }
            }
            else
            {
                ConsoleELog << "SoftmaxX86avx Type is not support";
            }
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Softmax_x86_avx, SoftmaxX86avx);
}
#endif