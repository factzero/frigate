#include "layer/layer_relu.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    Relu::Relu(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Relu::load_param(const ParamDict& pd)
    {
        slope = pd.get(0, 0.f);

        return 0;
    }

    int Relu::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (bottom_blobs.size() != top_blobs.size())
        {
            ConsoleELog << "Relu ERROR: bottom_blobs size(" << bottom_blobs.size() << ") != top_blobs size(" << top_blobs.size() << ")";
            return -1;
        }

        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            int w = bottom_blob.m_w;
            int h = bottom_blob.m_h;
            int channels = bottom_blob.m_c;

            top_blob.create(w, h, channels, bottom_blob.m_elemsize, bottom_blob.m_allocator);
            for (int q = 0; q < channels; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                for (int j = 0; j < w * h; j++)
                {
                    if (inptr[j] < 0)
                    {
                        outptr[j] = inptr[j] * slope;
                    }
                    else
                    {
                        outptr[j] = inptr[j];
                    }
                }
            }
        }

        return 0;
    }

    int Relu::forward_inplace(aMat& bottom_top_blob, const Option& opt) const
    {
        const aMat& blob = bottom_top_blob;
        int size = blob.m_w * blob.m_h;
        int channels = blob.m_c;
        for (int q = 0; q < channels; q++)
        {
            float* ioptr = blob.channel(q);
            for (int j = 0; j < size; j++)
            {
                if (ioptr[j] < 0)
                {
                    ioptr[j] = ioptr[j] * slope;
                }
            }
        }

        return 0;
    }

    int Relu::forward_inplace(std::vector<aMat>& bottom_top_blobs, const Option& opt) const
    {
        for (size_t i = 0; i < bottom_top_blobs.size(); i++)
        {
            const aMat& blob = bottom_top_blobs[i];
            int size = blob.m_w * blob.m_h;
            int channels = blob.m_c;
            for (int q = 0; q < channels; q++)
            {
                float* ioptr = blob.channel(q);
                for (int j = 0; j < size; j++)
                {
                    if (ioptr[j] < 0)
                    {
                        ioptr[j] = ioptr[j] * slope;
                    }
                }
            }
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(ReLU, Relu);
}