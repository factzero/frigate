#include "layer/layer_concat.h"
#include "layer/layer_factory.h"

namespace ACNN
{
    Concat::Concat(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Concat::load_param(const ParamDict& pd)
    {
        axis = pd.get(0, 0);

        return 0;
    }

    int Concat::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        int w = bottom_blobs[0].m_w;
        int h = bottom_blobs[0].m_h;

        int top_channels = 0;
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            top_channels += bottom_blob.m_c;
        }

        aMat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels, bottom_blobs[0].m_elemsize, bottom_blobs[0].m_allocator);
        int q = 0;
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            int size = bottom_blob.total();
            const float* inptr = bottom_blob;
            float* outptr = top_blob.channel(q);
            memcpy((void*)outptr, (const void*)inptr, size * sizeof(float));
            q += bottom_blob.m_c;
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Concat, Concat);
}