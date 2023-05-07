#include "layer/layer_flatten.h"
#include "layer/layer_factory.h"


namespace ACNN
{
    Flatten::Flatten(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Flatten::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        const int w = bottom_blobs[0].m_w;
        const int h = bottom_blobs[0].m_h;
        const int channels = bottom_blobs[0].m_c;
        const size_t elemsize = bottom_blobs[0].m_elemsize;
        int size = w * h;

        top_blobs[0].create(size * channels, elemsize, bottom_blobs[0].m_allocator);
        if (top_blobs[0].empty())
        {
            return -100;
        }

        for (int q = 0; q < channels; q++)
        {
            const unsigned char* ptr = bottom_blobs[0].channel(q);
            unsigned char* outptr = (unsigned char*)top_blobs[0] + size * elemsize * q;

            memcpy(outptr, ptr, size * elemsize);
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Flatten, Flatten);
}