#include "layer/layer_quantize.h"
#include "layer/layer_factory.h"
#include "quantize_tools.h"


namespace ACNN
{
    Quantize::Quantize(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Quantize::load_param(const ParamDict& pd)
    {
        scale_data_size = pd.get(0, 1);

        return 0;
    }

    int Quantize::load_model(const ModelBin& mb)
    {
        scale_data = mb.load(scale_data_size, 1);
        if (scale_data.empty())
            return -100;

        return 0;
    }

    int Quantize::forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const
    {
        int w = bottom_blob.m_w;
        int h = bottom_blob.m_h;
        int channels = bottom_blob.m_c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, bottom_blob.m_allocator);
        if (top_blob.empty())
            return -100;

        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            for (int i = 0; i < size; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale);
            }
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Quantize, Quantize);
}