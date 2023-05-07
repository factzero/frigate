#include "layer/layer_dequantize.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    Dequantize::Dequantize(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Dequantize::load_param(const ParamDict& pd)
    {
        scale_data_size = pd.get(0, 1);
        bias_data_size = pd.get(1, 0);

        return 0;
    }

    int Dequantize::load_model(const ModelBin& mb)
    {
        scale_data = mb.load(scale_data_size, 1);
        if (scale_data.empty())
        {
            return -100;
        }

        if (bias_data_size)
        {
            bias_data = mb.load(bias_data_size, 1);
            if (bias_data.empty())
            {
                return -100;
            }
        }

        return 0;
    }

    int Dequantize::forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const
    {
        int dims = bottom_blob.m_dims;

        if (1 == dims)
        {
            ConsoleELog << "Dequantize::forward dims=1 not support";
            return -100;
        }

        if (2 == dims)
        {
            ConsoleELog << "Dequantize::forward dims=1 not support";
            return -100;
        }

        if (3 == dims)
        {
            int w = bottom_blob.m_w;
            int h = bottom_blob.m_h;
            int channels = bottom_blob.m_c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)4u, bottom_blob.m_allocator);
            if (top_blob.empty())
            {
                return -100;
            }

            if (bias_data_size == 0)
            {
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = intptr[i] * scale;
                    }
                }
            }
            else
            {
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                    const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = intptr[i] * scale + bias;
                    }
                }
            }
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Dequantize, Dequantize);
}