#include "layer/layer_requantize.h"
#include "layer/layer_factory.h"
#include "layer/layer_fused_activation.h"
#include "quantize_tools.h"
#include "logger.h"


namespace ACNN
{
    Requantize::Requantize(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Requantize::load_param(const ParamDict& pd)
    {
        scale_in_data_size = pd.get(0, 1);
        scale_out_data_size = pd.get(1, 1);
        bias_data_size = pd.get(2, 0);
        activation_type = pd.get(3, 0);
        activation_params = pd.get(4, aMat());

        return 0;
    }

    int Requantize::load_model(const ModelBin& mb)
    {
        scale_in_data = mb.load(scale_in_data_size, 1);
        if (scale_in_data.empty())
            return -100;

        scale_out_data = mb.load(scale_out_data_size, 1);
        if (scale_out_data.empty())
            return -100;

        if (bias_data_size)
        {
            bias_data = mb.load(bias_data_size, 1);
            if (bias_data.empty())
                return -100;
        }

        return 0;
    }

    int Requantize::forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const
    {
        int dims = bottom_blob.m_dims;

        if (dims == 1)
        {
            ConsoleELog << "Requantize::forward dims=1 not support";
            return -100;
        }

        if (dims == 2)
        {
            ConsoleELog << "Requantize::forward dims=2 not support";
            return -100;
        }

        if (dims == 3)
        {
            int w = bottom_blob.m_w;
            int h = bottom_blob.m_h;
            int channels = bottom_blob.m_c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)1u, bottom_blob.m_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    signed char* ptr = top_blob.channel(q);

                    const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[q];
                    const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[q];

                    for (int i = 0; i < size; i++)
                    {
                        float v = intptr[i] * scale_in;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                }
            }
            else
            {
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    signed char* ptr = top_blob.channel(q);

                    const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[q];
                    const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[q];
                    const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                    for (int i = 0; i < size; i++)
                    {
                        float v = intptr[i] * scale_in + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                }
            }
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Requantize, Requantize);
}