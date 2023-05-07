#include "layer/layer_padding.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    template<typename T>
    static void copy_make_border_image(const aMat& src, aMat& dst, int top, int left, int type, T v)
    {
        int w = dst.m_w;
        int h = dst.m_h;

        const T* ptr = src;
        T* outptr = dst;

        if (type == 0)
        {
            int y = 0;
            // fill top
            for (; y < top; y++)
            {
                int x = 0;
                for (; x < w; x++)
                {
                    outptr[x] = v;
                }
                outptr += w;
            }
            // fill center
            for (; y < (top + src.m_h); y++)
            {
                int x = 0;
                for (; x < left; x++)
                {
                    outptr[x] = v;
                }

                memcpy(outptr + left, ptr, src.m_w * sizeof(T));
                x += src.m_w;

                for (; x < w; x++)
                {
                    outptr[x] = v;
                }
                ptr += src.m_w;
                outptr += w;
            }
            // fill bottom
            for (; y < h; y++)
            {
                int x = 0;
                for (; x < w; x++)
                {
                    outptr[x] = v;
                }
                outptr += w;
            }
        }

        if (type == 1)
        {
            ConsoleELog << "copy_make_border_image type=1 not support";
        }

        if (type == 2)
        {
            ConsoleELog << "copy_make_border_image type=2 not support";
        }
    }

    Padding::Padding(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Padding::load_param(const ParamDict& pd)
    {
        top = pd.get(0, 0);
        bottom = pd.get(1, 0);
        left = pd.get(2, 0);
        right = pd.get(3, 0);
        type = pd.get(4, 0);
        value = pd.get(5, 0.f);
        per_channel_pad_data_size = pd.get(6, 0);
        front = pd.get(7, 0);
        behind = pd.get(8, 0);

        return 0;
    }

    int Padding::load_model(const ModelBin& mb)
    {
        if (per_channel_pad_data_size)
        {
            per_channel_pad_data = mb.load(per_channel_pad_data_size, 1);
        }

        return 0;
    }

    int Padding::forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const
    {
        if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        int w = bottom_blob.m_w;
        int h = bottom_blob.m_h;
        int channels = bottom_blob.m_c;
        int dims = bottom_blob.m_dims;
        size_t elemsize = bottom_blob.m_elemsize;

        int outw = w + left + right;

        if (dims == 1)
        {
            top_blob.create(outw, elemsize, bottom_blob.m_allocator);
            if (top_blob.empty())
                return -100;

            if (elemsize == 1)
            {
                copy_make_border_image<signed char>(bottom_blob, top_blob, 0, left, type, static_cast<signed char>(value));
            }
            if (elemsize == 2)
            {
                ConsoleELog << "Padding::forward dims=1 and elemsize=2 not support";
            }
            if (elemsize == 4)
            {
                copy_make_border_image<float>(bottom_blob, top_blob, 0, left, type, value);
            }

            return 0;
        }

        int outh = h + top + bottom;

        if (dims == 2)
        {
            top_blob.create(outw, outh, elemsize, bottom_blob.m_allocator);
            if (top_blob.empty())
                return -100;

            if (elemsize == 1)
            {
                copy_make_border_image<signed char>(bottom_blob, top_blob, top, left, type, static_cast<signed char>(value));
            }
            if (elemsize == 2)
            {
                ConsoleELog << "Padding::forward dims=2 and elemsize=2 not support";
            }
            if (elemsize == 4)
            {
                copy_make_border_image<float>(bottom_blob, top_blob, top, left, type, value);
            }

            return 0;
        }

        if (dims == 3)
        {
            int outc = channels + front + behind;

            top_blob.create(outw, outh, outc, elemsize, bottom_blob.m_allocator);
            if (top_blob.empty())
                return -100;

            for (int q = 0; q < outc; q++)
            {
                aMat borderm = top_blob.channel(q);

                float pad_value = per_channel_pad_data_size ? per_channel_pad_data[q] : value;

                //Channel padding
                if (((q < front) || (q >= (channels + front))) && type == 0)
                {
                    if (elemsize == 1)
                    {
                        borderm.fill(static_cast<signed char>(pad_value));
                    }
                    if (elemsize == 2)
                    {
                        ConsoleELog << "Padding::forward dims=3 and elemsize=2 not support";
                    }
                    if (elemsize == 4)
                    {
                        borderm.fill(pad_value);
                    }
                }
                else
                {
                    int q_ = q - front;

                    if (type == 1)
                    {
                        q_ = q_ <= 0 ? 0 : q_;
                        q_ = q_ >= channels - 1 ? channels - 1 : q_;
                    }
                    if (type == 2)
                    {
                        q_ = abs(q_);
                        q_ = (channels - 1) - abs(q_ - (channels - 1));
                    }
                    const aMat m = bottom_blob.channel(q_);
                    if (elemsize == 1)
                    {
                        copy_make_border_image<signed char>(m, borderm, top, left, type, static_cast<signed char>(pad_value));
                    }   
                    if (elemsize == 2)
                    {
                        ConsoleELog << "Padding::forward dims=3 and elemsize=2 not support";
                    }
                    if (elemsize == 4)
                        copy_make_border_image<float>(m, borderm, top, left, type, pad_value);
                }
            }

            return 0;
        }

        if (dims == 4)
        {
            ConsoleELog << "Padding::forward dims=4 not support";
            return -100;
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Padding, Padding);
}