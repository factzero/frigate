#include <algorithm>
#include "layer/layer_binaryop.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    struct binary_op_add
    {
        float operator()(const float& x, const float& y) const
        {
            return x + y;
        }
    };

    struct binary_op_sub
    {
        float operator()(const float& x, const float& y) const
        {
            return x - y;
        }
    };

    struct binary_op_mul
    {
        float operator()(const float& x, const float& y) const
        {
            return x * y;
        }
    };

    struct binary_op_div
    {
        float operator()(const float& x, const float& y) const
        {
            return x / y;
        }
    };

    struct binary_op_max
    {
        float operator()(const float& x, const float& y) const
        {
            return std::max(x, y);
        }
    };

    struct binary_op_min
    {
        float operator()(const float& x, const float& y) const
        {
            return std::min(x, y);
        }
    };

    struct binary_op_pow
    {
        float operator()(const float& x, const float& y) const
        {
            return (float)pow(x, y);
        }
    };

    struct binary_op_rsub
    {
        float operator()(const float& x, const float& y) const
        {
            return y - x;
        }
    };

    struct binary_op_rdiv
    {
        float operator()(const float& x, const float& y) const
        {
            return y / x;
        }
    };

    template<typename Op>
    static int binary_op(const aMat& a, const aMat& b, aMat& c, const Option& opt)
    {
        Op op;

        int wa = a.m_w;
        int ha = a.m_h;
        int ca = a.m_c;
        int sizea = wa * ha;

        int wb = b.m_w;
        int hb = b.m_h;
        int cb = b.m_c;

        if (a.m_dims == 3)
        {
            if (b.m_dims == 3)
            {
                if (wa == wb && ha == hb && ca == cb)
                {
                    c.create(wa, ha, ca, a.m_elemsize, a.m_allocator);
                    if (c.empty())
                    {
                        return -100;
                    }

                    for (int q = 0; q < ca; q++)
                    {
                        const float *ptra = a.channel(q);
                        const float* ptrb = b.channel(q);
                        float* outptr = c.channel(q);

                        for (int i = 0; i < sizea; i++)
                        {
                            outptr[i] = op(ptra[i], ptrb[i]);
                        }
                    }

                    return 0;
                }
                else
                {
                    ConsoleELog << "BinaryOp not suppport";
                    return -100;
                }
            }
            else
            {
                ConsoleELog << "BinaryOp not suppport";
                return -100;
            }
        }
        else
        {
            ConsoleELog << "BinaryOp not suppport";
            return -100;
        }

        return 0;
    }

    BinaryOp::BinaryOp(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int BinaryOp::load_param(const ParamDict& pd)
    {
        op_type = pd.get(0, 0);
        with_scalar = pd.get(1, 0);
        b = pd.get(2, 0.f);

        return 0;
    }

    int BinaryOp::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        const aMat& bottom_blob = bottom_blobs[0];
        const aMat& bottom_blob1 = bottom_blobs[1];

        aMat& top_blob = top_blobs[0];

        if (op_type == Operation_ADD)
            return binary_op<binary_op_add>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op<binary_op_sub>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op<binary_op_mul>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op<binary_op_div>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op<binary_op_max>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op<binary_op_min>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op<binary_op_pow>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op<binary_op_sub>(bottom_blob1, bottom_blob, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op<binary_op_div>(bottom_blob1, bottom_blob, top_blob, opt);

        return 0;
    }

    REGISTER_LAYER_CLASS(BinaryOp, BinaryOp);
}