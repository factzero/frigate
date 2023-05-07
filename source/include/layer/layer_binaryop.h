#pragma once
#include "layer/layer.h"


namespace ACNN
{
    class BinaryOp : public Layer
    {
    public:
        BinaryOp(const LayerParam& layer_param);
        virtual ~BinaryOp() {}

        virtual int load_param(const ParamDict& pd) override;
        virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

        enum OperationType
        {
            Operation_ADD = 0,
            Operation_SUB = 1,
            Operation_MUL = 2,
            Operation_DIV = 3,
            Operation_MAX = 4,
            Operation_MIN = 5,
            Operation_POW = 6,
            Operation_RSUB = 7,
            Operation_RDIV = 8
        };

    public:
        // param
        int op_type;
        int with_scalar;
        float b;
    };
}