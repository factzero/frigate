#include "common.h"
#include "layer/layer.h"
#include "layer/layer_factory.h"


namespace ACNN
{
    void copy_make_border(const aMat& src, aMat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt)
    {
        LayerParam param;
        param.layer_type = "Padding";
        param.layer_name = "copy_make_border";
        std::shared_ptr<Layer> padding = LayerRegistry::CreateLayer(param);

        ParamDict pd;
        pd.set(0, top);
        pd.set(1, bottom);
        pd.set(2, left);
        pd.set(3, right);
        pd.set(4, type);
        pd.set(5, v);
        padding->load_param(pd);

        padding->forward(src, dst, opt);
    }
}