#include "layer/layer.h"


namespace ACNN
{
    Layer::Layer(const LayerParam& layer_param)
    {
        m_layer_type = layer_param.layer_type;
        m_layer_name = layer_param.layer_name;

        support_inplace = false;
        support_int8_storage = false;
    }

    int Layer::load_param(const ParamDict& pd)
    {
        return 0;
    }

    int Layer::load_model(const ModelBin& mb)
    {
        return 0;
    }

    int Layer::create_pipeline(const Option& opt)
    {
        return 0;
    }

    int Layer::forward(const aMat& bottom_blob, aMat& top_blob, const Option& opt) const
    {
        return 0;
    }

    int Layer::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        return 0;
    }

    int Layer::forward_inplace(aMat& bottom_top_blob, const Option& opt) const
    {
        return 0;
    }

    int Layer::forward_inplace(std::vector<aMat>& bottom_top_blobs, const Option& opt) const
    {
        return 0;
    }
}