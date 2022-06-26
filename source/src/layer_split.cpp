#include "layer_split.h"
#include "layer_factory.h"


namespace ACNN
{
    Split::Split(const LayerParam& layer_param)
        : Layer(layer_param)
    {}

    int Split::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        const aMat& bottom_blob = bottom_blobs[0];
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            top_blobs[i] = bottom_blob;
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Split, Split);
}