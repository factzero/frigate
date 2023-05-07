#pragma once
#include "layer/layer.h"


namespace ACNN
{
    float activation_ss(float v, int activation_type, const aMat& activation_params);
    std::shared_ptr<Layer> create_activation_layer(int activation_type, const aMat& activation_params, const Option& opt);
}