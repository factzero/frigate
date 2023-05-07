#include "layer/layer_fused_activation.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    float activation_ss(float v, int activation_type, const aMat& activation_params)
    {
        switch (activation_type)
        {
        case 1:
        {
            v = fmax(v, 0.f);
            break;
        }
        case 2:
        {
            float slope = activation_params[0];
            v = v > 0.f ? v : v * slope;
            break;
        }
        case 3:
        {
            float min = activation_params[0];
            float max = activation_params[1];
            if (v < min)
                v = min;
            if (v > max)
                v = max;
            break;
        }
        case 4:
        {
            v = std::min(v, 88.3762626647949f);
            v = std::max(v, -88.3762626647949f);
            v = 1.f / (1.f + exp(-v));
            break;
        }
        case 5:
        {
            v = v * tanh(log(exp(v) + 1.f));
            break;
        }
        case 6:
        {
            float alpha = activation_params[0];
            float beta = activation_params[1];
            float lower = -beta / alpha;
            float upper = (1.f / alpha) + lower;
            if (v < lower)
                v = 0.f;
            else if (v > upper)
                ;
            else
                v = v * (v * alpha + beta);
            break;
        }
        }

        return v;
    }

    std::shared_ptr<Layer> create_activation_layer(int activation_type, const aMat& activation_params, const Option& opt)
    {
        std::shared_ptr<Layer> activation = nullptr;

        if (activation_type == 1)
        {
            LayerParam param;
            param.layer_type = "ReLU";
            param.layer_name = "create_activation_layer";
            activation = LayerRegistry::CreateLayer(param);

            ParamDict pd;
            activation->load_param(pd);
        }
        else if (activation_type == 2)
        {
            ConsoleELog << "create_activation_layer activation_type=2 not support";
        }
        else if (activation_type == 3)
        {
            ConsoleELog << "create_activation_layer activation_type=3 not support";
        }
        else if (activation_type == 4)
        {
            ConsoleELog << "create_activation_layer activation_type=4 not support";
        }
        else if (activation_type == 5)
        {
            ConsoleELog << "create_activation_layer activation_type=5 not support";
        }
        else if (activation_type == 6)
        {
            ConsoleELog << "create_activation_layer activation_type=6 not support";
        }

        if (activation)
        {
            activation->create_pipeline(opt);
        }

        return activation;
    }
}