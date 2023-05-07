#include "quantize_tools.h"
#include "layer/layer.h"
#include "layer/layer_factory.h"


namespace ACNN
{
    void quantize_to_int8(const aMat& src, aMat& dst, const aMat& scale_data, const Option& opt)
    {
        LayerParam param;
        param.layer_type = "Quantize";
        param.layer_name = "quantize_to_int8";
        std::shared_ptr<Layer> quantize = LayerRegistry::CreateLayer(param);

        ParamDict pd;
        pd.set(0, scale_data.m_w);
        quantize->load_param(pd);

        std::vector<aMat> weights;
        weights.push_back(scale_data);
        quantize->load_model(ModelBinFromMatArray(weights));

        quantize->forward(src, dst, opt);
    }

    void requantize_int32_to_int8(const aMat& src, aMat& dst, const aMat& scale_in_data,
        const aMat& scale_out_data, const aMat& bias_data, int activation_type, const aMat& activation_params, const Option& opt)
    {
        LayerParam param;
        param.layer_type = "Requantize";
        param.layer_name = "requantize_int32_to_int8";
        std::shared_ptr<Layer> requantize = LayerRegistry::CreateLayer(param);

        ParamDict pd;
        pd.set(0, scale_in_data.m_w);
        pd.set(1, scale_out_data.m_w);
        pd.set(2, bias_data.m_w);
        pd.set(3, activation_type);
        pd.set(4, activation_params);
        requantize->load_param(pd);

        std::vector<aMat> weights;
        weights.push_back(scale_in_data);
        weights.push_back(scale_out_data);
        weights.push_back(bias_data);
        requantize->load_model(ModelBinFromMatArray(weights));

        requantize->forward(src, dst, opt);
    }

    void dequantize_int32_to_fp32(const aMat& src, aMat& dst, const aMat& scale_data, const aMat& bias_data, const Option& opt)
    {
        LayerParam param;
        param.layer_type = "Dequantize";
        param.layer_name = "dequantize_int32_to_fp32";
        std::shared_ptr<Layer> dequantize = LayerRegistry::CreateLayer(param);

        ParamDict pd;
        pd.set(0, scale_data.m_w);
        pd.set(1, bias_data.m_w);
        dequantize->load_param(pd);

        std::vector<aMat> weights;
        weights.push_back(scale_data);
        weights.push_back(bias_data);
        dequantize->load_model(ModelBinFromMatArray(weights));

        dequantize->forward(src, dst, opt);
    }
}