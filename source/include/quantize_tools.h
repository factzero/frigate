#pragma once
#include "amat.h"
#include "option.h"


namespace ACNN
{
    void quantize_to_int8(const aMat& src, aMat& dst, const aMat& scale_data, const Option& opt);

    static inline signed char float2int8(float v)
    {
        int int32 = static_cast<int>(round(v));
        if (int32 > 127) return 127;
        if (int32 < -127) return -127;
        return (signed char)int32;
    }

    void requantize_int32_to_int8(const aMat& src, aMat& dst, const aMat& scale_in_data,
        const aMat& scale_out_data, const aMat& bias_data, int activation_type, const aMat& activation_params, const Option& opt);
    void dequantize_int32_to_fp32(const aMat& src, aMat& dst, const aMat& scale_data, const aMat& bias_data, const Option& opt);
}