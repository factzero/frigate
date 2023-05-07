#pragma once
#if __AVX__

namespace ACNN
{
    float AVXMax(const float* inptr, int size);
}
#endif