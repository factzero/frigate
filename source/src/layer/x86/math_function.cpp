#if __AVX__
#include <immintrin.h>
#include <algorithm>
#include "layer/x86/math_function.h"


namespace ACNN
{
    float AVXMax(const float* inptr, int size)
    {
        float max = -FLT_MAX;
        __m256 _max = _mm256_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 7 < size; j += 8)
        {
            __m256 _p = _mm256_loadu_ps(inptr + j);
            _max = _mm256_max_ps(_max, _p);
        }

        float afsum[8];
        _mm256_storeu_ps(afsum, _max);
        max = std::max(max, afsum[0]);
        max = std::max(max, afsum[1]);
        max = std::max(max, afsum[2]);
        max = std::max(max, afsum[3]);
        max = std::max(max, afsum[4]);
        max = std::max(max, afsum[5]);
        max = std::max(max, afsum[6]);
        max = std::max(max, afsum[7]);

        for (; j < size; j++)
        {
            max = std::max(max, inptr[j]);
        }

        return max;
    }
}

#endif