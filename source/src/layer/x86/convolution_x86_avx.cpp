#if __AVX__
#include <immintrin.h>
#include "layer/x86/convolution_x86_avx.h"
#include "layer/layer_factory.h"
#include "logger.h"


namespace ACNN
{
    void conv_im2col_sgemm_transform_kernel_avx_8x8(const aMat& kernel, aMat& kernel_tm, const int kernel_w, const int kernel_h, const int inch, const int outch)
    {
        const int kernel_size = kernel_w * kernel_h;

        // src = kernel_size-inch-outch
        // dst = 8b-kernel_size-inch-outch/8b
        kernel_tm.create(8 * kernel_size, inch, outch / 8 + outch % 8, kernel.m_elemsize, kernel.m_allocator);

        const float* kernel_data = kernel;
        int q = 0;
        for (; q + 7 < outch; q += 8)
        {
            const float* k0 = kernel_data + inch * kernel_size * (q + 0);
            const float* k1 = kernel_data + inch * kernel_size * (q + 1);
            const float* k2 = kernel_data + inch * kernel_size * (q + 2);
            const float* k3 = kernel_data + inch * kernel_size * (q + 3);
            const float* k4 = kernel_data + inch * kernel_size * (q + 4);
            const float* k5 = kernel_data + inch * kernel_size * (q + 5);
            const float* k6 = kernel_data + inch * kernel_size * (q + 6);
            const float* k7 = kernel_data + inch * kernel_size * (q + 7);
            float* postr = kernel_tm.channel(q / 8);
            for (int i = 0; i < inch * kernel_size; i++)
            {
                postr[0] = k0[i];
                postr[1] = k1[i];
                postr[2] = k2[i];
                postr[3] = k3[i];
                postr[4] = k4[i];
                postr[5] = k5[i];
                postr[6] = k6[i];
                postr[7] = k7[i];
                postr += 8;
            }
        }

        for (; q < outch; q++)
        {
            const float* k0 = kernel_data + inch * kernel_size * q;
            float* postr = kernel_tm.channel(q / 8 + q % 8);
            for (int i = 0; i < inch * kernel_size; i++)
            {
                postr[0] = k0[i];
            }
        }

        return;
    }

    void im2col_sgemm_avx(const aMat& bottom_im2col, aMat& top_blob, const aMat& kernel, const aMat& bias, const int kernel_w, const int kernel_h,
        const int inch, const int outch, const int outw, const int outh)
    {
        const int kernel_size = kernel_w * kernel_h;
        const int out_size = outw * outh;
        // bottom_im2col memory packed 8x8
        aMat bottom_tm(8 * kernel_size, inch, out_size / 8 + out_size % 8, bottom_im2col.m_elemsize, bottom_im2col.m_allocator);
        {
            int i = 0;
            for (; i + 7 < out_size; i += 8)
            {
                const float* img0 = bottom_im2col.channel(0);
                img0 += i;
                float* tmpptr = bottom_tm.channel(i / 8);
                for (int q = 0; q < inch * kernel_size; q++)
                {
                    _mm256_storeu_ps(tmpptr, _mm256_loadu_ps(img0));
                    tmpptr += 8;
                    img0 += out_size;
                }
            }

            for (; i < out_size; i++)
            {
                const float* img0 = bottom_im2col.channel(0);
                img0 += i;
                float* tmpptr = bottom_tm.channel(i / 8 + i % 8);
                for (int q = 0; q < inch * kernel_size; q++)
                {
                    tmpptr[q] = img0[0];
                    img0 += out_size;
                }
            }
        }

        // sgemm(int M, int N, int L, float* A, float* B, float* C)
        {
            int N = outw * outh;
            int L = kernel_w * kernel_w * inch;
            const float* bias_data = bias;

            int pp = 0;
            for (; pp + 7 < outch; pp += 8)
            {
                float* output0 = top_blob.channel(pp + 0);
                float* output1 = top_blob.channel(pp + 1);
                float* output2 = top_blob.channel(pp + 2);
                float* output3 = top_blob.channel(pp + 3);
                float* output4 = top_blob.channel(pp + 4);
                float* output5 = top_blob.channel(pp + 5);
                float* output6 = top_blob.channel(pp + 6);
                float* output7 = top_blob.channel(pp + 7);

                const float zeros[8] = { 0.f };
                const float* biasptr = bias_data ? bias_data + pp : zeros;
                int j = 0;
                for (; j + 7 < N; j += 8)
                {
                    const float* vb = bottom_tm.channel(j / 8);
                    const float* va = kernel.channel(pp / 8);
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();

                    int k = 0;
                    for (; k + 3 < L; k += 4)
                    {
                        __m256 _val = _mm256_loadu_ps(vb);
                        __m256 _w0 = _mm256_broadcast_ss(va + 0);
                        __m256 _w1 = _mm256_broadcast_ss(va + 1);
                        _sum0 = _mm256_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_fmadd_ps(_val, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(va + 2);
                        __m256 _w3 = _mm256_broadcast_ss(va + 3);
                        _sum2 = _mm256_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_fmadd_ps(_val, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(va + 4);
                        __m256 _w5 = _mm256_broadcast_ss(va + 5);
                        _sum4 = _mm256_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_fmadd_ps(_val, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(va + 6);
                        __m256 _w7 = _mm256_broadcast_ss(va + 7);
                        _sum6 = _mm256_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_fmadd_ps(_val, _w7, _sum7);
                        va += 8;
                        vb += 8;

                        _val = _mm256_loadu_ps(vb);
                        _w0 = _mm256_broadcast_ss(va + 0);
                        _w1 = _mm256_broadcast_ss(va + 1);
                        _sum0 = _mm256_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(va + 2);
                        _w3 = _mm256_broadcast_ss(va + 3);
                        _sum2 = _mm256_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(va + 4);
                        _w5 = _mm256_broadcast_ss(va + 5);
                        _sum4 = _mm256_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(va + 6);
                        _w7 = _mm256_broadcast_ss(va + 7);
                        _sum6 = _mm256_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_fmadd_ps(_val, _w7, _sum7);
                        va += 8;
                        vb += 8;

                        _val = _mm256_loadu_ps(vb);
                        _w0 = _mm256_broadcast_ss(va + 0);
                        _w1 = _mm256_broadcast_ss(va + 1);
                        _sum0 = _mm256_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(va + 2);
                        _w3 = _mm256_broadcast_ss(va + 3);
                        _sum2 = _mm256_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(va + 4);
                        _w5 = _mm256_broadcast_ss(va + 5);
                        _sum4 = _mm256_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(va + 6);
                        _w7 = _mm256_broadcast_ss(va + 7);
                        _sum6 = _mm256_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_fmadd_ps(_val, _w7, _sum7);
                        va += 8;
                        vb += 8;

                        _val = _mm256_loadu_ps(vb);
                        _w0 = _mm256_broadcast_ss(va + 0);
                        _w1 = _mm256_broadcast_ss(va + 1);
                        _sum0 = _mm256_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(va + 2);
                        _w3 = _mm256_broadcast_ss(va + 3);
                        _sum2 = _mm256_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(va + 4);
                        _w5 = _mm256_broadcast_ss(va + 5);
                        _sum4 = _mm256_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(va + 6);
                        _w7 = _mm256_broadcast_ss(va + 7);
                        _sum6 = _mm256_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_fmadd_ps(_val, _w7, _sum7);
                        va += 8;
                        vb += 8;
                    }

                    for (; k < L; k++)
                    {
                        __m256 _val = _mm256_loadu_ps(vb);
                        __m256 _w0 = _mm256_broadcast_ss(va + 0);
                        __m256 _w1 = _mm256_broadcast_ss(va + 1);
                        _sum0 = _mm256_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_fmadd_ps(_val, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(va + 2);
                        __m256 _w3 = _mm256_broadcast_ss(va + 3);
                        _sum2 = _mm256_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_fmadd_ps(_val, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(va + 4);
                        __m256 _w5 = _mm256_broadcast_ss(va + 5);
                        _sum4 = _mm256_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_fmadd_ps(_val, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(va + 6);
                        __m256 _w7 = _mm256_broadcast_ss(va + 7);
                        _sum6 = _mm256_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_fmadd_ps(_val, _w7, _sum7);
                        va += 8;
                        vb += 8;
                    }

                    _sum0 = _mm256_add_ps(_sum0, _mm256_broadcast_ss(biasptr + 0));
                    _sum1 = _mm256_add_ps(_sum1, _mm256_broadcast_ss(biasptr + 1));
                    _sum2 = _mm256_add_ps(_sum2, _mm256_broadcast_ss(biasptr + 2));
                    _sum3 = _mm256_add_ps(_sum3, _mm256_broadcast_ss(biasptr + 3));
                    _sum4 = _mm256_add_ps(_sum4, _mm256_broadcast_ss(biasptr + 4));
                    _sum5 = _mm256_add_ps(_sum5, _mm256_broadcast_ss(biasptr + 5));
                    _sum6 = _mm256_add_ps(_sum6, _mm256_broadcast_ss(biasptr + 6));
                    _sum7 = _mm256_add_ps(_sum7, _mm256_broadcast_ss(biasptr + 7));

                    _mm256_storeu_ps(output0, _sum0);
                    _mm256_storeu_ps(output1, _sum1);
                    _mm256_storeu_ps(output2, _sum2);
                    _mm256_storeu_ps(output3, _sum3);
                    _mm256_storeu_ps(output4, _sum4);
                    _mm256_storeu_ps(output5, _sum5);
                    _mm256_storeu_ps(output6, _sum6);
                    _mm256_storeu_ps(output7, _sum7);

                    output0 += 8;
                    output1 += 8;
                    output2 += 8;
                    output3 += 8;
                    output4 += 8;
                    output5 += 8;
                    output6 += 8;
                    output7 += 8;
                }

                for (; j < N; j++)
                {
                    const float* vb = bottom_tm.channel(j / 8 + j % 8);
                    const float* va = kernel.channel(pp / 8);
                    __m256 _sum = _mm256_setzero_ps();

                    for (int k = 0; k < L; k++)
                    {
                        __m256 _val = _mm256_broadcast_ss(vb);
                        __m256 _w = _mm256_loadu_ps(va);
                        _sum = _mm256_fmadd_ps(_val, _w, _sum);
                        va += 8;
                        vb += 1;
                    }

                    float sum[8];
                    _mm256_storeu_ps(sum, _sum);

                    output0[0] = sum[0] + biasptr[0];
                    output1[0] = sum[1] + biasptr[1];
                    output2[0] = sum[2] + biasptr[2];
                    output3[0] = sum[3] + biasptr[3];
                    output4[0] = sum[4] + biasptr[4];
                    output5[0] = sum[5] + biasptr[5];
                    output6[0] = sum[6] + biasptr[6];
                    output7[0] = sum[7] + biasptr[7];

                    output0++;
                    output1++;
                    output2++;
                    output3++;
                    output4++;
                    output5++;
                    output6++;
                    output7++;
                }
            }

            for (; pp < outch; pp++)
            {
                float* output = top_blob.channel(pp);
                const float bias0 = bias_data ? bias_data[pp] : 0.f;

                int j = 0;
                for (; j + 7 < N; j += 8)
                {
                    const float* vb = bottom_tm.channel(j / 8);
                    const float* va = kernel.channel(pp / 8 + pp % 8);
                    float sum[8] = { 0.f };

                    int k = 0;
                    for (; k + 7 < L; k += 8)
                    {
                        for (int n = 0; n < 8; n++)
                        {
                            sum[n] += va[0] * vb[n];
                            sum[n] += va[1] * vb[n + 8];
                            sum[n] += va[2] * vb[n + 16];
                            sum[n] += va[3] * vb[n + 24];
                            sum[n] += va[4] * vb[n + 32];
                            sum[n] += va[5] * vb[n + 40];
                            sum[n] += va[6] * vb[n + 48];
                            sum[n] += va[7] * vb[n + 56];
                        }
                        va += 8;
                        vb += 64;
                    }

                    for (; k < L; k++)
                    {
                        for (int n = 0; n < 8; n++)
                        {
                            sum[n] += va[0] * vb[n];
                        }
                        va += 1;
                        vb += 8;
                    }

                    for (int n = 0; n < 8; n++)
                    {
                        output[n] = sum[n] + bias0;
                    }
                    output += 8;
                }

                for (; j < N; j++)
                {
                    const float* vb = bottom_tm.channel(j / 8 + j % 8);
                    const float* va = kernel.channel(pp / 8 + pp % 8);
                    float sum0 = 0.f;

                    for (int k = 0; k < L; k++)
                    {
                        sum0 += va[k] * vb[k];
                    }
                    output[0] = sum0;
                    output++;
                }
            }
        }

        return;
    }

    //void im2col_sgemm_avx(const aMat& bottom_im2col, aMat& top_blob, const aMat& kernel, const aMat& bias, const int kernel_w, const int kernel_h,
    //    const int inch, const int outch, const int outw, const int outh)
    //{
    //    const int kernel_size = kernel_w * kernel_h;
    //    const int out_size = outw * outh;
    //    // bottom_im2col memory packed 8x8
    //    aMat bottom_tm(8 * kernel_size, inch, out_size / 8 + out_size % 8, bottom_im2col.m_elemsize, bottom_im2col.m_allocator);
    //    {
    //        int i = 0;
    //        for (; i + 7 < out_size; i += 8)
    //        {
    //            const float* img0 = bottom_im2col.channel(0);
    //            img0 += i;
    //            float* tmpptr = bottom_tm.channel(i / 8);
    //            for (int q = 0; q < inch * kernel_size; q++)
    //            {
    //                tmpptr[0] = img0[0];
    //                tmpptr[1] = img0[1];
    //                tmpptr[2] = img0[2];
    //                tmpptr[3] = img0[3];
    //                tmpptr[4] = img0[4];
    //                tmpptr[5] = img0[5];
    //                tmpptr[6] = img0[6];
    //                tmpptr[7] = img0[7];
    //                tmpptr += 8;
    //                img0 += out_size;
    //            }
    //        }

    //        for (; i < out_size; i++)
    //        {
    //            const float* img0 = bottom_im2col.channel(0);
    //            img0 += i;
    //            float* tmpptr = bottom_tm.channel(i / 8 + i % 8);
    //            for (int q = 0; q < inch * kernel_size; q++)
    //            {
    //                tmpptr[q] = img0[0];
    //                img0 += out_size;
    //            }
    //        }
    //    }

    //    // sgemm(int M, int N, int L, float* A, float* B, float* C)
    //    {
    //        int N = outw * outh;
    //        int L = kernel_w * kernel_w * inch;
    //        const float* bias_data = bias;

    //        int pp = 0;
    //        for (; pp + 7 < outch; pp += 8)
    //        {
    //            float* output0 = top_blob.channel(pp + 0);
    //            float* output1 = top_blob.channel(pp + 1);
    //            float* output2 = top_blob.channel(pp + 2);
    //            float* output3 = top_blob.channel(pp + 3);
    //            float* output4 = top_blob.channel(pp + 4);
    //            float* output5 = top_blob.channel(pp + 5);
    //            float* output6 = top_blob.channel(pp + 6);
    //            float* output7 = top_blob.channel(pp + 7);

    //            const float zeros[8] = { 0.f };
    //            const float* biasptr = bias_data ? bias_data + pp : zeros;
    //            int j = 0;
    //            for (; j + 7 < N; j += 8)
    //            {
    //                const float* vb = bottom_tm.channel(j / 8);
    //                const float* va = kernel.channel(pp / 8);
    //                float sum0[8] = { 0.f };
    //                float sum1[8] = { 0.f };
    //                float sum2[8] = { 0.f };
    //                float sum3[8] = { 0.f };
    //                float sum4[8] = { 0.f };
    //                float sum5[8] = { 0.f };
    //                float sum6[8] = { 0.f };
    //                float sum7[8] = { 0.f };

    //                int k = 0;
    //                for (; k + 7 < L; k += 8)
    //                {
    //                    for (int n = 0; n < 8; n++)
    //                    {
    //                        sum0[n] += va[0] * vb[n];
    //                        sum1[n] += va[1] * vb[n];
    //                        sum2[n] += va[2] * vb[n];
    //                        sum3[n] += va[3] * vb[n];
    //                        sum4[n] += va[4] * vb[n];
    //                        sum5[n] += va[5] * vb[n];
    //                        sum6[n] += va[6] * vb[n];
    //                        sum7[n] += va[7] * vb[n];
    //                        va += 8;

    //                        sum0[n] += va[0] * vb[n + 8];
    //                        sum1[n] += va[1] * vb[n + 8];
    //                        sum2[n] += va[2] * vb[n + 8];
    //                        sum3[n] += va[3] * vb[n + 8];
    //                        sum4[n] += va[4] * vb[n + 8];
    //                        sum5[n] += va[5] * vb[n + 8];
    //                        sum6[n] += va[6] * vb[n + 8];
    //                        sum7[n] += va[7] * vb[n + 8];
    //                        va += 8;

    //                        sum0[n] += va[0] * vb[n + 16];
    //                        sum1[n] += va[1] * vb[n + 16];
    //                        sum2[n] += va[2] * vb[n + 16];
    //                        sum3[n] += va[3] * vb[n + 16];
    //                        sum4[n] += va[4] * vb[n + 16];
    //                        sum5[n] += va[5] * vb[n + 16];
    //                        sum6[n] += va[6] * vb[n + 16];
    //                        sum7[n] += va[7] * vb[n + 16];
    //                        va += 8;

    //                        sum0[n] += va[0] * vb[n + 24];
    //                        sum1[n] += va[1] * vb[n + 24];
    //                        sum2[n] += va[2] * vb[n + 24];
    //                        sum3[n] += va[3] * vb[n + 24];
    //                        sum4[n] += va[4] * vb[n + 24];
    //                        sum5[n] += va[5] * vb[n + 24];
    //                        sum6[n] += va[6] * vb[n + 24];
    //                        sum7[n] += va[7] * vb[n + 24];
    //                        va += 8;

    //                        sum0[n] += va[0] * vb[n + 32];
    //                        sum1[n] += va[1] * vb[n + 32];
    //                        sum2[n] += va[2] * vb[n + 32];
    //                        sum3[n] += va[3] * vb[n + 32];
    //                        sum4[n] += va[4] * vb[n + 32];
    //                        sum5[n] += va[5] * vb[n + 32];
    //                        sum6[n] += va[6] * vb[n + 32];
    //                        sum7[n] += va[7] * vb[n + 32];
    //                        va += 8;

    //                        sum0[n] += va[0] * vb[n + 40];
    //                        sum1[n] += va[1] * vb[n + 40];
    //                        sum2[n] += va[2] * vb[n + 40];
    //                        sum3[n] += va[3] * vb[n + 40];
    //                        sum4[n] += va[4] * vb[n + 40];
    //                        sum5[n] += va[5] * vb[n + 40];
    //                        sum6[n] += va[6] * vb[n + 40];
    //                        sum7[n] += va[7] * vb[n + 40];
    //                        va += 8;

    //                        sum0[n] += va[0] * vb[n + 48];
    //                        sum1[n] += va[1] * vb[n + 48];
    //                        sum2[n] += va[2] * vb[n + 48];
    //                        sum3[n] += va[3] * vb[n + 48];
    //                        sum4[n] += va[4] * vb[n + 48];
    //                        sum5[n] += va[5] * vb[n + 48];
    //                        sum6[n] += va[6] * vb[n + 48];
    //                        sum7[n] += va[7] * vb[n + 48];
    //                        va += 8;

    //                        sum0[n] += va[0] * vb[n + 56];
    //                        sum1[n] += va[1] * vb[n + 56];
    //                        sum2[n] += va[2] * vb[n + 56];
    //                        sum3[n] += va[3] * vb[n + 56];
    //                        sum4[n] += va[4] * vb[n + 56];
    //                        sum5[n] += va[5] * vb[n + 56];
    //                        sum6[n] += va[6] * vb[n + 56];
    //                        sum7[n] += va[7] * vb[n + 56];
    //                        va -= 56;
    //                    }
    //                    va += 64;
    //                    vb += 64;
    //                }

    //                for (; k < L; k++)
    //                {
    //                    for (int n = 0; n < 8; n++)
    //                    {
    //                        sum0[n] += va[0] * vb[n];
    //                        sum1[n] += va[1] * vb[n];
    //                        sum2[n] += va[2] * vb[n];
    //                        sum3[n] += va[3] * vb[n];
    //                        sum4[n] += va[4] * vb[n];
    //                        sum5[n] += va[5] * vb[n];
    //                        sum6[n] += va[6] * vb[n];
    //                        sum7[n] += va[7] * vb[n];
    //                    }

    //                    va += 8;
    //                    vb += 8;
    //                }

    //                for (int n = 0; n < 8; n++)
    //                {
    //                    output0[n] = sum0[n] + biasptr[0];
    //                    output1[n] = sum1[n] + biasptr[1];
    //                    output2[n] = sum2[n] + biasptr[2];
    //                    output3[n] = sum3[n] + biasptr[3];
    //                    output4[n] = sum4[n] + biasptr[4];
    //                    output5[n] = sum5[n] + biasptr[5];
    //                    output6[n] = sum6[n] + biasptr[6];
    //                    output7[n] = sum7[n] + biasptr[7];
    //                }

    //                output0 += 8;
    //                output1 += 8;
    //                output2 += 8;
    //                output3 += 8;
    //                output4 += 8;
    //                output5 += 8;
    //                output6 += 8;
    //                output7 += 8;
    //            }

    //            for (; j < N; j++)
    //            {
    //                const float* vb = bottom_tm.channel(j / 8 + j % 8);
    //                const float* va = kernel.channel(pp / 8);
    //                float sum0 = 0.f;
    //                float sum1 = 0.f;
    //                float sum2 = 0.f;
    //                float sum3 = 0.f;
    //                float sum4 = 0.f;
    //                float sum5 = 0.f;
    //                float sum6 = 0.f;
    //                float sum7 = 0.f;

    //                for (int k = 0; k < L; k++)
    //                {
    //                    sum0 += va[0] * vb[0];
    //                    sum1 += va[1] * vb[0];
    //                    sum2 += va[2] * vb[0];
    //                    sum3 += va[3] * vb[0];
    //                    sum4 += va[4] * vb[0];
    //                    sum5 += va[5] * vb[0];
    //                    sum6 += va[6] * vb[0];
    //                    sum7 += va[7] * vb[0];
    //                    va += 8;
    //                    vb += 1;
    //                }

    //                output0[0] = sum0 + biasptr[0];
    //                output1[0] = sum1 + biasptr[1];
    //                output2[0] = sum2 + biasptr[2];
    //                output3[0] = sum3 + biasptr[3];
    //                output4[0] = sum4 + biasptr[4];
    //                output5[0] = sum5 + biasptr[5];
    //                output6[0] = sum6 + biasptr[6];
    //                output7[0] = sum7 + biasptr[7];

    //                output0++;
    //                output1++;
    //                output2++;
    //                output3++;
    //                output4++;
    //                output5++;
    //                output6++;
    //                output7++;
    //            }
    //        }

    //        for (; pp < outch; pp++)
    //        {
    //            float* output = top_blob.channel(pp);
    //            const float bias0 = bias_data ? bias_data[pp] : 0.f;

    //            int j = 0;
    //            for (; j + 7 < N; j += 8)
    //            {
    //                const float* vb = bottom_tm.channel(j / 8);
    //                const float* va = kernel.channel(pp / 8 + pp % 8);
    //                float sum[8] = { 0.f };

    //                int k = 0;
    //                for (; k + 7 < L; k += 8)
    //                {
    //                    for (int n = 0; n < 8; n++)
    //                    {
    //                        sum[n] += va[0] * vb[n];
    //                        sum[n] += va[1] * vb[n + 8];
    //                        sum[n] += va[2] * vb[n + 16];
    //                        sum[n] += va[3] * vb[n + 24];
    //                        sum[n] += va[4] * vb[n + 32];
    //                        sum[n] += va[5] * vb[n + 40];
    //                        sum[n] += va[6] * vb[n + 48];
    //                        sum[n] += va[7] * vb[n + 56];
    //                    }
    //                    va += 8;
    //                    vb += 64;
    //                }

    //                for (; k < L; k++)
    //                {
    //                    for (int n = 0; n < 8; n++)
    //                    {
    //                        sum[n] += va[0] * vb[n];
    //                    }
    //                    va += 1;
    //                    vb += 8;
    //                }

    //                for (int n = 0; n < 8; n++)
    //                {
    //                    output[n] = sum[n] + bias0;
    //                }
    //                output += 8;
    //            }

    //            for (; j < N; j++)
    //            {
    //                const float* vb = bottom_tm.channel(j / 8 + j % 8);
    //                const float* va = kernel.channel(pp / 8 + pp % 8);
    //                float sum0 = 0.f;

    //                for (int k = 0; k < L; k++)
    //                {
    //                    sum0 += va[k] * vb[k];
    //                }
    //                output[0] = sum0;
    //                output++;
    //            }
    //        }
    //    }

    //    return;
    //}

    ConvolutionX86avx::ConvolutionX86avx(const LayerParam& layer_param)
        : Convolution(layer_param)
    {}

    int ConvolutionX86avx::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
    {
        if (bottom_blobs.size() != top_blobs.size())
        {
            ConsoleELog << "Convolution ERROR: bottom_blobs size(" << bottom_blobs.size() << ") != top_blobs size(" << top_blobs.size() << ")";
            return -1;
        }

        return forward_sgemm_avx(bottom_blobs, top_blobs);
    }

    int ConvolutionX86avx::forward_sgemm_avx(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs) const
    {
        conv_im2col_sgemm_transform_kernel_avx_8x8(weight_data, weight_sgemm_data, kernel_w, kernel_h, bottom_blobs[0].m_c, num_output);

        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            const aMat& bottom_blob = bottom_blobs[i];
            aMat& top_blob = top_blobs[i];
            aMat bottom_blob_bordered;
            make_padding(bottom_blob, bottom_blob_bordered);

            const int w = bottom_blob_bordered.m_w;
            const int h = bottom_blob_bordered.m_h;
            const int inch = bottom_blob_bordered.m_c;
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
            const int outw = (w - kernel_extent_w) / stride_w + 1;
            const int outh = (h - kernel_extent_h) / stride_h + 1;

            aMat bottom_im2col;
            conv_im2col(bottom_blob_bordered, bottom_im2col, kernel_w, kernel_h, stride_w, stride_h, outw, outh);

            top_blob.create(outw, outh, num_output, bottom_blob_bordered.m_elemsize, bottom_blob_bordered.m_allocator);
            im2col_sgemm_avx(bottom_im2col, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, inch, num_output, outw, outh);
        }

        return 0;
    }

    REGISTER_LAYER_CLASS(Convolution_x86_avx, ConvolutionX86avx);
}

#endif