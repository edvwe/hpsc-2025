#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main()
{
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for (int i = 0; i < N; i++)
  {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  int idx_array[16];
  for (int k = 0; k < 16; ++k)
  {
    idx_array[k] = k;
  }

  for (int i = 0; i < N; i++)
  ¨

    // float ry = y[i] - y[j];  
    __m512 vxi = _mm512_set1_ps(x[i]); // Broadcasts the scalar value x[i] to all 16 lanes of a 512-bit SIMD register (each lane holds the same double)
    __m512 vyi = _mm512_set1_ps(y[i]);
    __m512 vyj = _mm512_loadu_ps(y); // Load values of y into a 512-bit SIMD register
    __m512 vxj = _mm512_loadu_ps(x);
    __m512 vrx = _mm512_sub_ps(vxi, vxj);
    __m512 vry = _mm512_sub_ps(vyi, vyj);

    // float r = std::sqrt(rx * rx + ry * ry);
    __m512 vr = _mm512_add_ps(_mm512_mul_ps(vrx, vrx), _mm512_mul_ps(vry, vry));

    // fx[i] -= rx * m[j] / (r * r * r);
    // fy[i] -= ry * m[j] / (r * r * r);
    __m512 vr_inv = _mm512_rsqrt14_ps(vr);
    __m512 vr_3_4 = _mm512_mul_ps(vr_inv, _mm512_mul_ps(vr_inv, vr_inv));
    __m512 vmj = _mm512_loadu_ps(m);
    __m512 vx_div = _mm512_mul_ps(_mm512_mul_ps(vmj, vrx), vr_3_4);
    __m512 vy_div = _mm512_mul_ps(_mm512_mul_ps(vmj, vry), vr_3_4);

    // (i != j) check
    // Using SIMD it is faster to do every calculation and just mask out the ones we don't want rather than do conditionals or branches
    __m512i indices = _mm512_loadu_si512(idx_array);
    __mmask16 mask = _mm512_cmp_epi32_mask(indices, _mm512_set1_epi32(i), _MM_CMPINT_NE);

    __m512 vfx = _mm512_mask_sub_ps(_mm512_setzero_ps(), mask, _mm512_setzero_ps(), vx_div);
    __m512 vfy = _mm512_mask_sub_ps(_mm512_setzero_ps(), mask, _mm512_setzero_ps(), vy_div);

    fx[i] = _mm512_reduce_add_ps(vfx);
    fy[i] = _mm512_reduce_add_ps(vfy);
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
