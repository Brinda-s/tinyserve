// Attention forward: O = softmax(Q K^T * scale [+ causal mask]) V.
// Q, K, V, O are [B*H, N, D] contiguous (D = head dim = 64).
//
//   v0 unfused     cuBLAS batched GEMM -> our softmax -> cuBLAS batched GEMM. Materialises the
//                  N x N score matrix in DRAM: O(N^2) memory traffic. (See bench_attention.cu.)
//   v1 flash fp32  FlashAttention tiling, scalar fp32. One thread per query row, K/V tiles in shared
//                  memory, online softmax keeps (m, l) per row so S/P never leave the SM.
//   v2 flash wmma  Same algorithm on tensor cores: fp16 Q/K/V, fp32 accumulate. One warp per 16 query
//                  rows; S = Q K^T and O += P V are WMMA, softmax runs in fp32 out of shared memory,
//                  exp2 with the log2(e) factor folded into the scale.
//   v3 flash mma   FlashAttention-2 style: raw PTX mma.sync.m16n8k8 instead of WMMA, because its register
//                  layout is documented. S, P and the O accumulator all stay in registers: no shared-memory
//                  round trip per tile, half the shared memory of v2 (so 3 blocks/SM on a T4 instead of 1),
//                  and P is reused straight from S's accumulator registers as the A operand of P*V.
#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#include "common.cuh"
#include "reduce.cuh"

namespace attention {

constexpr int HEAD_DIM = 64;

// ------------------------------------------------------- v1 flash, fp32
template <int D, int BR, int BC>
__global__ void __launch_bounds__(BR)
    flash_attn_fp32(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
                    float* __restrict__ O, int N, float scale, bool causal) {
  __shared__ float Ks[BC][D];
  __shared__ float Vs[BC][D];

  const size_t base = (size_t)blockIdx.y * N * D;
  Q += base; K += base; V += base; O += base;
  const int q_row = blockIdx.x * BR + threadIdx.x;
  const bool valid = q_row < N;

  float q[D], o[D];
#pragma unroll
  for (int d = 0; d < D; ++d) {
    q[d] = valid ? Q[(size_t)q_row * D + d] * scale : 0.f;
    o[d] = 0.f;
  }
  float m = -INFINITY, l = 0.f;

  // Causal: this block never needs keys beyond its last query row.
  const int kv_end = causal ? min(N, (int)(blockIdx.x + 1) * BR) : N;
  for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
    __syncthreads();  // previous tile fully consumed
    for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
      const int r = idx / D, c = idx % D, g = kv0 + r;
      Ks[r][c] = g < N ? K[(size_t)g * D + c] : 0.f;
      Vs[r][c] = g < N ? V[(size_t)g * D + c] : 0.f;
    }
    __syncthreads();

    // S row for this query: every thread reads the same Ks[j][d] -> shared-memory broadcast.
    float s[BC];
    float tile_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < BC; ++j) {
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < D; ++d) acc += q[d] * Ks[j][d];
      const int kj = kv0 + j;
      if (kj >= N || (causal && kj > q_row)) acc = -INFINITY;
      s[j] = acc;
      tile_max = fmaxf(tile_max, acc);
    }

    const float m_new = fmaxf(m, tile_max);
    if (m_new == -INFINITY) continue;  // row fully masked so far (only padding rows)
    const float alpha = __expf(m - m_new);  // rescale the old partial results
    l *= alpha;
#pragma unroll
    for (int d = 0; d < D; ++d) o[d] *= alpha;
#pragma unroll
    for (int j = 0; j < BC; ++j) {
      const float p = __expf(s[j] - m_new);
      l += p;
#pragma unroll
      for (int d = 0; d < D; ++d) o[d] += p * Vs[j][d];
    }
    m = m_new;
  }

  if (valid) {
    const float inv = 1.f / l;
#pragma unroll
    for (int d = 0; d < D; ++d) O[(size_t)q_row * D + d] = o[d] * inv;
  }
}

// ------------------------------------------------- v2 flash, tensor cores
namespace flash_wmma {
constexpr int D = HEAD_DIM;
constexpr int WARPS = 4;
constexpr int BR = 16 * WARPS;  // 64 query rows per block
constexpr int BC = 32;          // keys per tile: exactly one per lane in the softmax step
constexpr int LDKV = D + 8;     // half, padded
constexpr int LDS = BC + 4;     // float
constexpr int LDP = BC + 8;     // half
constexpr int LDO = D + 4;      // float
}  // namespace flash_wmma

__global__ void __launch_bounds__(flash_wmma::WARPS * 32)
    flash_attn_wmma(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
                    float* __restrict__ O, int N, float scale_log2, bool causal) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  using namespace flash_wmma;
  // 40 KB of static shared memory; every fragment pointer below is 32-byte aligned.
  __shared__ __align__(32) half Ks[BC * LDKV];
  __shared__ __align__(32) half Vs[BC * LDKV];
  __shared__ __align__(32) float Ss[WARPS][16 * LDS];
  __shared__ __align__(32) half Ps[WARPS][16 * LDP];
  __shared__ __align__(32) float Os[WARPS][16 * LDO];

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const size_t base = (size_t)blockIdx.y * N * D;
  Q += base; K += base; V += base; O += base;
  const int q0 = blockIdx.x * BR + warp * 16;  // first query row of this warp
  float* S = Ss[warp];
  half* P = Ps[warp];
  float* Ow = Os[warp];

  // Q fragments stay in registers for the whole kernel (16 x 64 = 4 k-steps).
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[D / 16];
#pragma unroll
  for (int k = 0; k < D / 16; ++k) wmma::load_matrix_sync(qf[k], Q + (size_t)q0 * D + k * 16, D);

  for (int i = lane; i < 16 * LDO; i += 32) Ow[i] = 0.f;
  // Per-row running max / sum. Every lane keeps all 16 (warp-uniform values).
  float m[16], l[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) { m[r] = -INFINITY; l[r] = 0.f; }

  const int kv_end = causal ? min(N, (int)(blockIdx.x + 1) * BR) : N;
  for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
    __syncthreads();
    // K, V tiles: 32 x 64 halves = 256 x 16-byte vectors each, 2 per thread.
    for (int v = threadIdx.x; v < BC * D / 8; v += WARPS * 32) {
      const int r = v / (D / 8), c = (v % (D / 8)) * 8;
      *reinterpret_cast<int4*>(&Ks[r * LDKV + c]) = *reinterpret_cast<const int4*>(&K[(size_t)(kv0 + r) * D + c]);
      *reinterpret_cast<int4*>(&Vs[r * LDKV + c]) = *reinterpret_cast<const int4*>(&V[(size_t)(kv0 + r) * D + c]);
    }
    __syncthreads();

    // S (16 x 32) = Q_w (16 x 64) * K_tile^T. K stored row-major [BC][D] *is* K^T in col-major,
    // so we load it as a col_major matrix_b -- no transpose needed.
#pragma unroll
    for (int nt = 0; nt < BC / 16; ++nt) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
      wmma::fill_fragment(sf, 0.f);
#pragma unroll
      for (int k = 0; k < D / 16; ++k) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
        wmma::load_matrix_sync(kf, &Ks[(nt * 16) * LDKV + k * 16], LDKV);
        wmma::mma_sync(sf, qf[k], kf, sf);
      }
      wmma::store_matrix_sync(&S[nt * 16], sf, LDS, wmma::mem_row_major);
    }
    __syncwarp();

    // Online softmax, one row at a time; lane j owns key kv0 + j.
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int q_row = q0 + r, kj = kv0 + lane;
      float s = S[r * LDS + lane] * scale_log2;
      if (causal && kj > q_row) s = -INFINITY;
      const float m_new = fmaxf(m[r], reduce::warp_max(s));  // finite: key 0 is always visible
      const float p = exp2f(s - m_new);
      const float alpha = exp2f(m[r] - m_new);
      l[r] = l[r] * alpha + reduce::warp_sum(p);
      m[r] = m_new;
      P[r * LDP + lane] = __float2half(p);
      Ow[r * LDO + lane] *= alpha;       // D = 64 = 2 columns per lane
      Ow[r * LDO + lane + 32] *= alpha;
    }
    __syncwarp();

    // O_w (16 x 64) += P (16 x 32) * V_tile (32 x 64). The accumulator round-trips through
    // shared memory because WMMA's fragment->row mapping is opaque, so the per-row alpha
    // rescale above has to happen on a plain array.
#pragma unroll
    for (int nt = 0; nt < D / 16; ++nt) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> of;
      wmma::load_matrix_sync(of, &Ow[nt * 16], LDO, wmma::mem_row_major);
#pragma unroll
      for (int k = 0; k < BC / 16; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
        wmma::load_matrix_sync(pf, &P[k * 16], LDP);
        wmma::load_matrix_sync(vf, &Vs[(k * 16) * LDKV + nt * 16], LDKV);
        wmma::mma_sync(of, pf, vf, of);
      }
      wmma::store_matrix_sync(&Ow[nt * 16], of, LDO, wmma::mem_row_major);
    }
    __syncwarp();
  }

#pragma unroll
  for (int r = 0; r < 16; ++r) {
    const float inv = 1.f / l[r];
    O[(size_t)(q0 + r) * D + lane] = Ow[r * LDO + lane] * inv;
    O[(size_t)(q0 + r) * D + lane + 32] = Ow[r * LDO + lane + 32] * inv;
  }
#endif
}

// ---------------------------------- v3 flash, mma.sync, O in registers
// Fragment layouts for mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 (PTX ISA), with
// g = lane / 4 ("group") and t = lane % 4 ("thread in group"):
//   A 16x8 (row-major):  a[0] = {A[g][2t],   A[g][2t+1]}     a[1] = {A[g+8][2t], A[g+8][2t+1]}
//   B  8x8 (col-major):  b    = {B[2t][g],   B[2t+1][g]}
//   C 16x8 (fp32):       c0,c1 = C[g][2t], C[g][2t+1]        c2,c3 = C[g+8][2t], C[g+8][2t+1]
// Two consequences make this kernel work:
//   * each thread owns exactly rows g and g+8, so the per-row rescale by alpha is a plain multiply;
//   * the C layout of an S tile (16 queries x 8 keys) equals the A layout of P for the next MMA
//     (16 queries x 8 keys as the k dimension), so P never leaves registers.
// Requires sm_75+ (m16n8k8 with f16 inputs was introduced with Turing).
namespace flash_mma {
constexpr int D = HEAD_DIM;      // 64
constexpr int WARPS = 4;
constexpr int BR = 16 * WARPS;   // 64 query rows per block
constexpr int BC = 64;           // keys per tile
constexpr int LDKV = D + 8;      // 72 halves: makes both K and V fragment reads bank-conflict-free
constexpr int NT_S = BC / 8;     // 8 n-tiles of S per warp
constexpr int KS_Q = D / 8;      // 8 k-steps over the head dim
constexpr int NT_O = D / 8;      // 8 n-tiles of O per warp
}  // namespace flash_mma

__device__ __forceinline__ void mma_m16n8k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
#if __CUDA_ARCH__ >= 750
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b));
#endif
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);  // .x (low 16 bits) = lo
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_halves(half lo, half hi) {
  __half2 h = __halves2half2(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void __launch_bounds__(flash_mma::WARPS * 32)
    flash_attn_mma(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
                   float* __restrict__ O, int N, float scale_log2, bool causal) {
#if __CUDA_ARCH__ >= 750
  using namespace flash_mma;
  __shared__ __align__(16) half Ks[BC * LDKV];  // 9 KB
  __shared__ __align__(16) half Vs[BC * LDKV];  // 9 KB

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t = lane % 4;
  const size_t base = (size_t)blockIdx.y * N * D;
  Q += base; K += base; V += base; O += base;
  const int q0 = blockIdx.x * BR + warp * 16;
  const int row0 = q0 + g, row1 = q0 + g + 8;  // the two query rows this thread owns

  // Q as A fragments, loaded once from global: 8 k-steps x 2 registers.
  uint32_t qa[KS_Q][2];
#pragma unroll
  for (int ks = 0; ks < KS_Q; ++ks) {
    qa[ks][0] = *reinterpret_cast<const uint32_t*>(&Q[(size_t)row0 * D + ks * 8 + 2 * t]);
    qa[ks][1] = *reinterpret_cast<const uint32_t*>(&Q[(size_t)row1 * D + ks * 8 + 2 * t]);
  }

  float o[NT_O][4];
#pragma unroll
  for (int i = 0; i < NT_O; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const int kv_end = causal ? min(N, (int)(blockIdx.x + 1) * BR) : N;
  for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
    __syncthreads();
    for (int v = threadIdx.x; v < BC * D / 8; v += WARPS * 32) {
      const int r = v / (D / 8), c = (v % (D / 8)) * 8;
      *reinterpret_cast<int4*>(&Ks[r * LDKV + c]) = *reinterpret_cast<const int4*>(&K[(size_t)(kv0 + r) * D + c]);
      *reinterpret_cast<int4*>(&Vs[r * LDKV + c]) = *reinterpret_cast<const int4*>(&V[(size_t)(kv0 + r) * D + c]);
    }
    __syncthreads();

    // ---- S = Q K^T (16 x 64 per warp), in registers. B[k=d][n=key] = K[key][d]: the two
    // halves a thread needs are adjacent in a K row, so it is a single 32-bit shared load.
    float s[NT_S][4];
#pragma unroll
    for (int nt = 0; nt < NT_S; ++nt) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < KS_Q; ++ks) {
        const uint32_t b = *reinterpret_cast<const uint32_t*>(&Ks[(nt * 8 + g) * LDKV + ks * 8 + 2 * t]);
        mma_m16n8k8(s[nt], qa[ks][0], qa[ks][1], b);
      }
    }

    // ---- scale, mask, row max. A row's 64 scores live in the 4 threads of a quad
    // (16 per thread), so the row reduction is local + 2 shuffles.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NT_S; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int key = kv0 + nt * 8 + 2 * t + (i & 1);
        const int row = i < 2 ? row0 : row1;
        float x = s[nt][i] * scale_log2;
        if (causal && key > row) x = -INFINITY;
        s[nt][i] = x;
      }
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);  // finite: key 0 is in the first tile
    const float alpha0 = exp2f(m0 - mn0), alpha1 = exp2f(m1 - mn1);

    // ---- P = exp2(S - m), row sums
    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < NT_S; ++nt) {
      s[nt][0] = exp2f(s[nt][0] - mn0);
      s[nt][1] = exp2f(s[nt][1] - mn0);
      s[nt][2] = exp2f(s[nt][2] - mn1);
      s[nt][3] = exp2f(s[nt][3] - mn1);
      sum0 += s[nt][0] + s[nt][1];
      sum1 += s[nt][2] + s[nt][3];
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, off);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, off);
    }
    l0 = l0 * alpha0 + sum0;
    l1 = l1 * alpha1 + sum1;
    m0 = mn0;
    m1 = mn1;

    // ---- rescale O in registers: c0,c1 are row g, c2,c3 are row g+8. This is the step WMMA
    // could not do without a shared-memory round trip.
#pragma unroll
    for (int dt = 0; dt < NT_O; ++dt) {
      o[dt][0] *= alpha0; o[dt][1] *= alpha0;
      o[dt][2] *= alpha1; o[dt][3] *= alpha1;
    }

    // ---- O += P V. S n-tile kt (keys kt*8..+8) in C layout == P's A fragment for k-step kt.
    // B[k=key][n=d] = V[key][d]: the two halves are in adjacent V rows, so two 16-bit loads.
#pragma unroll
    for (int kt = 0; kt < NT_S; ++kt) {
      const uint32_t pa0 = pack_half2(s[kt][0], s[kt][1]);
      const uint32_t pa1 = pack_half2(s[kt][2], s[kt][3]);
#pragma unroll
      for (int dt = 0; dt < NT_O; ++dt) {
        const int krow = kt * 8 + 2 * t, dcol = dt * 8 + g;
        const uint32_t b = pack_halves(Vs[krow * LDKV + dcol], Vs[(krow + 1) * LDKV + dcol]);
        mma_m16n8k8(o[dt], pa0, pa1, b);
      }
    }
  }

  // ---- epilogue: normalise and store (two adjacent floats per row per n-tile)
  const float inv0 = 1.f / l0, inv1 = 1.f / l1;
#pragma unroll
  for (int dt = 0; dt < NT_O; ++dt) {
    const int col = dt * 8 + 2 * t;
    *reinterpret_cast<float2*>(&O[(size_t)row0 * D + col]) = make_float2(o[dt][0] * inv0, o[dt][1] * inv0);
    *reinterpret_cast<float2*>(&O[(size_t)row1 * D + col]) = make_float2(o[dt][2] * inv1, o[dt][3] * inv1);
  }
#endif
}

// ------------------------------------------------------------- launchers
inline void launch_flash_fp32(const float* Q, const float* K, const float* V, float* O, int BH, int N, bool causal,
                              cudaStream_t s = 0) {
  constexpr int BR = 64, BC = 32;
  dim3 grid(ceil_div(N, BR), BH);
  flash_attn_fp32<HEAD_DIM, BR, BC><<<grid, BR, 0, s>>>(Q, K, V, O, N, 1.f / std::sqrt((float)HEAD_DIM), causal);
  CUDA_CHECK_LAST();
}

inline void launch_flash_wmma(const half* Q, const half* K, const half* V, float* O, int BH, int N, bool causal,
                              cudaStream_t s = 0) {
  if (N % flash_wmma::BR) {
    fprintf(stderr, "flash_attn_wmma needs N %% %d == 0\n", flash_wmma::BR);
    std::exit(1);
  }
  const float scale_log2 = 1.4426950408889634f / std::sqrt((float)HEAD_DIM);  // log2(e) / sqrt(D)
  dim3 grid(N / flash_wmma::BR, BH);
  flash_attn_wmma<<<grid, flash_wmma::WARPS * 32, 0, s>>>(Q, K, V, O, N, scale_log2, causal);
  CUDA_CHECK_LAST();
}

inline void launch_flash_mma(const half* Q, const half* K, const half* V, float* O, int BH, int N, bool causal,
                             cudaStream_t s = 0) {
  static int cc = [] {
    int dev = 0, major = 0, minor = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    return major * 10 + minor;
  }();
  if (cc < 75) {
    fprintf(stderr, "flash_attn_mma needs sm_75+ (mma.sync.m16n8k8)\n");
    std::exit(1);
  }
  if (N % flash_mma::BR) {
    fprintf(stderr, "flash_attn_mma needs N %% %d == 0\n", flash_mma::BR);
    std::exit(1);
  }
  const float scale_log2 = 1.4426950408889634f / std::sqrt((float)HEAD_DIM);
  dim3 grid(N / flash_mma::BR, BH);
  flash_attn_mma<<<grid, flash_mma::WARPS * 32, 0, s>>>(Q, K, V, O, N, scale_log2, causal);
  CUDA_CHECK_LAST();
}

}  // namespace attention
