// Warp / block reduction primitives shared by softmax, LayerNorm and attention.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace reduce {

constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
  return v;  // every lane holds the result (butterfly)
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
  return v;
}

// ---- Online softmax state: running max m and running sum d = sum(exp(x - m)).
struct MD {
  float m, d;
};

__device__ __forceinline__ MD md_identity() { return {-INFINITY, 0.f}; }

__device__ __forceinline__ MD md_update(MD s, float x) {
  const float m = fmaxf(s.m, x);
  if (m == -INFINITY) return s;  // masked input (-inf) on an empty state: nothing to add
  return {m, s.d * __expf(s.m - m) + __expf(x - m)};
}

__device__ __forceinline__ MD md_merge(MD a, MD b) {
  const float m = fmaxf(a.m, b.m);
  if (m == -INFINITY) return md_identity();  // both empty: avoid exp(-inf - -inf) = NaN
  return {m, a.d * __expf(a.m - m) + b.d * __expf(b.m - m)};
}

__device__ __forceinline__ MD warp_reduce_md(MD v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    MD other{__shfl_xor_sync(FULL_MASK, v.m, o), __shfl_xor_sync(FULL_MASK, v.d, o)};
    v = md_merge(v, other);
  }
  return v;
}

// ---- Welford state for mean / variance: count, mean, M2 = sum((x - mean)^2).
struct Welford {
  float n, mean, m2;
};

__device__ __forceinline__ Welford welford_update(Welford s, float x) {
  const float n = s.n + 1.f;
  const float delta = x - s.mean;
  const float mean = s.mean + delta / n;
  return {n, mean, s.m2 + delta * (x - mean)};
}

__device__ __forceinline__ Welford welford_merge(Welford a, Welford b) {
  const float n = a.n + b.n;
  if (n == 0.f) return {0.f, 0.f, 0.f};
  const float delta = b.mean - a.mean;
  const float wb = b.n / n;
  return {n, a.mean + delta * wb, a.m2 + b.m2 + delta * delta * a.n * wb};
}

__device__ __forceinline__ Welford warp_reduce_welford(Welford v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    Welford other{__shfl_xor_sync(FULL_MASK, v.n, o), __shfl_xor_sync(FULL_MASK, v.mean, o),
                  __shfl_xor_sync(FULL_MASK, v.m2, o)};
    v = welford_merge(v, other);
  }
  return v;
}

// Generic block reduce: warp reduce -> one value per warp in smem -> warp 0 reduces those.
// Result is broadcast to every thread. BLOCK must be a multiple of 32.
template <int BLOCK, typename T, typename WarpReduce>
__device__ __forceinline__ T block_reduce(T v, T identity, WarpReduce warp_reduce) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "block size");
  __shared__ T partial[BLOCK / 32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warp_reduce(v);
  if (lane == 0) partial[wid] = v;
  __syncthreads();
  if (wid == 0) {
    T t = lane < BLOCK / 32 ? partial[lane] : identity;
    t = warp_reduce(t);
    if (lane == 0) partial[0] = t;
  }
  __syncthreads();
  T result = partial[0];
  __syncthreads();  // so a second call in the same kernel can't overwrite before everyone reads
  return result;
}

}  // namespace reduce
