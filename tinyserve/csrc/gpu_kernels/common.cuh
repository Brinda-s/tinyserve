// Shared helpers for every benchmark: error checks, timing, random init,
// correctness comparison and a tiny CLI parser.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Call after a kernel launch: catches bad launch configs immediately.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

__host__ __device__ inline constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// CLI: every bench understands --profile (one launch per kernel, for ncu),
// --iters N, --warmup N and --csv path.
// ---------------------------------------------------------------------------
struct BenchArgs {
  bool profile = false;
  int warmup = 10;
  int iters = 50;
  std::string csv;

  static BenchArgs parse(int argc, char** argv) {
    BenchArgs a;
    for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--profile")) {
        a.profile = true;
        a.warmup = 0;
        a.iters = 1;
      } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
        a.iters = std::atoi(argv[++i]);
      } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
        a.warmup = std::atoi(argv[++i]);
      } else if (!strcmp(argv[i], "--csv") && i + 1 < argc) {
        a.csv = argv[++i];
      }
    }
    return a;
  }
};

// Average time of `f()` in milliseconds using CUDA events.
template <typename F>
float time_ms(F&& f, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) f();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) f();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK_LAST();
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

inline void fill_uniform(std::vector<float>& v, uint32_t seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : v) x = dist(rng);
}

// Round-trip through fp16 so the CPU reference sees exactly what the GPU sees.
inline std::vector<float> round_to_half(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __half2float(__float2half(v[i]));
  return out;
}

inline std::vector<half> to_half(const std::vector<float>& v) {
  std::vector<half> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __float2half(v[i]);
  return out;
}

// |out - ref| <= atol + rtol * |ref| for every element.
inline bool check_close(const char* name, const float* ref, const float* out, size_t n,
                        float atol, float rtol, bool verbose = true) {
  double max_abs = 0.0;
  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double diff = std::fabs((double)out[i] - (double)ref[i]);
    if (!(diff <= atol + rtol * std::fabs((double)ref[i]))) {  // also catches NaN
      if (bad == 0) first_bad = i;
      ++bad;
    }
    if (diff > max_abs || std::isnan(diff)) max_abs = std::isnan(diff) ? INFINITY : diff;
  }
  if (verbose || bad) {
    printf("  [%s] %-28s max_abs_err=%.3e", bad ? "FAIL" : " ok ", name, max_abs);
    if (bad)
      printf("  (%zu bad, first at %zu: got %f want %f)", bad, first_bad, out[first_bad],
             ref[first_bad]);
    printf("\n");
  }
  return bad == 0;
}

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t n = 0;
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : n(count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  void upload(const std::vector<T>& h) { CUDA_CHECK(cudaMemcpy(ptr, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> download() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  void zero() { CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T))); }
};

// Achievable DRAM bandwidth ceiling for a memory-bound kernel that reads
// `bytes` and writes `bytes`: a device-to-device memcpy of the same size.
inline double memcpy_roofline_gbps(size_t bytes, int warmup, int iters) {
  DeviceBuffer<char> src(bytes), dst(bytes);
  float ms = time_ms([&] { CUDA_CHECK(cudaMemcpyAsync(dst.ptr, src.ptr, bytes, cudaMemcpyDeviceToDevice)); },
                     warmup, iters);
  return 2.0 * bytes / (ms * 1e-3) / 1e9;
}

inline void print_device() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("Device: %s (sm_%d%d, %d SMs, %.1f GB)\n", p.name, p.major, p.minor, p.multiProcessorCount,
         p.totalGlobalMem / 1e9);
}

// Appends one result row: op,version,shape,ms,metric,unit
struct CsvWriter {
  FILE* f = nullptr;
  explicit CsvWriter(const std::string& path) {
    if (path.empty()) return;
    FILE* probe = fopen(path.c_str(), "r");
    bool exists = probe != nullptr;
    if (probe) fclose(probe);
    f = fopen(path.c_str(), "a");
    if (f && !exists) fprintf(f, "op,version,shape,ms,metric,unit\n");
  }
  ~CsvWriter() { if (f) fclose(f); }
  void row(const char* op, const char* version, const std::string& shape, double ms, double metric,
           const char* unit) {
    if (f) fprintf(f, "%s,%s,%s,%.5f,%.3f,%s\n", op, version, shape.c_str(), ms, metric, unit);
  }
};
