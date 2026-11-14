// PyTorch binding for the hand-written FlashAttention kernel from gpu-kernels
// (https://github.com/Brinda-s/gpu-kernels, kernels/attention.cuh, version v3:
// PTX mma.sync.m16n8k8 with S, P and the output accumulator in registers).
//
// The headers in gpu_kernels/ are copied unchanged from gpu-kernels commit d096a51.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "gpu_kernels/attention.cuh"

// q, k, v: [batch*heads, N, 64] fp16, contiguous, N a multiple of 64. Returns fp32 output.
torch::Tensor causal_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == torch::kHalf && k.scalar_type() == torch::kHalf && v.scalar_type() == torch::kHalf,
              "inputs must be float16");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == attention::HEAD_DIM, "expected [batch*heads, N, 64]");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k and v must have the same shape");
  const int BH = q.size(0), N = q.size(1);
  TORCH_CHECK(N % attention::flash_mma::BR == 0, "N must be a multiple of 64 (pad on the Python side)");

  auto o = torch::empty(q.sizes(), q.options().dtype(torch::kFloat));
  attention::launch_flash_mma(reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                              reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                              reinterpret_cast<const half*>(v.data_ptr<at::Half>()), o.data_ptr<float>(), BH, N,
                              /*causal=*/true, at::cuda::getCurrentCUDAStream());
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("causal_attention", &causal_attention, "causal FlashAttention forward (gpu-kernels v3)");
}
