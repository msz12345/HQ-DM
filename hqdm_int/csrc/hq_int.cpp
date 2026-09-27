#include <torch/extension.h>

torch::Tensor fwht_quant_lastdim_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t qmax);

torch::Tensor fwht_quant_nchw_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t qmax);

torch::Tensor quantize_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t qmax);

torch::Tensor im2col_nhwc_cuda(
    torch::Tensor input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w);

torch::Tensor scale_bias_cuda(
    torch::Tensor accum_low,
    torch::Tensor accum_high,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    int64_t high_multiplier,
    double normalizer);

torch::Tensor scale_bias_nchw_cuda(
    torch::Tensor accum_low,
    torch::Tensor accum_high,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    int64_t high_multiplier,
    double normalizer,
    int64_t batch,
    int64_t output_h,
    int64_t output_w);

#ifdef HQDM_NO_CUTLASS_BINDINGS

torch::Tensor cutlass_conv2d_nhwc_cuda(
    torch::Tensor,
    torch::Tensor,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t) {
  TORCH_CHECK(
      false,
      "cutlass_conv2d_nhwc is unavailable in this HQ-DM vendor fallback; "
      "the Python dispatcher must use im2col_nhwc plus torch._int_mm");
  return {};
}

torch::Tensor cutlass_gemm_int8_cuda(torch::Tensor, torch::Tensor) {
  TORCH_CHECK(
      false,
      "cutlass_gemm_int8 is unavailable in this HQ-DM vendor fallback; "
      "the Python dispatcher must use torch._int_mm");
  return {};
}

#else

torch::Tensor cutlass_conv2d_nhwc_cuda(
    torch::Tensor q,
    torch::Tensor filter,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w);

torch::Tensor cutlass_gemm_int8_cuda(
    torch::Tensor a,
    torch::Tensor b_nk);

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fwht_quant_lastdim",
      &fwht_quant_lastdim_cuda,
      "Fused normalized block-FWHT32 and symmetric quantization (CUDA)");
  module.def(
      "fwht_quant_nchw",
      &fwht_quant_nchw_cuda,
      "Fused NCHW-to-NHWC block-FWHT32 and symmetric quantization (CUDA)");
  module.def(
      "quantize",
      &quantize_cuda,
      "Symmetric per-tensor quantization to int8 storage (CUDA)");
  module.def(
      "im2col_nhwc",
      &im2col_nhwc_cuda,
      "NHWC int8 im2col (CUDA)");
  module.def(
      "scale_bias",
      &scale_bias_cuda,
      "Combine split integer accumulators, scale, and add bias (CUDA)");
  module.def(
      "scale_bias_nchw",
      &scale_bias_nchw_cuda,
      "Combine split convolution accumulators into NCHW float output (CUDA)");
  module.def(
      "cutlass_gemm_int8",
      &cutlass_gemm_int8_cuda,
      "Architecture-selected INT8 GEMM entry point with INT32 output (CUDA)");
  module.def(
      "cutlass_conv2d_nhwc",
      &cutlass_conv2d_nhwc_cuda,
      "Architecture-selected INT8 Conv2d entry point with INT32 NHWC output (CUDA)");
}
