#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

namespace {

using ElementInput = int8_t;
using ElementFilter = int8_t;
using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;
using Layout = cutlass::layout::TensorNHWC;

// Keep the epilogue arithmetic integral.  In particular, using float here can
// round an otherwise exact INT32 accumulator once its magnitude exceeds 2^24.
using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute>;

using Conv2dFpropKernel =
    typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInput,
        Layout,
        ElementFilter,
        Layout,
        ElementOutput,
        Layout,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        OutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided,
        16,
        16>::Kernel;

using Conv2dFprop =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

void check_cutlass(cutlass::Status status, const char* operation) {
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS ",
      operation,
      " failed with status ",
      static_cast<int>(status),
      " (",
      cutlass::cutlassGetStatusString(status),
      ")");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(
      value >= 0 && value <= std::numeric_limits<int>::max(),
      name,
      " does not fit in a non-negative 32-bit integer: ",
      value);
  return static_cast<int>(value);
}

}  // namespace

// q must be a contiguous NHWC INT8 tensor.  filter must be a contiguous KRSC
// INT8 tensor, represented with CUTLASS TensorNHWC because its packed order is
// likewise the last dimension contiguous.  The returned tensor is contiguous
// NPQK (NHWC) INT32.
torch::Tensor cutlass_conv2d_nhwc_cuda(
    torch::Tensor q,
    torch::Tensor filter,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(filter.is_cuda(), "filter must be a CUDA tensor");
  TORCH_CHECK(
      q.device() == filter.device(),
      "q and filter must be on the same CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kChar, "q must have dtype torch.int8");
  TORCH_CHECK(
      filter.scalar_type() == at::kChar,
      "filter must have dtype torch.int8");
  TORCH_CHECK(q.dim() == 4, "q must have shape [N, H, W, C]");
  TORCH_CHECK(
      filter.dim() == 4,
      "filter must have shape [K, R, S, C] (KRSC)");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous NHWC storage");
  TORCH_CHECK(
      filter.is_contiguous(),
      "filter must be contiguous KRSC storage");
  TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "padding must be non-negative");
  TORCH_CHECK(
      dilation_h > 0 && dilation_w > 0,
      "dilation must be positive");

  const int64_t n64 = q.size(0);
  const int64_t h64 = q.size(1);
  const int64_t w64 = q.size(2);
  const int64_t c64 = q.size(3);
  const int64_t k64 = filter.size(0);
  const int64_t r64 = filter.size(1);
  const int64_t s64 = filter.size(2);

  TORCH_CHECK(
      filter.size(3) == c64,
      "filter Cin (",
      filter.size(3),
      ") must equal q channels (",
      c64,
      ")");
  TORCH_CHECK(
      n64 > 0 && h64 > 0 && w64 > 0 && c64 > 0 && k64 > 0 && r64 > 0 &&
          s64 > 0,
      "q and filter dimensions must all be non-zero");

  const int n = checked_int(n64, "N");
  const int h = checked_int(h64, "H");
  const int w = checked_int(w64, "W");
  const int c = checked_int(c64, "C");
  const int k = checked_int(k64, "K");
  const int r = checked_int(r64, "R");
  const int s = checked_int(s64, "S");
  const int stride_h_int = checked_int(stride_h, "stride_h");
  const int stride_w_int = checked_int(stride_w, "stride_w");
  const int pad_h_int = checked_int(pad_h, "pad_h");
  const int pad_w_int = checked_int(pad_w, "pad_w");
  const int dilation_h_int = checked_int(dilation_h, "dilation_h");
  const int dilation_w_int = checked_int(dilation_w, "dilation_w");

  // All factors have been bounded by INT_MAX above, so these products fit in
  // signed int64_t even for adversarial shapes.
  const int64_t effective_r =
      static_cast<int64_t>(dilation_h_int) * (r - 1) + 1;
  const int64_t effective_s =
      static_cast<int64_t>(dilation_w_int) * (s - 1) + 1;
  const int64_t padded_h = h64 + 2 * pad_h;
  const int64_t padded_w = w64 + 2 * pad_w;
  TORCH_CHECK(
      padded_h >= effective_r && padded_w >= effective_s,
      "filter does not fit the padded input: input=",
      h64,
      "x",
      w64,
      ", filter=",
      r64,
      "x",
      s64,
      ", padding=",
      pad_h,
      "x",
      pad_w,
      ", dilation=",
      dilation_h,
      "x",
      dilation_w);

  const int64_t p64 = (padded_h - effective_r) / stride_h + 1;
  const int64_t q_out64 = (padded_w - effective_s) / stride_w + 1;

  const int p = checked_int(p64, "P");
  const int q_out = checked_int(q_out64, "Q");

  const c10::cuda::CUDAGuard device_guard(q.device());
  torch::Tensor output = torch::empty(
      {n64, p64, q_out64, k64},
      q.options().dtype(torch::kInt32));

  const cutlass::Tensor4DCoord input_extent(n, h, w, c);
  const cutlass::Tensor4DCoord filter_extent(k, r, s, c);
  const cutlass::Tensor4DCoord output_extent(n, p, q_out, k);
  const cutlass::Tensor4DCoord padding(
      pad_h_int, pad_h_int, pad_w_int, pad_w_int);
  const cutlass::MatrixCoord stride(stride_h_int, stride_w_int);
  const cutlass::MatrixCoord dilation(dilation_h_int, dilation_w_int);

  const cutlass::conv::Conv2dProblemSize problem_size(
      input_extent,
      filter_extent,
      padding,
      stride,
      dilation,
      output_extent,
      cutlass::conv::Mode::kCrossCorrelation,
      1);

  const Layout input_layout = Layout::packed(input_extent);
  const Layout filter_layout = Layout::packed(filter_extent);
  const Layout output_layout = Layout::packed(output_extent);

  typename Conv2dFprop::Arguments arguments{
      problem_size,
      {q.data_ptr<ElementInput>(), input_layout},
      {filter.data_ptr<ElementFilter>(), filter_layout},
      {output.data_ptr<ElementOutput>(), output_layout},
      {output.data_ptr<ElementOutput>(), output_layout},
      typename OutputOp::Params(ElementCompute(1), ElementCompute(0))};

  Conv2dFprop operation;
  check_cutlass(operation.can_implement(arguments), "can_implement");

  const size_t workspace_size = operation.get_workspace_size(arguments);
  torch::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    TORCH_CHECK(
        workspace_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "CUTLASS workspace is too large: ",
        workspace_size,
        " bytes");
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        q.options().dtype(torch::kUInt8));
    workspace_ptr = workspace.data_ptr();
  }

  const auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
  check_cutlass(
      operation.initialize(arguments, workspace_ptr, stream.stream()),
      "initialize");
  check_cutlass(operation(stream.stream()), "operator()");
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}
