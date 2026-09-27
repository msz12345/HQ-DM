#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

namespace {

using ElementA = int8_t;
using ElementB = int8_t;
using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    4,
    ElementAccumulator,
    ElementAccumulator>;

// HQ-DM uses both small-M attention/timestep projections and the larger matrix
// shapes produced by eligible 1x1 convolutions. This reviewed tile is shared by
// those fixed shape families.
using Gemm = cutlass::gemm::device::Gemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 128, 64>,
    cutlass::gemm::GemmShape<16, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    OutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    16,
    16,
    false,
    cutlass::arch::OpMultiplyAddSaturate>;

void check_cutlass(cutlass::Status status, const char* operation) {
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS GEMM ",
      operation,
      " failed with status ",
      static_cast<int>(status),
      " (",
      cutlass::cutlassGetStatusString(status),
      ")");
}

int checked_positive_int(int64_t value, const char* name) {
  TORCH_CHECK(
      value > 0 && value <= std::numeric_limits<int>::max(),
      name,
      " must fit in a positive 32-bit integer, got ",
      value);
  return static_cast<int>(value);
}

}  // namespace

// a is row-major [M, K].  b_nk is physically row-major [N, K], which is the
// same byte layout as the logical column-major [K, N] operand CUTLASS consumes.
torch::Tensor cutlass_gemm_int8_cuda(
    torch::Tensor a,
    torch::Tensor b_nk) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b_nk.is_cuda(), "b_nk must be a CUDA tensor");
  TORCH_CHECK(a.device() == b_nk.device(), "GEMM operands must share a device");
  TORCH_CHECK(a.scalar_type() == at::kChar, "a must have dtype torch.int8");
  TORCH_CHECK(
      b_nk.scalar_type() == at::kChar,
      "b_nk must have dtype torch.int8");
  TORCH_CHECK(a.dim() == 2, "a must have shape [M, K]");
  TORCH_CHECK(b_nk.dim() == 2, "b_nk must have shape [N, K]");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous row-major");
  TORCH_CHECK(b_nk.is_contiguous(), "b_nk must be contiguous [N, K]");
  TORCH_CHECK(
      a.size(1) == b_nk.size(1),
      "GEMM K mismatch: ",
      a.size(1),
      " vs ",
      b_nk.size(1));

  const int m = checked_positive_int(a.size(0), "M");
  const int n = checked_positive_int(b_nk.size(0), "N");
  const int k = checked_positive_int(a.size(1), "K");
  TORCH_CHECK(k % 16 == 0, "K must be divisible by the 16-element alignment");
  TORCH_CHECK(n % 4 == 0, "N must be divisible by the output vector width 4");

  const c10::cuda::CUDAGuard device_guard(a.device());
  auto output = torch::empty({m, n}, a.options().dtype(at::kInt));
  typename Gemm::Arguments arguments(
      {m, n, k},
      {a.data_ptr<ElementA>(), k},
      {b_nk.data_ptr<ElementB>(), k},
      {output.data_ptr<ElementOutput>(), n},
      {output.data_ptr<ElementOutput>(), n},
      typename OutputOp::Params(ElementAccumulator(1), ElementAccumulator(0)));

  Gemm operation;
  check_cutlass(operation.can_implement(arguments), "can_implement");

  const size_t workspace_size = operation.get_workspace_size(arguments);
  torch::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    TORCH_CHECK(
        workspace_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "CUTLASS GEMM workspace is too large");
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        a.options().dtype(torch::kUInt8));
    workspace_ptr = workspace.data_ptr();
  }

  const auto stream = at::cuda::getCurrentCUDAStream(a.device().index());
  check_cutlass(
      operation.initialize(arguments, workspace_ptr, stream.stream()),
      "initialize");
  check_cutlass(operation(stream.stream()), "operator()");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
