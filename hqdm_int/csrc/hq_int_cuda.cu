#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr float kInvSqrt32 = 0.1767766952966369f;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) \
  TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

inline int blocks_for(int64_t count, int threads = kThreads) {
  TORCH_CHECK(
      count <= static_cast<int64_t>(std::numeric_limits<int>::max()) * threads,
      "launch is too large");
  return static_cast<int>((count + threads - 1) / threads);
}

template <typename scalar_t, bool kNchwInput>
__global__ void fwht32_quant_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale,
    int64_t rows,
    int channels,
    int spatial,
    int qmax) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t warp =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int channel_blocks = channels / kWarpSize;
  const int64_t total_warps = rows * channel_blocks;
  if (warp >= total_warps) {
    return;
  }

  const int64_t row = warp / channel_blocks;
  const int channel = static_cast<int>(warp % channel_blocks) * kWarpSize + lane;
  int64_t input_index;
  if constexpr (kNchwInput) {
    const int64_t batch = row / spatial;
    const int64_t pixel = row - batch * spatial;
    input_index = (batch * channels + channel) * spatial + pixel;
  } else {
    input_index = row * channels + channel;
  }

  float value = static_cast<float>(input[input_index]);
#pragma unroll
  for (int mask = 1; mask < kWarpSize; mask <<= 1) {
    const float peer = __shfl_xor_sync(0xffffffffu, value, mask);
    value = (lane & mask) ? (peer - value) : (value + peer);
  }

  const float inv_scale = 1.0f / scale[0];
  float quantized = nearbyintf(value * kInvSqrt32 * inv_scale);
  quantized = fminf(static_cast<float>(qmax), quantized);
  quantized = fmaxf(static_cast<float>(-qmax), quantized);
  output[row * channels + channel] = static_cast<int8_t>(quantized);
}

template <typename scalar_t>
__global__ void quantize_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale,
    int64_t count,
    int qmax) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  float quantized = nearbyintf(static_cast<float>(input[index]) / scale[0]);
  quantized = fminf(static_cast<float>(qmax), quantized);
  quantized = fmaxf(static_cast<float>(-qmax), quantized);
  output[index] = static_cast<int8_t>(quantized);
}

__global__ void im2col_nhwc_kernel(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int batch,
    int input_h,
    int input_w,
    int channels,
    int output_h,
    int output_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const int kernel_area = kernel_h * kernel_w;
  const int reduction = channels * kernel_area;
  const int64_t row = index / reduction;
  const int k = static_cast<int>(index - row * reduction);
  const int channel = k / kernel_area;
  const int kernel_offset = k - channel * kernel_area;
  const int kernel_y = kernel_offset / kernel_w;
  const int kernel_x = kernel_offset - kernel_y * kernel_w;

  const int output_spatial = output_h * output_w;
  const int n = static_cast<int>(row / output_spatial);
  const int output_pixel = static_cast<int>(row - n * output_spatial);
  const int output_y = output_pixel / output_w;
  const int output_x = output_pixel - output_y * output_w;
  const int input_y = output_y * stride_h - pad_h + kernel_y * dilation_h;
  const int input_x = output_x * stride_w - pad_w + kernel_x * dilation_w;

  int8_t value = 0;
  if (input_y >= 0 && input_y < input_h && input_x >= 0 && input_x < input_w) {
    const int64_t input_index =
        ((static_cast<int64_t>(n) * input_h + input_y) * input_w + input_x) *
            channels +
        channel;
    value = input[input_index];
  }
  output[index] = value;
}

__global__ void scale_bias_kernel(
    const int32_t* __restrict__ accum_low,
    const int32_t* __restrict__ accum_high,
    float* __restrict__ output,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    int64_t count,
    int columns,
    int high_multiplier,
    float normalizer,
    bool has_high,
    bool has_bias) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int column = static_cast<int>(index % columns);
  int64_t accumulator = accum_low[index];
  if (has_high) {
    accumulator += static_cast<int64_t>(high_multiplier) * accum_high[index];
  }
  float value = static_cast<float>(accumulator) * activation_scale[0] *
      weight_scale[column] * normalizer;
  if (has_bias) {
    value += bias[column];
  }
  output[index] = value;
}

__global__ void scale_bias_nchw_kernel(
    const int32_t* __restrict__ accum_low,
    const int32_t* __restrict__ accum_high,
    float* __restrict__ output,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    int64_t count,
    int columns,
    int spatial,
    int high_multiplier,
    float normalizer,
    bool has_high,
    bool has_bias) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int column = static_cast<int>(index % columns);
  const int64_t row = index / columns;
  const int64_t n = row / spatial;
  const int pixel = static_cast<int>(row - n * spatial);
  int64_t accumulator = accum_low[index];
  if (has_high) {
    accumulator += static_cast<int64_t>(high_multiplier) * accum_high[index];
  }
  float value = static_cast<float>(accumulator) * activation_scale[0] *
      weight_scale[column] * normalizer;
  if (has_bias) {
    value += bias[column];
  }
  output[(n * columns + column) * spatial + pixel] = value;
}

void check_common_quant_inputs(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    int64_t qmax) {
  CHECK_CUDA(input);
  CHECK_CUDA(scale);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(scale);
  CHECK_FLOAT32(scale);
  TORCH_CHECK(scale.numel() == 1, "scale must have exactly one element");
  TORCH_CHECK(qmax > 0 && qmax <= 127, "qmax must be in [1, 127]");
  TORCH_CHECK(input.device() == scale.device(), "input and scale devices differ");
}

void check_epilogue_inputs(
    const torch::Tensor& accum_low,
    const torch::Tensor& accum_high,
    const torch::Tensor& activation_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& bias) {
  CHECK_CUDA(accum_low);
  CHECK_CUDA(accum_high);
  CHECK_CUDA(activation_scale);
  CHECK_CUDA(weight_scale);
  CHECK_CUDA(bias);
  CHECK_CONTIGUOUS(accum_low);
  CHECK_CONTIGUOUS(accum_high);
  CHECK_CONTIGUOUS(activation_scale);
  CHECK_CONTIGUOUS(weight_scale);
  CHECK_CONTIGUOUS(bias);
  TORCH_CHECK(accum_low.scalar_type() == at::kInt, "accum_low must be int32");
  TORCH_CHECK(
      accum_high.numel() == 0 || accum_high.scalar_type() == at::kInt,
      "accum_high must be empty or int32");
  CHECK_FLOAT32(activation_scale);
  CHECK_FLOAT32(weight_scale);
  TORCH_CHECK(
      bias.numel() == 0 || bias.scalar_type() == at::kFloat,
      "bias must be empty or float32");
  TORCH_CHECK(accum_low.dim() == 2, "accum_low must be a matrix");
  TORCH_CHECK(
      accum_high.numel() == 0 || accum_high.sizes() == accum_low.sizes(),
      "accumulator matrix shapes differ");
  TORCH_CHECK(activation_scale.numel() == 1, "activation_scale must be scalar");
  TORCH_CHECK(
      weight_scale.numel() == accum_low.size(1),
      "weight_scale length must equal output columns");
  TORCH_CHECK(
      bias.numel() == 0 || bias.numel() == accum_low.size(1),
      "bias length must equal output columns");
}

}  // namespace

torch::Tensor fwht_quant_lastdim_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t qmax) {
  check_common_quant_inputs(input, scale, qmax);
  TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
  const int channels = static_cast<int>(input.size(-1));
  TORCH_CHECK(channels % kWarpSize == 0, "last dimension must be divisible by 32");
  const int64_t rows = input.numel() / channels;
  auto output = torch::empty(input.sizes(), input.options().dtype(at::kChar));
  const int64_t warps = rows * (channels / kWarpSize);
  const int blocks = blocks_for(warps, kThreads / kWarpSize);
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fwht_quant_lastdim_cuda",
      [&] {
        fwht32_quant_kernel<scalar_t, false><<<blocks, kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            rows,
            channels,
            1,
            static_cast<int>(qmax));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fwht_quant_nchw_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t qmax) {
  check_common_quant_inputs(input, scale, qmax);
  TORCH_CHECK(input.dim() == 4, "input must be NCHW");
  const int batch = static_cast<int>(input.size(0));
  const int channels = static_cast<int>(input.size(1));
  const int height = static_cast<int>(input.size(2));
  const int width = static_cast<int>(input.size(3));
  TORCH_CHECK(channels % kWarpSize == 0, "channel dimension must be divisible by 32");
  const int spatial = height * width;
  const int64_t rows = static_cast<int64_t>(batch) * spatial;
  auto output = torch::empty(
      {batch, height, width, channels}, input.options().dtype(at::kChar));
  const int64_t warps = rows * (channels / kWarpSize);
  const int blocks = blocks_for(warps, kThreads / kWarpSize);
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fwht_quant_nchw_cuda",
      [&] {
        fwht32_quant_kernel<scalar_t, true><<<blocks, kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            rows,
            channels,
            spatial,
            static_cast<int>(qmax));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor quantize_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t qmax) {
  check_common_quant_inputs(input, scale, qmax);
  auto output = torch::empty(input.sizes(), input.options().dtype(at::kChar));
  const int blocks = blocks_for(input.numel());
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "quantize_cuda",
      [&] {
        quantize_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            input.numel(),
            static_cast<int>(qmax));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor im2col_nhwc_cuda(
    torch::Tensor input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  TORCH_CHECK(input.scalar_type() == at::kChar, "input must be int8");
  TORCH_CHECK(input.dim() == 4, "input must be NHWC");
  TORCH_CHECK(
      kernel_h > 0 && kernel_w > 0 && stride_h > 0 && stride_w > 0 &&
          dilation_h > 0 && dilation_w > 0,
      "kernel, stride, and dilation values must be positive");
  const int batch = static_cast<int>(input.size(0));
  const int input_h = static_cast<int>(input.size(1));
  const int input_w = static_cast<int>(input.size(2));
  const int channels = static_cast<int>(input.size(3));
  const int output_h = static_cast<int>(
      (input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1);
  const int output_w = static_cast<int>(
      (input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1);
  TORCH_CHECK(output_h > 0 && output_w > 0, "invalid output shape");
  const int64_t rows = static_cast<int64_t>(batch) * output_h * output_w;
  const int64_t reduction = static_cast<int64_t>(channels) * kernel_h * kernel_w;
  auto output = torch::empty({rows, reduction}, input.options());
  const int64_t count = rows * reduction;
  const int blocks = blocks_for(count);
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  im2col_nhwc_kernel<<<blocks, kThreads, 0, stream>>>(
      input.data_ptr<int8_t>(),
      output.data_ptr<int8_t>(),
      batch,
      input_h,
      input_w,
      channels,
      output_h,
      output_w,
      static_cast<int>(kernel_h),
      static_cast<int>(kernel_w),
      static_cast<int>(stride_h),
      static_cast<int>(stride_w),
      static_cast<int>(pad_h),
      static_cast<int>(pad_w),
      static_cast<int>(dilation_h),
      static_cast<int>(dilation_w),
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor scale_bias_cuda(
    torch::Tensor accum_low,
    torch::Tensor accum_high,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    int64_t high_multiplier,
    double normalizer) {
  check_epilogue_inputs(
      accum_low, accum_high, activation_scale, weight_scale, bias);
  auto output = torch::empty(accum_low.sizes(), accum_low.options().dtype(at::kFloat));
  const int64_t count = accum_low.numel();
  const int blocks = blocks_for(count);
  const c10::cuda::CUDAGuard device_guard(accum_low.device());
  const auto stream = at::cuda::getCurrentCUDAStream(accum_low.device().index());
  scale_bias_kernel<<<blocks, kThreads, 0, stream>>>(
      accum_low.data_ptr<int32_t>(),
      accum_high.numel() ? accum_high.data_ptr<int32_t>() : nullptr,
      output.data_ptr<float>(),
      activation_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      bias.numel() ? bias.data_ptr<float>() : nullptr,
      count,
      static_cast<int>(accum_low.size(1)),
      static_cast<int>(high_multiplier),
      static_cast<float>(normalizer),
      accum_high.numel() != 0,
      bias.numel() != 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

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
    int64_t output_w) {
  check_epilogue_inputs(
      accum_low, accum_high, activation_scale, weight_scale, bias);
  const int64_t rows = batch * output_h * output_w;
  TORCH_CHECK(rows == accum_low.size(0), "convolution row count does not match output shape");
  const int64_t columns = accum_low.size(1);
  auto output = torch::empty(
      {batch, columns, output_h, output_w},
      accum_low.options().dtype(at::kFloat));
  const int64_t count = accum_low.numel();
  const int blocks = blocks_for(count);
  const c10::cuda::CUDAGuard device_guard(accum_low.device());
  const auto stream = at::cuda::getCurrentCUDAStream(accum_low.device().index());
  scale_bias_nchw_kernel<<<blocks, kThreads, 0, stream>>>(
      accum_low.data_ptr<int32_t>(),
      accum_high.numel() ? accum_high.data_ptr<int32_t>() : nullptr,
      output.data_ptr<float>(),
      activation_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      bias.numel() ? bias.data_ptr<float>() : nullptr,
      count,
      static_cast<int>(columns),
      static_cast<int>(output_h * output_w),
      static_cast<int>(high_multiplier),
      static_cast<float>(normalizer),
      accum_high.numel() != 0,
      bias.numel() != 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
