"""Integer inference path for HQ-DM Single-Hadamard layers.

The original HQ-DM module performs both Hadamard transforms and weight unpacking
in floating point on every invocation.  This module keeps the paper's integer
identity exactly (up to the final float scale/bias epilogue):

    Q(X H_norm) @ (W_int H_raw)^T * S_x * S_w / sqrt(32)

``W_int H_raw`` needs more than eight bits.  We split it losslessly as
``low + 128 * high`` and execute one or two INT8 matrix operations.  The
activation stays at the checkpoint's configured range (for example A4 is
stored as int8 values in [-7, 7]); int8 is only the physical container.  The
architecture registry reports whether those operations are native CUTLASS
TensorOp kernels or an explicit vendor-library fallback.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

try:  # Package import when installed; script import in the HQ-DM source tree.
    from .extension_loader import (
        load_extension as _load_extension,
        load_extension_details,
    )
except ImportError:  # pragma: no cover - exercised by deployment layout
    from extension_loader import (
        load_extension as _load_extension,
        load_extension_details,
    )


_HIGH_MULTIPLIER = 128
_INV_SQRT_32 = 1.0 / math.sqrt(32.0)


def load_extension(verbose: bool = False, device=None):
    """Compatibility wrapper for the v4 architecture-aware loader."""

    return _load_extension(verbose=verbose, device=device)


def _raw_hadamard_32(device: torch.device) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    for _ in range(5):
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, Iterable):
        result = tuple(int(x) for x in value)
        if len(result) != 2:
            raise ValueError(f"expected a pair, got {value!r}")
        return result
    scalar = int(value)
    return scalar, scalar


def _effective_weight(module: nn.Module) -> torch.Tensor:
    """Reproduce SingleH32QuantModule's eval-time effective fake-quant weight."""
    if module.intn_dequantizer is None:
        raise RuntimeError("intn_dequantizer must be attached before conversion")
    base = module.intn_dequantizer(module.weight)
    if module.fwd_func is torch.nn.functional.linear:
        lora = module.loraB.weight @ module.loraA.weight
    elif module.fwd_func is torch.nn.functional.conv2d:
        lora = torch.einsum(
            "or,rijk->oijk",
            module.loraB.weight.squeeze(-1).squeeze(-1),
            module.loraA.weight,
        )
    else:
        raise TypeError(f"unsupported HQ-DM operation: {module.fwd_func}")
    combined = base + lora

    quantizer = module.weight_quantizer
    delta = quantizer.delta.detach()
    zero_point = quantizer.zero_point.detach()
    levels = int(quantizer.n_levels)
    unsigned = torch.round(combined / delta) + zero_point
    unsigned = torch.clamp(unsigned, 0, levels - 1)
    # HQ-DM checkpoint zero-points are integral.  Refuse a silently inexact
    # integer path if a future checkpoint changes that invariant.
    max_fraction = (zero_point - zero_point.round()).abs().max().item()
    if max_fraction > 1e-6:
        raise ValueError(
            f"non-integral weight zero-point (max fractional part {max_fraction})"
        )
    return torch.round(unsigned - zero_point).to(torch.int16)


def _rotate_channels_raw_h32(weight: torch.Tensor) -> torch.Tensor:
    """Return integer W @ H_raw, applying H32 independently per channel block."""
    channels = int(weight.shape[1])
    if channels % 32:
        raise ValueError(f"HQ-DM H32 requires channels divisible by 32, got {channels}")
    hadamard = _raw_hadamard_32(weight.device)

    if weight.dim() == 2:
        output_channels = int(weight.shape[0])
        blocks = weight.reshape(output_channels, channels // 32, 32).float()
        rotated = torch.matmul(blocks, hadamard)
        return rotated.round().to(torch.int16).reshape(output_channels, channels)

    if weight.dim() == 4:
        output_channels, _, kernel_h, kernel_w = map(int, weight.shape)
        nhwc_filter = weight.permute(0, 2, 3, 1).contiguous()
        blocks = nhwc_filter.reshape(
            output_channels * kernel_h * kernel_w, channels // 32, 32
        ).float()
        rotated = torch.matmul(blocks, hadamard).round().to(torch.int16)
        return (
            rotated.reshape(output_channels, kernel_h, kernel_w, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    raise ValueError(f"expected a 2-D or 4-D weight, got {tuple(weight.shape)}")


def _split_int16_exact(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Losslessly split int16 values into signed-int8 low/high matrices."""
    weight_i32 = weight.to(torch.int32)
    # In trained W4 HQ-DM checkpoints H32-weight sums usually still fit int8.
    # Preserve that common case as a single tensor-core operation.
    if weight_i32.min().item() >= -128 and weight_i32.max().item() <= 127:
        return weight_i32.to(torch.int8), torch.zeros_like(weight_i32, dtype=torch.int8)
    high = torch.div(
        weight_i32 + (_HIGH_MULTIPLIER // 2),
        _HIGH_MULTIPLIER,
        rounding_mode="floor",
    )
    low = weight_i32 - high * _HIGH_MULTIPLIER
    if low.min().item() < -128 or low.max().item() > 127:
        raise AssertionError("low split does not fit int8")
    if high.min().item() < -128 or high.max().item() > 127:
        raise AssertionError("high split does not fit int8")
    reconstructed = low + high * _HIGH_MULTIPLIER
    if not torch.equal(reconstructed, weight_i32):
        raise AssertionError("integer weight split is not exact")
    return low.to(torch.int8), high.to(torch.int8)


class HQIntModule(nn.Module):
    """Drop-in inference replacement for ``SingleH32QuantModule``."""

    def __init__(
        self,
        source: nn.Module,
        qualified_name: str = "",
        channels_last_outputs: bool = False,
    ):
        super().__init__()
        if not source.use_weight_quant or not source.use_act_quant:
            raise ValueError("source module must have weight and activation quantization enabled")
        if not source.weight.is_cuda:
            raise ValueError("convert after moving the checkpoint to CUDA")
        loaded_build = load_extension_details(device=source.weight.device)
        backend = loaded_build.backend
        self.qualified_name = qualified_name
        # Record the dispatch contract used when this module was converted.
        # Forward verifies it again so moving a converted module to a GPU from
        # another architecture cannot silently use the wrong cached extension.
        self.backend_id = backend.backend_id
        self.backend_execution_kind = backend.execution_kind
        self.backend_kernel_status = backend.kernel_status
        self.backend_gemm_status = backend.gemm_status
        self.backend_convolution_status = backend.convolution_status
        self.backend_validation_status = backend.validation_status
        self.backend_capability = backend.capability
        self.backend_direct_spatial_convolution = backend.direct_spatial_convolution
        self.backend_device_index = int(source.weight.device.index)
        # Bypass nn.Module registration: this is a process-local pybind module,
        # not serializable model state. __getstate__ below drops the handle.
        object.__setattr__(self, "_extension_handle", loaded_build.module)
        # The integer core always accumulates in INT32 and applies scales in
        # FP32.  Keep the surrounding UNet's dtype at the module boundary so
        # the non-quantized normalization, attention and residual operations
        # can run in FP16 when the source UNet has been converted to half.
        self.output_dtype = source.loraA.weight.dtype
        self.channels_last_outputs = bool(channels_last_outputs)
        self.is_conv = source.fwd_func is torch.nn.functional.conv2d
        self.input_channels = int(source.ori_shape[1])
        self.output_channels = int(source.ori_shape[0])
        self.qmax = 2 ** (int(source.act_quantizer.n_bits) - 1) - 1
        self.total_steps = int(source.act_quantizer.total_steps)
        self.current_step = self.total_steps - 1

        activation_scales = source.act_quantizer.delta_list.detach().float().contiguous()
        if activation_scales.numel() != self.total_steps:
            raise ValueError(
                f"{qualified_name}: expected {self.total_steps} activation scales, "
                f"got {activation_scales.numel()}"
            )
        self.register_buffer("activation_scales", activation_scales)
        self.register_buffer(
            "weight_scale",
            source.weight_quantizer.delta.detach().float().reshape(-1).contiguous(),
        )
        bias = (
            source.bias.detach().float().reshape(-1).contiguous()
            if source.bias is not None
            else torch.empty(0, dtype=torch.float32, device=source.weight.device)
        )
        self.register_buffer("bias", bias)
        self.register_buffer(
            "empty_int32",
            torch.empty(0, dtype=torch.int32, device=source.weight.device),
            persistent=False,
        )

        with torch.no_grad():
            integer_weight = _effective_weight(source)
            rotated = _rotate_channels_raw_h32(integer_weight)
            reduction_bound = (
                rotated.to(torch.int64)
                .abs()
                .reshape(self.output_channels, -1)
                .sum(dim=1)
                * self.qmax
            )
            max_bound = int(reduction_bound.max().item())
            if max_bound >= torch.iinfo(torch.int32).max:
                raise OverflowError(
                    f"{qualified_name}: conservative int32 accumulation bound "
                    f"{max_bound} is unsafe"
                )
            self.max_accumulator_bound = max_bound
            low, high = _split_int16_exact(rotated)
            self._register_matrix_weights("rotated", low, high)
            if self.is_conv:
                self.register_buffer(
                    "rotated_weight_low_krsc",
                    low.permute(0, 2, 3, 1).contiguous(),
                )
                if torch.count_nonzero(high).item():
                    self.register_buffer(
                        "rotated_weight_high_krsc",
                        high.permute(0, 2, 3, 1).contiguous(),
                    )
                else:
                    self.register_buffer(
                        "rotated_weight_high_krsc",
                        torch.empty(0, dtype=torch.int8, device=low.device),
                    )

            # Linear layers in the original implementation only use Hadamard for
            # rank-2 inputs.  Attention projections commonly receive rank-3
            # tensors, so retain the exact non-Hadamard integer matrix as well.
            if not self.is_conv:
                raw = integer_weight.to(torch.int8)
                self.register_buffer("raw_weight_nk", raw.contiguous())

        if self.is_conv:
            self.kernel_size = tuple(int(x) for x in source.ori_shape[2:])
            self.stride = _pair(source.fwd_kwargs["stride"])
            self.padding = _pair(source.fwd_kwargs["padding"])
            self.dilation = _pair(source.fwd_kwargs["dilation"])
            self.groups = int(source.fwd_kwargs["groups"])
            if self.groups != 1:
                raise ValueError(f"{qualified_name}: grouped convolution is not supported")
        else:
            self.kernel_size = (1, 1)
            self.stride = (1, 1)
            self.padding = (0, 0)
            self.dilation = (1, 1)
            self.groups = 1

    def _register_matrix_weights(
        self, prefix: str, low: torch.Tensor, high: torch.Tensor
    ) -> None:
        low_matrix = low.reshape(low.shape[0], -1).contiguous()
        high_matrix = high.reshape(high.shape[0], -1).contiguous()
        self.register_buffer(f"{prefix}_weight_low_nk", low_matrix)
        if torch.count_nonzero(high).item():
            self.register_buffer(f"{prefix}_weight_high_nk", high_matrix)
            self.uses_high_digit = True
        else:
            self.register_buffer(
                f"{prefix}_weight_high_nk",
                torch.empty(0, dtype=torch.int8, device=low.device),
            )
            self.uses_high_digit = False

    def _extension_for_input(self, input: torch.Tensor):
        if not input.is_cuda:
            raise ValueError("HQIntModule requires a CUDA input")
        input_index = int(input.device.index)
        if input_index != self.backend_device_index:
            capability = tuple(torch.cuda.get_device_capability(input_index))
        else:
            capability = self.backend_capability
        if capability != self.backend_capability:
            raise RuntimeError(
                f"{self.qualified_name or 'HQIntModule'} was converted for "
                f"{self.backend_id} ({self.backend_capability}), but input "
                f"{input.device} has capability {capability}; reconvert after "
                "moving it between GPU architectures"
            )
        if input_index != self.backend_device_index:
            # A move to another GPU of the same capability needs no rebuild.
            self.backend_device_index = input_index
        extension = getattr(self, "_extension_handle", None)
        if extension is None:
            # This occurs after unpickling a converted module in a fresh
            # process. It is intentionally outside the steady-state path.
            loaded_build = load_extension_details(device=input.device)
            if loaded_build.backend.backend_id != self.backend_id:
                raise RuntimeError(
                    f"serialized module backend {self.backend_id} does not match "
                    f"the current device backend {loaded_build.backend.backend_id}"
                )
            extension = loaded_build.module
            object.__setattr__(self, "_extension_handle", extension)
            self.backend_device_index = input_index
        return extension

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.pop("_extension_handle", None)
        return state

    def reset_step(self, step: Optional[int] = None) -> None:
        self.current_step = self.total_steps - 1 if step is None else int(step)
        if self.current_step < 0 or self.current_step >= self.total_steps:
            raise ValueError(f"step {self.current_step} is outside the scale table")

    def _next_scale(self) -> torch.Tensor:
        scale = self.activation_scales[self.current_step]
        self.current_step = (
            self.total_steps - 1 if self.current_step == 0 else self.current_step - 1
        )
        return scale

    def _integer_mm(
        self,
        extension,
        activations: torch.Tensor,
        low_weight: torch.Tensor,
        high_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backend_execution_kind == "vendor_int8_im2col":
            if not hasattr(torch, "_int_mm"):
                raise RuntimeError(
                    f"{self.backend_id} requires torch._int_mm/cuBLASLt, but this "
                    "PyTorch build does not expose it"
                )
            rows = int(activations.shape[0])
            # torch._int_mm's Volta backend rejects M <= 16. Padding zero rows
            # is mathematically exact and harmless on the other vendor
            # fallback targets.
            if rows <= 16:
                padded = torch.nn.functional.pad(
                    activations, (0, 0, 0, 32 - rows), value=0
                )
            else:
                padded = activations

            def dp4a_mm(weight_nk: torch.Tensor) -> torch.Tensor:
                result = torch._int_mm(padded, weight_nk.t())
                return result[:rows] if rows <= 16 else result

            low_accum = dp4a_mm(low_weight)
            high_accum = (
                dp4a_mm(high_weight)
                if high_weight.numel()
                else self.empty_int32
            )
        else:
            low_accum = extension.cutlass_gemm_int8(activations, low_weight)
            high_accum = (
                extension.cutlass_gemm_int8(activations, high_weight)
                if high_weight.numel()
                else self.empty_int32
            )
        return low_accum, high_accum

    def _forward_linear(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        extension = self._extension_for_input(input)
        original_shape = input.shape
        contiguous = input.contiguous()
        if input.dim() == 2:
            quantized = extension.fwht_quant_lastdim(contiguous, scale, self.qmax)
            low_weight = self.rotated_weight_low_nk
            high_weight = self.rotated_weight_high_nk
            normalizer = _INV_SQRT_32
        else:
            quantized = extension.quantize(contiguous, scale, self.qmax)
            low_weight = self.raw_weight_nk
            high_weight = self.empty_int32
            normalizer = 1.0

        activations = quantized.reshape(-1, self.input_channels)
        low_accum, high_accum = self._integer_mm(
            extension, activations, low_weight, high_weight
        )
        output = extension.scale_bias(
            low_accum,
            high_accum,
            scale,
            self.weight_scale,
            self.bias,
            _HIGH_MULTIPLIER,
            normalizer,
        )
        if output.dtype != self.output_dtype:
            output = output.to(dtype=self.output_dtype)
        return output.reshape(*original_shape[:-1], self.output_channels)

    def _forward_conv(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        extension = self._extension_for_input(input)
        if input.dim() != 4:
            raise ValueError(f"convolution expected NCHW input, got {tuple(input.shape)}")
        if self.channels_last_outputs and input.is_contiguous(
            memory_format=torch.channels_last
        ):
            # An NCHW channels-last tensor becomes a contiguous NHWC view, so
            # the FWHT warp reads adjacent channels instead of striding across
            # the spatial plane.
            quantized_nhwc = extension.fwht_quant_lastdim(
                input.permute(0, 2, 3, 1), scale, self.qmax
            )
        else:
            quantized_nhwc = extension.fwht_quant_nchw(
                input.contiguous(), scale, self.qmax
            )
        batch, _, input_h, input_w = map(int, input.shape)
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        output_h = (
            input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        output_w = (
            input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1

        is_pointwise = (
            self.kernel_size == (1, 1)
            and self.stride == (1, 1)
            and self.padding == (0, 0)
            and self.dilation == (1, 1)
        )
        if is_pointwise:
            activations = quantized_nhwc.reshape(-1, self.input_channels)
            low_accum, high_accum = self._integer_mm(
                extension,
                activations,
                self.rotated_weight_low_nk,
                self.rotated_weight_high_nk,
            )
        elif not self.backend_direct_spatial_convolution:
            columns = extension.im2col_nhwc(
                quantized_nhwc,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
            )
            low_accum, high_accum = self._integer_mm(
                extension,
                columns,
                self.rotated_weight_low_nk,
                self.rotated_weight_high_nk,
            )
        else:
            low_accum = extension.cutlass_conv2d_nhwc(
                quantized_nhwc,
                self.rotated_weight_low_krsc,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
            ).reshape(-1, self.output_channels)
            high_accum = (
                extension.cutlass_conv2d_nhwc(
                    quantized_nhwc,
                    self.rotated_weight_high_krsc,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    dilation_h,
                    dilation_w,
                ).reshape(-1, self.output_channels)
                if self.rotated_weight_high_krsc.numel()
                else self.empty_int32
            )
        if self.channels_last_outputs:
            output = extension.scale_bias(
                low_accum,
                high_accum,
                scale,
                self.weight_scale,
                self.bias,
                _HIGH_MULTIPLIER,
                _INV_SQRT_32,
            )
            output = output.reshape(
                batch, output_h, output_w, self.output_channels
            ).permute(0, 3, 1, 2)
        else:
            output = extension.scale_bias_nchw(
                low_accum,
                high_accum,
                scale,
                self.weight_scale,
                self.bias,
                _HIGH_MULTIPLIER,
                _INV_SQRT_32,
                batch,
                output_h,
                output_w,
            )
        if output.dtype != self.output_dtype:
            output = output.to(dtype=self.output_dtype)
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._next_scale()
        if self.is_conv:
            return self._forward_conv(input, scale)
        return self._forward_linear(input, scale)

    def extra_repr(self) -> str:
        kind = "conv2d" if self.is_conv else "linear"
        return (
            f"kind={kind}, in={self.input_channels}, out={self.output_channels}, "
            f"a_bits={int(round(math.log2(self.qmax + 1))) + 1}, "
            f"steps={self.total_steps}, integer_core=int8, "
            f"digits={2 if self.uses_high_digit else 1}, "
            f"backend={self.backend_id}, status={self.backend_kernel_status}"
        )


def replace_single_h_modules(
    model: nn.Module, channels_last_outputs: bool = False
) -> Dict[str, object]:
    """Replace every loaded SingleH32 module and release its fake-quant state."""
    from hqdm.quantization.single_hadamard import SingleH32QuantModule

    counts = {
        "linear": 0,
        "conv2d": 0,
        "one_digit": 0,
        "two_digit": 0,
        "backend": None,
        "kernel_status": None,
        "gemm_status": None,
        "convolution_status": None,
        "validation_status": None,
    }

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, SingleH32QuantModule):
                replacement = HQIntModule(
                    child,
                    qualified,
                    channels_last_outputs=channels_last_outputs,
                )
                if counts["backend"] is None:
                    counts.update(
                        backend=replacement.backend_id,
                        kernel_status=replacement.backend_kernel_status,
                        gemm_status=replacement.backend_gemm_status,
                        convolution_status=replacement.backend_convolution_status,
                    )
                    counts["validation_status"] = replacement.backend_validation_status
                elif counts["backend"] != replacement.backend_id:
                    raise RuntimeError(
                        "one model cannot mix HQIntModule backends during conversion: "
                        f"{counts['backend']} and {replacement.backend_id}"
                    )
                setattr(parent, child_name, replacement)
                counts["conv2d" if replacement.is_conv else "linear"] += 1
                counts["two_digit" if replacement.uses_high_digit else "one_digit"] += 1
            else:
                visit(child, qualified)

    visit(model)
    return counts


def reset_timestep_scales(model: nn.Module, step: Optional[int] = None) -> None:
    for module in model.modules():
        if isinstance(module, HQIntModule):
            module.reset_step(step)
