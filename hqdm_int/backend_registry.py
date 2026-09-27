"""Pure-Python architecture registry for the HQ-DM integer extension.

This module deliberately does not import PyTorch.  Dispatch policy can
therefore be tested on a CPU-only development machine before a CUDA build is
attempted.  Kernel status is explicit: ``native`` means the implementation is
designed wholly for that ISA family, ``hybrid`` selects native and compatibility
paths by policy, and ``compat``/``fallback`` must never be reported as native.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union


Capability = Tuple[int, int]
CapabilityLike = Union[str, int, Iterable[int], Capability]

# Bump SOURCE_ABI whenever the binding contract, compilation policy, or source
# selection changes.  The loader also appends a content hash, so an edited
# kernel can never reuse a stale torch extension from the same Python process.
EXTENSION_ABI = "v4"
SOURCE_ABI = "20260927r1"


class UnsupportedArchitecture(RuntimeError):
    """Raised before compilation when no reviewed backend exists."""


class ToolchainMismatch(RuntimeError):
    """Raised when CUDA or CUTLASS is too old for the selected backend."""


class BackendUnavailable(RuntimeError):
    """Raised for a known architecture whose reviewed kernel is not present."""


@dataclass(frozen=True)
class BackendSpec:
    capability: Capability
    backend_id: str
    execution_kind: str
    kernel_status: str
    gemm_status: str
    convolution_status: str
    validation_status: str
    torch_arch_list: str
    extension_sources: Tuple[str, ...]
    compile_defines: Tuple[str, ...]
    requires_cutlass: bool
    minimum_cuda: str
    minimum_cutlass: Optional[str]
    uses_int8_tensor_cores: Optional[bool]
    direct_spatial_convolution: bool
    description: str
    available: bool = True

    @property
    def sm(self) -> int:
        return 10 * self.capability[0] + self.capability[1]

    def as_manifest_entry(self) -> Dict[str, object]:
        result = asdict(self)
        result["capability"] = list(self.capability)
        result["extension_sources"] = list(self.extension_sources)
        result["compile_defines"] = list(self.compile_defines)
        result["sm"] = self.sm
        return result


@dataclass(frozen=True)
class BuildPlan:
    backend: BackendSpec
    extension_name: str
    relative_sources: Tuple[str, ...]
    extra_cflags: Tuple[str, ...]
    extra_cuda_cflags: Tuple[str, ...]
    torch_arch_list: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend.as_manifest_entry(),
            "extension_name": self.extension_name,
            "relative_sources": list(self.relative_sources),
            "extra_cflags": list(self.extra_cflags),
            "extra_cuda_cflags": list(self.extra_cuda_cflags),
            "torch_arch_list": self.torch_arch_list,
        }


_COMMON_SOURCES = (
    "csrc/hq_int.cpp",
    "csrc/hq_int_cuda.cu",
)
_CUTLASS_SM80_SOURCES = _COMMON_SOURCES + (
    "csrc/cutlass_gemm.cu",
    "csrc/cutlass_conv.cu",
)


BACKENDS: Mapping[Capability, BackendSpec] = {
    (7, 0): BackendSpec(
        capability=(7, 0),
        backend_id="sm70_dp4a_im2col",
        execution_kind="vendor_int8_im2col",
        kernel_status="fallback",
        gemm_status="vendor-dp4a-fallback",
        convolution_status="im2col-plus-vendor-gemm-fallback",
        validation_status="validated-on-v100",
        torch_arch_list="7.0",
        extension_sources=_COMMON_SOURCES,
        compile_defines=("HQDM_NO_CUTLASS_BINDINGS=1",),
        requires_cutlass=False,
        minimum_cuda="11.8",
        minimum_cutlass=None,
        uses_int8_tensor_cores=False,
        direct_spatial_convolution=False,
        description=(
            "Volta CUDA-core INT8/DP4A fallback. Fused FWHT and epilogues are "
            "custom CUDA; GEMM uses torch._int_mm and spatial convolution "
            "materializes im2col. This is not a Tensor Core INT8 backend."
        ),
    ),
    (8, 0): BackendSpec(
        capability=(8, 0),
        backend_id="sm80_cutlass_tensorop",
        execution_kind="cutlass_sm80_tensorop",
        kernel_status="native",
        gemm_status="native-cutlass-sm80-tensorop",
        convolution_status="native-cutlass-sm80-implicit-gemm",
        validation_status="validated-on-a800",
        torch_arch_list="8.0",
        extension_sources=_CUTLASS_SM80_SOURCES,
        compile_defines=(),
        requires_cutlass=True,
        minimum_cuda="11.8",
        minimum_cutlass="3.5.1",
        uses_int8_tensor_cores=True,
        direct_spatial_convolution=True,
        description="Ampere SM80 CUTLASS INT8 TensorOp GEMM and implicit-GEMM convolution.",
    ),
    (9, 0): BackendSpec(
        capability=(9, 0),
        backend_id="sm90_cutlass_sm80_compat",
        execution_kind="cutlass_sm80_tensorop",
        kernel_status="compat",
        gemm_status="ampere-mma-compat",
        convolution_status="ampere-implicit-gemm-compat",
        validation_status="validated-on-h200",
        torch_arch_list="9.0",
        extension_sources=_CUTLASS_SM80_SOURCES,
        compile_defines=(),
        requires_cutlass=True,
        minimum_cuda="12.0",
        minimum_cutlass="3.5.1",
        uses_int8_tensor_cores=True,
        direct_spatial_convolution=True,
        description=(
            "Hopper-native cubin containing the reviewed SM80-style MMA "
            "compatibility algorithm."
        ),
    ),
    (12, 0): BackendSpec(
        capability=(12, 0),
        backend_id="sm120_cutlass_sm80_compat",
        execution_kind="cutlass_sm80_tensorop",
        kernel_status="compat",
        gemm_status="ampere-mma-compat",
        convolution_status="ampere-implicit-gemm-compat",
        validation_status="validated-on-rtx5090",
        torch_arch_list="12.0",
        extension_sources=_CUTLASS_SM80_SOURCES,
        compile_defines=(),
        requires_cutlass=True,
        minimum_cuda="13.0",
        minimum_cutlass="4.2.0",
        uses_int8_tensor_cores=True,
        direct_spatial_convolution=True,
        description=(
            "Blackwell-native cubin containing the reviewed SM80-style MMA "
            "compatibility algorithm. It requires CUDA 13/CUTLASS 4.2.0."
        ),
    ),
}


def normalize_capability(value: CapabilityLike) -> Capability:
    if isinstance(value, str):
        text = value.strip().lower().replace("compute_", "").replace("sm_", "")
        if text.startswith("sm"):
            text = text[2:]
        if "." in text:
            parts = text.split(".")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(f"invalid CUDA capability {value!r}")
            return int(parts[0]), int(parts[1])
        if text.isdigit() and len(text) >= 2:
            number = int(text)
            return number // 10, number % 10
        raise ValueError(f"invalid CUDA capability {value!r}")
    if isinstance(value, int):
        if value < 10:
            raise ValueError("integer capabilities must use SM notation, e.g. 80")
        return value // 10, value % 10
    parts = tuple(int(item) for item in value)
    if len(parts) != 2 or min(parts) < 0:
        raise ValueError(f"invalid CUDA capability {value!r}")
    return parts[0], parts[1]


def select_backend(
    capability: CapabilityLike, *, allow_unavailable: bool = False
) -> BackendSpec:
    normalized = normalize_capability(capability)
    try:
        backend = BACKENDS[normalized]
    except KeyError as error:
        supported = ", ".join(f"SM{10 * major + minor}" for major, minor in BACKENDS)
        raise UnsupportedArchitecture(
            f"HQ-DM v4 has no reviewed backend for SM{10 * normalized[0] + normalized[1]}; "
            f"supported targets are {supported}. Refusing an unvalidated fallback."
        ) from error
    if not backend.available and not allow_unavailable:
        raise BackendUnavailable(
            f"{backend.backend_id} is a registered architecture hook but has no "
            f"reviewed implementation: {backend.description}"
        )
    return backend


def _version_tuple(value: Optional[str]) -> Tuple[int, ...]:
    if value is None:
        return ()
    numbers = re.findall(r"\d+", str(value))
    if not numbers:
        raise ValueError(f"invalid version {value!r}")
    return tuple(int(number) for number in numbers[:3])


def version_at_least(actual: str, minimum: str) -> bool:
    left = _version_tuple(actual)
    right = _version_tuple(minimum)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def validate_toolchain(
    backend: BackendSpec,
    cuda_version: str,
    cutlass_version: Optional[str] = None,
) -> None:
    if not version_at_least(cuda_version, backend.minimum_cuda):
        raise ToolchainMismatch(
            f"{backend.backend_id} requires CUDA >= {backend.minimum_cuda}, "
            f"but PyTorch reports CUDA {cuda_version}"
        )
    if backend.requires_cutlass:
        if cutlass_version is None:
            raise ToolchainMismatch(
                f"{backend.backend_id} requires CUTLASS >= {backend.minimum_cutlass}; "
                "the header version could not be determined"
            )
        if not version_at_least(cutlass_version, str(backend.minimum_cutlass)):
            raise ToolchainMismatch(
                f"{backend.backend_id} requires CUTLASS >= {backend.minimum_cutlass}, "
                f"but found {cutlass_version}"
            )


def _tag(value: Optional[str], missing: str) -> str:
    if value is None:
        return missing
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()
    return cleaned or missing


def make_extension_name(
    backend: BackendSpec,
    cuda_version: str,
    cutlass_version: Optional[str] = None,
) -> str:
    cutlass_tag = (
        f"cutlass{_tag(cutlass_version, 'unknown')}"
        if backend.requires_cutlass
        else "nocutlass"
    )
    return (
        f"hqdm_int_cuda_{EXTENSION_ABI}_{SOURCE_ABI}_{backend.backend_id}_"
        f"cu{_tag(cuda_version, 'unknown')}_{cutlass_tag}"
    )


def make_build_plan(
    capability: CapabilityLike,
    cuda_version: str,
    cutlass_version: Optional[str] = None,
    *,
    validate_versions: bool = True,
) -> BuildPlan:
    backend = select_backend(capability)
    if validate_versions:
        validate_toolchain(backend, cuda_version, cutlass_version)
    definitions = tuple(f"-D{item}" for item in backend.compile_defines)
    return BuildPlan(
        backend=backend,
        extension_name=make_extension_name(backend, cuda_version, cutlass_version),
        relative_sources=backend.extension_sources,
        extra_cflags=("-O3",) + definitions,
        extra_cuda_cflags=("-O3", "--threads=2") + definitions,
        torch_arch_list=backend.torch_arch_list,
    )


def build_manifest() -> Dict[str, object]:
    return {
        "extension_abi": EXTENSION_ABI,
        "source_abi": SOURCE_ABI,
        "targets": [BACKENDS[key].as_manifest_entry() for key in sorted(BACKENDS)],
    }


def main() -> None:
    print(json.dumps(build_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
