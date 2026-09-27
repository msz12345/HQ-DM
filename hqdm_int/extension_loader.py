"""Architecture-aware builder/loader for the HQ-DM CUDA extension.

The dispatch policy lives in :mod:`backend_registry` and is intentionally
PyTorch-free.  This module is the only place that inspects the active CUDA
device or mutates build environment variables.  Mutations are scoped to one
``torch.utils.cpp_extension.load`` call and are restored in ``finally``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig
import threading
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union
import warnings

try:  # Package import when installed; script import in the HQ-DM source tree.
    from .backend_registry import (
        BackendSpec,
        BuildPlan,
        ToolchainMismatch,
        make_build_plan,
        select_backend,
        version_at_least,
    )
except ImportError:  # pragma: no cover - exercised by deployment layout
    from backend_registry import (
        BackendSpec,
        BuildPlan,
        ToolchainMismatch,
        make_build_plan,
        select_backend,
        version_at_least,
    )


DeviceLike = Union[int, str, object]

_PROJECT_ROOT = Path(__file__).resolve().parent
_REQUIRED_EXPORTS = (
    "fwht_quant_lastdim",
    "fwht_quant_nchw",
    "quantize",
    "im2col_nhwc",
    "scale_bias",
    "scale_bias_nchw",
    "cutlass_gemm_int8",
    "cutlass_conv2d_nhwc",
)


@dataclass(frozen=True)
class ResolvedBuild:
    """A fully resolved build, before compilation starts."""

    plan: BuildPlan
    source_paths: Tuple[Path, ...]
    # ``cuda_version`` is retained as the compatibility spelling for the
    # compiler toolkit version.  It must not be confused with either the CUDA
    # version used to build PyTorch or the libcudart loaded by this process.
    cuda_version: str
    torch_cuda_version: str
    compiler_cuda_version: str
    cuda_runtime_version: Optional[str]
    cuda_home: Path
    nvcc_path: Path
    nvcc_version_output: str
    toolchain_fingerprint: str
    cutlass_version: Optional[str]
    cutlass_root: Optional[Path]
    source_fingerprint: str
    cache_key: Tuple[object, ...]


@dataclass(frozen=True)
class LoadedBuild:
    """Loaded Python extension plus the policy used to build it."""

    module: object
    resolved: ResolvedBuild

    @property
    def backend(self) -> BackendSpec:
        return self.resolved.plan.backend


@dataclass(frozen=True)
class CudaCompilerIdentity:
    """The compiler executable that ``torch.cpp_extension`` will invoke."""

    cuda_home: Path
    nvcc_path: Path
    version: str
    version_output: str


_LOADED: Dict[Tuple[object, ...], LoadedBuild] = {}
_COMPILER_IDENTITIES: Dict[
    Tuple[object, ...], Tuple[CudaCompilerIdentity, Tuple[object, ...]]
] = {}
_CUDA_RUNTIME_VERSIONS: Dict[Tuple[object, ...], Optional[str]] = {}
_RESOLVED_BUILDS: Dict[
    Tuple[object, ...], Tuple[ResolvedBuild, Tuple[Tuple[object, ...], ...]]
] = {}
_BUILD_LOCK = threading.RLock()


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on deployment
        raise RuntimeError("PyTorch is required to build the HQ-DM extension") from error
    return torch


def parse_nvcc_version(output: str) -> str:
    """Return ``major.minor`` from the output of the selected nvcc binary."""

    match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", output, flags=re.IGNORECASE)
    if match is None:
        # Some wrappers retain only the Vxx.yy.zz token from nvcc's banner.
        match = re.search(r"\bV(\d+)\.(\d+)(?:\.\d+)?\b", output)
    if match is None:
        raise RuntimeError(
            "could not parse a CUDA toolkit version from `nvcc --version`: "
            + output.strip()
        )
    return f"{int(match.group(1))}.{int(match.group(2))}"


def _resolve_executable(value: Union[str, Path]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    located = shutil.which(str(value))
    if located:
        return Path(located).resolve()
    raise FileNotFoundError(f"CUDA compiler executable does not exist: {value}")


def _path_state(path: Path) -> Tuple[object, ...]:
    """Return a cheap change token without re-hashing a file's contents."""

    try:
        status = path.stat()
    except OSError:
        return (str(path), None, None, None)
    return (
        str(path),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _cuda_compiler_environment(torch) -> Tuple[Path, Tuple[object, ...]]:
    """Resolve CUDA_HOME and form the environment part of the nvcc cache key."""

    # Import through the supplied torch installation.  cpp_extension computes
    # CUDA_HOME once at module import, and that exact value is what its loader
    # will use rather than a later direct read of os.environ["CUDA_HOME"].
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise FileNotFoundError(
            "torch.utils.cpp_extension.CUDA_HOME is unset; install a CUDA "
            "toolkit and set CUDA_HOME before building the HQ-DM extension"
        )
    cuda_home = Path(CUDA_HOME).expanduser().resolve()
    override = os.environ.get("PYTORCH_NVCC")
    # PATH and cwd matter only when PYTORCH_NVCC is a relative/bare name.  They
    # are harmless in this small key and make changes to wrapper selection
    # visible without spawning nvcc again on every HQIntModule conversion.
    key = (
        str(cuda_home),
        override,
        os.environ.get("PATH") if override else None,
        os.getcwd() if override else None,
    )
    return cuda_home, key


def discover_cuda_compiler(torch) -> CudaCompilerIdentity:
    """Resolve CUDA_HOME and fingerprint the nvcc used by the JIT builder."""

    cuda_home, environment_key = _cuda_compiler_environment(torch)
    with _BUILD_LOCK:
        cached = _COMPILER_IDENTITIES.get(environment_key)
        if cached is not None and _path_state(cached[0].nvcc_path) == cached[1]:
            return cached[0]

    override = os.environ.get("PYTORCH_NVCC")
    if override:
        nvcc_path = _resolve_executable(override)
    else:
        executable = "nvcc.exe" if os.name == "nt" else "nvcc"
        nvcc_path = _resolve_executable(cuda_home / "bin" / executable)
    try:
        completed = subprocess.run(
            [str(nvcc_path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except OSError as error:
        raise RuntimeError(f"failed to execute CUDA compiler {nvcc_path}: {error}") from error
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"CUDA compiler {nvcc_path} --version failed with exit code "
            f"{completed.returncode}: {output}"
        )
    version = parse_nvcc_version(output)
    identity = CudaCompilerIdentity(
        cuda_home=cuda_home,
        nvcc_path=nvcc_path,
        version=version,
        version_output=output,
    )
    with _BUILD_LOCK:
        _COMPILER_IDENTITIES[environment_key] = (identity, _path_state(nvcc_path))
    return identity


def cuda_runtime_version_from_integer(value: int) -> str:
    """Convert CUDA's integer runtime encoding (for example 12080) to 12.8."""

    if value <= 0:
        raise ValueError(f"invalid cudaRuntimeGetVersion result {value}")
    return f"{value // 1000}.{(value % 1000) // 10}"


def _runtime_version_from_library(library) -> Optional[str]:
    try:
        function = library.cudaRuntimeGetVersion
    except AttributeError:
        return None
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    encoded = ctypes.c_int()
    status = int(function(ctypes.byref(encoded)))
    if status != 0:
        raise RuntimeError(f"cudaRuntimeGetVersion failed with CUDA status {status}")
    return cuda_runtime_version_from_integer(int(encoded.value))


def _loaded_linux_cudart_paths() -> Tuple[Path, ...]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return ()
    paths = []
    seen = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "libcudart.so" not in fields[5]:
            continue
        path_text = fields[5]
        deleted_suffix = " (deleted)"
        if path_text.endswith(deleted_suffix):
            path_text = path_text[: -len(deleted_suffix)]
        path = Path(path_text)
        if path not in seen and path.is_file():
            seen.add(path)
            paths.append(path)
    return tuple(paths)


def _windows_cudart_names(versions: Sequence[str]) -> Tuple[str, ...]:
    names = []
    for version in versions:
        numbers = re.findall(r"\d+", version)
        if len(numbers) < 2:
            continue
        major, minor = int(numbers[0]), int(numbers[1])
        candidates = (
            f"cudart64_{major}.dll",
            f"cudart64_{major}{minor}.dll",
            f"cudart64_{major}0.dll",
        )
        for candidate in candidates:
            if candidate not in names:
                names.append(candidate)
    return tuple(names)


def query_loaded_cuda_runtime_version(
    torch,
    *,
    version_hints: Sequence[str] = (),
) -> Optional[str]:
    """Query the libcudart already loaded by this process without replacing it.

    This deliberately does not load a runtime from ``CUDA_HOME``.  Doing that
    could report a library different from the one to which PyTorch is bound and
    cannot repair a process that started with the wrong loader search path.
    """

    torch.cuda.init()
    libraries = []
    try:
        libraries.append(ctypes.CDLL(None))
    except OSError:
        pass
    if os.name == "nt":  # pragma: no cover - exercised on deployment hosts
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        for name in _windows_cudart_names(version_hints):
            handle = kernel32.GetModuleHandleW(name)
            if handle:
                libraries.append(ctypes.WinDLL(name, handle=handle))
    else:
        for path in _loaded_linux_cudart_paths():
            try:
                libraries.append(ctypes.CDLL(str(path)))
            except OSError:
                continue
    for library in libraries:
        version = _runtime_version_from_library(library)
        if version is not None:
            return version
    return None


def _cached_cuda_runtime_version(
    torch,
    *,
    torch_cuda_version: str,
    compiler: CudaCompilerIdentity,
) -> Optional[str]:
    """Query the process runtime once for each effective toolchain identity."""

    key = (
        str(getattr(torch, "__version__", "unknown")),
        torch_cuda_version,
        compiler.version,
        str(compiler.nvcc_path),
    )
    with _BUILD_LOCK:
        if key in _CUDA_RUNTIME_VERSIONS:
            return _CUDA_RUNTIME_VERSIONS[key]
    version = query_loaded_cuda_runtime_version(
        torch,
        version_hints=(torch_cuda_version, compiler.version),
    )
    with _BUILD_LOCK:
        _CUDA_RUNTIME_VERSIONS[key] = version
    return version


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


def validate_cuda_runtime_identity(
    backend: BackendSpec,
    *,
    torch_cuda_version: str,
    compiler_cuda_version: str,
    cuda_runtime_version: Optional[str],
    strict: bool = False,
) -> Tuple[str, ...]:
    """Validate known incompatibilities and return non-fatal diagnostics."""

    diagnostics = []
    torch_major = int(re.findall(r"\d+", torch_cuda_version)[0])
    compiler_major = int(re.findall(r"\d+", compiler_cuda_version)[0])
    if torch_major != compiler_major:
        raise ToolchainMismatch(
            "PyTorch was built with CUDA "
            f"{torch_cuda_version}, but torch.cpp_extension selected nvcc "
            f"{compiler_cuda_version}. PyTorch CUDA extensions require matching "
            "toolkit major versions. Select a compatible CUDA_HOME/nvcc."
        )

    if cuda_runtime_version is None:
        message = (
            "could not query cudaRuntimeGetVersion from the libcudart already "
            "loaded by this process; runtime/compiler compatibility is unknown"
        )
        if strict:
            raise ToolchainMismatch(message)
        diagnostics.append(message)
        return tuple(diagnostics)

    if not version_at_least(cuda_runtime_version, backend.minimum_cuda):
        raise ToolchainMismatch(
            f"{backend.backend_id} requires a CUDA runtime >= "
            f"{backend.minimum_cuda}, but this process loaded libcudart "
            f"{cuda_runtime_version}"
        )

    runtime_major = int(re.findall(r"\d+", cuda_runtime_version)[0])
    if runtime_major < compiler_major:
        raise ToolchainMismatch(
            f"nvcc {compiler_cuda_version} is newer by a major version than "
            f"the loaded libcudart {cuda_runtime_version}. Restart Python after "
            "selecting a compatible runtime with the system loader; changing "
            "LD_PRELOAD inside a running process cannot repair this."
        )
    if not version_at_least(cuda_runtime_version, compiler_cuda_version):
        message = (
            f"nvcc {compiler_cuda_version} is newer than the loaded libcudart "
            f"{cuda_runtime_version}. CUDA minor-version combinations can be "
            "compatible, so this is a warning by default; if extension loading "
            "reports missing runtime symbols, restart Python with the matching "
            "CUDA runtime selected before startup. Set "
            "HQDM_STRICT_CUDA_RUNTIME=1 to reject this configuration pre-build."
        )
        if strict:
            raise ToolchainMismatch(message)
        diagnostics.append(message)
    return tuple(diagnostics)


def _identity_tag(value: Optional[str], missing: str = "unknown") -> str:
    if value is None:
        return missing
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()
    return cleaned or missing


def toolchain_fingerprint(
    *,
    torch_cuda_version: str,
    compiler: CudaCompilerIdentity,
    cuda_runtime_version: Optional[str],
) -> str:
    payload = {
        "torch_cuda_version": torch_cuda_version,
        "compiler_cuda_version": compiler.version,
        "cuda_runtime_version": cuda_runtime_version,
        "cuda_home": str(compiler.cuda_home),
        "nvcc_path": str(compiler.nvcc_path),
        "nvcc_version_output": compiler.version_output,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _device_index(torch, device: Optional[DeviceLike]) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required to load the HQ-DM extension")
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return int(device)
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {parsed}")
    return int(torch.cuda.current_device() if parsed.index is None else parsed.index)


def _candidate_cutlass_roots() -> Iterator[Path]:
    for variable in ("HQDM_CUTLASS_PATH", "CUTLASS_PATH"):
        value = os.environ.get(variable)
        if value:
            yield Path(value).expanduser()
    try:
        yield Path(sysconfig.get_paths()["purelib"]) / "cutlass_library" / "source"
    except (KeyError, TypeError):
        pass
    yield _PROJECT_ROOT / "cutlass"
    yield _PROJECT_ROOT.parent / "cutlass"
    yield _PROJECT_ROOT.parent.parent / "cutlass"


def discover_cutlass_root(explicit: Optional[Union[str, Path]] = None) -> Path:
    """Return a CUTLASS root containing ``include/cutlass`` or fail clearly."""

    candidates = (Path(explicit).expanduser(),) if explicit is not None else tuple(
        _candidate_cutlass_roots()
    )
    checked = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(str(candidate))
        if (candidate / "include" / "cutlass").is_dir():
            return candidate
    rendered = "\n  - ".join(checked) if checked else "(no candidates)"
    raise FileNotFoundError(
        "CUTLASS headers were not found. Set HQDM_CUTLASS_PATH (or "
        f"CUTLASS_PATH) to a checkout root. Checked:\n  - {rendered}"
    )


def read_cutlass_version(root: Path) -> str:
    """Parse CUTLASS's public version header without importing its Python API."""

    header = root / "include" / "cutlass" / "version.h"
    if not header.is_file():
        override = os.environ.get("HQDM_CUTLASS_VERSION")
        if override:
            return override
        raise FileNotFoundError(f"CUTLASS version header is missing: {header}")
    text = header.read_text(encoding="utf-8", errors="replace")

    def component(name: str) -> Optional[str]:
        patterns = (
            rf"^\s*#\s*define\s+CUTLASS_{name}\s+(\d+)\b",
            rf"^\s*#\s*define\s+CUTLASS_VERSION_{name}\s+(\d+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.MULTILINE)
            if match:
                return match.group(1)
        return None

    pieces = tuple(component(name) for name in ("MAJOR", "MINOR", "PATCH"))
    if all(piece is not None for piece in pieces):
        return ".".join(str(piece) for piece in pieces)
    override = os.environ.get("HQDM_CUTLASS_VERSION")
    if override:
        return override
    raise RuntimeError(
        f"could not parse CUTLASS version from {header}; set "
        "HQDM_CUTLASS_VERSION only after verifying the checkout"
    )


def _validate_sources(plan: BuildPlan) -> Tuple[Path, ...]:
    paths = tuple((_PROJECT_ROOT / relative).resolve() for relative in plan.relative_sources)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "backend source selection is incomplete; missing:\n  - "
            + "\n  - ".join(missing)
        )
    return paths


def _source_fingerprint(
    plan: BuildPlan,
    source_paths: Tuple[Path, ...],
    cutlass_root: Optional[Path],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(plan.as_dict(), sort_keys=True).encode("utf-8"))
    for relative, path in zip(plan.relative_sources, source_paths):
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    if cutlass_root is not None:
        # The absolute path distinguishes two checkouts that report the same
        # version but contain locally patched headers.
        digest.update(str(cutlass_root).encode("utf-8"))
        version_header = cutlass_root / "include" / "cutlass" / "version.h"
        if version_header.is_file():
            digest.update(version_header.read_bytes())
    return digest.hexdigest()[:12]


def _resolved_input_state(resolved: ResolvedBuild) -> Tuple[Tuple[object, ...], ...]:
    """Cheaply detect source/toolchain edits before reusing a resolved build."""

    paths = list(resolved.source_paths)
    paths.append(resolved.nvcc_path)
    if resolved.cutlass_root is not None:
        paths.append(resolved.cutlass_root / "include" / "cutlass" / "version.h")
    return tuple(_path_state(path) for path in paths)


def _resolution_request_key(
    torch,
    *,
    capability: Tuple[int, int],
    torch_cuda_version: str,
    cutlass_path: Optional[Union[str, Path]],
    strict_runtime: bool,
) -> Tuple[object, ...]:
    """Form a fast key using only cheap state inspected on every conversion."""

    _, compiler_environment = _cuda_compiler_environment(torch)
    explicit_cutlass = (
        str(Path(cutlass_path).expanduser().resolve())
        if cutlass_path is not None
        else None
    )
    return (
        capability,
        str(getattr(torch, "__version__", "unknown")),
        torch_cuda_version,
        compiler_environment,
        explicit_cutlass,
        os.environ.get("HQDM_CUTLASS_PATH"),
        os.environ.get("CUTLASS_PATH"),
        os.environ.get("HQDM_CUTLASS_VERSION"),
        strict_runtime,
    )


def resolve_build(
    device: Optional[DeviceLike] = None,
    *,
    cutlass_path: Optional[Union[str, Path]] = None,
) -> ResolvedBuild:
    """Resolve device, toolchain, sources, and a collision-safe cache name."""

    torch = _torch()
    index = _device_index(torch, device)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
    # Reject GPUs outside the deliberately small production matrix before
    # probing CUDA_HOME, nvcc, the runtime, or CUTLASS.  Otherwise an unrelated
    # toolchain error can hide the actual unsupported-architecture diagnosis.
    backend = select_backend(capability)
    torch_cuda_version = torch.version.cuda
    if not torch_cuda_version:
        raise RuntimeError("this PyTorch build does not report a CUDA toolkit version")
    strict_runtime = _environment_flag("HQDM_STRICT_CUDA_RUNTIME")
    request_key = _resolution_request_key(
        torch,
        capability=capability,
        torch_cuda_version=torch_cuda_version,
        cutlass_path=cutlass_path,
        strict_runtime=strict_runtime,
    )
    with _BUILD_LOCK:
        cached_resolution = _RESOLVED_BUILDS.get(request_key)
        if (
            cached_resolution is not None
            and _resolved_input_state(cached_resolution[0]) == cached_resolution[1]
        ):
            return cached_resolution[0]

    compiler = discover_cuda_compiler(torch)
    cuda_runtime_version = _cached_cuda_runtime_version(
        torch,
        torch_cuda_version=torch_cuda_version,
        compiler=compiler,
    )

    # make_build_plan below performs the authoritative version validation.
    cutlass_root = discover_cutlass_root(cutlass_path) if backend.requires_cutlass else None
    cutlass_version = read_cutlass_version(cutlass_root) if cutlass_root else None
    # Architecture support is a compiler property, so validate the selected
    # nvcc rather than torch.version.cuda (which identifies PyTorch's build).
    plan = make_build_plan(capability, compiler.version, cutlass_version)
    diagnostics = validate_cuda_runtime_identity(
        backend,
        torch_cuda_version=torch_cuda_version,
        compiler_cuda_version=compiler.version,
        cuda_runtime_version=cuda_runtime_version,
        strict=strict_runtime,
    )
    for diagnostic in diagnostics:
        warnings.warn(diagnostic, RuntimeWarning, stacklevel=2)
    source_paths = _validate_sources(plan)
    source_fingerprint = _source_fingerprint(plan, source_paths, cutlass_root)
    identity_fingerprint = toolchain_fingerprint(
        torch_cuda_version=torch_cuda_version,
        compiler=compiler,
        cuda_runtime_version=cuda_runtime_version,
    )
    plan = replace(
        plan,
        extension_name=(
            f"{plan.extension_name}_torchcu{_identity_tag(torch_cuda_version)}_"
            f"rt{_identity_tag(cuda_runtime_version)}_tool{identity_fingerprint}_"
            f"src{source_fingerprint}"
        ),
    )
    cache_key = (
        plan.backend.capability,
        torch_cuda_version,
        compiler.version,
        cuda_runtime_version,
        str(compiler.cuda_home),
        str(compiler.nvcc_path),
        compiler.version_output,
        cutlass_version,
        plan.backend.backend_id,
        source_fingerprint,
    )
    resolved = ResolvedBuild(
        plan=plan,
        source_paths=source_paths,
        cuda_version=compiler.version,
        torch_cuda_version=torch_cuda_version,
        compiler_cuda_version=compiler.version,
        cuda_runtime_version=cuda_runtime_version,
        cuda_home=compiler.cuda_home,
        nvcc_path=compiler.nvcc_path,
        nvcc_version_output=compiler.version_output,
        toolchain_fingerprint=identity_fingerprint,
        cutlass_version=cutlass_version,
        cutlass_root=cutlass_root,
        source_fingerprint=source_fingerprint,
        cache_key=cache_key,
    )
    with _BUILD_LOCK:
        _RESOLVED_BUILDS[request_key] = (resolved, _resolved_input_state(resolved))
    return resolved


@contextmanager
def _temporary_build_environment(plan: BuildPlan) -> Iterator[None]:
    """Set the exact architecture for one build and restore caller state."""

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = plan.torch_arch_list
    if previous_jobs is None:
        os.environ["MAX_JOBS"] = "4"
    try:
        yield
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs


def load_extension_details(
    verbose: bool = False,
    device: Optional[DeviceLike] = None,
    *,
    cutlass_path: Optional[Union[str, Path]] = None,
) -> LoadedBuild:
    """Build/load the extension selected for ``device`` and return its policy."""

    with _BUILD_LOCK:
        resolved = resolve_build(device, cutlass_path=cutlass_path)
        cached = _LOADED.get(resolved.cache_key)
        if cached is not None:
            return cached

        from torch.utils.cpp_extension import load

        include_paths = []
        if resolved.cutlass_root is not None:
            include_paths.append(str(resolved.cutlass_root / "include"))
            # CUTLASS 3.x collective examples and SM90 utilities expose
            # packed_stride.hpp from this separate public utility include.
            tools_util = resolved.cutlass_root / "tools" / "util" / "include"
            if tools_util.is_dir():
                include_paths.append(str(tools_util))
        with _temporary_build_environment(resolved.plan):
            module = load(
                name=resolved.plan.extension_name,
                sources=[str(path) for path in resolved.source_paths],
                extra_include_paths=include_paths,
                extra_cflags=list(resolved.plan.extra_cflags),
                extra_cuda_cflags=list(resolved.plan.extra_cuda_cflags),
                with_cuda=True,
                verbose=verbose,
            )
        missing_exports = [name for name in _REQUIRED_EXPORTS if not hasattr(module, name)]
        if missing_exports:
            raise RuntimeError(
                f"{resolved.plan.extension_name} violates the v4 binding contract; "
                f"missing exports: {', '.join(missing_exports)}"
            )
        loaded = LoadedBuild(module=module, resolved=resolved)
        _LOADED[resolved.cache_key] = loaded
        return loaded


def load_extension(
    verbose: bool = False,
    device: Optional[DeviceLike] = None,
    *,
    cutlass_path: Optional[Union[str, Path]] = None,
):
    """Compatibility wrapper returning only the loaded pybind module."""

    return load_extension_details(
        verbose=verbose, device=device, cutlass_path=cutlass_path
    ).module


def clear_build_caches(*, clear_loaded: bool = False) -> None:
    """Clear process-local discovery caches used by development workflows.

    Normal callers do not need this: source, selected nvcc, and CUTLASS version
    header timestamps are checked before a resolved build is reused.  Call this
    after deliberately replacing a compiler wrapper or header while preserving
    its file metadata.  ``clear_loaded`` only forgets HQ-DM's registry; Python
    cannot unload a shared library that has already been imported.
    """

    with _BUILD_LOCK:
        _COMPILER_IDENTITIES.clear()
        _CUDA_RUNTIME_VERSIONS.clear()
        _RESOLVED_BUILDS.clear()
        if clear_loaded:
            _LOADED.clear()


def loaded_builds() -> Tuple[LoadedBuild, ...]:
    """Return an immutable snapshot of in-process architecture builds."""

    with _BUILD_LOCK:
        return tuple(_LOADED.values())
