"""CUDA-graph wrapper for the fixed-shape diffusion UNet call."""

from __future__ import annotations

import math
from typing import List

import torch

try:
    from .hq_int import reset_timestep_scales
except ImportError:  # Direct script execution from the source tree.
    from hq_int import reset_timestep_scales


class GraphUNetRunner:
    """Capture one UNet graph per HQ-DM timestep scale and replay in order.

    HQ-DM owns a distinct learned activation scale for every denoising step.  A
    graph therefore captures a fixed scale index; the graphs share a memory pool
    and are replayed in the same order in which they were captured.
    """

    def __init__(
        self,
        model,
        example_image: torch.Tensor,
        example_timestep: torch.Tensor,
        example_context: torch.Tensor,
        scale_steps: int,
        warmup: int = 3,
        compute_dtype: torch.dtype = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA")
        self.model = model
        self.eager_apply_model = model.apply_model
        self.static_image = example_image.clone()
        self.static_timestep = example_timestep.clone()
        self.static_context = example_context.clone()
        self.scale_steps = int(scale_steps)
        self.graphs: List[torch.cuda.CUDAGraph] = []
        self.outputs: List[torch.Tensor] = []
        self.call_index = 0
        self.compute_dtype = compute_dtype

        # The upstream implementation builds the sinusoidal frequency vector on
        # CPU and copies it to CUDA on every UNet call.  Host-to-device creation
        # during stream capture is illegal, so cache the vector on-device during
        # warmup and keep the otherwise identical formula.
        from ldm.modules.diffusionmodules import openaimodel

        frequency_cache = {}

        def graph_safe_timestep_embedding(
            timesteps, dim, max_period=10000, repeat_only=False
        ):
            if repeat_only:
                return timesteps.reshape(-1, 1).repeat(1, dim)
            half = dim // 2
            key = (timesteps.device, dim, max_period)
            frequencies = frequency_cache.get(key)
            if frequencies is None:
                # Match the upstream arithmetic exactly: arange/exp happen on
                # CPU, followed by one cached H2D copy during eager warmup.
                frequencies = torch.exp(
                    -math.log(max_period)
                    * torch.arange(
                        start=0,
                        end=half,
                        dtype=torch.float32,
                    )
                    / half
                ).to(device=timesteps.device)
                frequency_cache[key] = frequencies
            arguments = timesteps[:, None].float() * frequencies[None]
            embedding = torch.cat(
                (torch.cos(arguments), torch.sin(arguments)), dim=-1
            )
            if dim % 2:
                embedding = torch.cat(
                    (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
                )
            if self.compute_dtype is not None:
                embedding = embedding.to(dtype=self.compute_dtype)
            return embedding

        openaimodel.timestep_embedding = graph_safe_timestep_embedding

        current_stream = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup):
                reset_timestep_scales(model, self.scale_steps - 1)
                self.eager_apply_model(
                    self.static_image, self.static_timestep, self.static_context
                )
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize()

        pool = torch.cuda.graph_pool_handle()
        for scale_index in range(self.scale_steps - 1, -1, -1):
            reset_timestep_scales(model, scale_index)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                output = self.eager_apply_model(
                    self.static_image, self.static_timestep, self.static_context
                )
            self.graphs.append(graph)
            self.outputs.append(output)
        torch.cuda.synchronize()

    def reset(self):
        self.call_index = 0

    def __call__(self, image, timestep, context):
        if image.shape != self.static_image.shape:
            raise ValueError(
                f"captured image shape {tuple(self.static_image.shape)}, got {tuple(image.shape)}"
            )
        if context.shape != self.static_context.shape:
            raise ValueError(
                f"captured context shape {tuple(self.static_context.shape)}, got {tuple(context.shape)}"
            )
        index = self.call_index
        if index >= len(self.graphs):
            raise RuntimeError("more UNet calls than captured denoising steps")
        self.static_image.copy_(image)
        self.static_timestep.copy_(timestep)
        self.static_context.copy_(context)
        self.graphs[index].replay()
        self.call_index += 1
        return self.outputs[index]
