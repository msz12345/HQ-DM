"""HQ-DM multi-architecture integer inference extension."""

from .backend_registry import BACKENDS, BackendSpec, BuildPlan, select_backend

__all__ = ["BACKENDS", "BackendSpec", "BuildPlan", "select_backend"]
