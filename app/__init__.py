# app/__init__.py
"""
RAG backend package.

Lightweight exports:
- settings: loaded configuration from app.common.config (env-based)
- get_api_app(): lazy-import FastAPI app from app.api.server
- get_pipeline(name): lazy-import a pipeline module by name ("standard" or "hybrid")

We intentionally avoid importing heavy modules (HF models, torch) at import time.
"""

from __future__ import annotations

import importlib
import os
from typing import Literal

# Cheap import: just reads env + small helpers (no model load)
from .common import settings  # noqa: F401

__all__ = ["settings", "get_api_app", "get_pipeline", "__version__"]

# Optional version (override via env if you want to stamp builds)
__version__ = os.getenv("RAG_BACKEND_VERSION", "0.1.0")


def get_api_app():
    """
    Lazy-load and return the FastAPI app (avoids importing FastAPI during package import).
    Usage:
        from app import get_api_app
        app = get_api_app()
    """
    mod = importlib.import_module("app.api.server")
    return getattr(mod, "app")


def get_pipeline(name: Literal["standard", "hybrid"] = "standard"):
    """
    Lazy-load a pipeline module by name without triggering model load at import time.
    Returns the module so you can call its functions (e.g., run_query).

    Example:
        pipe = get_pipeline("hybrid")
        result = pipe.run_query("question", filters={})
    """
    if name == "standard":
        return importlib.import_module("app.pipelines.standard")
    if name == "hybrid":
        return importlib.import_module("app.pipelines.hybrid_rerank")
    raise ValueError(f"Unknown pipeline '{name}'. Expected 'standard' or 'hybrid'.")
