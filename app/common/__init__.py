# app/common/__init__.py
"""
Common utilities for the RAG backend.

This package exposes:
- settings: loaded configuration (env-based)
- Pinecone helpers: get_pc, get_index, VECTOR_DIM
- HF helpers: get_model, generate
- Text helpers: build_context_string, build_pinecone_filter, flatten_locations_payload
"""

from .config import settings
from .pinecone_utils import get_pc, get_index, VECTOR_DIM
from .hf_utils import get_model, generate
from .text_utils import (
    build_context_string,
    build_pinecone_filter,
    flatten_locations_payload,
)

__all__ = [
    "settings",
    "get_pc",
    "get_index",
    "VECTOR_DIM",
    "get_model",
    "generate",
    "build_context_string",
    "build_pinecone_filter",
    "flatten_locations_payload",
]
