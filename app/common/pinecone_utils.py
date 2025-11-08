# app/common/pinecone_utils.py
from __future__ import annotations

import pinecone
from .config import settings

_pc = None
_index = None

VECTOR_DIM = 1024  # llama-text-embed-v2


def _ensure_modern_sdk():
    ver = getattr(pinecone, "__version__", "0.0.0")
    # Require >=7.0 which bundles .inference and the Pinecone client
    try:
        from packaging.version import Version
        if Version(ver) < Version("7.0.0"):
            raise RuntimeError(
                f"Pinecone SDK >=7.0.0 required, found {ver}. "
                "Run: pip uninstall -y pinecone pinecone-client pinecone-plugin-inference && "
                "pip install 'pinecone==7.3.0'"
            )
    except Exception:
        # If packaging isn't installed, just try attribute presence
        if not hasattr(pinecone, "Pinecone"):
            raise RuntimeError(
                "Pinecone SDK missing Pinecone client. "
                "Run: pip uninstall -y pinecone pinecone-client pinecone-plugin-inference && "
                "pip install 'pinecone==7.3.0'"
            )


def get_pc():
    """Lazy-init Pinecone client (works with SDK >=7)."""
    global _pc
    if _pc is None:
        _ensure_modern_sdk()
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set")
        _pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_index():
    """Lazy-init Pinecone index handle."""
    global _index
    if _index is None:
        if not settings.pinecone_index:
            raise ValueError("PINECONE_INDEX_NAME is not set")
        _index = get_pc().Index(settings.pinecone_index)
    return _index
