# app/common/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


def _csv_env(name: str, default: str | None = None) -> List[str]:
    """Parse comma-separated env var; '*' -> ['*']; empty -> [] or default."""
    raw = os.getenv(name)
    if raw is None:
        return [] if default is None else ([default] if default != "*" else ["*"])
    raw = raw.strip()
    if raw == "*":
        return ["*"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # ==== Pinecone ====
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index: str   = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", ""))
    pinecone_env: str     = field(default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1"))
    pinecone_ns: str      = field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "__default__"))

    # ==== Hugging Face ====
    hf_model_id: str = field(default_factory=lambda: os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"))
    hf_token: str    = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))  # optional if model is public

    # ==== Cross-encoder (Pipeline B) ====
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # ==== Backend / API ====
    api_host: str         = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int         = field(default_factory=lambda: _int_env("API_PORT", 8000))
    cors_allow: list[str] = field(default_factory=lambda: _csv_env("CORS_ALLOW_ORIGINS", "*"))
    api_bearer_token: str = field(default_factory=lambda: os.getenv("API_BEARER_TOKEN", ""))  # optional

    # ==== Misc ====
    debug: bool           = field(default_factory=lambda: _bool_env("DEBUG", False))
    require_secrets: bool = field(default_factory=lambda: _bool_env("REQUIRE_SECRETS", False))

    # Derived / constants
    vector_dim: int = 1024  # for llama-text-embed-v2
    metric: str     = "dotproduct"

    def validate(self) -> None:
        if not self.require_secrets:
            return
        missing = []
        if not self.pinecone_api_key: missing.append("PINECONE_API_KEY")
        if not self.pinecone_index:   missing.append("PINECONE_INDEX_NAME")
        if not self.hf_model_id:      missing.append("HF_MODEL_ID")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


settings = Settings()
settings.validate()
