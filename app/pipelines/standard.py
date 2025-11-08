# app/pipelines/standard.py
from __future__ import annotations

import pinecone
from typing import Dict, Any, List

from app.common.config import settings
from app.common.pinecone_utils import get_index

# Create one client and one index handle
_pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
_index = get_index()


def _embed_query(text: str) -> List[float]:
    """
    Get a 1024-dim dense embedding for the query using Pinecone Integrated Inference.
    Matches serverless index (dotproduct, 1024d).
    """
    out = _pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=text,
        parameters={"input_type": "query", "truncate": "END"},
    )
    return out[0]["values"]  # list[float] length 1024


def retrieve_chunks(query: str, meta_filter: Dict[str, Any] | None, k: int = 5):
    """
    Dense retrieval against Pinecone. Set meta_filter to a Pinecone metadata filter
    dict (or None). Returns the raw query response.
    """
    vector = _embed_query(query)
    kwargs = dict(
        namespace=settings.pinecone_ns,
        top_k=k,
        vector=vector,
        include_metadata=True,
        include_values=False,
    )
    if meta_filter:
        kwargs["filter"] = meta_filter
    return _index.query(**kwargs)


def _to_matches(resp) -> list[dict]:
    matches = []
    for m in resp.get("matches", []):
        matches.append(
            {
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": m.get("metadata", {}),
            }
        )
    return matches


def run_query(question: str, filters: Dict[str, Any] | None = None, *, filter_only: bool = False, max_ctx: int = 10):
    """
    Standard pipeline:
      1) embed query (dense)
      2) retrieve top-k chunks with optional metadata filters
      3) if filter_only==True → just return matches (no generation)
      4) else → build context and call generator
    """
    # 1–2) retrieve
    resp = retrieve_chunks(question, filters or {}, k=10 if filter_only else 5)
    matches = _to_matches(resp)

    if filter_only:
        return {"answer": "", "matches": matches}

    # 3) build small context window
    # (Assumes your metadata contains "chunk_text"; adapt if your key differs)
    ctx_parts: list[str] = []
    for m in matches[:max_ctx]:
        meta = m.get("metadata", {})
        txt = meta.get("chunk_text") or meta.get("text") or ""
        if txt:
            ctx_parts.append(txt.strip())
    context = "\n\n---\n\n".join(ctx_parts)

    # 4) generate with your HF helper
    from app.common.hf_utils import generate

    system = "You are a helpful legal research assistant. Answer succinctly and cite sections when possible."
    prompt = f"Question: {question}\n\nRelevant context:\n{context}\n\nAnswer:"
    answer = generate(prompt, max_new_tokens=256, temperature=0.2, top_p=0.9, system=system)

    return {"answer": answer, "matches": matches}

