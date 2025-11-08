from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from ..common.pinecone_utils import get_index
from ..common.text_utils import build_context_string, build_pinecone_filter, flatten_locations_payload
from ..common.hf_utils import generate
from ..common.config import settings


index = get_index()
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker




def retrieve_chunks_hybrid(query: str, filter_object: dict, k: int = 100) -> List[Dict[str, Any]]:
    from pinecone import Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)


    dense = pc.inference.embed(model="llama-text-embed-v2", inputs=query,
                                parameters={"input_type": "query", "truncate": "END"})
    sparse = pc.inference.embed(model="pinecone-sparse-english-v0", inputs=query,
                                parameters={"input_type": "query", "truncate": "END"})
    dv = dense[0]["values"]
    s = sparse[0]


    res = index.query(
        namespace=settings.pinecone_ns,
        top_k=k,
        vector=dv,
        sparse_vector={"indices": s["sparse_indices"], "values": s["sparse_values"]},
        include_metadata=True,
        include_values=False,
        filter=filter_object or {}
        )
    return res.get("matches", [])




def rerank(query: str, matches: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    reranker = get_reranker()
    pairs = []
    for m in matches:
        text = m.get("metadata", {}).get("chunk_text", "")
        pairs.append([query, text])
    scores = reranker.predict(pairs)
    for m, sc in zip(matches, scores):
        m["rerank_score"] = float(sc)
    matches.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return matches[:top_n]




def run_query(query: str, filters: dict, top_n: int = 10) -> dict:
    f = build_pinecone_filter(flatten_locations_payload(filters))
    matches = retrieve_chunks_hybrid(query, f, k=100)
    matches = rerank(query, matches, top_n=top_n)
    context = build_context_string(matches, max_chunks=top_n)


    prompt = (
    "You are a legal assistant. Answer based only on the CONTEXT.\n\n"
    f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\n"
    "If the context is insufficient, say so and outline what is missing."
    )
    answer = generate(prompt)
    return {"answer": answer, "matches": matches}