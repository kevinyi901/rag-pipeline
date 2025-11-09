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


def build_context_string(matches, max_chunks=None):
    """
    Build enriched context string with metadata for LLM.
    """
    if not matches:
        return "No documents were retrieved."
    
    matches_to_process = matches
    if max_chunks is not None:
        matches_to_process = matches[:max_chunks]
    
    context_string = ""
    for i, match in enumerate(matches_to_process):
        metadata = match.get('metadata', {})
        score = match.get('score', 0)
        
        chunk_text = metadata.get('chunk_text', 'N/A')
        state = metadata.get('state', 'N/A')
        county = metadata.get('county', 'N/A')
        section = metadata.get('section', 'N/A')
        
        tags = []
        if metadata.get('obligation') == 'Y':
            tags.append("Obligation")
        if metadata.get('penalty') == 'Y':
            tags.append("Penalty")
        if metadata.get('permission') == 'Y':
            tags.append("Permission")
        if metadata.get('prohibition') == 'Y':
            tags.append("Prohibition")
        
        context_string += f"[Chunk {i+1}]\n"
        context_string += f"Score: {score:.4f}\n"
        context_string += f"State: {state}\n"
        context_string += f"County: {county}\n"
        context_string += f"Section: {section}\n"
        
        if tags:
            context_string += f"Tags: {', '.join(tags)}\n"
        
        context_string += f"Text: \"{chunk_text}\"\n\n"
    
    return context_string


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
        # For filter-only search, generate summary of sample
        context = build_context_string(matches, max_chunks=10)
        num_total_chunks = len(matches)
        
        system = """You are a highly intelligent legal analyst.
You will be given a sample of the top-retrieved legal documents.
Your task is to provide a high-level summary of the main themes found in this sample.

- DO NOT try to answer a question.
- DO NOT say "I cannot find an answer."
- Simply summarize what you see. Group similar topics together.
- Start your response with: "The documents in this sample primarily discuss..." """
        
        prompt = f"""**Retrieved Chunks (Sample):**
{context}"""
        
        from app.common.hf_utils import generate
        response_text = generate(prompt, system=system, max_new_tokens=1024, temperature=0.2, top_p=0.9)
        
        answer = (
            f"Found {num_total_chunks} laws matching your filters. "
            f"A full list is available in the generated CSV file.\n\n"
            f"Here is a quick summary of the first 10 results:\n\n"
            f"{response_text}"
        )
        
        return {"answer": answer, "matches": matches}

    # Standard query with answer generation
    context = build_context_string(matches, max_chunks=max_ctx)
    
    system = """You are a highly intelligent legal analyst. Your goal is to help a user understand the legal information provided.
You will be given the user's original question and a list of 'Retrieved Chunks' from a legal database.

Your task is to generate a natural language response. You MUST follow these rules:
1. Base your answer ONLY on the information inside the "Retrieved Chunks". Do not use any outside knowledge.
2. Use the 'Score, State, County, Section, Tags' fields for quick understanding, but use the full 'Text' field to find the specific answer.
3. If the chunks do not contain a clear answer to the user's question, you MUST respond only with the text: 'The information was not found in the provided documents.'
4. If the chunks do contain an answer, summarize it and cite the chunks.

TEMPLATE FOR A SUCCESSFUL ANSWER:
### Summary of Findings
[Your summary of the answer found in the chunks. Cite the chunks, e.g., "The law prohibits owners from letting their dog disturb the peace [Chunk 1]."]

### How This Was Generated
To answer your question, this tool performed a search on the UnBarred 2.0 legal database. The "Retrieved Chunks" represent the top most relevant sections of the law found by our search. This summary is based only on the information in those chunks."""
    
    prompt = f"""**User's Question:**
{question}

**Retrieved Chunks:**
{context}"""
    
    from app.common.hf_utils import generate
    answer = generate(prompt, system=system, max_new_tokens=1024, temperature=0.2, top_p=0.9)

    return {"answer": answer, "matches": matches}