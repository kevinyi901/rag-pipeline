# RAG API (Hugging Face + Pinecone) for EC2

FastAPI backend exposing `/query` for two RAG pipelines:

- `standard`: dense semantic + filters (supports `filter_only`)
- `hybrid`: dense + sparse + cross-encoder reranking

Frontends (Streamlit/React/etc.) call this API over HTTP. CORS and optional bearer auth are configurable via `.env`.

---

## 1) Prereqs

- Python 3.11 (for local dev) or Docker
- Pinecone serverless index (1024 dim, dotproduct)
- Hugging Face account + read token (accept model license)

**Environment: `.env` (do not commit)**
```env
# Pinecone
PINECONE_API_KEY=__FILL_ME__
PINECONE_INDEX_NAME=hybrid-search-index
PINECONE_ENVIRONMENT=us-east-1
PINECONE_NAMESPACE=__default__

# Hugging Face
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=__FILL_ME__

# Reranker (hybrid pipeline)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# API
API_HOST=0.0.0.0
API_PORT=8000
CORS_ALLOW_ORIGINS=*
API_BEARER_TOKEN=__OPTIONAL__

