from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ..common.config import settings
from ..pipelines import standard, hybrid_rerank


app = FastAPI(title="RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Optional: simple bearer check so only your frontend can call the API
API_BEARER = None # set via env and import here if you want


def _auth(bearer: str | None):
    if API_BEARER and bearer != f"Bearer {API_BEARER}":
        raise HTTPException(status_code=401, detail="Unauthorized")


class QueryIn(BaseModel):
    query: str
    filters: dict = {}
    pipeline: str = "standard" # or "hybrid"
    filter_only: bool = False # standard-only


@app.get("/healthz")


def healthz():
    return {"status": "ok"}


@app.post("/query")


def query_endpoint(inp: QueryIn, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if inp.pipeline == "hybrid":
        return hybrid_rerank.run_query(inp.query, inp.filters)
    else:
        return standard.run_query(inp.query, inp.filters, filter_only=inp.filter_only, max_ctx=10)