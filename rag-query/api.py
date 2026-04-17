import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline import RAGPipeline
from config import Config

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from the React dev server

# Initialize BOTH pipelines ONCE
print("Initializing RAG Pipelines...")
baseline_pipeline = RAGPipeline(use_reranking=False)
hybrid_pipeline = RAGPipeline(use_reranking=True)
print("Pipelines ready!")

def serialize_chunks(chunks):
    """Convert retrieved chunks to JSON-serializable format."""
    serialized = []
    for chunk in chunks:
        chunk_data = {
            'id': chunk.get('id'),
            'score': float(chunk.get('score', 0)),
        }
        if 'rerank_score' in chunk:
            chunk_data['rerank_score'] = float(chunk.get('rerank_score', 0))
        if 'metadata' in chunk:
            for key, value in chunk['metadata'].items():
                chunk_data[key] = value.item() if hasattr(value, 'item') else value
        serialized.append(chunk_data)
    return serialized


# ─── Health ──────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


# ─── Auth ────────────────────────────────────────────────────────────────────

@app.route('/auth/login', methods=['POST'])
def auth_login():
    """Server-side password check for the main login gate."""
    correct = os.getenv('LOGIN_PASS', '')
    if not correct:
        return jsonify({"error": "LOGIN_PASS not configured on server"}), 500
    pw = (request.json or {}).get('password', '')
    if pw == correct:
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 401


@app.route('/auth/upload', methods=['POST'])
def auth_upload():
    """Server-side password check for the document-upload gate."""
    correct = os.getenv('DATA_ADD_PASS', '')
    if not correct:
        return jsonify({"error": "DATA_ADD_PASS not configured on server"}), 500
    pw = (request.json or {}).get('password', '')
    if pw == correct:
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 401


# ─── Stats ───────────────────────────────────────────────────────────────────

@app.route('/stats', methods=['GET'])
def stats():
    pc = baseline_pipeline.pc
    s = baseline_pipeline.pinecone_index.describe_index_stats()
    all_indexes = [
        {"name": idx.name, "dimension": idx.dimension, "metric": idx.metric}
        for idx in pc.list_indexes()
    ]
    return jsonify({
        "current_index": Config.PINECONE_INDEX_NAME,
        "total_vector_count": s.total_vector_count,
        "namespaces": {k: v.vector_count for k, v in s.namespaces.items()},
        "dimension": s.dimension,
        "all_indexes": all_indexes,
    })


# ─── Query ───────────────────────────────────────────────────────────────────

@app.route('/query', methods=['POST'])
def query():
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        data = request.json
        query_text = data.get('query', '')
        filters = data.get('filters', {})
        mode = data.get('mode', 'hybrid')

        if not isinstance(query_text, str):
            return jsonify({"error": "query must be a string"}), 400
        if not query_text.strip():
            return jsonify({"error": "query cannot be empty"}), 400
        if not isinstance(filters, dict):
            return jsonify({"error": "filters must be a dictionary"}), 400
        if mode not in ['hybrid', 'baseline']:
            return jsonify({"error": "mode must be 'hybrid' or 'baseline'"}), 400

        pipeline = hybrid_pipeline if mode == 'hybrid' else baseline_pipeline
        llm_output, retrieved_chunks = pipeline.run(query_text, filters)
        serialized_chunks = serialize_chunks(retrieved_chunks)

        return jsonify({
            "response": llm_output,
            "chunks": serialized_chunks,
            "mode": mode,
        })

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
