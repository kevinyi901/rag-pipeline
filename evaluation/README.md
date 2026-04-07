# Research Funding Retrieval Evaluator

An evaluation framework for the **Foundation Fighting Blindness (FFB)** RAG pipeline using **LLM-as-a-Judge** methodology. Evaluates retrieval quality by comparing retrieved research funding documents against ground truth using NVIDIA's Nemotron model.

## Overview

This tool assesses retrieval engine performance for research funding inquiries by:

1. **Querying** a research funding retrieval API with test questions
2. **Comparing** retrieved chunks against golden truth answers
3. **Judging** results using an LLM (NVIDIA Nemotron-Nano via NIMs API)
4. **Computing** comprehensive retrieval and quality metrics

## Metrics

### Retrieval Metrics
| Metric | Description |
|--------|-------------|
| **Top-5 Recall** | Percentage of queries where the correct document appears in the top 5 results |
| **MRR** | Mean Reciprocal Rank — average of 1/rank for the first correct result |
| **Chunk Coverage** | How complete the retrieved chunk is compared to the golden truth (0-100%) |

### Metadata Accuracy
The evaluator checks four research funding metadata flags on retrieved documents:

| Flag | Description |
|------|-------------|
| **Funding Available** | Does the text describe grants, awards, RFAs, or available financial support? |
| **Clinical Trial** | Does the text relate to clinical trials, patient recruitment, or Phase 1/2/3 studies? |
| **Research Focus** | Does the text describe a specific methodology? (gene therapy, CRISPR, optogenetics, etc.) |
| **Eligibility** | Does the text describe requirements or qualification criteria for applicants? |

### Negative Test Support
Evaluates cases where **no document should exist** for a query, measuring:
- **True Negatives**: System correctly identifies no relevant funding document
- **False Positives**: System incorrectly claims a document exists

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- pandas >= 2.0.0
- requests >= 2.28.0
- tqdm >= 4.65.0
- openpyxl >= 3.1.0 (for Excel support)

## Configuration

Set your NVIDIA NIMs API key as an environment variable (required before running):

```bash
export NIMS_API_KEY="your-nvidia-nims-api-key"
```

The following constants can also be edited at the top of `research_funding_evaluator.py`:

```python
RETRIEVAL_ENDPOINT = "http://your-api-endpoint:8000/query"
NIMS_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-8b-v1"
```

## Usage

### Basic Usage

```bash
python research_funding_evaluator.py
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `eval_dataset_research_funding.csv` | Input evaluation dataset (CSV or Excel) |
| `--output` | `-o` | `research_funding_results.csv` | Output file for per-query results |
| `--summary` | `-s` | `evaluation_summary.json` | Output file for summary metrics |
| `--limit` | `-l` | None | Limit number of queries (for testing) |
| `--delay` | | `1.0` | Delay between API calls (seconds) |
| `--mode` | `-m` | `hybrid` | Retrieval mode: `hybrid` or `baseline` |
| `--dry-run` | | `false` | Validate CSV and preview prompts — no API calls made |

### Examples

```bash
# Evaluate with hybrid mode (default)
python research_funding_evaluator.py -i eval_dataset_research_funding.csv -o results.csv -s summary.json

# Evaluate with baseline mode
python research_funding_evaluator.py -m baseline -o results_baseline.csv -s summary_baseline.json

# Quick test with first 5 queries
python research_funding_evaluator.py --limit 5 --delay 0.5

# Validate dataset and preview prompts without spending API quota
python research_funding_evaluator.py --dry-run

# Use Excel input/output
python research_funding_evaluator.py -i dataset.xlsx -o results.xlsx
```

## Input Dataset Format

The evaluation dataset should be a CSV or Excel file with these columns:

| Column | Description |
|--------|-------------|
| `Disease_Type` | Disease or condition (e.g., "Stargardt", "Retinitis_Pigmentosa") |
| `Research_Area` | Scientific domain (e.g., "Gene_Therapy", "Stem_Cell", "Optogenetics") |
| `Difficulty` | Difficulty level: "Easy", "Medium", or "Hard" |
| `Question` | The research funding query to evaluate |
| `Answer` | Golden truth: expected document text (or "NO_LAW_EXISTS" for negative tests) |
| `Document_ID` | Document reference (e.g., "NEI-RFA-EY-23-001", "NCT03496012") or "N/A" for negative tests |
| `Funding_Source` | Funding body filter (e.g., "NIH", "FFB") — optional, leave blank for negative tests |

### Negative Test Cases
To create a negative test (where no document should exist):
- Set `Answer` to `NO_LAW_EXISTS`
- Set `Document_ID` to `N/A`
- Leave `Funding_Source` blank

### Document ID Formats Supported
The evaluator normalizes document IDs for matching and handles all of these formats:
- Hyphen-separated: `NEI-RFA-EY-23-001`, `FFB-IIRA-2023`, `PA-22-196`
- Compact alphanumeric: `NCT03496012`, `HHSN263201200001C`
- Numeric section codes (legacy): `5.08.010`

## Output

### Per-Query Results (`research_funding_results.csv`)

Each row contains:
- Query metadata (ID, disease type, research area, difficulty, question)
- Golden document ID and chunk text
- Retrieved document ID(s) and chunk text
- Match status (`found_in_top5`, `rank`, `chunk_coverage`)
- Metadata flag comparisons for all four flags (golden vs retrieved)
- LLM reasoning explanation

### Summary Metrics (`evaluation_summary.json`)

```json
{
  "total_queries": 12,
  "valid_queries": 12,
  "failed_queries": 0,
  "positive_test_count": 10,
  "top5_recall": 0.85,
  "mrr": 0.78,
  "avg_chunk_coverage": 0.92,
  "avg_metadata_accuracy": 0.88,
  "funding_available_accuracy": 0.91,
  "clinical_trial_accuracy": 0.87,
  "research_focus_accuracy": 0.89,
  "eligibility_accuracy": 0.85,
  "negative_test_count": 2,
  "true_negatives": 2,
  "false_positives": 0,
  "negative_accuracy": 1.0,
  "by_difficulty": {
    "Easy": {"count": 4, "top5_recall": 0.95, "mrr": 0.92},
    "Medium": {"count": 4, "top5_recall": 0.83, "mrr": 0.75},
    "Hard": {"count": 2, "top5_recall": 0.72, "mrr": 0.65}
  },
  "composite_score": 0.86
}
```

### Composite Score

A weighted combination of metrics (for positive tests only):
- **30%** Top-5 Recall
- **30%** MRR
- **20%** Chunk Coverage
- **20%** Metadata Accuracy

## API Requirements

### Retrieval Engine API

The evaluator sends POST requests with research funding filters:

```json
{
  "query": "What NIH grants are available for Stargardt gene therapy?",
  "filters": {
    "disease_types": ["Stargardt"],
    "research_areas": ["Gene_Therapy"],
    "funding_sources": ["NIH"]
  },
  "mode": "hybrid"
}
```

Expected response (tries these field names in order for chunks and system response):
```json
{
  "results": [
    {
      "document_id": "NEI-RFA-EY-23-001",
      "chunk_text": "The NEI offers R01 grants..."
    }
  ],
  "response": "LLM-generated answer..."
}
```

**Chunk field fallbacks**: `results` → `chunks` → `documents`

**System response field fallbacks**: `response` → `answer` → `llm_response` → `generated_text` → `output`

**Chunk document ID field fallbacks**: `section` → `document_id` → `doc_id` → `title`

### NVIDIA NIMs API

Requires a valid NVIDIA NIMs API key with access to the Nemotron model for LLM-as-a-Judge evaluation.

## License

Internal use only.
