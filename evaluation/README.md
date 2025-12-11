# Legal Retrieval Engine Evaluator

An evaluation framework for legal retrieval systems using **LLM-as-a-Judge** methodology. Evaluates retrieval quality by comparing retrieved legal documents against ground truth using NVIDIA's Nemotron model.

## Overview

This tool assesses legal retrieval engine performance by:

1. **Querying** a legal retrieval API with test questions
2. **Comparing** retrieved chunks against golden truth answers
3. **Judging** results using an LLM (NVIDIA Nemotron-Nano via NIMs API)
4. **Computing** comprehensive retrieval and quality metrics

## Metrics

### Retrieval Metrics
| Metric | Description |
|--------|-------------|
| **Top-5 Recall** | Percentage of queries where the correct law appears in the top 5 results |
| **MRR** | Mean Reciprocal Rank — average of 1/rank for the first correct result |
| **Chunk Coverage** | How complete the retrieved chunk is compared to the golden truth (0-100%) |

### Metadata Accuracy
The evaluator checks four legal metadata flags on retrieved documents:

| Flag | Description |
|------|-------------|
| **Penalty/Fine** | Does the text describe monetary penalties, fines, or punitive consequences? |
| **Prohibition** | Does the text prohibit actions? ("shall not", "may not", "unlawful") |
| **Obligation** | Does the text impose mandatory requirements? ("must", "shall", "required") |
| **Permission** | Does the text grant authorization? ("may", "permitted", "allowed") |

### Negative Test Support
Evaluates cases where **no law should exist** for a query, measuring:
- **True Negatives**: System correctly identifies no relevant law
- **False Positives**: System incorrectly claims a law exists

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

Edit the configuration variables in `legal_retrieval_evaluator.py`:

```python
RETRIEVAL_ENDPOINT = "http://your-api-endpoint:8000/query"
NIMS_API_KEY = "your-nvidia-nims-api-key"
NIMS_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-8b-v1"
```

## Usage

### Basic Usage

```bash
python legal_retrieval_evaluator.py
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `eval_dataset_final.csv` | Input evaluation dataset (CSV or Excel) |
| `--output` | `-o` | `evaluation_results.csv` | Output file for per-query results |
| `--summary` | `-s` | `evaluation_summary.json` | Output file for summary metrics |
| `--limit` | `-l` | None | Limit number of queries (for testing) |
| `--delay` | | `1.0` | Delay between API calls (seconds) |
| `--mode` | `-m` | `hybrid` | Retrieval mode: `hybrid` or `baseline` |

### Examples

```bash
# Evaluate with hybrid mode (default)
python legal_retrieval_evaluator.py -i eval_dataset.csv -o results.csv -s summary.json

# Evaluate with baseline mode
python legal_retrieval_evaluator.py -m baseline -o results_baseline.csv -s summary_baseline.json

# Quick test with first 10 queries
python legal_retrieval_evaluator.py --limit 10 --delay 0.5

# Use Excel input/output
python legal_retrieval_evaluator.py -i dataset.xlsx -o results.xlsx
```

## Input Dataset Format

The evaluation dataset should be a CSV or Excel file with these columns:

| Column | Description |
|--------|-------------|
| `State` | State code (e.g., "CA", "GA") |
| `County` | County name (e.g., "Alameda", "Los Angeles") |
| `Question` | The legal query to evaluate |
| `Answer` | Golden truth: expected law text (or "NO_LAW_EXISTS" for negative tests) |
| `Section` | Section reference (e.g., "5.08.010") or "N/A" for negative tests |
| `Difficulty Column` | Difficulty level: "Easy", "Medium", or "Hard" |

### Negative Test Cases
To create a negative test (where no law should exist):
- Set `Answer` to `NO_LAW_EXISTS`
- Set `Section` to `N/A`

## Output

### Per-Query Results (`evaluation_results.csv`)

Each row contains:
- Query metadata (ID, state, county, difficulty, question)
- Golden section and chunk text
- Retrieved section(s) and chunk text
- Match status (`found_in_top5`, `rank`, `chunk_coverage`)
- Metadata flag comparisons (golden vs retrieved)
- LLM reasoning explanation

### Summary Metrics (`evaluation_summary.json`)

```json
{
  "total_queries": 110,
  "valid_queries": 108,
  "failed_queries": 2,
  "positive_test_count": 100,
  "top5_recall": 0.85,
  "mrr": 0.78,
  "avg_chunk_coverage": 0.92,
  "avg_metadata_accuracy": 0.88,
  "penalty_fine_accuracy": 0.91,
  "prohibition_accuracy": 0.87,
  "obligation_accuracy": 0.89,
  "permission_accuracy": 0.85,
  "negative_test_count": 10,
  "true_negatives": 8,
  "false_positives": 2,
  "negative_accuracy": 0.80,
  "by_difficulty": {
    "Easy": {"count": 40, "top5_recall": 0.95, "mrr": 0.92},
    "Medium": {"count": 35, "top5_recall": 0.83, "mrr": 0.75},
    "Hard": {"count": 25, "top5_recall": 0.72, "mrr": 0.65}
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

The evaluator expects a retrieval API endpoint that accepts POST requests:

```json
{
  "query": "Is there a noise ordinance?",
  "filters": {
    "locations": [
      {"state": "ca", "county": ["alameda-county"]}
    ]
  },
  "mode": "hybrid"
}
```

Expected response:
```json
{
  "results": [
    {
      "section": "8.36.010",
      "chunk_text": "It shall be unlawful..."
    }
  ],
  "response": "LLM-generated answer..."
}
```

### NVIDIA NIMs API

Requires a valid NVIDIA NIMs API key with access to the Nemotron model for LLM-as-a-Judge evaluation.

## License

Internal use only.

