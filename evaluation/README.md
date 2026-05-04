# LLM-as-a-Judge Evaluator

Uses **Anthropic Claude** to judge LLM-generated answers against human gold standard answers. Scores each answer on a 1–5 scale and outputs a results CSV with scores and reasoning.

## Overview

This tool evaluates answer quality by:

1. **Reading** a CSV containing questions, LLM answers, and human gold answers
2. **Judging** each LLM answer using Anthropic Claude (claude-sonnet-4-20250514)
3. **Scoring** answers 1–5 based on accuracy and completeness
4. **Outputting** a CSV with scores, reasoning, and summary statistics

## Scoring Rubric

| Score | Meaning |
|-------|---------|
| **5** | Excellent: covers all key facts and intent from gold answers accurately |
| **4** | Good: covers most key facts with only minor omissions or imprecision |
| **3** | Partial: addresses the question but misses important facts or details |
| **2** | Poor: barely relevant, contains significant gaps or factual errors |
| **1** | Wrong: incorrect, off-topic, or refused to answer |

## Installation

```bash
pip install anthropic pandas
```

### Requirements
- Python 3.10+
- anthropic
- pandas

## Configuration

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Basic Usage

```bash
python llm_judge.py --input your_results.csv --output judged.csv
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Input CSV path |
| `--output` | `judged_results.csv` | Output CSV path |
| `--question-col` | `Question` | Column name for the question |
| `--llm-col` | `LLM_Answer` | Column name for the model's answer |
| `--gold-cols` | `Answers Answers.1 Answers.2 Answers.3` | Gold answer column names (space-separated) |
| `--delay` | `0.5` | Seconds to wait between API calls |
| `--limit` | `None` | Only judge first N rows (for testing) |

### Examples

```bash
# Basic usage
python llm_judge.py --input results.csv --output judged.csv

# Custom column names
python llm_judge.py --input results.csv --llm-col "Model Response" --question-col "Q"

# Quick test with first 5 rows
python llm_judge.py --input results.csv --limit 5 --delay 0.5
```

## Input CSV Format

The input CSV should contain these columns:

| Column | Description |
|--------|-------------|
| `Question` | The question text |
| `LLM_Answer` | Your model's answer (add this column to your CSV) |
| `Answers` | Human gold answer 1 |
| `Answers.1` | Human gold answer 2 (optional) |
| `Answers.2` | Human gold answer 3 (optional) |
| `Answers.3` | Human gold answer 4 (optional) |
| `Category` | Question category (optional, passed through) |
| `Difficulty` | Easy/Medium/Hard (optional, passed through) |

## Output

### Judged CSV (`judged_results.csv`)

The output CSV adds these columns to your original data:
- `score` — 1–5 judge score
- `reasoning` — Judge's explanation (2–3 sentences)
- `gold_used` — Number of gold answers available for that question

### Summary Statistics (printed to console)

```
Summary (50 scored rows):
  Average score : 3.82/5
  Distribution  : 1★ x2 | 2★ x5 | 3★ x13 | 4★ x20 | 5★ x10
  Skipped (no gold answers): 0 rows
```
