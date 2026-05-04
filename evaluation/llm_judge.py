"""
LLM-as-a-Judge: Compare model answers to human gold answers from CSV.

Input CSV expected columns:
  - Question        : the question text
  - LLM_Answer      : your model's answer (add this column to your CSV)
  - Answers         : human gold answer 1
  - Answers.1       : human gold answer 2  (optional)
  - Answers.2       : human gold answer 3  (optional)
  - Answers.3       : human gold answer 4  (optional)
  - Category        : question category    (optional, passed through)
  - Difficulty      : Easy/Medium/Hard     (optional, passed through)

Output CSV adds:
  - score           : 1-5 judge score
  - reasoning       : judge's explanation
  - gold_used       : how many gold answers were available

Usage:
  pip install anthropic pandas
  export ANTHROPIC_API_KEY=sk-ant-...
  python llm_judge.py --input your_results.csv --output judged.csv

  # Custom column names:
  python llm_judge.py --input results.csv --llm-col "Model Response" --question-col "Q"
"""

import argparse
import json
import os
import sys
import time

import anthropic
import pandas as pd

GOLD_COLS = ["Answers", "Answers.1", "Answers.2", "Answers.3"]

JUDGE_PROMPT = """You are an expert evaluator assessing a language model's answer against human-written gold standard answers.

QUESTION:
{question}

HUMAN GOLD ANSWERS ({n_gold} provided):
{gold_block}

MODEL ANSWER:
{llm_answer}

Score the model answer from 1 to 5 using this rubric:
  5 - Excellent: covers all key facts and intent from the gold answers accurately
  4 - Good: covers most key facts with only minor omissions or imprecision
  3 - Partial: addresses the question but misses important facts or details
  2 - Poor: barely relevant, contains significant gaps or factual errors
  1 - Wrong: incorrect, off-topic, or refused to answer

Respond ONLY with valid JSON (no markdown fences):
{{"score": <1-5>, "reasoning": "<2-3 sentences explaining your score>"}}"""


def build_gold_block(row, gold_cols):
    answers = []
    for col in gold_cols:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            answers.append(str(val).strip())
    return answers


def judge_answer(client, question, gold_answers, llm_answer, retries=3):
    if not gold_answers:
        return {"score": None, "reasoning": "No gold answers available for this question."}
    if not llm_answer or str(llm_answer).strip() == "":
        return {"score": 1, "reasoning": "Model provided no answer."}

    gold_block = "\n\n".join(
        f"Gold answer {i+1}:\n{a}" for i, a in enumerate(gold_answers)
    )

    prompt = JUDGE_PROMPT.format(
        question=question,
        n_gold=len(gold_answers),
        gold_block=gold_block,
        llm_answer=str(llm_answer).strip(),
    )

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON parse error on attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                return {"score": None, "reasoning": f"Parse error: {raw[:200]}"}
            time.sleep(1)
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  API error on attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                return {"score": None, "reasoning": f"API error: {str(e)[:200]}"}
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge CSV evaluator")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="judged_results.csv", help="Output CSV path")
    parser.add_argument("--question-col", default="Question", help="Column name for the question")
    parser.add_argument("--llm-col", default="LLM_Answer", help="Column name for the model's answer")
    parser.add_argument("--gold-cols", nargs="+", default=GOLD_COLS,
                        help="Gold answer column names (space-separated)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to wait between API calls (default 0.5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only judge first N rows (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input)

    if args.question_col not in df.columns:
        sys.exit(f"Error: question column '{args.question_col}' not found. Available: {df.columns.tolist()}")
    if args.llm_col not in df.columns:
        sys.exit(f"Error: LLM answer column '{args.llm_col}' not found. Available: {df.columns.tolist()}")

    gold_cols_present = [c for c in args.gold_cols if c in df.columns]
    if not gold_cols_present:
        sys.exit(f"Error: none of the gold columns {args.gold_cols} found in CSV.")

    print(f"  {len(df)} rows | question col: '{args.question_col}' | llm col: '{args.llm_col}'")
    print(f"  Gold columns found: {gold_cols_present}")

    rows = df.head(args.limit) if args.limit else df
    scores, reasonings, gold_counts = [], [], []

    for i, (_, row) in enumerate(rows.iterrows()):
        question = str(row[args.question_col])
        llm_answer = row[args.llm_col]
        gold_answers = build_gold_block(row, gold_cols_present)

        print(f"[{i+1}/{len(rows)}] Q: {question[:70]}...")
        result = judge_answer(client, question, gold_answers, llm_answer)

        score = result.get("score")
        reasoning = result.get("reasoning", "")
        scores.append(score)
        reasonings.append(reasoning)
        gold_counts.append(len(gold_answers))

        label = f"{score}/5" if score is not None else "n/a"
        print(f"  Score: {label} | {reasoning[:80]}...")

        if args.delay > 0:
            time.sleep(args.delay)

    # Write output — pad with None if --limit was used
    if args.limit and len(df) > args.limit:
        pad = len(df) - args.limit
        scores += [None] * pad
        reasonings += [""] * pad
        gold_counts += [None] * pad

    df["score"] = scores
    df["reasoning"] = reasonings
    df["gold_used"] = gold_counts

    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # Summary stats
    valid = [s for s in scores if s is not None]
    if valid:
        avg = sum(valid) / len(valid)
        dist = {s: valid.count(s) for s in range(1, 6)}
        print(f"\nSummary ({len(valid)} scored rows):")
        print(f"  Average score : {avg:.2f}/5")
        print(f"  Distribution  : " + " | ".join(f"{k}★ x{v}" for k, v in dist.items()))
        no_gold = sum(1 for g in gold_counts if g == 0)
        if no_gold:
            print(f"  Skipped (no gold answers): {no_gold} rows")


if __name__ == "__main__":
    main()
