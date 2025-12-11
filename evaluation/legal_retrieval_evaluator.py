#!/usr/bin/env python3
"""
Legal Retrieval Engine Evaluator

Evaluates a legal retrieval engine using LLM-as-a-Judge methodology with
NVIDIA Nemotron-Nano-9B-v2 model through the NIMs API.

Metrics computed:
- Top-5 Recall: Whether the correct law appears in top 5 results
- MRR (Mean Reciprocal Rank): 1/rank of first correct result
- Chunk Coverage: How complete the retrieved chunk is
- Metadata Accuracy: Correctness of Penalty/Fine, Prohibition, Obligation, Permission flags
"""

import json
import re
import time
from typing import Optional
import pandas as pd
import requests
from dataclasses import dataclass, asdict
from tqdm import tqdm


# Configuration
RETRIEVAL_ENDPOINT = "http://3.234.136.27:8000/query"
NIMS_API_KEY = ""
NIMS_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-8b-v1"  # Nemotron Nano model via NIMs


def sanitize_for_csv(text: str) -> str:
    """
    Sanitize text for CSV output by handling special characters.
    
    - Replaces newlines with spaces
    - Normalizes smart quotes to standard quotes
    - Removes or replaces problematic characters
    """
    if not text:
        return ""
    
    # Replace newlines and carriage returns with spaces
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    
    # Normalize smart/curly quotes to standard quotes
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('"', '"').replace('"', '"')
    
    # Replace other problematic Unicode characters
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('…', '...')
    
    # Remove any remaining control characters (except spaces)
    text = ''.join(char if char.isprintable() or char == ' ' else ' ' for char in text)
    
    # Collapse multiple spaces into one
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    return text.strip()


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: int
    state: str
    county: str
    difficulty: str
    question: str
    
    # System's LLM-generated response
    system_response: str = ""
    
    # For negative tests: did system correctly say no law exists?
    is_negative_test: bool = False
    system_says_no_law: Optional[bool] = None  # True if system correctly says no relevant law
    negative_test_correct: Optional[bool] = None
    
    # Retrieval metrics (for positive tests)
    found_in_top5: bool = False
    rank: int = 0  # 0 means not found
    chunk_coverage: float = 0.0  # 0.0 to 1.0
    
    # Section comparison (golden | retrieved side by side)
    golden_section: str = ""
    retrieved_section: str = ""  # Section of the matched chunk
    
    # All top 5 retrieved sections (for analysis)
    retrieved_section_1: str = ""
    retrieved_section_2: str = ""
    retrieved_section_3: str = ""
    retrieved_section_4: str = ""
    retrieved_section_5: str = ""
    
    # Chunk comparison (golden | retrieved side by side)
    golden_chunk: str = ""
    retrieved_chunk: str = ""
    
    # Penalty/Fine flag comparison
    golden_penalty_fine: Optional[bool] = None
    retrieved_penalty_fine: Optional[bool] = None
    penalty_fine_correct: Optional[bool] = None
    
    # Prohibition flag comparison
    golden_prohibition: Optional[bool] = None
    retrieved_prohibition: Optional[bool] = None
    prohibition_correct: Optional[bool] = None
    
    # Obligation flag comparison
    golden_obligation: Optional[bool] = None
    retrieved_obligation: Optional[bool] = None
    obligation_correct: Optional[bool] = None
    
    # Permission flag comparison
    golden_permission: Optional[bool] = None
    retrieved_permission: Optional[bool] = None
    permission_correct: Optional[bool] = None
    
    # Overall metrics
    metadata_accuracy: Optional[float] = None  # None if not found in top 5
    llm_reasoning: str = ""  # LLM's explanation




def query_retrieval_engine(question: str, state: str, county: str, mode: str = "hybrid") -> dict:
    """
    Query the legal retrieval engine.
    
    Args:
        question: The query text
        state: State code (e.g., "CA", "GA")
        county: County name (e.g., "Alameda")
        mode: Retrieval mode - "hybrid" or "baseline"
    
    Returns the API response containing top-5 retrieved chunks.
    """
    # Format county name for API (convert to lowercase with hyphens)
    formatted_county = county.lower().replace(" ", "-")
    if not formatted_county.endswith("-county"):
        formatted_county = f"{formatted_county}-county"
    
    payload = {
        "query": question,
        "filters": {
            "locations": [
                {
                    "state": state.lower(),
                    "county": [formatted_county]
                }
            ]
        },
        "mode": mode
    }
    
    try:
        response = requests.post(RETRIEVAL_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def create_evaluation_prompt(
    question: str,
    golden_answer: str,
    golden_section: str,
    retrieved_chunks: list,
    is_negative_test: bool = False,
    system_response: str = ""
) -> str:
    """
    Create a structured evaluation prompt for the LLM judge.
    
    The LLM will determine:
    1. Whether the correct law appears in top-5
    2. Its rank and chunk coverage
    3. Metadata flags for BOTH golden truth AND retrieved chunks
    
    For negative tests (where no law should exist), the LLM evaluates if
    the system's response correctly indicates no relevant law was found.
    """
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks[:5], 1):
        # API uses 'chunk_text' field for the text content
        chunk_text = (chunk.get("chunk_text") or 
                      chunk.get("text") or 
                      chunk.get("content") or "")
        chunk_section = chunk.get("section", chunk.get("metadata", {}).get("section", "Unknown"))
        
        chunks_text += f"""
--- Chunk {i} ---
Section: {chunk_section}
Text: {chunk_text[:2000]}
"""
    
    if is_negative_test:
        # Special prompt for negative test cases - evaluate the SYSTEM'S RESPONSE
        prompt = f"""You are a legal retrieval evaluation expert. This is a NEGATIVE TEST - the question asks about something for which NO relevant law should exist.

## Query
Question: {question}

## Expected Result
There should be NO relevant law for this query. The question asks about something fictional, hypothetical, or not covered by any real municipal ordinance.

## System's LLM-Generated Response
{system_response}

## Your Evaluation Task
Evaluate whether the SYSTEM'S RESPONSE acknowledges that no relevant law exists for this specific query.

The response is CORRECT if the system:
- Says "no law exists" or "no regulations found" for the specific topic
- Says "the law does not mention" or "does not address" the specific topic
- Says "information was not found" for the query
- Acknowledges the topic is not covered, even if it provides some tangentially related information afterward

The response is INCORRECT only if the system:
- Claims to have found a law that DIRECTLY addresses the fictional/hypothetical query
- Provides legal guidance as if a specific law exists for the topic
- Does NOT acknowledge that no relevant law exists

IMPORTANT: If the system says something like "The law does not explicitly mention X" or "No regulations specifically related to X" - this counts as CORRECT, even if the system then mentions some related but different laws.

Provide your evaluation in the following JSON format:

```json
{{
    "system_says_no_law": true/false,
    "negative_test_correct": true/false,
    "reasoning": "Brief explanation"
}}
```

Respond ONLY with the JSON object, no additional text."""
    else:
        # Standard prompt for positive test cases
        prompt = f"""You are a legal retrieval evaluation expert. Evaluate the retrieval results for the following query.

## Query
Question: {question}

## Golden Truth (Expected Answer)
Section Reference: {golden_section}
Law Text: {golden_answer[:3000]}

## Retrieved Chunks (Top 5)
{chunks_text}

## Metadata Flag Definitions
You must analyze the text and determine if each flag applies. Use your legal expertise to understand context, not just keyword matching:

1. **Penalty/Fine Flag (Y/N)**: Does the text describe monetary penalties, fines, fees as punishment, imprisonment, or other punitive consequences for violations?

2. **Prohibition Flag (Y/N)**: Does the text prohibit certain actions? Look for language like "may not," "shall not," "prohibited," "unlawful," "it shall be unlawful," "no person shall," or similar prohibitive language.

3. **Obligation Flag (Y/N)**: Does the text impose mandatory requirements or duties? Look for language like "must," "shall" (when imposing a duty, NOT "shall not"), "required," "duty to," or similar obligatory language.

4. **Permission Flag (Y/N)**: Does the text grant permission or authorization? Look for language like "may" (when granting permission, NOT "may not"), "authorized," "permitted," "allowed," "entitled to," or similar permissive language.

## Your Evaluation Task
First, check if ANY retrieved chunk contains the ACTUAL TEXT from the golden truth section.

STEP 1: Compare each chunk's SECTION NUMBER with the golden truth section reference.
STEP 2: If a chunk has the SAME section number, verify it contains the SAME TEXT (not just references to it).
STEP 3: If NO chunk matches, set found_in_top5 to FALSE, rank to 0, and matching_chunk_index to null.

Provide your evaluation in JSON format:

```json
{{
    "found_in_top5": false,
    "rank": 0,
    "chunk_coverage": 0.0,
    "matching_chunk_index": null,
    "golden_metadata": {{
        "penalty_fine": true/false,
        "prohibition": true/false,
        "obligation": true/false,
        "permission": true/false
    }},
    "retrieved_metadata": {{
        "penalty_fine": false,
        "prohibition": false,
        "obligation": false,
        "permission": false
    }},
    "reasoning": "Explain which chunks you checked and why none/one matched"
}}
```

### CRITICAL RULES:
- **found_in_top5 = true** ONLY if a chunk contains the SAME SECTION and SAME TEXT as golden truth
- **found_in_top5 = false** if chunks only MENTION the section number but have DIFFERENT content
- **matching_chunk_index = null** when found_in_top5 is false
- **rank = 0** when found_in_top5 is false

### BE CONSISTENT: 
If your reasoning says "not found" or "does not appear", then found_in_top5 MUST be false.
If your reasoning says a chunk matches, then found_in_top5 MUST be true.

Respond ONLY with the JSON object, no additional text."""

    return prompt


def call_llm_judge(prompt: str) -> dict:
    """
    Call the NVIDIA NIMs API with the evaluation prompt.
    """
    headers = {
        "Authorization": f"Bearer {NIMS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert legal retrieval evaluator. Respond only with valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(NIMS_ENDPOINT, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        # Extract the content from the response
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"content": content, "raw": result}
        
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def normalize_section(section: str) -> str:
    """Normalize section reference for comparison."""
    if not section:
        return ""
    # Remove common variations: "5.08.010 - Running" vs "5.08.010. Running"
    # Extract just the numeric section code
    import re
    # Match patterns like "5.08.010", "Sec. 78-38", etc.
    match = re.search(r'[\d]+[.\-][\d]+[.\-]?[\d]*', str(section))
    if match:
        return match.group(0).replace('-', '.').replace('..', '.')
    return str(section).lower().strip()


def find_matching_chunk(golden_section: str, chunks: list) -> tuple:
    """
    Find a chunk that matches the golden section programmatically.
    Returns (found, rank, matching_chunk) or (False, 0, None).
    """
    golden_norm = normalize_section(golden_section)
    
    for i, chunk in enumerate(chunks[:5], 1):
        chunk_section = chunk.get("section", chunk.get("title", ""))
        chunk_norm = normalize_section(chunk_section)
        
        if golden_norm and chunk_norm and golden_norm == chunk_norm:
            return (True, i, chunk)
    
    return (False, 0, None)


def parse_llm_response(response_content: str) -> dict:
    """
    Parse the LLM's JSON response.
    """
    try:
        # Try to extract JSON from the response
        # Sometimes the LLM wraps it in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response_content)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {"error": "No JSON found in response"}
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}


def evaluate_single_query(
    query_id: int,
    row: pd.Series,
    mode: str = "hybrid"
) -> EvaluationResult:
    """
    Evaluate a single query from the dataset.
    
    Args:
        query_id: Index of the query
        row: DataFrame row with query data
        mode: Retrieval mode - "hybrid" or "baseline"
    """
    golden_answer = row['Answer']
    golden_section = row['Section']
    
    # Detect negative test cases (questions where no law should exist)
    is_negative_test = (
        golden_answer.upper() == "NO_LAW_EXISTS" or 
        str(golden_section).upper() == "N/A" or
        str(golden_section).upper() == "NAN"
    )
    
    result = EvaluationResult(
        query_id=query_id,
        state=row['State'],
        county=row['County'],
        difficulty=row['Difficulty Column'],
        question=row['Question'],
        golden_section=golden_section if not is_negative_test else "NO_LAW_EXISTS",
        golden_chunk=sanitize_for_csv(golden_answer) if not is_negative_test else "NO_LAW_EXISTS"
    )
    
    # Step 1: Query the retrieval engine
    retrieval_response = query_retrieval_engine(
        row['Question'],
        row['State'],
        row['County'],
        mode=mode
    )
    
    if "error" in retrieval_response:
        result.llm_reasoning = f"Retrieval error: {retrieval_response['error']}"
        return result
    
    # Extract chunks from response
    # The response structure may vary - try common patterns
    chunks = retrieval_response.get("results", 
              retrieval_response.get("chunks",
              retrieval_response.get("documents", [])))
    
    # Capture the system's LLM-generated response
    system_response = retrieval_response.get("response", "")
    result.system_response = sanitize_for_csv(system_response)
    result.is_negative_test = is_negative_test
    
    # Store all top 5 retrieved sections
    for i, chunk in enumerate(chunks[:5]):
        section = chunk.get("section", chunk.get("title", ""))
        section_str = sanitize_for_csv(str(section)) if section else ""
        if i == 0:
            result.retrieved_section_1 = section_str
        elif i == 1:
            result.retrieved_section_2 = section_str
        elif i == 2:
            result.retrieved_section_3 = section_str
        elif i == 3:
            result.retrieved_section_4 = section_str
        elif i == 4:
            result.retrieved_section_5 = section_str
    
    if not chunks:
        # For negative tests, no chunks is actually correct
        if is_negative_test:
            result.found_in_top5 = False
            result.system_says_no_law = True
            result.negative_test_correct = True
            result.llm_reasoning = "Correctly identified: No relevant law exists (no chunks returned)"
        else:
            result.llm_reasoning = "No chunks returned from retrieval"
        return result
    
    # Step 2: Create evaluation prompt and call LLM judge
    prompt = create_evaluation_prompt(
        row['Question'],
        golden_answer,
        golden_section,
        chunks,
        is_negative_test=is_negative_test,
        system_response=system_response
    )
    
    llm_response = call_llm_judge(prompt)
    
    if "error" in llm_response:
        result.llm_reasoning = f"LLM error: {llm_response['error']}"
        return result
    
    raw_content = llm_response.get("content", "")
    
    # Step 3: Parse LLM response
    parsed = parse_llm_response(raw_content)
    
    if "error" in parsed:
        result.llm_reasoning = f"Parse error: {parsed['error']}"
        return result
    
    # Extract reasoning
    result.llm_reasoning = sanitize_for_csv(parsed.get("reasoning", ""))
    
    # Handle negative tests differently
    if is_negative_test:
        result.system_says_no_law = parsed.get("system_says_no_law", False)
        result.negative_test_correct = parsed.get("negative_test_correct", False)
        # For negative tests, we don't need to extract chunk/metadata info
        return result
    
    # === POSITIVE TEST EVALUATION ===
    # First, try programmatic section matching (more reliable than LLM)
    prog_found, prog_rank, prog_chunk = find_matching_chunk(golden_section, chunks)
    
    if prog_found:
        # Use programmatic result - section numbers match
        result.found_in_top5 = True
        result.rank = prog_rank
        matching_idx = prog_rank
        matched_chunk = prog_chunk
    else:
        # Fall back to LLM evaluation
        result.found_in_top5 = parsed.get("found_in_top5", False)
        result.rank = parsed.get("rank", 0)
        matching_idx = parsed.get("matching_chunk_index")
        matched_chunk = chunks[matching_idx - 1] if matching_idx and 1 <= matching_idx <= len(chunks) else None
    
    # Extract the matched chunk text and section if found
    if result.found_in_top5 and matched_chunk:
        # Try different field names for chunk text (API uses 'chunk_text')
        chunk_text = (matched_chunk.get("chunk_text") or
                      matched_chunk.get("text") or 
                      matched_chunk.get("content") or 
                      matched_chunk.get("chunk") or
                      matched_chunk.get("page_content") or
                      matched_chunk.get("document") or "")
        
        # Try different field names for section
        chunk_section = (matched_chunk.get("section") or
                        matched_chunk.get("title") or
                        matched_chunk.get("metadata", {}).get("section") or
                        matched_chunk.get("metadata", {}).get("title") or
                        "Unknown")
        
        result.retrieved_chunk = sanitize_for_csv(str(chunk_text))  # Full chunk text
        result.retrieved_section = sanitize_for_csv(str(chunk_section))
        
        # Compute chunk coverage programmatically instead of relying on LLM estimate
        # Normalize both texts for comparison
        golden_normalized = golden_answer.strip().lower()
        retrieved_normalized = chunk_text.strip().lower()
        
        if golden_normalized == retrieved_normalized:
            # Exact match = 100% coverage
            result.chunk_coverage = 1.0
        elif golden_normalized in retrieved_normalized:
            # Golden is fully contained in retrieved = 100% coverage
            result.chunk_coverage = 1.0
        elif retrieved_normalized in golden_normalized:
            # Retrieved is subset of golden - compute ratio
            result.chunk_coverage = len(retrieved_normalized) / len(golden_normalized) if golden_normalized else 0.0
        else:
            # Compute word overlap for partial matches
            golden_words = set(golden_normalized.split())
            retrieved_words = set(retrieved_normalized.split())
            if golden_words:
                overlap = len(golden_words & retrieved_words) / len(golden_words)
                result.chunk_coverage = min(overlap, 1.0)
            else:
                result.chunk_coverage = parsed.get("chunk_coverage", 0.0)
    else:
        result.chunk_coverage = parsed.get("chunk_coverage", 0.0)
    
    # Extract golden metadata flags (determined by LLM)
    # Handle both "penalty_fine" and "penalty/fine" variants
    golden_meta = parsed.get("golden_metadata", {})
    result.golden_penalty_fine = golden_meta.get("penalty_fine", golden_meta.get("penalty/fine", False))
    result.golden_prohibition = golden_meta.get("prohibition", False)
    result.golden_obligation = golden_meta.get("obligation", False)
    result.golden_permission = golden_meta.get("permission", False)
    
    # Extract retrieved metadata flags (determined by LLM)
    retrieved_meta = parsed.get("retrieved_metadata", {})
    result.retrieved_penalty_fine = retrieved_meta.get("penalty_fine", retrieved_meta.get("penalty/fine"))
    result.retrieved_prohibition = retrieved_meta.get("prohibition")
    result.retrieved_obligation = retrieved_meta.get("obligation")
    result.retrieved_permission = retrieved_meta.get("permission")
    
    # Only compute flag correctness if the answer was found in top 5
    if result.found_in_top5:
        # Compare LLM's assessment of golden vs retrieved flags
        if result.retrieved_penalty_fine is not None and result.golden_penalty_fine is not None:
            result.penalty_fine_correct = (result.retrieved_penalty_fine == result.golden_penalty_fine)
        if result.retrieved_prohibition is not None and result.golden_prohibition is not None:
            result.prohibition_correct = (result.retrieved_prohibition == result.golden_prohibition)
        if result.retrieved_obligation is not None and result.golden_obligation is not None:
            result.obligation_correct = (result.retrieved_obligation == result.golden_obligation)
        if result.retrieved_permission is not None and result.golden_permission is not None:
            result.permission_correct = (result.retrieved_permission == result.golden_permission)
        
        # Calculate overall metadata accuracy
        metadata_scores = []
        if result.penalty_fine_correct is not None:
            metadata_scores.append(1.0 if result.penalty_fine_correct else 0.0)
        if result.prohibition_correct is not None:
            metadata_scores.append(1.0 if result.prohibition_correct else 0.0)
        if result.obligation_correct is not None:
            metadata_scores.append(1.0 if result.obligation_correct else 0.0)
        if result.permission_correct is not None:
            metadata_scores.append(1.0 if result.permission_correct else 0.0)
        
        if metadata_scores:
            result.metadata_accuracy = sum(metadata_scores) / len(metadata_scores)
    # If not found in top 5, flag correctness and metadata_accuracy remain None
    
    return result


def compute_aggregate_metrics(results: list[EvaluationResult]) -> dict:
    """
    Compute aggregate metrics across all evaluation results.
    """
    # Valid results are those where LLM reasoning doesn't indicate an error
    valid_results = [r for r in results if not r.llm_reasoning.startswith(("Retrieval error:", "LLM error:", "Parse error:"))]
    
    if not valid_results:
        return {"error": "No valid results to compute metrics"}
    
    # Separate positive and negative test cases
    positive_results = [r for r in valid_results if not r.is_negative_test]
    negative_results = [r for r in valid_results if r.is_negative_test]
    
    n = len(valid_results)
    n_positive = len(positive_results)
    n_negative = len(negative_results)
    
    # === POSITIVE TEST METRICS ===
    # Top-5 Recall (only for positive tests)
    if n_positive > 0:
        top5_hits = sum(1 for r in positive_results if r.found_in_top5)
        top5_recall = top5_hits / n_positive
        
        # MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for r in positive_results:
            if r.rank > 0:
                reciprocal_ranks.append(1.0 / r.rank)
            else:
                reciprocal_ranks.append(0.0)
        mrr = sum(reciprocal_ranks) / n_positive
        
        # Average Chunk Coverage (only for queries where answer was found)
        coverages = [r.chunk_coverage for r in positive_results if r.found_in_top5]
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        
        # Metadata Accuracy
        metadata_accuracies = [r.metadata_accuracy for r in positive_results 
                              if r.found_in_top5 and r.metadata_accuracy is not None]
        avg_metadata_accuracy = sum(metadata_accuracies) / len(metadata_accuracies) if metadata_accuracies else 0.0
        
        # Per-flag accuracy
        found_results = [r for r in positive_results if r.found_in_top5]
        n_found = len(found_results)
        
        penalty_correct = sum(1 for r in found_results if r.penalty_fine_correct is True)
        prohibition_correct = sum(1 for r in found_results if r.prohibition_correct is True)
        obligation_correct = sum(1 for r in found_results if r.obligation_correct is True)
        permission_correct = sum(1 for r in found_results if r.permission_correct is True)
    else:
        top5_recall = None
        mrr = None
        avg_coverage = None
        avg_metadata_accuracy = None
        n_found = 0
        penalty_correct = prohibition_correct = obligation_correct = permission_correct = 0
    
    # === NEGATIVE TEST METRICS ===
    # Evaluate based on whether system correctly said no law exists
    if n_negative > 0:
        # True Negatives: system correctly identified no relevant law
        true_negatives = sum(1 for r in negative_results if r.negative_test_correct is True)
        # False Positives: system incorrectly claimed a law exists
        false_positives = sum(1 for r in negative_results if r.negative_test_correct is False)
        # Pending: not yet evaluated (e.g., errors)
        pending = sum(1 for r in negative_results if r.negative_test_correct is None)
        negative_accuracy = true_negatives / n_negative if n_negative > 0 else None
    else:
        true_negatives = 0
        false_positives = 0
        pending = 0
        negative_accuracy = None
    
    # By difficulty
    difficulty_metrics = {}
    for diff in ['Easy', 'Medium', 'Hard']:
        diff_results = [r for r in positive_results if r.difficulty == diff]
        if diff_results:
            diff_hits = sum(1 for r in diff_results if r.found_in_top5)
            diff_mrr_values = [1.0/r.rank if r.rank > 0 else 0.0 for r in diff_results]
            difficulty_metrics[diff] = {
                "count": len(diff_results),
                "top5_recall": diff_hits / len(diff_results),
                "mrr": sum(diff_mrr_values) / len(diff_results)
            }
    
    # Composite Score (weighted average) - only for positive tests
    if n_positive > 0 and top5_recall is not None:
        composite_score = (
            0.30 * (top5_recall or 0) +
            0.30 * (mrr or 0) +
            0.20 * (avg_coverage or 0) +
            0.20 * (avg_metadata_accuracy or 0)
        )
    else:
        composite_score = None
    
    return {
        "total_queries": len(results),
        "valid_queries": n,
        "failed_queries": len(results) - n,
        
        # Positive test metrics
        "positive_test_count": n_positive,
        "top5_recall": top5_recall,
        "mrr": mrr,
        "avg_chunk_coverage": avg_coverage,
        "avg_metadata_accuracy": avg_metadata_accuracy,
        
        "penalty_fine_accuracy": penalty_correct / n_found if n_found > 0 else None,
        "prohibition_accuracy": prohibition_correct / n_found if n_found > 0 else None,
        "obligation_accuracy": obligation_correct / n_found if n_found > 0 else None,
        "permission_accuracy": permission_correct / n_found if n_found > 0 else None,
        "queries_with_answer_in_top5": n_found,
        
        # Negative test metrics
        "negative_test_count": n_negative,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "negative_accuracy": negative_accuracy,
        
        "by_difficulty": difficulty_metrics,
        
        "composite_score": composite_score
    }


def main():
    """
    Main evaluation pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal Retrieval Engine Evaluator")
    parser.add_argument(
        "--input", "-i",
        default="eval_dataset_final.csv",
        help="Input evaluation dataset (CSV or Excel)"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation_results.csv",
        help="Output file for per-query results"
    )
    parser.add_argument(
        "--summary", "-s",
        default="evaluation_summary.json",
        help="Output file for summary metrics"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (for testing)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds"
    )
    parser.add_argument(
        "--mode", "-m",
        default="hybrid",
        choices=["hybrid", "baseline"],
        help="Retrieval mode: 'hybrid' or 'baseline'"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading evaluation dataset from {args.input}...")
    if args.input.endswith('.xlsx') or args.input.endswith('.xls'):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input, encoding='utf-8-sig')
    
    print(f"Loaded {len(df)} queries")
    print(f"Retrieval mode: {args.mode}")
    
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} queries")
    
    # Evaluate each query
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        result = evaluate_single_query(idx, row, mode=args.mode)
        results.append(result)
        
        # Rate limiting
        time.sleep(args.delay)
    
    # Convert results to DataFrame
    results_data = [asdict(r) for r in results]
    results_df = pd.DataFrame(results_data)
    
    # Remove negative test columns from output (keep logic internal only)
    columns_to_drop = ['system_response', 'is_negative_test', 'system_says_no_law', 'negative_test_correct']
    results_df = results_df.drop(columns=[c for c in columns_to_drop if c in results_df.columns])
    
    # Save per-query results
    print(f"\nSaving per-query results to {args.output}...")
    if args.output.endswith('.xlsx'):
        results_df.to_excel(args.output, index=False)
    else:
        # Use proper CSV quoting to handle special characters
        import csv
        results_df.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    
    # Compute and save aggregate metrics
    print("Computing aggregate metrics...")
    metrics = compute_aggregate_metrics(results)
    
    print(f"\nSaving summary metrics to {args.summary}...")
    with open(args.summary, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Queries:       {metrics.get('total_queries', 'N/A')}")
    print(f"Valid Queries:       {metrics.get('valid_queries', 'N/A')}")
    print(f"Failed Queries:      {metrics.get('failed_queries', 'N/A')}")
    
    # Positive test metrics
    n_positive = metrics.get('positive_test_count', 0)
    if n_positive > 0:
        print()
        print(f"=== POSITIVE TESTS (laws that should exist): {n_positive} ===")
        print(f"Found in Top-5:      {metrics.get('queries_with_answer_in_top5', 'N/A')}")
        top5 = metrics.get('top5_recall')
        mrr_val = metrics.get('mrr')
        cov = metrics.get('avg_chunk_coverage')
        meta = metrics.get('avg_metadata_accuracy')
        print(f"Top-5 Recall:        {top5:.2%}" if top5 is not None else "Top-5 Recall:        N/A")
        print(f"MRR:                 {mrr_val:.4f}" if mrr_val is not None else "MRR:                 N/A")
        print(f"Avg Chunk Coverage:  {cov:.2%}" if cov is not None else "Avg Chunk Coverage:  N/A")
        print(f"Avg Metadata Acc:    {meta:.2%}" if meta is not None else "Avg Metadata Acc:    N/A")
        print()
        print("Metadata Flag Accuracy:")
        pf_acc = metrics.get('penalty_fine_accuracy')
        pr_acc = metrics.get('prohibition_accuracy')
        ob_acc = metrics.get('obligation_accuracy')
        pe_acc = metrics.get('permission_accuracy')
        print(f"  Penalty/Fine:      {pf_acc:.2%}" if pf_acc is not None else "  Penalty/Fine:      N/A")
        print(f"  Prohibition:       {pr_acc:.2%}" if pr_acc is not None else "  Prohibition:       N/A")
        print(f"  Obligation:        {ob_acc:.2%}" if ob_acc is not None else "  Obligation:        N/A")
        print(f"  Permission:        {pe_acc:.2%}" if pe_acc is not None else "  Permission:        N/A")
        print()
        print("By Difficulty:")
        for diff, diff_metrics in metrics.get('by_difficulty', {}).items():
            print(f"  {diff}: Recall={diff_metrics['top5_recall']:.2%}, MRR={diff_metrics['mrr']:.4f} (n={diff_metrics['count']})")
    
    # Negative test metrics
    n_negative = metrics.get('negative_test_count', 0)
    if n_negative > 0:
        print()
        print(f"=== NEGATIVE TESTS (no law should exist): {n_negative} ===")
        print(f"True Negatives:      {metrics.get('true_negatives', 0)} (correctly identified)")
        print(f"False Positives:     {metrics.get('false_positives', 0)} (incorrectly claimed law exists)")
        neg_acc = metrics.get('negative_accuracy')
        print(f"Negative Accuracy:   {neg_acc:.2%}" if neg_acc is not None else "Negative Accuracy:   N/A")
    
    print()
    comp_score = metrics.get('composite_score')
    if comp_score is not None:
        print(f"COMPOSITE SCORE:     {comp_score:.2%}")
    else:
        print("COMPOSITE SCORE:     N/A (no positive tests)")
    print("="*60)


if __name__ == "__main__":
    main()

