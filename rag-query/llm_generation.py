"""
LLM generation utilities for RAG pipeline.
"""
import anthropic
from typing import List, Dict, Any, Optional

from config import Config


def build_context_string(retrieved_chunks: List[dict], max_chunks: Optional[int] = None) -> str:
    """
    Send only useful metadata to the LLM.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        max_chunks: Maximum number of chunks to include (for filter-only search)

    Returns:
        Formatted context string for LLM
    """
    context_string = ""
    if not retrieved_chunks:
        return "No documents were retrieved."

    matches_to_process = retrieved_chunks  # standard search
    if max_chunks is not None:  # filter-only search, only send the first N chunks to LLM
        matches_to_process = matches_to_process[:max_chunks]

    for i, match in enumerate(matches_to_process):
        metadata = match.get('metadata', {})
        score = match.get('score', 0)

        chunk_text = metadata.get('text', 'N/A')
        pi = metadata.get('Primary Investigator (Award Period) (Award Reports)', 'N/A')
        project_title = metadata.get('Project Title (Award Period) (Award Reports)', 'N/A')
        award_period = metadata.get('Award Period', 'N/A')
        program_area = metadata.get('Program Area (Award Period) (Award Reports)', 'N/A')
        award_type = metadata.get('Award Type (Award Period) (Award Reports)', 'N/A')
        award_amount = metadata.get('Award Amount (Award Period) (Award Reports)', 'N/A')
        name = metadata.get('Name', 'N/A')
        page = metadata.get('page', 'N/A')
        report_link = metadata.get('Report Link', 'N/A')

        context_string += f"[Chunk {i+1}]\n"
        context_string += f"Score: {score:.4f}\n"
        context_string += f"PI: {pi}\n"
        context_string += f"Project Title: {project_title}\n"
        context_string += f"Award Period: {award_period}\n"
        context_string += f"Program Area: {program_area}\n"
        context_string += f"Award Type: {award_type}\n"
        context_string += f"Award Amount: {award_amount}\n"
        context_string += f"Name: {name}\n"
        context_string += f"Page: {page}\n"
        context_string += f"Report Link: {report_link}\n"
        context_string += f"Text: \"{chunk_text}\"\n\n"

    return context_string


def generate_llm_response(query_text: str, context_string: str) -> str:
    """
    Generate LLM response for standard search queries.

    Args:
        query_text: User's query
        context_string: Context from retrieved chunks

    Returns:
        Generated response text
    """
    system_prompt = """
    You are a highly intelligent program evaluation analyst with a scientific background. Your goal is to help a
    user understand the scientific research your organization has funded.
    You will be given the user’s original question and a list of ‘Retrieved Chunks’ from the organization’s database.
    Each chunk includes metadata: PI, Project Title, Award Period, Program Area, Award Type, Award Amount, Name, and Page.

    Rules:
    1. Base your answer *ONLY* on the information inside the "Retrieved Chunks" and metadata. Do not use outside knowledge.
    2. If the context doesn’t answer the question, say so. Never guess.
    3. Cite every claim using only the structured metadata fields: [PI, Project Title, Award Period, Page]. Never extract citation info from the chunk text itself.
    4. Neutral tone. No "impressive," "promising," or "significant." State what was proposed, done, and incomplete.
    5. Lead with numbers over adjectives. Flag contradictions between stated completion % and described work.
    6. Use "The investigator reports..." not declarative statements. Mark synthesis across sources.
    7. Only use gene/protein/compound names that appear in the context above.
  """

    user_prompt = f"""
    **User's Question:**
    {query_text}

    **Retrieved Chunks:**
    {context_string}
  """

    client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=Config.CLAUDE_MODEL,
        max_tokens=Config.MAX_NEW_TOKENS,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )

    return message.content[0].text


# TODO: get rid of this function probs
def generate_llm_response_filter_only_search(
    query_text: str,
    context_string: str,
    num_total_chunks: int
) -> str:
    """
    Generate LLM response for filter-only searches.

    Args:
        query_text: User's query (empty for filter-only)
        context_string: Context from retrieved chunks sample
        num_total_chunks: Total number of chunks retrieved

    Returns:
        Generated response text with summary
    """
    system_prompt = """
    You are a highly intelligent program evaluation analyst with a scientific background.
    You will be given a sample of retrieved research funding reports.
    Your task is to provide a high-level summary of the main themes found in this sample.

    - DO NOT try to answer a question.
    - DO NOT say "I cannot find an answer."
    - Simply summarize what you see. Group similar topics together.
    - Start your response with: "The reports in this sample primarily discuss..."
  """

    user_prompt = f"""
    **Retrieved Chunks (Sample):**
    {context_string}
  """

    client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=Config.CLAUDE_MODEL,
        max_tokens=Config.MAX_NEW_TOKENS,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text

    return (
        f"Found {num_total_chunks} reports matching your filters.\n\n"
        f"Here is a quick summary of the first 10 results:\n\n"
        f"{response_text}"
    )