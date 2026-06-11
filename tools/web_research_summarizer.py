#!/usr/bin/env python3
"""Query-aware summarize-at-fetch for web_research — a fork-owned overlay.

This module holds Alex's query-aware summarization so that ``tools/web_tools.py`` stays
byte-identical to its NousResearch upstream parent (eliminating the worst future-merge
conflict surface). It reuses the STABLE, upstream-owned seam — ``_resolve_web_extract_auxiliary``
(from ``tools.web_tools``) plus ``async_call_llm`` / ``extract_content_or_reasoning`` (from
``agent.auxiliary_client``) — and reproduces, behaviourally identically, what
``process_content_with_llm(..., query=query)`` used to do: run each fetched page through the
cheap ``auxiliary.web_extract`` side-model with the search query and return a query-relevant
markdown extract (or ``None`` on short/empty content, a truncated-raw banner on timeout, a
``"[Content too large…]"`` placeholder above 2M chars, a ``"[Failed to process…]"`` placeholder
when every chunk fails).

The ONLY consumer is ``tools/web_research_tool.py`` (``_fetch`` → ``summarize_for_query``).
Net contract is unchanged from the previous in-``web_tools`` query path; the sentinel-guard
fallback in ``_fetch`` (None/placeholder → capped raw body) is unchanged and still applies.
"""

import asyncio
import logging
from typing import Optional

from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from tools.web_tools import DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION, _resolve_web_extract_auxiliary

logger = logging.getLogger(__name__)

# Size thresholds — inlined from the former process_content_with_llm (byte-identical values).
MAX_CONTENT_SIZE = 2_000_000  # 2M chars - refuse entirely above this
CHUNK_THRESHOLD = 500_000     # 500k chars - use chunked processing above this
CHUNK_SIZE = 100_000          # 100k chars per chunk
MAX_OUTPUT_SIZE = 5000        # Hard cap on final output size


async def _call_query_summarizer(
    content: str,
    context_str: str,
    query: str,
    model: Optional[str],
    max_tokens: int = 20000,
    is_chunk: bool = False,
    chunk_info: str = "",
) -> Optional[str]:
    """Single query-focused LLM call (the former _call_summarizer_llm `if query:` branch).

    Returns the extracted markdown, or None on failure / no auxiliary model. Same low-retry
    contract as the original (summarization is a nice-to-have; the caller falls back to
    truncated content).
    """
    if is_chunk:
        system_prompt = """You are an expert research assistant processing a SECTION of a larger document. Extract ONLY information in THIS SECTION that is directly relevant to the user's research query.

Guidelines:
1. Do NOT write introductions or conclusions - this is a partial document.
2. Extract all relevant facts, figures, data points, quotes, and insights; keep numbers, dates, names, and formulas verbatim.
3. If this section contains nothing relevant to the query, respond with exactly: "No relevant information found."
4. Use bullet points; your output will be combined with other sections."""

        user_prompt = f"""Research Query: {query}

{context_str}{chunk_info}

SECTION CONTENT:
{content}

Extract only the information in this section relevant to the query '{query}'."""
    else:
        system_prompt = """You are an expert research assistant. Your task is to extract information from the web page content that is directly relevant to the user's research query.

Guidelines:
1. Extract all relevant facts, data points, figures, quotes, and insights.
2. Be concise but thorough: keep only what is useful for answering the query.
3. Keep specific numbers, dates, names, and formulas verbatim.
4. If the content does not contain any information relevant to the query, respond with exactly: "No relevant information found."
5. Format your output using clear markdown (e.g. bullet points)."""

        user_prompt = f"""Research Query: {query}

{context_str}CONTENT TO PROCESS:
{content}

Extract all important information relevant to the query '{query}'."""

    # Call the LLM with retry logic — keep retries low since summarization
    # is a nice-to-have; the caller falls back to truncated content on failure.
    max_retries = 2
    retry_delay = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            aux_client, effective_model, extra_body = _resolve_web_extract_auxiliary(model)
            if aux_client is None or not effective_model:
                logger.warning("No auxiliary model available for web content processing")
                return None
            call_kwargs = {
                "task": "web_extract",
                "model": effective_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                # No explicit timeout — async_call_llm reads auxiliary.web_extract.timeout
                # from config.yaml. Fresh configs ship with 360s; if the key is absent
                # the runtime default is 30s (_DEFAULT_AUX_TIMEOUT in
                # agent/auxiliary_client.py). Users with slow local models should set
                # or increase auxiliary.web_extract.timeout in config.yaml.
            }
            if extra_body:
                call_kwargs["extra_body"] = extra_body
            response = await async_call_llm(**call_kwargs)
            text = extract_content_or_reasoning(response)
            if text:
                return text
            # Reasoning-only / empty response — let the retry loop handle it
            logger.warning("LLM returned empty content (attempt %d/%d), retrying", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                continue
            return text  # Return whatever we got after exhausting retries
        except RuntimeError:
            logger.warning("No auxiliary model available for web content processing")
            return None
        except Exception as api_error:
            last_error = api_error
            if attempt < max_retries - 1:
                logger.warning("LLM API call failed (attempt %d/%d): %s", attempt + 1, max_retries, str(api_error)[:100])
                logger.warning("Retrying in %ds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                raise last_error

    return None


async def _summarize_large_chunked(
    content: str,
    context_str: str,
    query: str,
    model: Optional[str],
    chunk_size: int,
    max_output_size: int,
) -> Optional[str]:
    """Chunk large content, summarize each chunk in parallel, then synthesize.

    The former _process_large_content_chunked `query` path — query-only synthesis prompt.
    """
    # Split content into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        chunks.append(chunk)

    logger.info("Split into %d chunks of ~%d chars each", len(chunks), chunk_size)

    # Summarize each chunk in parallel
    async def summarize_chunk(chunk_idx: int, chunk_content: str) -> tuple[int, Optional[str]]:
        """Summarize a single chunk."""
        try:
            chunk_info = f"[Processing chunk {chunk_idx + 1} of {len(chunks)}]"
            summary = await _call_query_summarizer(
                chunk_content,
                context_str,
                query,
                model,
                max_tokens=10000,
                is_chunk=True,
                chunk_info=chunk_info,
            )
            if summary:
                logger.info("Chunk %d/%d summarized: %d -> %d chars", chunk_idx + 1, len(chunks), len(chunk_content), len(summary))
            return chunk_idx, summary
        except Exception as e:
            logger.warning("Chunk %d/%d failed: %s", chunk_idx + 1, len(chunks), str(e)[:50])
            return chunk_idx, None

    # Run all chunk summarizations in parallel
    tasks = [summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    # Use return_exceptions=True so a single task failure does not discard
    # all other successfully summarized chunks.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions, then collect successful summaries in order
    successful_results = []
    for result_item in results:
        if isinstance(result_item, BaseException):
            logger.warning("Chunk summarization task failed: %s", result_item)
            continue
        successful_results.append(result_item)

    summaries = []
    for chunk_idx, summary in sorted(successful_results, key=lambda x: x[0]):
        if summary:
            summaries.append(f"## Section {chunk_idx + 1}\n{summary}")

    if not summaries:
        logger.debug("All chunk summarizations failed")
        return "[Failed to process large content: all chunk summarizations failed]"

    logger.info("Got %d/%d chunk summaries", len(summaries), len(chunks))

    # If only one chunk succeeded, just return it (with cap)
    if len(summaries) == 1:
        result = summaries[0]
        if len(result) > max_output_size:
            result = result[:max_output_size] + "\n\n[... truncated ...]"
        return result

    # Synthesize the summaries into a final summary
    logger.info("Synthesizing %d summaries...", len(summaries))

    combined_summaries = "\n\n---\n\n".join(summaries)

    synthesis_prompt = f"""You have been given extracts from different sections of a large document, all relevant to a research query.
Synthesize them into ONE cohesive, non-redundant extraction that:
1. Keeps only information relevant to the query: {query}
2. Preserves all key facts, figures, numbers, dates, names, and quotes verbatim
3. Is well-organized markdown, under {max_output_size} characters
4. If no section had relevant information, respond with exactly: "No relevant information found."

{context_str}SECTION EXTRACTS:
{combined_summaries}

Create a single, unified markdown extraction relevant to the query."""

    try:
        aux_client, effective_model, extra_body = _resolve_web_extract_auxiliary(model)
        if aux_client is None or not effective_model:
            logger.warning("No auxiliary model for synthesis, concatenating summaries")
            fallback = "\n\n".join(summaries)
            if len(fallback) > max_output_size:
                fallback = fallback[:max_output_size] + "\n\n[... truncated ...]"
            return fallback

        call_kwargs = {
            "task": "web_extract",
            "model": effective_model,
            "messages": [
                {"role": "system", "content": "You synthesize multiple summaries into one cohesive, comprehensive summary. Be thorough but concise."},
                {"role": "user", "content": synthesis_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 20000,
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        response = await async_call_llm(**call_kwargs)
        final_summary = extract_content_or_reasoning(response)

        # Retry once on empty content (reasoning-only response)
        if not final_summary:
            logger.warning("Synthesis LLM returned empty content, retrying once")
            response = await async_call_llm(**call_kwargs)
            final_summary = extract_content_or_reasoning(response)

        # If still None after retry, fall back to concatenated summaries
        if not final_summary:
            logger.warning("Synthesis failed after retry — concatenating chunk summaries")
            fallback = "\n\n".join(summaries)
            if len(fallback) > max_output_size:
                fallback = fallback[:max_output_size] + "\n\n[... truncated ...]"
            return fallback

        # Enforce hard cap
        if len(final_summary) > max_output_size:
            final_summary = final_summary[:max_output_size] + "\n\n[... summary truncated for context management ...]"

        original_len = len(content)
        final_len = len(final_summary)
        compression = final_len / original_len if original_len > 0 else 1.0

        logger.info("Synthesis complete: %d -> %d chars (%.2f%%)", original_len, final_len, compression * 100)
        return final_summary

    except Exception as e:
        logger.warning("Synthesis failed: %s", str(e)[:100])
        # Fall back to concatenated summaries with truncation
        fallback = "\n\n".join(summaries)
        if len(fallback) > max_output_size:
            fallback = fallback[:max_output_size] + "\n\n[... truncated due to synthesis failure ...]"
        return fallback


async def summarize_for_query(
    content: str,
    query: str,
    *,
    url: str = "",
    title: str = "",
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION,
) -> Optional[str]:
    """Extract the query-relevant portion of ``content`` via the cheap aux model.

    Behaviourally identical to the former ``process_content_with_llm(content, url=url,
    title=title, model=model, min_length=min_length, query=query)``:
    - ``> MAX_CONTENT_SIZE`` (2M) → a ``"[Content too large…]"`` placeholder string.
    - ``< min_length`` → ``None`` (caller keeps the raw page).
    - ``> CHUNK_THRESHOLD`` (500k) → chunk + synthesize.
    - else → single-pass; output hard-capped at ``MAX_OUTPUT_SIZE`` (5000).
    - on exception → truncated raw content + a timeout banner.
    """
    try:
        content_len = len(content)

        # Refuse if content is absurdly large
        if content_len > MAX_CONTENT_SIZE:
            size_mb = content_len / 1_000_000
            logger.warning("Content too large (%.1fMB > 2MB limit). Refusing to process.", size_mb)
            return f"[Content too large to process: {size_mb:.1f}MB. Try a more focused source URL.]"

        # Skip processing if content is too short
        if content_len < min_length:
            logger.debug("Content too short (%d < %d chars), skipping LLM processing", content_len, min_length)
            return None

        # Create context information
        context_info = []
        if title:
            context_info.append(f"Title: {title}")
        if url:
            context_info.append(f"Source: {url}")
        context_str = "\n".join(context_info) + "\n\n" if context_info else ""

        # Check if we need chunked processing
        if content_len > CHUNK_THRESHOLD:
            logger.info("Content large (%d chars). Using chunked processing...", content_len)
            return await _summarize_large_chunked(
                content, context_str, query, model, CHUNK_SIZE, MAX_OUTPUT_SIZE
            )

        # Standard single-pass processing for normal content
        logger.info("Processing content with LLM (%d characters)", content_len)

        processed_content = await _call_query_summarizer(content, context_str, query, model)

        if processed_content:
            # Enforce output cap
            if len(processed_content) > MAX_OUTPUT_SIZE:
                processed_content = processed_content[:MAX_OUTPUT_SIZE] + "\n\n[... summary truncated for context management ...]"

            # Log compression metrics
            processed_length = len(processed_content)
            compression_ratio = processed_length / content_len if content_len > 0 else 1.0
            logger.info("Content processed: %d -> %d chars (%.1f%%)", content_len, processed_length, compression_ratio * 100)

        return processed_content

    except Exception as e:
        logger.warning(
            "web_research query summarization failed (%s). "
            "Tip: increase auxiliary.web_extract.timeout in config.yaml "
            "or switch to a faster auxiliary model.",
            str(e)[:120],
        )
        # Fall back to truncated raw content instead of returning a useless
        # error message.  The first ~5000 chars are almost always more useful
        # to the model than "[Failed to process content: ...]".
        truncated = content[:MAX_OUTPUT_SIZE]
        if len(content) > MAX_OUTPUT_SIZE:
            truncated += (
                f"\n\n[Content truncated — showing first {MAX_OUTPUT_SIZE:,} of "
                f"{len(content):,} chars. LLM summarization timed out. "
                f"To fix: increase auxiliary.web_extract.timeout in config.yaml, "
                f"or use a faster auxiliary model. Use browser_navigate for the full page.]"
            )
        return truncated
