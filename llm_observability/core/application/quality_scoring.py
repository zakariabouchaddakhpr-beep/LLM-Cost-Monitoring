"""
Judge-powered enrichment: quality scoring and query classification.

Both answers come from ONE call. The judge already receives the prompt and the
response and already returns JSON, so asking it for a classification as well
costs nothing extra — where a dedicated classifier would have been a third API
call on every request, adding both money and latency to the hot path.
"""

import json
import logging

import litellm

from llm_observability.config.settings import JUDGE_MODEL

logger = logging.getLogger(__name__)

# A closed set, on purpose. A judge answering in its own words produces many
# spellings of the same idea and GROUP BY query_type stops meaning anything.
# The split is chosen to line up with the routing question — could a cheaper
# model have handled this — and a bounded set is also the only case where
# ClickHouse's LowCardinality pays off.
QUERY_TYPES = (
    "summarization", "question-answering", "code-generation", "extraction",
    "classification", "translation", "reasoning", "creative-writing",
    "conversation", "other",
)

_EMPTY = {"quality_score": None, "quality_reasoning": None, "query_type": None}


def _coerce_query_type(raw) -> str | None:
    """Anything the judge invents becomes 'other' rather than polluting the
    vocabulary or being dropped silently."""
    if raw is None:
        return None
    label = str(raw).strip().lower()
    return label if label in QUERY_TYPES else "other"


def _build_prompt(original_prompt: str, response_text: str,
                  want_score: bool, want_query_type: bool) -> str:
    asks, shape = [], []
    if want_score:
        asks.append("Rate the RESPONSE from 1 (poor) to 10 (excellent) on how "
                    "relevant, clear and complete it is as an answer to the PROMPT.")
        shape.append('"score": <integer 1-10>, "reasoning": "<one short sentence>"')
    if want_query_type:
        asks.append("Classify what kind of request the PROMPT is, choosing "
                    f"exactly one of: {', '.join(QUERY_TYPES)}.")
        shape.append('"query_type": "<one of the listed values>"')

    return f"""You are an impartial evaluator.
{chr(10).join(asks)}

PROMPT:
{original_prompt}

RESPONSE:
{response_text}

Reply with ONLY a valid JSON object, no markdown, no code fences, in this
exact format:
{{{', '.join(shape)}}}
"""


def evaluate(original_prompt: str, response_text: str, *,
             want_score: bool = False, want_query_type: bool = False,
             api_key: str | None = None) -> dict:
    """
    Ask the judge for a quality score, a query classification, or both.

    Returns a dict with quality_score, quality_reasoning and query_type. Any
    field not requested, or not obtainable, is None — a failure here must never
    interrupt the call being measured.
    """
    if not (want_score or want_query_type):
        return dict(_EMPTY)
    if not response_text:
        return dict(_EMPTY)

    try:
        extra = {"api_key": api_key} if api_key else {}
        judge_response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": _build_prompt(
                original_prompt, response_text, want_score, want_query_type)}],
            temperature=0,
            **extra,
        )
        raw = judge_response.choices[0].message.content.strip()

        # Defensive cleanup in case the judge wraps output in code fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)
        out = dict(_EMPTY)
        if want_score and parsed.get("score") is not None:
            out["quality_score"] = max(1, min(10, int(parsed["score"])))
            out["quality_reasoning"] = parsed.get("reasoning")
        if want_query_type:
            out["query_type"] = _coerce_query_type(parsed.get("query_type"))
        return out

    except Exception as exc:
        logger.warning("Judge call failed, continuing without it: %s", exc)
        return dict(_EMPTY)
