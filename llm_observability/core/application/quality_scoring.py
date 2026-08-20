"""
Using a second AI model as a "judge" to mark the first one's answer.
"""

import json
import logging

import litellm

from llm_observability.config.settings import JUDGE_MODEL

logger = logging.getLogger(__name__)

QUERY_TYPES = (
    "summarization", "question-answering", "code-generation", "extraction",
    "classification", "translation", "reasoning", "creative-writing",
    "conversation", "other",
)

_EMPTY = {"quality_score": None, "quality_reasoning": None, "query_type": None}


def _clean_query_type(raw) -> str | None:
    if raw is None:
        return None

    label = str(raw).strip().lower()
    return label if label in QUERY_TYPES else "other"


def _build_prompt(original_prompt: str, response_text: str,
                  want_score: bool, want_query_type: bool) -> str:

    instructions = []
    json_fields = []

    if want_score:
        instructions.append(
            "Rate the RESPONSE from 1 (poor) to 10 (excellent) on how "
            "relevant, clear and complete it is as an answer to the PROMPT."
        )
        json_fields.append(
            '"score": <integer 1-10>, "reasoning": "<one short sentence>"'
        )

    if want_query_type:
        allowed = ", ".join(QUERY_TYPES)
        instructions.append(
            "Classify what kind of request the PROMPT is, choosing "
            f"exactly one of: {allowed}."
        )
        json_fields.append('"query_type": "<one of the listed values>"')
    instruction_text = "\n".join(instructions)
    json_shape = "{" + ", ".join(json_fields) + "}"


    return f"""You are an impartial evaluator.
{instruction_text}

PROMPT:
{original_prompt}

RESPONSE:
{response_text}

Reply with ONLY a valid JSON object, no markdown, no code fences, in this
exact format:
{json_shape}
"""


def _strip_code_fences(raw: str) -> str:
    if not raw.startswith("```"):
        return raw
    raw = raw.strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return raw


def evaluate(original_prompt: str, response_text: str, *,
             want_score: bool = False, want_query_type: bool = False,
             api_key: str | None = None) -> dict:

    if not (want_score or want_query_type):
        return dict(_EMPTY)

    if not response_text:
        return dict(_EMPTY)

    try:
        auth = {"api_key": api_key} if api_key else {}
        judge_prompt = _build_prompt(original_prompt, response_text,
                                     want_score, want_query_type)

        judge_response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],

            temperature=0,
            **auth,
        )

        raw = judge_response.choices[0].message.content.strip()
        parsed = json.loads(_strip_code_fences(raw))

        result = dict(_EMPTY)

        if want_score and parsed.get("score") is not None:
            score = int(parsed["score"])
            result["quality_score"] = max(1, min(10, score))
            result["quality_reasoning"] = parsed.get("reasoning")

        if want_query_type:
            result["query_type"] = _clean_query_type(parsed.get("query_type"))
        return result

    except Exception as exc:
        logger.warning("Judge call failed, continuing without it: %s", exc)
        return dict(_EMPTY)
