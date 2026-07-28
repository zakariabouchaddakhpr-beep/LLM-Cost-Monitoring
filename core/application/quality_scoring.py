"""
Quality/accuracy scoring via an LLM judge.
"""

import json
import litellm

from config.settings import JUDGE_MODEL


def score_quality(original_prompt: str, response_text: str) -> dict:
    """
    Uses a judge LLM to rate a response from 1 to 10 on relevance,
    clarity, and completeness relative to the original prompt.
    Returns a dict with the score and short reasoning, or None values
    if judging itself fails.
    """
    if not response_text:
        return {"quality_score": None, "quality_reasoning": None}

    judge_prompt = f"""You are an impartial evaluator. Rate the RESPONSE below on a scale
from 1 (poor) to 10 (excellent), based on how relevant, clear, and complete
it is as an answer to the PROMPT.

PROMPT:
{original_prompt}

RESPONSE:
{response_text}

Reply with ONLY a valid JSON object, no markdown, no code fences, in this
exact format:
{{"score": <integer 1-10>, "reasoning": "<one short sentence>"}}
"""

    try:
        judge_response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )
        raw = judge_response.choices[0].message.content.strip()

        # Defensive cleanup in case the judge wraps output in code fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)
        score = int(parsed.get("score"))
        reasoning = parsed.get("reasoning")

        score = max(1, min(10, score))

        return {"quality_score": score, "quality_reasoning": reasoning}

    except Exception as e:
        return {"quality_score": None, "quality_reasoning": None}
