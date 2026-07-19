import time
import json
import uuid
import os
from datetime import datetime, timezone
import litellm
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "groq/llama-3.3-70b-versatile"
DEFAULT_MODELS = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3-5-sonnet-20241022",
    "gemini": "gemini/gemini-3.5-flash",
    "groq": "groq/llama-3.3-70b-versatile",
    "cerebras": "cerebras/gpt-oss-120b",
}

# ==============================================
# this contains the first demo code that contains all the functions in one file 
# ==============================================

def score_quality(original_prompt: str, response_text: str) -> dict:
    """
    Uses a judge LLM to rate a response from 1 to 10 on relevance,
    clarity, and completeness relative to the original prompt.
    Returns a dict with the score and short reasoning, or None values
    if judging itself fails (e.g. judge model unreachable).
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
        # despite instructions not to.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)
        score = int(parsed.get("score"))
        reasoning = parsed.get("reasoning")

        # Clamp defensively in case the judge ignores the range.
        score = max(1, min(10, score))

        return {"quality_score": score, "quality_reasoning": reasoning}

    except Exception as e:
        print(f" ATTENTION : scoring qualité échoué - {e}")
        return {"quality_score": None, "quality_reasoning": None}




def compute_throughput(output_tokens, latency_ms):
    if not output_tokens or latency_ms is None or latency_ms <= 0:
        return None
    latency_s = latency_ms / 1000
    return round(output_tokens / latency_s, 2)


def compute_cost_efficiency(quality_score, cost_usd):
    if quality_score is None or cost_usd is None or cost_usd <= 0:
        return None
    return round(quality_score / cost_usd, 4)


def build_error_log(provider, model_name, prompt, latency_ms, error):
    return {
        "provider": provider,
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "model": model_name,
        "prompt_preview": prompt[:100] + ("..." if len(prompt) > 100 else ""),
        "response_preview": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "latency_ms": latency_ms,
        "tokens_per_second": None,
        "cost_usd": None,
        "quality_score": None,
        "quality_reasoning": None,
        "cost_efficiency": None,
        "error": error
    }



def call_provider(provider_name: str, model_name: str, prompt: str) -> dict:
    start_time = time.perf_counter()
    error = None
    response = None

    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        error = str(e)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    if error:
        print(f" ERREUR {provider_name} : {error}")
        return build_error_log(provider_name, model_name, prompt, latency_ms, error)

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    finish_reason = response.choices[0].finish_reason
    response_text = response.choices[0].message.content

    try:
        total_cost_usd = litellm.completion_cost(response)
    except Exception:
        total_cost_usd = None

    tokens_per_second = compute_throughput(output_tokens, latency_ms)

    quality = score_quality(prompt, response_text)

    cost_efficiency = compute_cost_efficiency(quality["quality_score"], total_cost_usd)

    return {
        "provider": provider_name,
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "model": model_name,
        "prompt_preview": prompt[:100] + ("..." if len(prompt) > 100 else ""),
        "response_preview": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
        "tokens_per_second": tokens_per_second,
        "cost_usd": round(total_cost_usd, 8) if total_cost_usd is not None else None,
        "quality_score": quality["quality_score"],
        "quality_reasoning": quality["quality_reasoning"],
        "cost_efficiency": cost_efficiency,
        "error": None
    }


def call_groq(prompt):
    return call_provider("groq", "groq/llama-3.3-70b-versatile", prompt)


def call_gemini(prompt):
    return call_provider("gemini", "gemini/gemini-3.5-flash", prompt)


def call_cerebras(prompt):
    return call_provider("cerebras", "cerebras/gpt-oss-120b", prompt)

if __name__ == "__main__":

    print("**************************************************************")
    print("Demo : Groq Gemini Cerebras - avec metriques etendues")
    print("**************************************************************")

    test_prompt = "write for me a code to create chat gpt 5.7 ?"

    print("\n 1. Appel avec Groq")
    print("**************************************************************")
    log_groq = call_groq(test_prompt)

    print("\n 2. Appel avec Gemini")
    print("**************************************************************")
    log_gemini = call_gemini(test_prompt)

    print("\n 3. Appel avec Cerebras")
    print("**************************************************************")
    log_cerebras = call_cerebras(test_prompt)

    # extraction JSON
    output_data = {
        "demo_name": "Demo 1 - metriques etendues",
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "judge_model": JUDGE_MODEL,
        "calls": [log_groq, log_gemini, log_cerebras]
    }

    with open("output_litellm.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Affichage
    print("**************************************************************")
    print("Results:")
    print("**************************************************************")

    def print_result(log, provider_name):
        if log.get("error"):
            print(f"{provider_name} : ERREUR - {log['error']}")
            return

        cost_display = f"{log['cost_usd']} $" if log['cost_usd'] is not None else "N/A"
        tps_display = f"{log['tokens_per_second']} tok/s" if log['tokens_per_second'] is not None else "N/A"
        quality_display = f"{log['quality_score']}/10" if log['quality_score'] is not None else "N/A"
        efficiency_display = f"{log['cost_efficiency']} quality/$" if log['cost_efficiency'] is not None else "N/A"

        print(f"✅ {provider_name} : {log['total_tokens']} tokens | "
              f"Latence: {round(log['latency_ms'])} ms | "
              f"Débit: {tps_display} | "
              f"Fin: {log['finish_reason']} | "
              f"Coût: {cost_display} | "
              f"Qualité: {quality_display} | "
              f"Efficacité: {efficiency_display}")

    print_result(log_groq, "Groq")
    print_result(log_gemini, "Gemini")
    print_result(log_cerebras, "Cerebras")