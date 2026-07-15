"""
Example usage of the LLM monitoring library.

This is NOT a benchmark comparing providers against each other — it
demonstrates how a developer would use LLMMonitor as a middleman
around whichever single LLM their application actually uses.

Run with: python main.py
"""

import json
import os

from core.monitor import LLMMonitor


def print_result(result: dict):
    if result.get("error"):
        print(f"Error ({result['error_type']}): {result['error']}")
        return

    cost_display = f"{result['cost_usd']} $" if result['cost_usd'] is not None else "N/A"
    tps_display = f"{result['tokens_per_second']} tok/s" if result['tokens_per_second'] is not None else "N/A"
    quality_display = f"{result['quality_score']}/10" if result['quality_score'] is not None else "N/A"

    print(f"✅ {result['provider']} ({result['model']})")
    print(f"   Tokens: {result['total_tokens']} | Latency: {round(result['latency_ms'])} ms | "
          f"Débit: {tps_display}")
    print(f"   Coût: {cost_display} | Qualité: {quality_display} | "
          f"Fin: {result['finish_reason']} | Tentatives: {result['attempts_used']}")


def main():
    # The developer picks ONE provider their app actually uses.
    # Swap "anthropic" for "openai", "gemini", "groq", or "cerebras" —
    # nothing else below needs to change, thanks to the Factory pattern.
    monitor = LLMMonitor(provider_name="gemini")

    prompt = "Explain the Factory design pattern in 3 sentences."
    result = monitor.call(prompt)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "call_log.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print_result(result)


if __name__ == "__main__":
    main()