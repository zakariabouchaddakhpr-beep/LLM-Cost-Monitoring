"""
Entry point. Run with: python main.py
Loops over the providers defined in config/settings.py, calls each one,
saves a JSON log, and prints a summary line per provider.
"""

import json
import os
from datetime import datetime, timezone

from config.settings import PROVIDERS, JUDGE_MODEL
from core.providers import call_provider


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


def main():
    print("**************************************************************")
    print("Demo : Groq Gemini Cerebras - avec metriques etendues")
    print("**************************************************************")

    test_prompt = "Help me pick a new manhwa to read, based on my preferences for action, fantasy, and romance."

    results = []
    for i, (provider_name, model_name) in enumerate(PROVIDERS, start=1):
        print(f"\n {i}. Appel avec {provider_name.capitalize()}")
        print("**************************************************************")
        log = call_provider(provider_name, model_name, test_prompt)
        results.append(log)

    output_data = {
        "demo_name": "Demo 1 - metriques etendues",
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "judge_model": JUDGE_MODEL,
        "calls": results
    }

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "output_litellm.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("**************************************************************")
    print("Results:")
    print("**************************************************************")

    for log, (provider_name, _) in zip(results, PROVIDERS):
        print_result(log, provider_name.capitalize())


if __name__ == "__main__":
    main()