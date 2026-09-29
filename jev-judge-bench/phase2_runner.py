"""Phase 2: shared runner. Fires a sample's payload at Jev's Decisions API
and logs prediction, confidence, latency, tokens, and cost in one schema
so every later system (LLM baselines, hybrid) writes the same shape."""
import json
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def call_jev(payload, question_key, timeout=30):
    t0 = time.monotonic()
    resp = httpx.post(
        DECISIONS_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json=payload,
        timeout=timeout,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    body = resp.json()
    prob_true = body["answers"][question_key]["noul"]
    return {
        "raw_request": payload,
        "raw_response": body,
        "prob_true": prob_true,
        "latency_ms": latency_ms,
        "input_tokens": body["usage"]["input_tokens"],
        "output_tokens": body["usage"]["output_tokens"],
        "cost": body["usage"]["cost"],
    }


def run_jev_on_file(in_path, out_path, question_key, ground_truth_key, threshold=0.5, limit=None):
    rows = load_jsonl(in_path)
    if limit:
        rows = rows[:limit]
    results = []
    for i, row in enumerate(rows):
        call = call_jev(row["payload"], question_key)
        predicted_true = call["prob_true"] >= threshold
        results.append(
            {
                "id": row["id"],
                "ground_truth": row[ground_truth_key],
                "predicted": predicted_true,
                "prob_true": call["prob_true"],
                "latency_ms": call["latency_ms"],
                "input_tokens": call["input_tokens"],
                "output_tokens": call["output_tokens"],
                "cost": call["cost"],
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")
    write_jsonl(out_path, results)
    total_cost = sum(r["cost"] for r in results)
    print(f"[{in_path}] n={len(results)} total_cost=${total_cost:.5f}")
    return results


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if which == "run_dev":
        print("=== risk dev, zero-shot ===")
        run_jev_on_file(
            "data/samples/risk_dev.jsonl",
            "results/raw/jev_zeroshot_risk_dev.jsonl",
            question_key="is_safe",
            ground_truth_key="ground_truth_is_safe",
        )
        print("\n=== quality dev, zero-shot ===")
        run_jev_on_file(
            "data/samples/quality_dev.jsonl",
            "results/raw/jev_zeroshot_quality_dev.jsonl",
            question_key="is_high_quality",
            ground_truth_key="ground_truth_is_high_quality",
        )
    elif which == "smoke":
        risk_rows = load_jsonl("data/samples/risk_dev.jsonl")
        quality_rows = load_jsonl("data/samples/quality_dev.jsonl")
        print("=" * 100)
        print("RISK - exact request body sent to Jev")
        print("=" * 100)
        print(json.dumps(risk_rows[0]["payload"], indent=2))
        risk_result = call_jev(risk_rows[0]["payload"], "is_safe")
        print("\nRESPONSE:")
        print(json.dumps(risk_result["raw_response"], indent=2))

        print("\n" + "=" * 100)
        print("QUALITY - exact request body sent to Jev")
        print("=" * 100)
        print(json.dumps(quality_rows[0]["payload"], indent=2))
        quality_result = call_jev(quality_rows[0]["payload"], "is_high_quality")
        print("\nRESPONSE:")
        print(json.dumps(quality_result["raw_response"], indent=2))
