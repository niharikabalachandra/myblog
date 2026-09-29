"""Run variant-A (refined criteria) against the same dev set, reconstructing
the per-example rubric level descriptions from the already-built payloads
so we don't have to re-download/re-shuffle the source datasets."""
import json
import re

from phase2_runner import call_jev, load_jsonl, write_jsonl
from phase4_tuned_payloads import make_risk_payload_tuned, make_quality_payload_tuned

LEVEL_RE = re.compile(r"\(Level (\d)\)\s*(.*?)(?=\s*\(Level \d\)|$)", re.DOTALL)


def parse_level_criteria(text):
    return {int(lvl): desc.strip() for lvl, desc in LEVEL_RE.findall(text)}


def run_risk_tuned(in_path, out_path):
    rows = load_jsonl(in_path)
    results = []
    for i, row in enumerate(rows):
        payload = make_risk_payload_tuned(row["raw"])
        call = call_jev(payload, "is_safe")
        predicted_true = call["prob_true"] >= 0.5
        results.append(
            {
                "id": row["id"],
                "ground_truth": row["ground_truth_is_safe"],
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
    print(f"[{out_path}] n={len(results)} total_cost=${sum(r['cost'] for r in results):.5f}")


def run_quality_tuned(in_path, out_path):
    rows = load_jsonl(in_path)
    results = []
    for i, row in enumerate(rows):
        old_criteria = row["payload"]["questions"]["is_high_quality"]["criteria"]
        levels = {**parse_level_criteria(old_criteria["true"]), **parse_level_criteria(old_criteria["false"])}
        fake_row = {
            "orig_instruction": row["raw"]["instruction"],
            "orig_response": row["raw"]["response"],
            "orig_criteria": row["orig_criteria"],
            **{f"orig_score{i}_description": levels[i] for i in range(1, 6)},
        }
        payload = make_quality_payload_tuned(fake_row)
        call = call_jev(payload, "is_high_quality")
        predicted_true = call["prob_true"] >= 0.5
        results.append(
            {
                "id": row["id"],
                "ground_truth": row["ground_truth_is_high_quality"],
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
    print(f"[{out_path}] n={len(results)} total_cost=${sum(r['cost'] for r in results):.5f}")


if __name__ == "__main__":
    print("=== risk dev, tuned (variant A) ===")
    run_risk_tuned("data/samples/risk_dev.jsonl", "results/raw/jev_tuned_a_risk_dev.jsonl")
    print("\n=== quality dev, tuned (variant A) ===")
    run_quality_tuned("data/samples/quality_dev.jsonl", "results/raw/jev_tuned_a_quality_dev.jsonl")
