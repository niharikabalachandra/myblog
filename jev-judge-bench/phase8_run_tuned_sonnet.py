import json
import time

from phase2_runner import load_jsonl, write_jsonl
from phase3_llm_runner import call_llm_judge
from phase8_sonnet_tuned_risk import build_risk_escalation_prompt_tuned
from phase8_tune_sonnet import build_escalation_dev_set


def run(ids, samples_path, question_key, ground_truth_key, out_path):
    samples = {r["id"]: r for r in load_jsonl(samples_path)}
    results = []
    for i, id_ in enumerate(sorted(ids)):
        row = samples[id_]
        prompt = build_risk_escalation_prompt_tuned(row["payload"], question_key)
        call = call_llm_judge("anthropic/claude-sonnet-5", prompt, max_tokens=600)
        results.append({
            "id": id_,
            "ground_truth": row[ground_truth_key],
            "predicted": call["verdict_true"],
            "confidence": call["confidence"],
            "raw_response_text": call["raw_response_text"],
            "cost": call["cost"],
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ids)} done")
        time.sleep(0.2)
    write_jsonl(out_path, results)
    print(f"[{out_path}] n={len(results)} total_cost=${sum(r['cost'] for r in results):.5f}")


if __name__ == "__main__":
    risk_esc_ids = build_escalation_dev_set("results/raw/jev_fewshot_b_risk_dev.jsonl")
    run(
        risk_esc_ids, "data/samples/risk_dev.jsonl", "is_safe", "ground_truth_is_safe",
        "results/raw/sonnet_tuned_risk_escalation_dev.jsonl",
    )
