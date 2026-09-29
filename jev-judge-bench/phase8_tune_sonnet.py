"""Phase 8: tune Sonnet's escalation prompt. The hybrid is only a fair
comparison if BOTH halves are tuned - Jev already went through two rounds,
Sonnet has been running the generic zero-shot judge prompt this whole time.
Builds a dev set restricted to cases Jev's frozen config would actually
escalate (not a general sample), mines Sonnet's own errors on it, and
tunes its prompt the same way we tuned Jev's."""
import json

from phase2_runner import load_jsonl, write_jsonl
from phase3_llm_runner import build_judge_prompt, call_llm_judge

CONF_BAND = 0.15


def escalation_ids(jev_results, rule):
    if rule == "outcome":
        return {r["id"] for r in jev_results if not r["predicted"]}
    elif rule == "confidence":
        return {r["id"] for r in jev_results if abs(r["prob_true"] - 0.5) <= CONF_BAND}
    raise ValueError(rule)


def build_escalation_dev_set(jev_dev_path):
    jev_dev = load_jsonl(jev_dev_path)
    union_ids = escalation_ids(jev_dev, "outcome") | escalation_ids(jev_dev, "confidence")
    return union_ids


def run_sonnet_zeroshot(ids, samples_path, question_key, ground_truth_key, out_path):
    samples = {r["id"]: r for r in load_jsonl(samples_path)}
    results = []
    for i, id_ in enumerate(sorted(ids)):
        row = samples[id_]
        prompt = build_judge_prompt(row["payload"], question_key)
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
        import time
        time.sleep(0.2)
    write_jsonl(out_path, results)
    print(f"[{out_path}] n={len(results)} total_cost=${sum(r['cost'] for r in results):.5f}")


if __name__ == "__main__":
    risk_esc_ids = build_escalation_dev_set("results/raw/jev_fewshot_b_risk_dev.jsonl")
    quality_esc_ids = build_escalation_dev_set("results/raw/jev_fewshot_b_quality_dev.jsonl")
    print(f"risk escalation dev set: {len(risk_esc_ids)} ids")
    print(f"quality escalation dev set: {len(quality_esc_ids)} ids")

    print("\n=== Sonnet zero-shot on risk escalation dev set ===")
    run_sonnet_zeroshot(
        risk_esc_ids, "data/samples/risk_dev.jsonl", "is_safe", "ground_truth_is_safe",
        "results/raw/sonnet_zeroshot_risk_escalation_dev.jsonl",
    )
    print("\n=== Sonnet zero-shot on quality escalation dev set ===")
    run_sonnet_zeroshot(
        quality_esc_ids, "data/samples/quality_dev.jsonl", "is_high_quality", "ground_truth_is_high_quality",
        "results/raw/sonnet_zeroshot_quality_escalation_dev.jsonl",
    )
