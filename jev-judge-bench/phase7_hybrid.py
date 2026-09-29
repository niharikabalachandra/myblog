"""Phase 7: hybrid escalation. Two rules - outcome-based (escalate every
Jev negative verdict) and confidence-based (escalate only near-0.5 calls) -
both escalating to Sonnet 5 for a reasoned second opinion. Calls Sonnet
once per unique escalated id (deduped across both rules) to avoid paying
twice for cases both rules would escalate."""
import json
import time

from phase2_runner import load_jsonl, write_jsonl
from phase3_llm_runner import build_judge_prompt, call_llm_judge

CONF_BAND = 0.15  # |prob_true - 0.5| <= this counts as "near-threshold"


def escalation_ids(jev_results, rule):
    if rule == "outcome":
        return {r["id"] for r in jev_results if not r["predicted"]}
    elif rule == "confidence":
        return {r["id"] for r in jev_results if abs(r["prob_true"] - 0.5) <= CONF_BAND}
    raise ValueError(rule)


def run_escalation(jev_results_path, samples_path, question_key, out_path, resume=True):
    jev_results = load_jsonl(jev_results_path)
    union_ids = escalation_ids(jev_results, "outcome") | escalation_ids(jev_results, "confidence")
    samples = {r["id"]: r for r in load_jsonl(samples_path)}

    done_ids = set()
    if resume:
        try:
            done_ids = {r["id"] for r in load_jsonl(out_path)}
        except FileNotFoundError:
            pass
    todo = sorted(union_ids - done_ids)

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as f:
        for i, id_ in enumerate(todo):
            row = samples[id_]
            prompt = build_judge_prompt(row["payload"], question_key)
            call = call_llm_judge("anthropic/claude-sonnet-5", prompt, max_tokens=600)
            result = {
                "id": id_,
                "predicted": call["verdict_true"],
                "confidence": call["confidence"],
                "parse_error": call["parse_error"],
                "content_filtered": call["content_filtered"],
                "cost": call["cost"],
                "latency_ms": call["latency_ms"],
                "raw_response_text": call["raw_response_text"],
            }
            f.write(json.dumps(result) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)} done ({len(done_ids)} resumed)")
            time.sleep(0.2)
    print(f"[{out_path}] escalated_total={len(union_ids)} this_run={len(todo)} resumed={len(done_ids)}")


if __name__ == "__main__":
    print("=== risk escalation -> Sonnet 5 ===")
    run_escalation(
        "results/raw/jev_final_risk_test.jsonl",
        "data/samples/risk_test.jsonl",
        "is_safe",
        "results/raw/hybrid_sonnet_risk_escalated.jsonl",
    )
    print("\n=== quality escalation -> Sonnet 5 ===")
    run_escalation(
        "results/raw/jev_final_quality_test.jsonl",
        "data/samples/quality_test.jsonl",
        "is_high_quality",
        "results/raw/hybrid_sonnet_quality_escalated.jsonl",
    )
