"""Redo the real test-set risk escalations with the tuned Sonnet prompt,
replacing the untuned Phase 7 risk escalation results. Quality escalations
are left untouched per the decision to report that disagreement as a
finding rather than tune toward GPT-4-generated ground truth."""
import json
import time

from phase2_runner import load_jsonl, write_jsonl
from phase3_llm_runner import call_llm_judge
from phase7_hybrid import escalation_ids
from phase8_sonnet_tuned_risk import build_risk_escalation_prompt_tuned


def run(resume=True):
    jev_results = load_jsonl("results/raw/jev_final_risk_test.jsonl")
    union_ids = escalation_ids(jev_results, "outcome") | escalation_ids(jev_results, "confidence")
    samples = {r["id"]: r for r in load_jsonl("data/samples/risk_test.jsonl")}

    out_path = "results/raw/hybrid_sonnet_risk_escalated_tuned.jsonl"
    done_ids = set()
    if resume:
        try:
            done_ids = {r["id"] for r in load_jsonl(out_path)}
        except FileNotFoundError:
            pass
    todo = sorted(union_ids - done_ids)

    with open(out_path, "a") as f:
        for i, id_ in enumerate(todo):
            row = samples[id_]
            prompt = build_risk_escalation_prompt_tuned(row["payload"], "is_safe")
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
    run()
