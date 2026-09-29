"""Phase 6: consistency/repeatability check. Re-run a fixed 100-example
subsample through each system a second time (same frozen config as the
Phase 5 test run) and measure how often the second pass agrees with the
first - including for the LLMs at temperature=0, where any disagreement
is itself a finding."""
import json

from phase2_runner import call_jev, load_jsonl, write_jsonl
from phase3_llm_runner import build_judge_prompt, call_llm_judge
from phase4_variant_b import make_risk_payload_fewshot, make_quality_payload_fewshot

N_SUBSAMPLE = 100
RISK_IDS = [f"risk-{i}" for i in range(N_SUBSAMPLE)]
QUALITY_IDS = [f"quality-{i}" for i in range(N_SUBSAMPLE)]


def rerun_jev_risk():
    rows = {r["id"]: r for r in load_jsonl("data/samples/risk_test.jsonl")}
    results = []
    for i, id_ in enumerate(RISK_IDS):
        row = rows[id_]
        payload = make_risk_payload_fewshot(row["raw"])
        call = call_jev(payload, "is_safe")
        results.append({"id": id_, "predicted": call["prob_true"] >= 0.5, "prob_true": call["prob_true"]})
        if (i + 1) % 25 == 0:
            print(f"  jev risk {i + 1}/{N_SUBSAMPLE}")
    write_jsonl("results/raw/consistency_jev_risk_rerun.jsonl", results)


def rerun_jev_quality():
    rows = {r["id"]: r for r in load_jsonl("data/samples/quality_test.jsonl")}
    results = []
    for i, id_ in enumerate(QUALITY_IDS):
        row = rows[id_]
        payload = make_quality_payload_fewshot(row)
        call = call_jev(payload, "is_high_quality")
        results.append({"id": id_, "predicted": call["prob_true"] >= 0.5, "prob_true": call["prob_true"]})
        if (i + 1) % 25 == 0:
            print(f"  jev quality {i + 1}/{N_SUBSAMPLE}")
    write_jsonl("results/raw/consistency_jev_quality_rerun.jsonl", results)


def rerun_llm(model_id, tag, samples_path, ids, question_key, out_path):
    rows = {r["id"]: r for r in load_jsonl(samples_path)}
    results = []
    for i, id_ in enumerate(ids):
        row = rows[id_]
        prompt = build_judge_prompt(row["payload"], question_key)
        call = call_llm_judge(model_id, prompt, max_tokens=600)
        results.append({"id": id_, "predicted": call["verdict_true"], "confidence": call["confidence"]})
        if (i + 1) % 25 == 0:
            print(f"  {tag} {i + 1}/{len(ids)}")
        import time
        time.sleep(0.2)
    write_jsonl(out_path, results)


def compare(pass1_path, pass2_path, ids, label):
    pass1 = {r["id"]: r for r in load_jsonl(pass1_path)}
    pass2 = {r["id"]: r for r in load_jsonl(pass2_path)}
    agree = sum(1 for id_ in ids if pass1[id_]["predicted"] == pass2[id_]["predicted"])
    flips = [id_ for id_ in ids if pass1[id_]["predicted"] != pass2[id_]["predicted"]]
    print(f"{label}: agreement={agree}/{len(ids)} ({100*agree/len(ids):.1f}%) flips={flips}")


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "jev"):
        print("=== Jev rerun (risk) ===")
        rerun_jev_risk()
        print("=== Jev rerun (quality) ===")
        rerun_jev_quality()

    if which in ("all", "haiku"):
        print("=== Haiku 4.5 rerun (risk) ===")
        rerun_llm(
            "anthropic/claude-haiku-4.5", "haiku-risk",
            "data/samples/risk_test.jsonl", RISK_IDS, "is_safe",
            "results/raw/consistency_haiku_risk_rerun.jsonl",
        )
        print("=== Haiku 4.5 rerun (quality) ===")
        rerun_llm(
            "anthropic/claude-haiku-4.5", "haiku-quality",
            "data/samples/quality_test.jsonl", QUALITY_IDS, "is_high_quality",
            "results/raw/consistency_haiku_quality_rerun.jsonl",
        )

    if which in ("all", "luna"):
        print("=== Luna rerun (risk) ===")
        rerun_llm(
            "openai/gpt-6-luna", "luna-risk",
            "data/samples/risk_test.jsonl", RISK_IDS, "is_safe",
            "results/raw/consistency_luna_risk_rerun.jsonl",
        )
        print("=== Luna rerun (quality) ===")
        rerun_llm(
            "openai/gpt-6-luna", "luna-quality",
            "data/samples/quality_test.jsonl", QUALITY_IDS, "is_high_quality",
            "results/raw/consistency_luna_quality_rerun.jsonl",
        )

    if which == "compare":
        # pass1 = original Phase 5 test results, filtered to the same 100 ids
        def filter_and_save(src, ids, out):
            rows = {r["id"]: r for r in load_jsonl(src)}
            write_jsonl(out, [rows[i] for i in ids])

        filter_and_save("results/raw/jev_final_risk_test.jsonl", RISK_IDS, "/tmp/jev_risk_pass1.jsonl")
        filter_and_save("results/raw/jev_final_quality_test.jsonl", QUALITY_IDS, "/tmp/jev_quality_pass1.jsonl")
        filter_and_save("results/raw/haiku45_zeroshot_risk_test.jsonl", RISK_IDS, "/tmp/haiku_risk_pass1.jsonl")
        filter_and_save("results/raw/haiku45_zeroshot_quality_test.jsonl", QUALITY_IDS, "/tmp/haiku_quality_pass1.jsonl")
        filter_and_save("results/raw/luna_zeroshot_risk_test.jsonl", RISK_IDS, "/tmp/luna_risk_pass1.jsonl")
        filter_and_save("results/raw/luna_zeroshot_quality_test.jsonl", QUALITY_IDS, "/tmp/luna_quality_pass1.jsonl")

        compare("/tmp/jev_risk_pass1.jsonl", "results/raw/consistency_jev_risk_rerun.jsonl", RISK_IDS, "Jev risk")
        compare("/tmp/jev_quality_pass1.jsonl", "results/raw/consistency_jev_quality_rerun.jsonl", QUALITY_IDS, "Jev quality")
        compare("/tmp/haiku_risk_pass1.jsonl", "results/raw/consistency_haiku_risk_rerun.jsonl", RISK_IDS, "Haiku risk")
        compare("/tmp/haiku_quality_pass1.jsonl", "results/raw/consistency_haiku_quality_rerun.jsonl", QUALITY_IDS, "Haiku quality")
        compare("/tmp/luna_risk_pass1.jsonl", "results/raw/consistency_luna_risk_rerun.jsonl", RISK_IDS, "Luna risk")
        compare("/tmp/luna_quality_pass1.jsonl", "results/raw/consistency_luna_quality_rerun.jsonl", QUALITY_IDS, "Luna quality")
