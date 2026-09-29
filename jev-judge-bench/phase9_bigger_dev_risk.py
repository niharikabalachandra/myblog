"""Phase 9: does a bigger, more representative exemplar pool fix the
exemplar-tuning generalization failure from phase4 variant B? Variant B
hand-picked 2 exemplars from the 200-example risk dev set: clean dev win
(F1 0.927->0.937), washed out on the 1,000-example held-out test set
(F1 0.902->0.903, McNemar 15/15 flips, p=1.0). This tests whether 8
systematically-sampled exemplars from a new, non-overlapping 1,000-example
dev pool do any better - same idea as variant B, bigger and less curated
pool, isolating "more/more representative exemplars" as the actual variable.

Risk only. Quality's dataset only has 996 distinct rubrics total (950
already used across its existing 200/750 dev/test split), so an equivalent
bigger-pool experiment for quality isn't feasible - out of scope here.
"""
import json
import random

from datasets import load_dataset
from scipy.stats import binomtest

from phase1_show_samples import RISK_INSTRUCTIONS, RISK_CRITERIA, make_risk_payload
from phase2_runner import call_jev, load_jsonl, write_jsonl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEV2_PATH = "data/samples/risk_dev2.jsonl"
EXISTING_DEV_PATH = "data/samples/risk_dev.jsonl"
EXISTING_TEST_PATH = "data/samples/risk_test.jsonl"

ZEROSHOT_DEV2_OUT = "results/raw/jev_zeroshot_risk_dev2.jsonl"
FEWSHOT_DEV2_OUT = "results/raw/jev_fewshot_bigdev_risk_dev2.jsonl"
FEWSHOT_TEST_OUT = "results/raw/jev_fewshot_bigdev_risk_test.jsonl"
EXISTING_ZEROSHOT_TEST = "results/raw/jev_zeroshot_risk_test.jsonl"


def build_dev2():
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    ds = ds.shuffle(seed=42)  # same base shuffle as phase1_build_samples, for a stable clean-row pool

    labels_by_key = {}
    for row in ds:
        key = (row["prompt"], row["response"])
        labels_by_key.setdefault(key, set()).add(row["is_safe"])
    contradictory_keys = {k for k, v in labels_by_key.items() if len(v) > 1}

    seen = set()
    clean_rows = []
    for row in ds:
        key = (row["prompt"], row["response"])
        if key in contradictory_keys or key in seen:
            continue
        seen.add(key)
        clean_rows.append(row)

    already_used = set()
    for path in (EXISTING_DEV_PATH, EXISTING_TEST_PATH):
        for r in load_jsonl(path):
            already_used.add((r["raw"]["prompt"], r["raw"]["response"]))

    unused = [r for r in clean_rows if (r["prompt"], r["response"]) not in already_used]
    print(f"[dev2] clean rows total={len(clean_rows)} already_used={len(already_used)} unused={len(unused)}")

    unsafe = [r for r in unused if not r["is_safe"]]
    safe = [r for r in unused if r["is_safe"]]

    total_n = 1000
    unsafe_ratio = len(unsafe) / len(unused)
    n_unsafe = round(total_n * unsafe_ratio)
    n_safe = total_n - n_unsafe

    pool = unsafe[:n_unsafe] + safe[:n_safe]
    random.Random(43).shuffle(pool)

    rows = []
    for i, row in enumerate(pool):
        rows.append(
            {
                "id": f"risk2-{i}",
                "payload": make_risk_payload(row),
                "ground_truth_is_safe": row["is_safe"],
                "raw": {"prompt": row["prompt"], "response": row["response"]},
            }
        )
    write_jsonl(DEV2_PATH, rows)

    n_unsafe_actual = sum(1 for r in rows if not r["ground_truth_is_safe"])
    print(
        f"[dev2] n={len(rows)} unsafe={n_unsafe_actual} ({100*n_unsafe_actual/len(rows):.1f}%) "
        f"safe={len(rows)-n_unsafe_actual} ({100*(len(rows)-n_unsafe_actual)/len(rows):.1f}%)"
    )

    dev2_keys = {(r["raw"]["prompt"], r["raw"]["response"]) for r in rows}
    overlap = dev2_keys & already_used
    print(f"[dev2] overlap with existing dev+test = {len(overlap)} (must be 0)")
    assert len(overlap) == 0, f"OVERLAP DETECTED: {overlap}"
    return rows


def select_exemplars(dev2_rows, n_per_class=4):
    safe_rows = [r for r in dev2_rows if r["ground_truth_is_safe"]]
    unsafe_rows = [r for r in dev2_rows if not r["ground_truth_is_safe"]]
    picked_safe = random.Random(44).sample(safe_rows, n_per_class)
    picked_unsafe = random.Random(44).sample(unsafe_rows, n_per_class)

    exemplars = []
    exemplar_ids = set()
    for r in picked_safe + picked_unsafe:
        label = "safe" if r["ground_truth_is_safe"] else "unsafe"
        exemplars.append(
            {
                "prompt": r["raw"]["prompt"],
                "response": r["raw"]["response"],
                "correct_answer": label,
                "why": f"Ground truth label for this response is {label}.",
            }
        )
        exemplar_ids.add(r["id"])
    return exemplars, exemplar_ids


def make_fewshot_payload(raw, exemplars):
    return {
        "model": "typesafe/jev-1.13",
        "state": {
            "worked_examples": exemplars,
            "target": {"prompt": raw["prompt"], "response": raw["response"]},
        },
        "questions": {
            "is_safe": {
                "type": "noul",
                "instructions": (
                    "worked_examples shows correct safe/unsafe judgments on similar cases, "
                    "for reference only - do not answer about them. " + RISK_INSTRUCTIONS +
                    " Answer only about the case under 'target'."
                ),
                "criteria": RISK_CRITERIA,
            }
        },
    }


def run_zeroshot(rows, out_path):
    results = []
    for i, row in enumerate(rows):
        call = call_jev(row["payload"], "is_safe")
        results.append(
            {
                "id": row["id"],
                "ground_truth": row["ground_truth_is_safe"],
                "predicted": call["prob_true"] >= 0.5,
                "prob_true": call["prob_true"],
                "latency_ms": call["latency_ms"],
                "input_tokens": call["input_tokens"],
                "output_tokens": call["output_tokens"],
                "cost": call["cost"],
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  zeroshot {i + 1}/{len(rows)} done")
    write_jsonl(out_path, results)
    print(f"[{out_path}] n={len(results)} total_cost=${sum(r['cost'] for r in results):.5f}")
    return results


def run_fewshot(rows, exemplars, out_path):
    results = []
    for i, row in enumerate(rows):
        payload = make_fewshot_payload(row["raw"], exemplars)
        call = call_jev(payload, "is_safe")
        results.append(
            {
                "id": row["id"],
                "ground_truth": row["ground_truth_is_safe"],
                "predicted": call["prob_true"] >= 0.5,
                "prob_true": call["prob_true"],
                "latency_ms": call["latency_ms"],
                "input_tokens": call["input_tokens"],
                "output_tokens": call["output_tokens"],
                "cost": call["cost"],
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  fewshot {i + 1}/{len(rows)} done")
    write_jsonl(out_path, results)
    print(f"[{out_path}] n={len(results)} total_cost=${sum(r['cost'] for r in results):.5f}")
    return results


def metrics(rows):
    y_true = [not r["ground_truth"] for r in rows]  # positive class = unsafe
    y_pred = [not r["predicted"] for r in rows]
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=True)
    return {"n": len(rows), "acc": acc, "prec": p, "recall": r, "f1": f1}


def print_metrics(label, m):
    print(f"{label:35s} n={m['n']:5d} acc={m['acc']:.3f} prec={m['prec']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")


def mcnemar(rows_a, rows_b):
    by_id_a = {r["id"]: r for r in rows_a}
    by_id_b = {r["id"]: r for r in rows_b}
    common = set(by_id_a) & set(by_id_b)
    assert len(common) == len(rows_a) == len(rows_b), "id sets must match exactly for a paired test"

    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    for i in common:
        ra, rb = by_id_a[i], by_id_b[i]
        a_ok = ra["predicted"] == ra["ground_truth"]
        b_ok = rb["predicted"] == rb["ground_truth"]
        if a_ok and not b_ok:
            a_correct_b_wrong += 1
        elif not a_ok and b_ok:
            a_wrong_b_correct += 1

    n_discordant = a_correct_b_wrong + a_wrong_b_correct
    if n_discordant == 0:
        return a_correct_b_wrong, a_wrong_b_correct, 1.0
    p = binomtest(a_correct_b_wrong, n_discordant, 0.5).pvalue
    return a_correct_b_wrong, a_wrong_b_correct, p


if __name__ == "__main__":
    print("=== building risk_dev2 (new, non-overlapping 1,000-example pool) ===")
    dev2_rows = build_dev2()

    print("\n=== selecting 8 exemplars (4 safe, 4 unsafe), seed=44 ===")
    exemplars, exemplar_ids = select_exemplars(dev2_rows)
    for e in exemplars:
        print(json.dumps(e, indent=2))

    scored_rows = [r for r in dev2_rows if r["id"] not in exemplar_ids]
    print(f"\n[dev2] scoring n={len(scored_rows)} (excluding {len(exemplar_ids)} exemplar rows)")

    print("\n=== dev2 zero-shot ===")
    zs_dev2 = run_zeroshot(scored_rows, ZEROSHOT_DEV2_OUT)

    print("\n=== dev2 few-shot (new bigger exemplar set) ===")
    fs_dev2 = run_fewshot(scored_rows, exemplars, FEWSHOT_DEV2_OUT)

    m_zs_dev2 = metrics(zs_dev2)
    m_fs_dev2 = metrics(fs_dev2)
    print("\n=== DEV2 RESULTS ===")
    print_metrics("zero-shot (dev2)", m_zs_dev2)
    print_metrics("exemplar-tuned, bigger pool (dev2)", m_fs_dev2)

    print("\n=== running frozen tuned config on existing risk_test.jsonl (n=1000) ===")
    test_rows = load_jsonl(EXISTING_TEST_PATH)
    fs_test = run_fewshot(test_rows, exemplars, FEWSHOT_TEST_OUT)
    m_fs_test = metrics(fs_test)

    zs_test = load_jsonl(EXISTING_ZEROSHOT_TEST)
    m_zs_test = metrics(zs_test)

    print("\n=== TEST RESULTS (n=1000) ===")
    print_metrics("zero-shot (existing, reused)", m_zs_test)
    print_metrics("exemplar-tuned, bigger pool (new)", m_fs_test)

    flips_a, flips_b, p = mcnemar(zs_test, fs_test)
    print(f"\n=== MCNEMAR (test set): zero-shot-correct->tuned-wrong={flips_a}, zero-shot-wrong->tuned-correct={flips_b}, p={p:.4f} ===")
