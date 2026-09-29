"""Phase 1: draw the real dev/test samples for both judges and materialize
the exact Jev request payload (with cleaned-up criteria) alongside ground
truth, so Phase 3+ just reads these files."""
import json
import os
import random

from datasets import load_dataset

from phase1_show_samples import make_risk_payload, make_quality_payload

SEED = 42
OUT_DIR = "data/samples"

RISK_DEV_N = 200
RISK_TEST_N = 1000
QUALITY_DEV_N = 200
QUALITY_TEST_N = 750


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_risk_samples():
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    ds = ds.shuffle(seed=SEED)

    # BeaverTails has a small number of (prompt, response) pairs that recur
    # with CONFLICTING is_safe labels across rows (real annotation noise, not
    # a bug here) - drop those entirely so no ambiguous example can land in
    # the sample, then dedupe same-label repeats down to one occurrence.
    labels_by_key = {}
    for row in ds:
        key = (row["prompt"], row["response"])
        labels_by_key.setdefault(key, set()).add(row["is_safe"])
    contradictory_keys = {k for k, v in labels_by_key.items() if len(v) > 1}
    print(f"[risk] dropping {len(contradictory_keys)} prompt/response pairs with conflicting is_safe labels")

    seen = set()
    clean_rows = []
    for row in ds:
        key = (row["prompt"], row["response"])
        if key in contradictory_keys or key in seen:
            continue
        seen.add(key)
        clean_rows.append(row)

    unsafe = [r for r in clean_rows if not r["is_safe"]]
    safe = [r for r in clean_rows if r["is_safe"]]

    total_n = RISK_DEV_N + RISK_TEST_N
    unsafe_ratio = len(unsafe) / len(clean_rows)
    n_unsafe = round(total_n * unsafe_ratio)
    n_safe = total_n - n_unsafe

    pool = unsafe[:n_unsafe] + safe[:n_safe]
    random.Random(SEED).shuffle(pool)

    dev, test = pool[:RISK_DEV_N], pool[RISK_DEV_N : RISK_DEV_N + RISK_TEST_N]

    def to_rows(rows):
        out = []
        for i, row in enumerate(rows):
            out.append(
                {
                    "id": f"risk-{i}",
                    "payload": make_risk_payload(row),
                    "ground_truth_is_safe": row["is_safe"],
                    "raw": {"prompt": row["prompt"], "response": row["response"]},
                }
            )
        return out

    dev_rows, test_rows = to_rows(dev), to_rows(test)
    _write_jsonl(f"{OUT_DIR}/risk_dev.jsonl", dev_rows)
    _write_jsonl(f"{OUT_DIR}/risk_test.jsonl", test_rows)

    for name, rows in [("dev", dev_rows), ("test", test_rows)]:
        n = len(rows)
        n_unsafe_actual = sum(1 for r in rows if not r["ground_truth_is_safe"])
        print(
            f"[risk/{name}] n={n} unsafe={n_unsafe_actual} "
            f"({100*n_unsafe_actual/n:.1f}%) safe={n - n_unsafe_actual} "
            f"({100*(n - n_unsafe_actual)/n:.1f}%)"
        )


def build_quality_samples():
    ds = load_dataset("prometheus-eval/Feedback-Collection", split="train")
    ds = ds.shuffle(seed=SEED)

    seen_rubrics = set()
    candidates = []
    for row in ds:
        rubric = row["orig_criteria"]
        if rubric in seen_rubrics:
            continue
        seen_rubrics.add(rubric)
        candidates.append(row)

    total_n = QUALITY_DEV_N + QUALITY_TEST_N
    random.Random(SEED).shuffle(candidates)
    pool = candidates[:total_n]

    dev, test = pool[:QUALITY_DEV_N], pool[QUALITY_DEV_N : QUALITY_DEV_N + QUALITY_TEST_N]

    def to_rows(rows):
        out = []
        for i, row in enumerate(rows):
            score = int(row["orig_score"])
            out.append(
                {
                    "id": f"quality-{i}",
                    "payload": make_quality_payload(row),
                    "ground_truth_is_high_quality": score > 3,
                    "orig_score": score,
                    "orig_criteria": row["orig_criteria"],
                    "raw": {
                        "instruction": row["orig_instruction"],
                        "response": row["orig_response"],
                    },
                }
            )
        return out

    dev_rows, test_rows = to_rows(dev), to_rows(test)
    _write_jsonl(f"{OUT_DIR}/quality_dev.jsonl", dev_rows)
    _write_jsonl(f"{OUT_DIR}/quality_test.jsonl", test_rows)

    print(f"[quality] distinct rubrics available: {len(candidates)}, used: {total_n}")
    for name, rows in [("dev", dev_rows), ("test", test_rows)]:
        n = len(rows)
        n_low = sum(1 for r in rows if not r["ground_truth_is_high_quality"])
        print(
            f"[quality/{name}] n={n} low={n_low} ({100*n_low/n:.1f}%) "
            f"high={n - n_low} ({100*(n - n_low)/n:.1f}%)"
        )


if __name__ == "__main__":
    build_risk_samples()
    print()
    build_quality_samples()
