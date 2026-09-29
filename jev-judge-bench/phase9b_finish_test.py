"""Resume phase9 after a transient httpx.ReadTimeout crashed the test-set
run partway through (no partial output was written - write_jsonl only
writes once the full loop completes). Reuses the already-built risk_dev2
pool and exemplar selection (both deterministic given fixed seeds), retries
transient network errors, and reruns only the missing steps: the frozen
tuned config on risk_test.jsonl, metrics, and the McNemar comparison."""
import time

from phase1_show_samples import RISK_INSTRUCTIONS, RISK_CRITERIA
from phase2_runner import call_jev, load_jsonl, write_jsonl
from phase9_bigger_dev_risk import (
    select_exemplars,
    make_fewshot_payload,
    metrics,
    print_metrics,
    mcnemar,
    DEV2_PATH,
    EXISTING_TEST_PATH,
    FEWSHOT_TEST_OUT,
    EXISTING_ZEROSHOT_TEST,
)


def call_jev_retry(payload, key, attempts=4):
    for attempt in range(attempts):
        try:
            return call_jev(payload, key)
        except Exception as e:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"  retrying after error: {e} (attempt {attempt + 1}/{attempts}, sleeping {wait}s)")
            time.sleep(wait)


def run_fewshot_retry(rows, exemplars, out_path):
    results = []
    for i, row in enumerate(rows):
        payload = make_fewshot_payload(row["raw"], exemplars)
        call = call_jev_retry(payload, "is_safe")
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


if __name__ == "__main__":
    dev2_rows = load_jsonl(DEV2_PATH)
    exemplars, exemplar_ids = select_exemplars(dev2_rows)
    print(f"re-derived {len(exemplars)} exemplars (ids={sorted(exemplar_ids)})")

    print("\n=== running frozen tuned config on existing risk_test.jsonl (n=1000), with retries ===")
    test_rows = load_jsonl(EXISTING_TEST_PATH)
    fs_test = run_fewshot_retry(test_rows, exemplars, FEWSHOT_TEST_OUT)
    m_fs_test = metrics(fs_test)

    zs_test = load_jsonl(EXISTING_ZEROSHOT_TEST)
    m_zs_test = metrics(zs_test)

    print("\n=== TEST RESULTS (n=1000) ===")
    print_metrics("zero-shot (existing, reused)", m_zs_test)
    print_metrics("exemplar-tuned, bigger pool (new)", m_fs_test)

    flips_a, flips_b, p = mcnemar(zs_test, fs_test)
    print(f"\n=== MCNEMAR (test set): zero-shot-correct->tuned-wrong={flips_a}, zero-shot-wrong->tuned-correct={flips_b}, p={p:.4f} ===")
