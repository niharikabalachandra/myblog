"""Compute accuracy/precision/recall/F1 from a results file, with the
positive class flipped to match our locked convention: unsafe=positive
for risk, low-quality=positive for quality (both are the noul=False case)."""
import json
import statistics
import sys

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def report(path, positive_label):
    rows = load_jsonl(path)
    y_true = [not r["ground_truth"] for r in rows]  # True = unsafe / low-quality
    y_pred = [not r["predicted"] for r in rows]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=True
    )
    latencies = [r["latency_ms"] for r in rows]
    total_cost = sum(r["cost"] for r in rows)

    print(f"\n=== {path} (positive class = {positive_label}) ===")
    print(f"n={len(rows)}")
    print(f"accuracy={acc:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print(
        f"latency ms: p50={statistics.median(latencies):.0f} "
        f"p95={sorted(latencies)[int(0.95*len(latencies))]:.0f} "
        f"mean={statistics.mean(latencies):.0f}"
    )
    print(f"total cost=${total_cost:.5f} (${total_cost/len(rows):.7f}/call)")


if __name__ == "__main__":
    report("results/raw/jev_zeroshot_risk_dev.jsonl", "unsafe")
    report("results/raw/jev_zeroshot_quality_dev.jsonl", "low quality")
