"""Phase 1: pull both datasets and report real class balance before
locking in sample sizes."""
from datasets import load_dataset


def inspect_beavertails():
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    n = len(ds)
    n_safe = sum(1 for x in ds["is_safe"] if x)
    n_unsafe = n - n_safe
    print(f"[beavertails] test split: {n} rows")
    print(f"              is_safe=True:  {n_safe} ({100*n_safe/n:.1f}%)")
    print(f"              is_safe=False: {n_unsafe} ({100*n_unsafe/n:.1f}%)")
    return ds


def inspect_feedback_collection():
    ds = load_dataset("prometheus-eval/Feedback-Collection", split="train")
    n = len(ds)
    scores = ds["orig_score"]
    from collections import Counter

    counts = Counter(scores)
    n_low = sum(v for k, v in counts.items() if int(k) <= 3)
    n_high = n - n_low
    print(f"[feedback-collection] train split: {n} rows")
    for level in sorted(counts):
        print(f"                       score={level}: {counts[level]} ({100*counts[level]/n:.1f}%)")
    print(f"                       LOW (<=3):  {n_low} ({100*n_low/n:.1f}%)")
    print(f"                       HIGH (>3):  {n_high} ({100*n_high/n:.1f}%)")
    n_rubrics = len(set(ds["orig_criteria"]))
    print(f"                       distinct rubrics: {n_rubrics}")
    return ds


if __name__ == "__main__":
    inspect_beavertails()
    print()
    inspect_feedback_collection()
