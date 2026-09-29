"""Phase 4: mine Jev's zero-shot dev errors (FN/FP, highest-confidence-wrong
first) and join back to the original sample content so we can read what's
actually going wrong before rewriting criteria text."""
import json


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def mine(results_path, samples_path, ground_truth_key, question_key, label_true, label_false):
    results = {r["id"]: r for r in load_jsonl(results_path)}
    samples = {r["id"]: r for r in load_jsonl(samples_path)}

    fns, fps = [], []
    for id_, res in results.items():
        sample = samples[id_]
        gt = res["ground_truth"]  # True = safe/high-quality (noul-true)
        pred = res["predicted"]
        prob_true = res["prob_true"]
        if gt and not pred:
            # false alarm: actually fine, Jev said no -> false positive on the
            # "unsafe/low-quality" positive-class framing
            fps.append((prob_true, id_, sample, res))
        elif not gt and pred:
            # missed: actually bad, Jev said fine -> false negative
            fns.append((prob_true, id_, sample, res))

    # highest-confidence-wrong first: FN sorted by highest prob_true (most
    # confidently "fine"), FP sorted by lowest prob_true (most confidently "bad")
    fns.sort(key=lambda x: -x[0])
    fps.sort(key=lambda x: x[0])
    return fns, fps


def print_case(prob_true, id_, sample, res, raw_fields):
    print(f"  id={id_} prob_true={prob_true:.2f} ground_truth_safe/high={res['ground_truth']}")
    for k in raw_fields:
        text = sample["raw"][k]
        print(f"    {k}: {text[:250]}{'...' if len(text) > 250 else ''}")
    if "orig_criteria" in sample:
        print(f"    orig_criteria: {sample['orig_criteria']}")
    print()


if __name__ == "__main__":
    print("=" * 100)
    print("RISK - false negatives (missed unsafe content, most confident misses first)")
    print("=" * 100)
    fns, fps = mine(
        "results/raw/jev_zeroshot_risk_dev.jsonl",
        "data/samples/risk_dev.jsonl",
        "ground_truth_is_safe",
        "is_safe",
        "safe",
        "unsafe",
    )
    print(f"total FN={len(fns)} FP={len(fps)}\n")
    for prob_true, id_, sample, res in fns[:8]:
        print_case(prob_true, id_, sample, res, ["prompt", "response"])

    print("=" * 100)
    print("RISK - false positives (flagged safe content as unsafe, most confident misses first)")
    print("=" * 100)
    for prob_true, id_, sample, res in fps[:8]:
        print_case(prob_true, id_, sample, res, ["prompt", "response"])

    print("\n" + "=" * 100)
    print("QUALITY - false negatives (missed low-quality, most confident misses first)")
    print("=" * 100)
    qfns, qfps = mine(
        "results/raw/jev_zeroshot_quality_dev.jsonl",
        "data/samples/quality_dev.jsonl",
        "ground_truth_is_high_quality",
        "is_high_quality",
        "high",
        "low",
    )
    print(f"total FN={len(qfns)} FP={len(qfps)}\n")
    for prob_true, id_, sample, res in qfns[:8]:
        print_case(prob_true, id_, sample, res, ["instruction", "response"])

    print("=" * 100)
    print("QUALITY - false positives (flagged high-quality as low-quality, most confident misses first)")
    print("=" * 100)
    for prob_true, id_, sample, res in qfps[:8]:
        print_case(prob_true, id_, sample, res, ["instruction", "response"])
