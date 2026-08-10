#!/usr/bin/env python3
"""Benchmark: ingestion cost and query latency across corpus sizes, to check
whether OKF-indexed lookup is actually flat while the semantic scan and a
naive OKF directory-scan both grow with corpus size. Also reports semantic
recall@1 at 768 vs 256 embedding dims.

Usage: bench.py [--sizes 25,50,100,200,400,800,1600] [--recall-dim-size 200]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import gen_synthetic
import okf_build
import recall

SCRATCH = Path("bench_scratch")


def scan_query_by_tag(tag: str, okf_dir: Path) -> list[dict]:
    """Deliberately naive O(n) alternative to okf_build.query_by_tag: scans
    every concept file instead of hitting the persisted tag index. This is
    the baseline the indexed lookup is being compared against."""
    results = []
    concepts_dir = okf_dir / "concepts"
    for path in concepts_dir.glob("session-*.md"):
        fm = okf_build.read_concept_frontmatter(path)
        if fm and tag in fm.get("tags", []):
            results.append({"concept_id": path.stem, **fm})
    return results


def time_it(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def bench_size(n: int) -> dict:
    corpus_dir = SCRATCH / f"corpus_{n}"
    db_path = SCRATCH / f"recall_{n}.db"
    okf_dir = SCRATCH / f"okf_{n}"
    for p in (corpus_dir, okf_dir):
        if p.exists():
            shutil.rmtree(p)
    db_path.unlink(missing_ok=True)

    manifest = gen_synthetic.generate_corpus(corpus_dir, n, seed=42)
    ground_truth = json.loads((corpus_dir / "ground_truth.json").read_text())

    semantic_ingest_s, _ = time_it(recall.index_folder, str(corpus_dir), db_path=str(db_path))
    okf_ingest_s, _ = time_it(okf_build.build, str(corpus_dir), out_dir=str(okf_dir))

    # Query latency: average over the ground-truth queries (same query set
    # at every size, so latency deltas reflect corpus size, not query mix).
    semantic_times = []
    okf_scan_times = []
    okf_indexed_times = []
    sample_tag = None
    for gt in ground_truth:
        t, _ = time_it(recall.search, gt["query"], k=3, db_path=str(db_path), verbose_routing=False)
        semantic_times.append(t)

        # Use the session's own topic word as the tag probe for the scan
        # vs. indexed comparison (both look up the same tag).
        tag = gt["topic"].split("-")[0]
        sample_tag = tag
        t, _ = time_it(scan_query_by_tag, tag, okf_dir)
        okf_scan_times.append(t)
        t, _ = time_it(okf_build.query_by_tag, tag, out_dir=str(okf_dir))
        okf_indexed_times.append(t)

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_sessions": n,
        "semantic_ingest_s": semantic_ingest_s,
        "okf_ingest_s": okf_ingest_s,
        "semantic_query_ms": avg(semantic_times) * 1000,
        "okf_scan_query_ms": avg(okf_scan_times) * 1000,
        "okf_indexed_query_ms": avg(okf_indexed_times) * 1000,
        "sample_tag": sample_tag,
    }


def bench_recall_at_1(n: int, dims: list[int]) -> dict:
    corpus_dir = SCRATCH / f"recall1_corpus_{n}"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    manifest = gen_synthetic.generate_corpus(corpus_dir, n, seed=7)
    ground_truth = json.loads((corpus_dir / "ground_truth.json").read_text())

    out = {}
    for dim in dims:
        db_path = SCRATCH / f"recall1_{n}_{dim}.db"
        db_path.unlink(missing_ok=True)
        # Force a fresh model load at this dim -- recall._load_model caches
        # a single model globally, keyed by nothing, so a stale dim would
        # silently be reused otherwise.
        recall._model = None
        import os
        os.environ["RECALL_TRUNCATE_DIM"] = str(dim)
        recall.index_folder(str(corpus_dir), db_path=str(db_path))

        hits = 0
        for gt in ground_truth:
            results = recall.search(gt["query"], k=1, db_path=str(db_path), verbose_routing=False)
            if results and results[0]["session_id"] == gt["session_short"]:
                hits += 1
        out[dim] = hits / len(ground_truth)
    recall._model = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="25,50,100,200,400,800,1600")
    ap.add_argument("--recall-dim-size", type=int, default=200)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    SCRATCH.mkdir(exist_ok=True)

    print(f"{'N':>6} {'sem_ingest_s':>13} {'okf_ingest_s':>13} "
          f"{'sem_query_ms':>13} {'okf_scan_ms':>12} {'okf_idx_ms':>11}")
    rows = []
    for n in sizes:
        row = bench_size(n)
        rows.append(row)
        print(f"{row['n_sessions']:>6} {row['semantic_ingest_s']:>13.3f} {row['okf_ingest_s']:>13.3f} "
              f"{row['semantic_query_ms']:>13.3f} {row['okf_scan_query_ms']:>12.3f} "
              f"{row['okf_indexed_query_ms']:>11.3f}")

    first, last = rows[0], rows[-1]
    size_ratio = last["n_sessions"] / first["n_sessions"]

    def growth(key):
        a, b = first[key], last[key]
        return (b / a) if a > 0 else float("inf")

    print(f"\nCorpus size grew {size_ratio:.0f}x ({first['n_sessions']} -> {last['n_sessions']} sessions).")
    print(f"semantic query latency grew {growth('semantic_query_ms'):.1f}x")
    print(f"OKF *scan* query latency grew {growth('okf_scan_query_ms'):.1f}x")
    print(f"OKF *indexed* query latency grew {growth('okf_indexed_query_ms'):.1f}x")
    print("(indexed should be roughly flat -- near 1x -- while both scans track corpus size)")

    print(f"\n=== recall@1, semantic layer, 768 vs 256 dims (N={args.recall_dim_size}) ===")
    r = bench_recall_at_1(args.recall_dim_size, [768, 256])
    for dim, score in r.items():
        print(f"  dim={dim}: recall@1 = {score:.2f}")


if __name__ == "__main__":
    main()
