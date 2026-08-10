#!/usr/bin/env python3
"""Composition layer: try the OKF tag index first (O(1)), fall through to
semantic search (O(n) scan) only on a miss. Routing is explicit and logged
to stderr so you can see which layer actually answered.

Usage: query.py "<question>" [--recall-db recall.db] [--okf-dir okf] [-k N]
"""

from __future__ import annotations

import argparse
import re
import sys

import okf_build
import recall

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")


def _candidate_tags(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in okf_build.STOPWORDS]


def query(text: str, recall_db: str = recall.DEFAULT_DB, okf_dir: str = okf_build.DEFAULT_OUT,
          k: int = 5) -> dict:
    """Returns {"layer": "okf"|"semantic", "results": [...]}."""
    for tag in _candidate_tags(text):
        hits = okf_build.query_by_tag(tag, out_dir=okf_dir)
        if hits:
            print(f"[query] layer=okf tag={tag!r} hits={len(hits)}", file=sys.stderr)
            return {"layer": "okf", "matched_tag": tag, "results": hits[:k]}

    print("[query] layer=semantic (no OKF tag matched, falling through to vector+lexical scan)",
          file=sys.stderr)
    results = recall.search(text, k=k, db_path=recall_db, verbose_routing=False)
    return {"layer": "semantic", "matched_tag": None, "results": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text")
    ap.add_argument("--recall-db", default=recall.DEFAULT_DB)
    ap.add_argument("--okf-dir", default=okf_build.DEFAULT_OUT)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    result = query(args.text, recall_db=args.recall_db, okf_dir=args.okf_dir, k=args.k)
    print(f"\n=== answered by: {result['layer']} ===")
    if result["layer"] == "okf":
        for r in result["results"]:
            print(f"\n[{r['concept_id']}] {r.get('title')}")
            print(f"    {r.get('description')}")
    else:
        for i, r in enumerate(result["results"], start=1):
            tiers = "+".join(r["tiers"])
            snippet = r["text"][:200].replace("\n", " ")
            print(f"\n[{i}] session={r['session_id']} score={r['score']:.4f} tiers={tiers}")
            print(f"    {snippet}...")


if __name__ == "__main__":
    main()
