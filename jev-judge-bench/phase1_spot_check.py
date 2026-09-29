"""Phase 1: broader spot-check across all 2,150 sampled rows - length
outliers, empty fields, exact duplicates, plus a random sample of full
payloads to eyeball."""
import json
import random
import statistics

FILES = [
    "data/samples/risk_dev.jsonl",
    "data/samples/risk_test.jsonl",
    "data/samples/quality_dev.jsonl",
    "data/samples/quality_test.jsonl",
]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def state_text(row):
    state = row["payload"]["state"]
    return " ".join(str(v) for v in state.values())


def check_file(path):
    rows = load(path)
    lengths = [len(state_text(r)) for r in rows]
    empties = [r["id"] for r in rows if any(v == "" or v is None for v in row_values(r))]
    raw_seen = {}
    dupes = []
    for r in rows:
        key = json.dumps(r["raw"], sort_keys=True)
        if key in raw_seen:
            dupes.append((raw_seen[key], r["id"]))
        raw_seen[key] = r["id"]

    print(f"\n=== {path} (n={len(rows)}) ===")
    print(
        f"state char length: min={min(lengths)} max={max(lengths)} "
        f"mean={statistics.mean(lengths):.0f} median={statistics.median(lengths):.0f}"
    )
    print(f"empty fields: {empties[:10]}{' ...' if len(empties) > 10 else ''} (total {len(empties)})")
    print(f"exact duplicate raw content: {len(dupes)}{(' e.g. ' + str(dupes[:5])) if dupes else ''}")

    shortest = min(rows, key=lambda r: len(state_text(r)))
    longest = max(rows, key=lambda r: len(state_text(r)))
    print(f"shortest ({len(state_text(shortest))} chars, id={shortest['id']}): {state_text(shortest)[:200]!r}")
    print(f"longest  ({len(state_text(longest))} chars, id={longest['id']}): {state_text(longest)[:200]!r}...")

    return rows


def row_values(r):
    return r["raw"].values()


def print_random_full_examples(all_rows_by_file, n_per_file=4, seed=7):
    rng = random.Random(seed)
    for path, rows in all_rows_by_file.items():
        print(f"\n--- random full examples: {path} ---")
        for r in rng.sample(rows, n_per_file):
            print(json.dumps(r, indent=2)[:1500])
            print("...")


if __name__ == "__main__":
    all_rows = {}
    for path in FILES:
        all_rows[path] = check_file(path)
    print_random_full_examples(all_rows)
