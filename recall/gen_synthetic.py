"""Generate a synthetic corpus of Claude Code session backups for testing
recall.py / okf_build.py without touching real transcripts.

Usage: python gen_synthetic.py <out_dir> <n_sessions> [--seed N]
Writes auto_backup_<timestamp>_<session_short>.jsonl files plus a
ground_truth.json mapping a handful of queries to the session_short that
should answer them (for recall@1 benchmarking).
"""

from __future__ import annotations

import argparse
import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

# Each topic has: a tag word, a set of decision sentences (what the "assistant"
# concluded), and filler exchange sentences so sessions aren't just decisions
# back to back. Deliberately distinct vocabulary per topic so semantic search
# and tag-based OKF lookup have real signal to separate on.
TOPICS = {
    "database-migration": {
        "tag": "database-migration",
        "decisions": [
            "We decided to run the schema migration in two phases to avoid locking the orders table during peak traffic.",
            "The migration will backfill the new column in batches of 5000 rows instead of one transaction.",
            "We're keeping the old column around for one release cycle before dropping it, in case rollback is needed.",
        ],
        "filler": [
            "Can you check how large the orders table is before we run this?",
            "The orders table has about 40 million rows.",
            "What's our rollback plan if the migration fails halfway?",
            "Let's add a dry-run flag to the migration script first.",
        ],
    },
    "auth-bug": {
        "tag": "auth-bug",
        "decisions": [
            "The root cause was a race condition where the refresh token was rotated before the old session finished validating.",
            "We fixed it by making token rotation idempotent within a five second window.",
            "We decided not to add a distributed lock here since the idempotency fix is simpler and sufficient.",
        ],
        "filler": [
            "Users are getting logged out randomly during checkout.",
            "Can you reproduce this locally with two concurrent requests?",
            "I see duplicate refresh token rows in the audit log around the same timestamp.",
            "Let's add a regression test for concurrent refresh requests.",
        ],
    },
    "rate-limiting": {
        "tag": "rate-limiting",
        "decisions": [
            "We decided on a token bucket limiter per API key instead of a fixed window counter, since it handles bursts better.",
            "The limiter will live in Redis with a Lua script for atomic check-and-decrement.",
            "We're setting the default bucket size to 100 requests with a refill rate of 10 per second.",
        ],
        "filler": [
            "Some clients are hammering the search endpoint and starving other tenants.",
            "Should the limit be per API key or per IP address?",
            "What happens when Redis is briefly unavailable?",
            "Let's fail open for five seconds if Redis times out, then fail closed after that.",
        ],
    },
    "caching-strategy": {
        "tag": "caching-strategy",
        "decisions": [
            "We decided to cache the product catalog response for 60 seconds with stale-while-revalidate.",
            "Cache invalidation will be event-driven off the product-updated queue rather than TTL alone.",
            "We're keying the cache by tenant id and locale to avoid cross-tenant leakage.",
        ],
        "filler": [
            "The catalog endpoint is our slowest one under load.",
            "How stale can the catalog data be before it's a real problem for merchants?",
            "Let's check whether the CDN is already caching part of this.",
            "We need a manual cache-bust endpoint for support to use.",
        ],
    },
    "onboarding-flow": {
        "tag": "onboarding-flow",
        "decisions": [
            "We decided to cut the onboarding flow from seven steps down to three by deferring optional profile fields.",
            "Email verification will happen asynchronously after signup instead of blocking account creation.",
            "We're keeping the old flow behind a flag for two weeks to compare conversion.",
        ],
        "filler": [
            "Drop-off is highest on the third onboarding screen.",
            "Which fields are actually required before someone can use the product?",
            "Can we A/B test this instead of a full cutover?",
            "Let's instrument each step so we can see exactly where people leave.",
        ],
    },
    "ci-flakiness": {
        "tag": "ci-flakiness",
        "decisions": [
            "The flaky integration test was caused by a shared test database not being reset between parallel test workers.",
            "We decided to give each CI worker its own ephemeral database instead of trying to serialize test runs.",
            "We're adding a CI check that fails the build if a test leaves rows behind after teardown.",
        ],
        "filler": [
            "This test passes locally but fails about one in ten times in CI.",
            "Are the workers actually running in parallel against the same database?",
            "Let's add logging around setup and teardown to confirm the theory.",
            "How much slower is CI if every worker gets its own database?",
        ],
    },
    "cost-optimization": {
        "tag": "cost-optimization",
        "decisions": [
            "We decided to move the nightly batch job from on-demand instances to spot instances with checkpointing.",
            "We're downsizing the staging environment outside business hours instead of running it 24/7.",
            "The biggest single line item was over-provisioned log retention, which we cut from 90 to 30 days.",
        ],
        "filler": [
            "Our cloud bill grew 40% this quarter without a matching traffic increase.",
            "Which service is actually driving most of the spend?",
            "Can the batch job tolerate being interrupted and resumed?",
            "Let's set a budget alert so this doesn't surprise us again.",
        ],
    },
    "search-relevance": {
        "tag": "search-relevance",
        "decisions": [
            "We decided to boost exact SKU matches above fuzzy text matches in the ranking function.",
            "We're adding a recency signal so newly restocked items don't rank below out-of-stock ones.",
            "We rejected re-ranking with a learned model for now since the rule-based boost already fixed most complaints.",
        ],
        "filler": [
            "Customers are saying they can't find items by exact product code.",
            "How is the current ranking function weighting text similarity versus popularity?",
            "Would a learned ranking model be worth the added complexity right now?",
            "Let's ship the rule-based fix first and measure before reaching for a model.",
        ],
    },
}

TOOL_NAMES = ["Bash", "Read", "Edit", "Grep"]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _rand_session_short(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=8))


def make_session_events(topic_key: str, rng: random.Random, start: datetime) -> list[dict]:
    topic = TOPICS[topic_key]
    events: list[dict] = []
    t = start
    fillers = topic["filler"][:]
    rng.shuffle(fillers)
    decisions = topic["decisions"][:]
    rng.shuffle(decisions)

    def add(role: str, content, use_list_content: bool):
        nonlocal t
        t += timedelta(seconds=rng.randint(15, 90))
        message = {"role": role}
        if use_list_content:
            message["content"] = content if isinstance(content, list) else [{"type": "text", "text": content}]
        else:
            message["content"] = content if isinstance(content, str) else content[0]["text"]
        events.append({"type": role, "timestamp": _iso(t), "message": message})

    # Interleave filler exchanges, a tool call + dropped tool_result, and
    # decision statements, alternating user/assistant. Roughly half the
    # sessions use string content, half use list-of-blocks content, so both
    # code paths in _text_from_content get exercised across the corpus.
    use_list = rng.random() < 0.5

    add("user", f"Here's the context: {fillers[0]}", use_list)
    add("assistant", f"Let me look into that. {fillers[1] if len(fillers) > 1 else ''}", use_list)

    # A tool_use + tool_result pair — tool_result must be dropped downstream.
    add("assistant", [
        {"type": "text", "text": "Checking the current state first."},
        {"type": "tool_use", "name": rng.choice(TOOL_NAMES), "input": {"command": "inspect"}},
    ], True)
    add("user", [
        {"type": "tool_result", "content": "x" * 400 + " raw noisy tool output that should never be indexed " + "y" * 400},
    ], True)

    add("user", fillers[2] if len(fillers) > 2 else "What should we do here?", use_list)
    add("assistant", decisions[0], use_list)

    if len(fillers) > 3:
        add("user", fillers[3], use_list)
    if len(decisions) > 1:
        add("assistant", decisions[1], use_list)
    if len(decisions) > 2:
        add("user", "Anything else we should decide now?", use_list)
        add("assistant", decisions[2], use_list)

    return events


def generate_corpus(out_dir: Path, n_sessions: int, seed: int = 1234) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    topic_keys = list(TOPICS.keys())
    base_time = datetime(2026, 6, 1, 9, 0, 0)
    manifest: list[tuple[str, str]] = []  # (session_short, topic_key)

    for i in range(n_sessions):
        topic_key = topic_keys[i % len(topic_keys)]
        session_short = _rand_session_short(rng)
        session_start = base_time + timedelta(minutes=i * 7, seconds=rng.randint(0, 59))
        events = make_session_events(topic_key, rng, session_start)
        ts_str = session_start.strftime("%Y%m%d_%H%M%S")
        fname = f"auto_backup_{ts_str}_{session_short}.jsonl"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
        manifest.append((session_short, topic_key))

    # A small labeled query -> session_short set for recall@1. Pick one
    # session per topic (the first occurrence) and pair it with a query
    # phrase drawn from that topic's decisions, using vocabulary that
    # doesn't literally quote the decision sentence verbatim.
    ground_truth = []
    seen_topics = set()
    QUERY_PHRASES = {
        "database-migration": "why did we phase the schema migration instead of one big transaction",
        "auth-bug": "what caused users to get logged out during checkout",
        "rate-limiting": "why token bucket instead of fixed window for the api limiter",
        "caching-strategy": "how is the product catalog cache invalidated",
        "onboarding-flow": "why did we cut onboarding down to three steps",
        "ci-flakiness": "root cause of the flaky integration test in CI",
        "cost-optimization": "what was driving the cloud bill increase this quarter",
        "search-relevance": "how do we rank exact sku matches in search",
    }
    for session_short, topic_key in manifest:
        if topic_key in seen_topics:
            continue
        seen_topics.add(topic_key)
        if topic_key in QUERY_PHRASES:
            ground_truth.append({
                "query": QUERY_PHRASES[topic_key],
                "session_short": session_short,
                "topic": topic_key,
            })

    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump([{"session_short": s, "topic": t} for s, t in manifest], f, indent=2)

    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("n_sessions", type=int)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    manifest = generate_corpus(Path(args.out_dir), args.n_sessions, seed=args.seed)
    print(f"Wrote {len(manifest)} synthetic sessions to {args.out_dir}")
