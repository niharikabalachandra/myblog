#!/usr/bin/env python3
"""Layer 2: OKF (Open Knowledge Format, spec v0.2) structured layer over
Claude Code session backups. The front layer — fast, readable, but only
covers what got a concept written for it. One markdown concept per session
under concepts/, a persisted tag -> [concept_ids] index for O(1) lookup, and
a root index.md for progressive disclosure.

Spec: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
  - Only `type` is a required frontmatter field; title/description/tags are
    recommended extras (spec explicitly allows producer-added keys).
  - Concept links use standard markdown links, bundle-relative (leading `/`).
  - index.md needs no frontmatter except an optional `okf_version` key, and
    lists concepts as "* [Title](url) - description" grouped under headings.
  - `index.md` / `log.md` are reserved filenames, never used for concepts.

Commands:
  okf_build.py build <backup_folder> [--out DIR] [--llm]
  okf_build.py query <tag> [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import _transcript as tr

DEFAULT_OUT = "okf"
OKF_VERSION = "0.2"

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of",
    "in", "on", "at", "for", "with", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "we",
    "our", "you", "your", "i", "do", "does", "did", "not", "no", "yes",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "have", "has", "had", "here", "there", "what", "when", "where", "why",
    "how", "which", "who", "whom", "let", "let's", "one", "any", "all",
    "some", "into", "out", "up", "down", "over", "under", "again", "just",
    "now", "than", "too", "very", "s", "t", "re", "ve", "ll", "d", "m",
    "user", "assistant", "used", "tool", "from", "based", "first", "here's",
    "context", "let's", "anything", "else", "check",
}

DECISION_PATTERNS = [
    re.compile(r"\bwe(?:'re| are)? (decided|chose|opted)\b", re.I),
    re.compile(r"\bwe(?:'re| are) (going with|keeping|setting|adding|dropping|rejecting)\b", re.I),
    re.compile(r"\bthe (root cause|fix|reason) (was|is)\b", re.I),
    re.compile(r"\bwe (decided|chose|opted)\b", re.I),
]

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")


def extract_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def rule_based_tags(all_text: str, top_n: int = 6) -> list[str]:
    words = [w.lower() for w in WORD_RE.findall(all_text)]
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_n)]


def rule_based_decisions(assistant_text_blocks: list[str]) -> list[str]:
    found = []
    for block in assistant_text_blocks:
        for sentence in extract_sentences(block):
            if any(p.search(sentence) for p in DECISION_PATTERNS):
                found.append(sentence)
    return found


def rule_based_distill(turns: list[tr.Turn]) -> dict:
    assistant_texts = [t.text for t in turns if t.role == "assistant"]
    all_text = "\n".join(t.text for t in turns)
    decisions = rule_based_decisions(assistant_texts)
    tags = rule_based_tags(all_text)
    if decisions:
        description = decisions[0][:200]
        title = decisions[0][:60].rstrip(",.;: ") or "Untitled session"
    elif turns:
        description = turns[0].text[:200]
        title = turns[0].text[:60].rstrip(",.;: ") or "Untitled session"
    else:
        description = "Empty session"
        title = "Untitled session"
    return {"tags": tags, "decisions": decisions, "title": title, "description": description}


# ── Agent-authored mode (one LLM call per session, costs API tokens) ───────

_LLM_MODEL = "claude-opus-5"


def llm_distill(turns: list[tr.Turn]) -> dict:
    """Requires the `anthropic` package and API credentials
    (ANTHROPIC_API_KEY, or `ant auth login`). Not exercised in this
    environment — no credentials were available when this was built; the
    rule-based path is what's actually been verified end to end."""
    from pydantic import BaseModel
    import anthropic

    class Distillation(BaseModel):
        title: str
        description: str
        tags: list[str]
        decisions: list[str]

    transcript_text = "\n".join(f"{t.role}: {t.text}" for t in turns)
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=_LLM_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "Distill this Claude Code session transcript into an OKF concept. "
                "title: short human-readable name. description: one sentence. "
                "tags: 3-8 short lowercase keyword tags. decisions: the concrete "
                "decisions/conclusions reached, as short standalone sentences "
                "(empty list if none).\n\n" + transcript_text
            ),
        }],
        output_format=Distillation,
    )
    d = response.parsed_output
    return {"tags": d.tags, "decisions": d.decisions, "title": d.title, "description": d.description}


# ── Concept file I/O ────────────────────────────────────────────────────

def _yaml_escape(s: str) -> str:
    return s.replace('"', '\\"')


def write_concept(concepts_dir: Path, session_id: str, distilled: dict,
                   timestamp: Optional[str], related: list[tuple[str, str]]) -> None:
    tags_yaml = "[" + ", ".join(f'"{_yaml_escape(t)}"' for t in distilled["tags"]) + "]"
    lines = [
        "---",
        'type: "Coding Session"',
        f'title: "{_yaml_escape(distilled["title"])}"',
        f'description: "{_yaml_escape(distilled["description"])}"',
        f"tags: {tags_yaml}",
        f'timestamp: "{timestamp or ""}"',
        "---",
        "",
        "# Decisions",
        "",
    ]
    if distilled["decisions"]:
        for d in distilled["decisions"]:
            lines.append(f"- {d}")
    else:
        lines.append("- (no decision-pattern sentences found)")
    lines.append("")
    lines.append("# Related")
    lines.append("")
    if related:
        for rel_id, rel_title in related:
            lines.append(f"- [{rel_title}](/concepts/{rel_id}.md)")
    else:
        lines.append("(no related concepts share a tag)")
    lines.append("")

    (concepts_dir / f"session-{session_id}.md").write_text("\n".join(lines), encoding="utf-8")


def read_concept_frontmatter(path: Path) -> Optional[dict]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    fm_text = text[3:end].strip("\n")
    fm: dict = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [_unquote(v.strip()) for v in value[1:-1].split(",") if v.strip()]
            fm[key] = items
        else:
            fm[key] = _unquote(value)
    return fm


def _unquote(value: str) -> str:
    """Strip exactly one leading/trailing quote (not a char-class strip,
    which over-strips when an escaped quote sits next to the closing one),
    then unescape \\" -> "."""
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value.replace('\\"', '"')


# ── Build ────────────────────────────────────────────────────────────────

def build(folder: str, out_dir: str = DEFAULT_OUT, use_llm: bool = False) -> dict:
    out = Path(out_dir)
    concepts_dir = out / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    state_path = out / ".build_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    files = tr.list_backup_files(Path(folder))
    n_built = 0
    n_skipped = 0

    for f in files:
        session_id = tr.session_id_from_filename(f)
        mtime, size = tr.file_fingerprint(f)
        key = str(f)
        if state.get(key) == [mtime, size] and (concepts_dir / f"session-{session_id}.md").exists():
            n_skipped += 1
            continue

        turns = list(tr.iter_turns(f))
        distilled = llm_distill(turns) if use_llm else rule_based_distill(turns)
        timestamp = turns[0].timestamp if turns else None
        # Related links computed in a second pass once every concept's tags
        # are known; write with an empty placeholder for now.
        write_concept(concepts_dir, session_id, distilled, timestamp, related=[])
        state[key] = [mtime, size]
        n_built += 1

    state_path.write_text(json.dumps(state, indent=2))

    # --- Second pass: build the tag index once, then fill in # Related and
    # write it to disk. This index (NOT the markdown files) is what makes
    # tag lookup O(1) instead of an O(n) directory scan. ---
    tag_index: dict[str, list[str]] = defaultdict(list)
    concept_meta: dict[str, dict] = {}
    for concept_path in concepts_dir.glob("session-*.md"):
        concept_id = concept_path.stem
        fm = read_concept_frontmatter(concept_path)
        if not fm:
            continue
        concept_meta[concept_id] = fm
        for tag in fm.get("tags", []):
            tag_index[tag].append(concept_id)

    for concept_id, fm in concept_meta.items():
        related_ids: set[str] = set()
        for tag in fm.get("tags", []):
            for other_id in tag_index.get(tag, []):
                if other_id != concept_id:
                    related_ids.add(other_id)
        related = [(rid, concept_meta[rid].get("title", rid)) for rid in sorted(related_ids)]
        distilled = {
            "tags": fm.get("tags", []),
            "decisions": _read_decisions_body(concepts_dir / f"{concept_id}.md"),
            "title": fm.get("title", concept_id),
            "description": fm.get("description", ""),
        }
        write_concept(concepts_dir, concept_id.removeprefix("session-"), distilled,
                       fm.get("timestamp"), related)

    (out / "tag_index.json").write_text(json.dumps(dict(tag_index), indent=2))
    _write_root_index(out, concept_meta)

    return {"built": n_built, "skipped": n_skipped, "total_concepts": len(concept_meta)}


def _read_decisions_body(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"# Decisions\n\n(.*?)\n\n# Related", text, re.S)
    if not m:
        return []
    lines = [l[2:].strip() for l in m.group(1).splitlines() if l.startswith("- ")]
    return [l for l in lines if l and l != "(no decision-pattern sentences found)"]


def _write_root_index(out: Path, concept_meta: dict[str, dict]) -> None:
    # Group under each concept's first (primary) tag for progressive
    # disclosure, per spec section 6.
    by_primary_tag: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for concept_id, fm in concept_meta.items():
        primary = (fm.get("tags") or ["untagged"])[0]
        by_primary_tag[primary].append((concept_id, fm))

    lines = [
        "---",
        f'okf_version: "{OKF_VERSION}"',
        "---",
        "",
        "# Session concepts",
        "",
    ]
    for tag in sorted(by_primary_tag):
        lines.append(f"## {tag}")
        lines.append("")
        for concept_id, fm in sorted(by_primary_tag[tag]):
            title = fm.get("title", concept_id)
            desc = fm.get("description", "")
            lines.append(f"* [{title}](/concepts/{concept_id}.md) - {desc}")
        lines.append("")

    (out / "index.md").write_text("\n".join(lines), encoding="utf-8")


# ── Query (O(1) via the persisted tag index — never scans concept files) ──

def query_by_tag(tag: str, out_dir: str = DEFAULT_OUT) -> Optional[list[dict]]:
    out = Path(out_dir)
    index_path = out / "tag_index.json"
    if not index_path.exists():
        return None
    tag_index = json.loads(index_path.read_text())
    concept_ids = tag_index.get(tag)
    if not concept_ids:
        return None
    results = []
    concepts_dir = out / "concepts"
    for cid in concept_ids:
        fm = read_concept_frontmatter(concepts_dir / f"{cid}.md")
        if fm:
            results.append({"concept_id": cid, **fm})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("folder")
    p_build.add_argument("--llm", action="store_true",
                          help="agent-authored distillation, one LLM call per session (costs API tokens)")

    p_query = sub.add_parser("query")
    p_query.add_argument("tag")

    args = ap.parse_args()

    if args.command == "build":
        result = build(args.folder, out_dir=args.out, use_llm=args.llm)
        print(f"Built {result['built']} concepts ({result['skipped']} unchanged, skipped), "
              f"{result['total_concepts']} total -> {args.out}/")
    elif args.command == "query":
        results = query_by_tag(args.tag, out_dir=args.out)
        if results is None:
            print(f"No tag index entry for {args.tag!r} (miss)")
        else:
            for r in results:
                print(f"{r['concept_id']}: {r.get('title')}")


if __name__ == "__main__":
    main()
