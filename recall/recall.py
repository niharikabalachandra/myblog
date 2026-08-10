#!/usr/bin/env python3
"""Layer 1: semantic + lexical (BM25) search over Claude Code session backups,
fused with Reciprocal Rank Fusion. The fallback layer — covers anything, at
O(n) scan cost.

Commands:
  recall.py index <folder>
  recall.py search <query> [-k N]
  recall.py stats
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

import _transcript as tr

DEFAULT_DB = "recall.db"
DEFAULT_TRUNCATE_DIM = 768
RRF_K = 60  # standard RRF damping constant
VECTOR_CANDIDATES = 50
LEXICAL_CANDIDATES = 50

_model = None  # lazy-loaded SentenceTransformer, module-level cache


def _get_truncate_dim() -> int:
    return int(os.environ.get("RECALL_TRUNCATE_DIM", DEFAULT_TRUNCATE_DIM))


def _load_model(truncate_dim: int):
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v2-moe",
        trust_remote_code=True,
        truncate_dim=truncate_dim,
    )
    return _model


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize rows. truncate_dim slices the model's built-in Normalize()
    output, which leaves norms != 1 (verified empirically) — always
    re-normalize explicitly rather than trusting the pipeline's own norm."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def embed_passages(texts: list[str], truncate_dim: int) -> np.ndarray:
    """Embed chunks being INDEXED. Kept as a separate function from
    embed_query so the two prompt types can never accidentally share a
    prefix."""
    model = _load_model(truncate_dim)
    vecs = model.encode(texts, prompt_name="passage", show_progress_bar=False)
    return _normalize(vecs)


def embed_query(text: str, truncate_dim: int) -> np.ndarray:
    """Embed a search QUERY. Kept separate from embed_passages — see above."""
    model = _load_model(truncate_dim)
    vecs = model.encode([text], prompt_name="query", show_progress_bar=False)
    return _normalize(vecs)[0]


# ── DB ────────────────────────────────────────────────────────────────────

def get_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            session_id TEXT NOT NULL,
            text TEXT NOT NULL,
            start_turn INTEGER,
            end_turn INTEGER,
            start_ts TEXT,
            end_ts TEXT,
            source_file TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED, text
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    con.commit()
    return con


def _get_meta(con: sqlite3.Connection, key: str, default=None):
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def _set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


# ── Indexing ─────────────────────────────────────────────────────────────

def index_folder(folder: str, db_path: str = DEFAULT_DB, batch_size: int = 32) -> dict:
    con = get_db(db_path)
    truncate_dim = _get_truncate_dim()

    indexed_dim = _get_meta(con, "embedding_dim")
    if indexed_dim is not None and int(indexed_dim) != truncate_dim:
        raise SystemExit(
            f"recall.db was built with embedding_dim={indexed_dim}, but "
            f"RECALL_TRUNCATE_DIM={truncate_dim} now. Re-index into a fresh "
            f"db (different --db) or unset the env var to match the existing index."
        )

    files = tr.list_backup_files(Path(folder))
    n_files_processed = 0
    n_files_skipped = 0
    n_chunks_indexed = 0

    for f in files:
        mtime, size = tr.file_fingerprint(f)
        row = con.execute("SELECT mtime, size FROM files WHERE path = ?", (str(f),)).fetchone()
        if row and row[0] == mtime and row[1] == size:
            n_files_skipped += 1
            continue

        # Drop old chunks for this SESSION, not just this file. The backup
        # folder accumulates multiple snapshots of the same growing session
        # over time (that's the whole point of the status-line + PreCompact
        # design) -- a newer snapshot re-covers the same turn ranges as an
        # older one, so deduping by source_file alone leaves stale chunks
        # whose chunk_id collides with the new file's on re-index. Files are
        # processed oldest-to-newest (list_backup_files sorts by filename,
        # which embeds a sortable timestamp), so the newest snapshot for a
        # session always wins.
        session_id = tr.session_id_from_filename(f)
        old_chunk_ids = [r[0] for r in con.execute(
            "SELECT chunk_id FROM chunks WHERE session_id = ?", (session_id,)).fetchall()]
        if old_chunk_ids:
            con.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(c,) for c in old_chunk_ids])
            con.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(c,) for c in old_chunk_ids])

        chunks = list(tr.iter_chunks(f))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vecs = embed_passages([c.text for c in batch], truncate_dim)
            for c, v in zip(batch, vecs):
                con.execute(
                    "INSERT INTO chunks (chunk_id, session_id, text, start_turn, end_turn, "
                    "start_ts, end_ts, source_file, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
                    (c.chunk_id, c.session_id, c.text, c.start_turn, c.end_turn,
                     c.start_ts, c.end_ts, c.source_file, v.astype(np.float32).tobytes()),
                )
                con.execute(
                    "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                    (c.chunk_id, c.text),
                )
        n_chunks_indexed += len(chunks)

        con.execute(
            "INSERT INTO files (path, mtime, size) VALUES (?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET mtime = excluded.mtime, size = excluded.size",
            (str(f), mtime, size),
        )
        n_files_processed += 1
        con.commit()

    if indexed_dim is None and n_chunks_indexed > 0:
        _set_meta(con, "embedding_dim", str(truncate_dim))
        con.commit()

    con.close()
    return {
        "files_processed": n_files_processed,
        "files_skipped": n_files_skipped,
        "chunks_indexed": n_chunks_indexed,
    }


# ── Search ───────────────────────────────────────────────────────────────

_FTS_SPECIAL = re.compile(r'["*^]')


def _fts_query(raw: str) -> str:
    """Turn a free-text query into a safe FTS5 MATCH expression: quote each
    token so punctuation in the query can't be parsed as FTS5 syntax."""
    tokens = re.findall(r"[A-Za-z0-9_]+", raw)
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


def vector_search(con: sqlite3.Connection, query: str, truncate_dim: int, top_n: int) -> list[tuple[str, float]]:
    qvec = embed_query(query, truncate_dim)
    rows = con.execute("SELECT chunk_id, embedding FROM chunks").fetchall()
    if not rows:
        return []
    ids = [r[0] for r in rows]
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    sims = mat @ qvec  # both sides pre-normalized -> dot product == cosine sim
    order = np.argsort(-sims)[:top_n]
    return [(ids[i], float(sims[i])) for i in order]


def lexical_search(con: sqlite3.Connection, query: str, top_n: int) -> list[tuple[str, float]]:
    fq = _fts_query(query)
    try:
        rows = con.execute(
            "SELECT chunk_id, bm25(chunks_fts) AS rank FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (fq, top_n),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    # bm25() in SQLite is *more negative = better*; keep as-is, rank order matters for RRF.
    return [(r[0], float(r[1])) for r in rows]


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[str, float]]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def search(query: str, k: int = 5, db_path: str = DEFAULT_DB, verbose_routing: bool = True) -> list[dict]:
    con = get_db(db_path)
    truncate_dim = int(_get_meta(con, "embedding_dim", _get_truncate_dim()))

    vec_ranked = vector_search(con, query, truncate_dim, VECTOR_CANDIDATES)
    lex_ranked = lexical_search(con, query, LEXICAL_CANDIDATES)
    fused = reciprocal_rank_fusion([vec_ranked, lex_ranked])[:k]

    vec_ids = {cid for cid, _ in vec_ranked}
    lex_ids = {cid for cid, _ in lex_ranked}

    results = []
    for chunk_id, score in fused:
        row = con.execute(
            "SELECT session_id, text, start_ts, end_ts, source_file FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if not row:
            continue
        session_id, text, start_ts, end_ts, source_file = row
        tiers = []
        if chunk_id in vec_ids:
            tiers.append("vector")
        if chunk_id in lex_ids:
            tiers.append("lexical")
        results.append({
            "chunk_id": chunk_id,
            "session_id": session_id,
            "score": score,
            "tiers": tiers,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "source_file": source_file,
            "text": text,
        })
    con.close()
    if verbose_routing:
        print(f"[recall] layer=semantic vector_hits={len(vec_ranked)} lexical_hits={len(lex_ranked)} fused_top_k={len(results)}",
              file=sys.stderr)
    return results


def stats(db_path: str = DEFAULT_DB) -> dict:
    con = get_db(db_path)
    n_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_sessions = con.execute("SELECT COUNT(DISTINCT session_id) FROM chunks").fetchone()[0]
    n_files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    dim = _get_meta(con, "embedding_dim", "(unset — nothing indexed yet)")
    db_size = Path(db_path).stat().st_size if Path(db_path).exists() else 0
    con.close()
    return {
        "chunks": n_chunks,
        "sessions": n_sessions,
        "files_indexed": n_files,
        "embedding_dim": dim,
        "db_bytes": db_size,
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB, help="path to recall.db (default: ./recall.db)")
    sub = ap.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index a folder of auto_backup_*.jsonl files")
    p_index.add_argument("folder")

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)

    sub.add_parser("stats", help="print index stats")

    args = ap.parse_args()

    if args.command == "index":
        result = index_folder(args.folder, db_path=args.db)
        print(f"Indexed {result['chunks_indexed']} chunks from "
              f"{result['files_processed']} files "
              f"({result['files_skipped']} unchanged, skipped) -> {args.db}")

    elif args.command == "search":
        results = search(args.query, k=args.k, db_path=args.db)
        if not results:
            print("No results. Did you run `recall.py index <folder>` first?")
            return
        for i, r in enumerate(results, start=1):
            tiers = "+".join(r["tiers"])
            snippet = r["text"][:220].replace("\n", " ")
            print(f"\n[{i}] session={r['session_id']} score={r['score']:.4f} tiers={tiers}")
            print(f"    {snippet}...")

    elif args.command == "stats":
        s = stats(db_path=args.db)
        print(f"chunks:        {s['chunks']}")
        print(f"sessions:      {s['sessions']}")
        print(f"files indexed: {s['files_indexed']}")
        print(f"embedding dim: {s['embedding_dim']}")
        print(f"db size:       {s['db_bytes'] / 1024:.1f} KB")


if __name__ == "__main__":
    main()
