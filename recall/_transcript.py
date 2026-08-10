"""Shared parsing and chunking for Claude Code session backups.

Backup files: ./claude_chat_backup/auto_backup_<timestamp>_<session_short>.jsonl
One JSON event per line. Role lives in the top-level `.type` field ("user" /
"assistant"); turn content lives under `.message.content`, which is either a
plain string or a list of typed content blocks (text / tool_use / tool_result
/ thinking / ...).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

FILENAME_RE = re.compile(r"^auto_backup_(?P<timestamp>\d{8}_\d{6})_(?P<session_short>.+)\.jsonl$")

CHUNK_WINDOW = 4
CHUNK_STRIDE = 2


def list_backup_files(folder: Path) -> list[Path]:
    return sorted(Path(folder).glob("auto_backup_*.jsonl"))


def session_id_from_filename(path: Path) -> str:
    m = FILENAME_RE.match(Path(path).name)
    if m:
        return m.group("session_short")
    # Fall back to the whole stem so unexpected filenames don't crash indexing.
    return Path(path).stem


def file_fingerprint(path: Path) -> tuple[float, int]:
    st = Path(path).stat()
    return (st.st_mtime, st.st_size)


@dataclass
class Turn:
    index: int
    role: str
    text: str
    timestamp: Optional[str]


def _text_from_content(content) -> str:
    """Flatten .message.content into text, dropping tool output, keeping
    short markers for tool calls."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"].strip())
        elif btype == "tool_use":
            name = block.get("name", "tool")
            parts.append(f"[used tool: {name}]")
        # tool_result, thinking, image, and anything else: dropped (high
        # volume, low retrieval signal).
    return "\n".join(p for p in parts if p)


def iter_turns(path: Path) -> Iterator[Turn]:
    """Yield one Turn per user/assistant event with non-empty text."""
    idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            role = event.get("type")
            if role not in ("user", "assistant"):
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            text = _text_from_content(message.get("content"))
            if not text:
                continue
            yield Turn(
                index=idx,
                role=role,
                text=text,
                timestamp=event.get("timestamp"),
            )
            idx += 1


@dataclass
class Chunk:
    session_id: str
    chunk_id: str
    text: str
    start_turn: int
    end_turn: int
    start_ts: Optional[str]
    end_ts: Optional[str]
    source_file: str


def chunk_turns(turns: list[Turn], session_id: str, source_file: str,
                 window: int = CHUNK_WINDOW, stride: int = CHUNK_STRIDE) -> Iterator[Chunk]:
    if not turns:
        return
    n = len(turns)
    start = 0
    made_any = False
    while start < n:
        window_turns = turns[start:start + window]
        if not window_turns:
            break
        text = "\n\n".join(f"{t.role}: {t.text}" for t in window_turns)
        yield Chunk(
            session_id=session_id,
            chunk_id=f"{session_id}:{window_turns[0].index}-{window_turns[-1].index}",
            text=text,
            start_turn=window_turns[0].index,
            end_turn=window_turns[-1].index,
            start_ts=window_turns[0].timestamp,
            end_ts=window_turns[-1].timestamp,
            source_file=source_file,
        )
        made_any = True
        if start + window >= n:
            break
        start += stride
    # window==stride edge case guard is unnecessary here since stride < window
    # by construction (2 < 4), but keep made_any for callers that care.
    _ = made_any


def iter_chunks(path: Path) -> Iterator[Chunk]:
    session_id = session_id_from_filename(path)
    turns = list(iter_turns(path))
    yield from chunk_turns(turns, session_id=session_id, source_file=str(path))
