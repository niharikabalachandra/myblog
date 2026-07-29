#!/usr/bin/env bash
# PreCompact hook (global): backs up the current session .jsonl before
# Claude auto-compacts the context window.
#
# Second layer of defense alongside the status line backup. Fires exactly once
# at compaction time, regardless of the status line's 5-minute throttle.
#
# Backup location: <repo_root>/claude_chat_backup/ (created if missing)
#
# Register in settings.json:
#   "hooks": { "PreCompact": [ { "matcher": "*", "hooks": [
#     { "type": "command", "command": "$HOME/.claude/hooks/pre-compact-backup.sh",
#       "timeout": 20 } ] } ] }

# NOT `set -e`. A PreCompact hook that exits non-zero can block compaction —
# a backup script must never do that. Every exit below is explicit and zero.
set -uo pipefail

LOCK_STALE_MIN=10   # reclaim a lockfile older than this

fail() { echo "pre-compact-backup: $1" >&2; exit 0; }

# --- resolve jq --------------------------------------------------------------
JQ=""
for candidate in jq /opt/homebrew/bin/jq /usr/local/bin/jq; do
  if command -v "$candidate" >/dev/null 2>&1; then
    JQ=$(command -v "$candidate"); break
  fi
done
[ -n "$JQ" ] || fail "jq not found, skipping"

# --- read the PreCompact payload from stdin ----------------------------------
# The payload carries transcript_path directly, so there is no need to
# reconstruct Claude's project-directory encoding. That encoding is an internal
# detail and has changed before; don't build on it if you don't have to.
payload=$(cat)

# Read loop rather than `mapfile`: mapfile is bash 4+, and macOS ships bash 3.2.
FIELDS=("" "" "")
_i=0
while IFS= read -r _line; do
  FIELDS[$_i]="$_line"; _i=$(( _i + 1 ))
done < <(
  printf '%s' "$payload" | "$JQ" -r '
    [ (.session_id // ""), (.transcript_path // ""), (.cwd // "") ] | .[]
  ' 2>/dev/null
)
SESSION_ID=${FIELDS[0]:-}
TRANSCRIPT_PATH=${FIELDS[1]:-}
PAYLOAD_CWD=${FIELDS[2]:-}

[ -n "$SESSION_ID" ] || fail "no session_id in payload, skipping"

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-${PAYLOAD_CWD:-$PWD}}"

# --- locate the session transcript -------------------------------------------
SESSION_JSONL=""
if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
  SESSION_JSONL="$TRANSCRIPT_PATH"
else
  # Fallback 1: find by UUID filename, independent of path encoding
  SESSION_JSONL=$(find "$HOME/.claude/projects" -maxdepth 2 \
                  -name "${SESSION_ID}.jsonl" 2>/dev/null | head -1)
fi

if [ -z "$SESSION_JSONL" ]; then
  # Fallback 2: most recently modified transcript. `ls -t` rather than
  # `stat -f`, which is BSD-only and fails on Linux. And `-exec ls -t {} +`
  # rather than `| xargs -0 ls -t`: with no matches GNU xargs still runs
  # `ls -t`, listing the current directory and returning an unrelated file.
  SESSION_JSONL=$(find "$HOME/.claude/projects" -maxdepth 2 -name '*.jsonl' \
                  -exec ls -t {} + 2>/dev/null | head -1)
fi

[ -n "$SESSION_JSONL" ] && [ -f "$SESSION_JSONL" ] \
  || fail "session file not found for ${SESSION_ID}"

# --- destination -------------------------------------------------------------
REPO_ROOT=$(git -C "$PROJECT_DIR" rev-parse --show-toplevel 2>/dev/null \
            || echo "$PROJECT_DIR")
BACKUP_DIR="${REPO_ROOT}/claude_chat_backup"
mkdir -p "$BACKUP_DIR" || fail "could not create ${BACKUP_DIR}"

SESSION_SHORT="${SESSION_ID:0:8}"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

# Session ID in the filename, matching the status line layer. Without it two
# agents compacting in the same second write to the same path.
DEST="${BACKUP_DIR}/auto_backup_${TIMESTAMP}_${SESSION_SHORT}.jsonl"

# Per-session lockfile, not a global one. A shared lock means that when two
# sessions compact at once, one silently loses its backup at the exact moment
# the backup matters most.
LOCKFILE="${BACKUP_DIR}/.backup_${SESSION_SHORT}.lock"

# Reclaim a stale lock left behind by a crash, or this session never backs up again.
if [ -e "$LOCKFILE" ] \
   && [ -z "$(find "$LOCKFILE" -mmin "-${LOCK_STALE_MIN}" 2>/dev/null)" ]; then
  rm -f "$LOCKFILE"
fi

if ! (set -C; echo $$ > "$LOCKFILE") 2>/dev/null; then
  fail "backup already in progress for ${SESSION_SHORT}, skipping"
fi
trap 'rm -f "$LOCKFILE"' EXIT

# The transcript is flushed asynchronously and can lag the live conversation.
# This matters more here than in the status line: it's the last capture before
# the context is compacted.
sleep 0.5

if cp "$SESSION_JSONL" "$DEST" 2>/dev/null; then
  echo "pre-compact-backup: saved ${DEST}" >&2
else
  echo "pre-compact-backup: copy failed for ${SESSION_ID}" >&2
fi

exit 0
