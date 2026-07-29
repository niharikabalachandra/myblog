#!/usr/bin/env bash
# Claude Code status line: context bar, git branch, and auto-backup.
# Backup location: claude_chat_backup/ at repo root (created if missing).
#
# Install: chmod +x ~/.claude/statusline.sh
# Register: "statusLine": { "type": "command", "command": "$HOME/.claude/statusline.sh" }

set -uo pipefail

BAR_WIDTH=10
BACKUP_AT=80        # start backing up at this % of compact budget
THROTTLE_MIN=5      # minutes between backups, per session
COMPACT_BUDGET=0.84 # auto-compact fires at ~84% of the full window

input=$(cat)

# --- locate jq: PATH first, then common Homebrew locations -------------------
JQ=""
for candidate in jq /opt/homebrew/bin/jq /usr/local/bin/jq; do
  if command -v "$candidate" >/dev/null 2>&1; then
    JQ=$(command -v "$candidate")
    break
  fi
done

# --- git branch --------------------------------------------------------------
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
BRANCH_PART=""
[ -n "$BRANCH" ] && BRANCH_PART="| $BRANCH"

# --- degrade gracefully if jq is missing -------------------------------------
if [ -z "$JQ" ]; then
  echo "claude | ctx ?? $BRANCH_PART"
  exit 0
fi

# --- parse the payload in a single jq pass -----------------------------------
# PCT is relative to Claude Code's actual compact budget (84% of the window),
# not the full window. That makes it match the "% until auto-compact" indicator.
# Two things to preserve here:
#  - Do NOT use `IFS=$'\t' read -r ...`. Tab is IFS *whitespace*, so bash
#    collapses consecutive tabs and drops empty fields; a session with no
#    transcript_path would shift PCT into the wrong variable and report 0%.
#  - Do NOT use `mapfile`. It is bash 4+, and macOS still ships bash 3.2 at
#    /bin/bash. The read loop below is equivalent and portable.
FIELDS=("" "" "" "")
_i=0
while IFS= read -r _line; do
  FIELDS[$_i]="$_line"; _i=$(( _i + 1 ))
done < <(
  printf '%s' "$input" | "$JQ" -r --argjson budget_frac "$COMPACT_BUDGET" '
    def n(x): (x // 0);
    ( n(.context_window.current_usage.cache_read_input_tokens)
    + n(.context_window.current_usage.cache_creation_input_tokens)
    + n(.context_window.current_usage.input_tokens) ) as $tokens
    | ( n(.context_window.context_window_size) * $budget_frac ) as $budget
    | [ (.model.display_name // "claude"),
        (.session_id // ""),
        (.transcript_path // ""),
        ( if $budget <= 0 then 0
          else ( $tokens / $budget * 100 | floor
                 | if . > 100 then 100 elif . < 0 then 0 else . end )
          end )
      ] | .[] | tostring
  ' 2>/dev/null
)

MODEL=${FIELDS[0]:-claude}
SESSION_ID=${FIELDS[1]:-}
TRANSCRIPT_PATH=${FIELDS[2]:-}
PCT=${FIELDS[3]:-0}
[[ "$PCT" =~ ^[0-9]+$ ]] || PCT=0
REMAINING=$(( 100 - PCT ))

# --- context bar -------------------------------------------------------------
FILLED=$(( PCT * BAR_WIDTH / 100 ))
(( FILLED > BAR_WIDTH )) && FILLED=$BAR_WIDTH
(( FILLED < 0 )) && FILLED=0
BAR="$(printf '%*s' "$FILLED" '' | tr ' ' '#')"
BAR+="$(printf '%*s' "$(( BAR_WIDTH - FILLED ))" '' | tr ' ' '.')"

# --- warning label -----------------------------------------------------------
if   [ "$REMAINING" -le 15 ]; then WARN=" COMPACT SOON"
elif [ "$REMAINING" -le 25 ]; then WARN=" !"
else                               WARN=""
fi

# --- auto-backup -------------------------------------------------------------
do_backup() {
  local repo_root backup_dir latest session_short recent timestamp lockfile

  repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
  backup_dir="$repo_root/claude_chat_backup"
  mkdir -p "$backup_dir" || return 0

  # Resolve the session's .jsonl
  latest=""
  if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
    # Best case: the payload gives it to us directly, no path-encoding guesswork
    latest="$TRANSCRIPT_PATH"
  elif [ -n "$SESSION_ID" ]; then
    # Fallback 1: find by UUID filename, independent of path encoding
    latest=$(find "$HOME/.claude/projects" -maxdepth 2 \
             -name "${SESSION_ID}.jsonl" 2>/dev/null | head -1)
  fi
  if [ -z "$latest" ]; then
    # Fallback 2: most recently modified transcript across all projects
    # `-exec ls -t {} +` rather than `| xargs -0 ls -t`: with no matches, GNU
    # xargs still runs `ls -t`, which lists the *current directory* and returns
    # an unrelated filename. -exec + simply does not run.
    latest=$(find "$HOME/.claude/projects" -maxdepth 2 -name '*.jsonl' \
             -exec ls -t {} + 2>/dev/null | head -1)
  fi
  [ -n "$latest" ] && [ -f "$latest" ] || return 0

  session_short="${SESSION_ID:0:8}"
  [ -n "$session_short" ] || session_short="nosess"

  # Per-session throttle: skip if this session was backed up recently.
  # -mmin, not -amin: access times are unreliable on relatime/noatime mounts.
  recent=$(find "$backup_dir" -name "auto_backup_*_${session_short}.jsonl" \
           -mmin "-${THROTTLE_MIN}" 2>/dev/null | head -1)
  [ -z "$recent" ] || return 0

  # Per-session lockfile so parallel agents don't block each other.
  # set -C makes the redirect fail if the file already exists.
  lockfile="${backup_dir}/.backup_${session_short}.lock"
  if ! (set -C; echo $$ > "$lockfile") 2>/dev/null; then
    return 0
  fi
  trap 'rm -f "$lockfile"' RETURN

  # The transcript is flushed asynchronously; give it a moment to catch up.
  sleep 0.5

  timestamp=$(date -u +%Y%m%d_%H%M%S)
  cp "$latest" "${backup_dir}/auto_backup_${timestamp}_${session_short}.jsonl"
}

if [ "$PCT" -ge "$BACKUP_AT" ]; then
  # Backgrounded and silenced: anything on stdout would become the status line,
  # and a large cp shouldn't stall the render.
  ( do_backup ) >/dev/null 2>&1 &
fi

# --- output (must be a single line) ------------------------------------------
echo "${MODEL} | ctx ${BAR} ${PCT}%${WARN}${BRANCH_PART:+ }${BRANCH_PART}"
