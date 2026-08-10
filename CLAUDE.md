# myblog

Niharika's personal blog, built with [Quarto](https://quarto.org/) (`project: type: website`).

## Structure

- `_quarto.yml` — main site config (nav, footer, theme, analytics)
- `quarto.yml` — secondary/legacy site metadata (title, description) — check both when changing site-wide settings
- `posts/` — blog posts, one subdirectory per post (e.g. `posts/welcome/`, `posts/image_recognition/`)
- `posts/_metadata.yml` — shared post defaults (author, freeze, comments via utterances, footer include)
- `index.qmd`, `about.qmd` — top-level pages
- `footer.html` — shared footer injected into posts
- `styles.css` — site-wide custom CSS
- `docs/` — rendered site output (publish target, `output-dir: docs`)
- `_site/` — local Quarto preview/build output
- `_freeze/`, `.quarto/` — Quarto caches, not hand-edited

## Working with this repo

- Render the site with `quarto render`; preview locally with `quarto preview`.
- New posts go in their own folder under `posts/` with an `index.qmd`; front matter inherits defaults from `posts/_metadata.yml` unless overridden.
- `docs/` is generated — don't hand-edit files there, re-render instead.
- Comments use utterances against the `niharikabalachandra/blog_comments` GitHub repo.

## Change management

- `main` is the only branch in normal use; work directly on it unless a change is large or experimental enough to warrant a feature branch.
- `docs/` is a build artifact (the Quarto `output-dir`). Re-render (`quarto render`) to update it rather than editing generated files by hand, and commit the re-rendered `docs/` output alongside the source change that caused it so the published site stays in sync.
- `_site/`, `.quarto/`, `_freeze/` are local/caching directories — don't commit stray changes from these unless they're the intended freeze update for a post.

## Sensitive data

- `claude_chat_backup/` (repo root) holds raw Claude Code session transcripts — everything ever pasted into that terminal. It's gitignored and denied to Claude's own Read/Grep/Glob via `.claude/settings.json`. Never remove either protection without deliberately deciding to.
- `recall/` (the retrieval system built over that archive) has its own `.gitignore` excluding `*.db`, `okf/`, `.venv/`, `synthetic_corpus/`, `bench_scratch/`, `__pycache__/` — none of that is real to commit, only the source (`recall.py`, `okf_build.py`, `query.py`, `bench.py`, `_transcript.py`, `gen_synthetic.py`, `requirements.txt`).
- A pre-commit hook backstops both of the above regardless of `.gitignore` state (catches even `git add -f`): `scripts/git-hooks/pre-commit`. It's **not** tracked by git itself (`.git/hooks/` never is), so after a fresh clone run `bash scripts/git-hooks/install.sh` once to reinstall it. If a commit is ever blocked and you're certain it's safe, don't reach for `--no-verify` — fix the blocklist in `scripts/git-hooks/pre-commit` instead so the exception is deliberate and visible in history.
- This repo is public. Verify with `git log --all --diff-filter=A --name-only | grep -E '<pattern>'` before assuming something was never committed, don't just trust the current `.gitignore`.

## Git commit policy

- Only commit when explicitly asked to. Don't commit automatically after making edits.
- Create new commits rather than amending, unless amending is explicitly requested.
- Stage specific files by name (not `git add -A`/`.`) so unrelated or generated files aren't swept in accidentally.
- Never force-push, reset --hard, or rewrite history on `main` without explicit confirmation.
- Never skip hooks (`--no-verify`) unless explicitly asked.
- Keep commit messages short and focused on *why* (e.g. new post, config/theme change, dependency bump), not a restatement of the diff.
- Don't push to the remote unless explicitly asked to.
