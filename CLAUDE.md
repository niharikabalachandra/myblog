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

## Git commit policy

- Only commit when explicitly asked to. Don't commit automatically after making edits.
- Create new commits rather than amending, unless amending is explicitly requested.
- Stage specific files by name (not `git add -A`/`.`) so unrelated or generated files aren't swept in accidentally.
- Never force-push, reset --hard, or rewrite history on `main` without explicit confirmation.
- Never skip hooks (`--no-verify`) unless explicitly asked.
- Keep commit messages short and focused on *why* (e.g. new post, config/theme change, dependency bump), not a restatement of the diff.
- Don't push to the remote unless explicitly asked to.
