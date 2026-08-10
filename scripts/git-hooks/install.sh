#!/usr/bin/env bash
# .git/hooks/ isn't tracked by git, so this hook has to be (re)installed
# manually after every fresh clone. Run this once: bash scripts/git-hooks/install.sh
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
cp "$repo_root/scripts/git-hooks/pre-commit" "$repo_root/.git/hooks/pre-commit"
chmod +x "$repo_root/.git/hooks/pre-commit"
echo "Installed pre-commit hook -> .git/hooks/pre-commit"
