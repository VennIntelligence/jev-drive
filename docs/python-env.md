# Python env

Read this when you need to run project code on the box or add a dependency.

- Deps live in `pyproject.toml`, pinned in the committed `uv.lock`. Python 3.11. Packages come from the Aliyun PyPI mirror.
- The venv lives on the data disk, not in the repo. Every `uv` command on the box needs:
  `export UV_PROJECT_ENVIRONMENT=$DATA_DIR/envs/jevdrive` (fish: `set -gx UV_PROJECT_ENVIRONMENT $DATA_DIR/envs/jevdrive`).
- Create or update it: `cd ~/data/jev-drive && uv sync`. Run code with `uv run python -m jevdrive....`.
- Add a dependency locally with `uv add <pkg>`, commit `pyproject.toml` and `uv.lock`, then pull and `uv sync` on the box.
- If uv cannot download Python 3.11 directly, `source /etc/network_turbo` for that one command.

Last verified: 2026-09-19
