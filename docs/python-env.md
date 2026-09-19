# Python env

Read this when you need to run project code on the box or add a dependency.

- Deps live in `pyproject.toml`, pinned in the committed `uv.lock`. Python 3.11. Packages come from the Aliyun PyPI mirror.
- torch 2.14.0 from that mirror is the cu130 build and already supports Blackwell (sm_120); no extra index needed.
- The venv lives on the data disk, not in the repo. Every `uv` command on the box needs:
  `export UV_PROJECT_ENVIRONMENT=$DATA_DIR/envs/jevdrive` (fish: `set -gx UV_PROJECT_ENVIRONMENT $DATA_DIR/envs/jevdrive`).
- Create or update it: `cd ~/data/jev-drive && uv sync`. Run code with `uv run python -m jevdrive....`.
- Add a dependency locally with `uv add <pkg>`, commit `pyproject.toml` and `uv.lock`, then pull and `uv sync` on the box.
- Installing Python 3.11 from GitHub hangs on the box (direct and via turbo). Use the npmmirror copy:
  `UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone uv python install 3.11`.

Last verified: 2026-09-20
