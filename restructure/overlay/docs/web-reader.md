# Web reader (showing docs without a push)

Read this when you want to show documents, research roadmaps or interactive reports to someone without pushing to
GitHub.

Moved verbatim from CLAUDE.md (2026-10 restructure):

- **Web presentation & interaction without GitHub push**: When presenting documents, research roadmaps, or
  interactive reports to the user without pushing to GitHub, run `./scripts/start_public_roadmap.sh` on the Mac mini.
  It launches the lightweight dynamic reader (`scripts/serve_research.py`) and Cloudflare Quick Tunnel (`trycloudflare.com`),
  instantly copying an open, read-only HTTPS link to the clipboard. All Markdown files in `research/`, `todos/`, `docs/`,
  and `tmp/` are scanned dynamically on every page load with zero build steps.

After the restructure `todos/` no longer exists; the reader should scan `experiments/` as well.

Last verified: 2026-10-01
