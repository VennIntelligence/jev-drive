# Repository web reader

The reader is no longer part of this repo. It is machine-wide infrastructure at `~/mycode/webdesk`
(see its `README.md`): one password-protected site, `https://desk.gaochengzhi.com`, that serves a read-only,
secret-filtering file browser under `/read/` and the Claude Code web UI everywhere else.

View jev-drive docs at `https://desk.gaochengzhi.com/read/r/jev-drive/<path>`, for example
`/read/r/jev-drive/docs/storage.md` or `/read/r/jev-drive/research/decisions.md`. The reader home lists recent
workspaces; the older `/read/r/mycode/jev-drive/<path>` links keep working. No push is needed: the reader shows the
working tree. Roots, restart, and troubleshooting are in the webdesk README.

Last verified: 2026-10-07
