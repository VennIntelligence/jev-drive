# Repository web reader

Run `./scripts/start_public_roadmap.sh` on the Mac mini. The script starts the Python reader on
`127.0.0.1:6006` and a Cloudflare Quick Tunnel. It prints a public HTTPS URL and copies it to the clipboard.
No build or GitHub push is required. The home page shows the repository root.

After changing reader code, run `./scripts/start_public_roadmap.sh --restart`. This restarts the reader
and reuses the running tunnel. The public URL stays the same while that tunnel is running. A new tunnel
can receive a different URL. Both processes must keep running. The script installs the reader as a macOS
LaunchAgent (`com.jevdrive.repository-reader.6006`). `launchd` keeps it running and starts it at login.
An existing tunnel is reused to preserve its URL. If no tunnel exists, the script installs a tunnel
LaunchAgent as well. A reused legacy tunnel is not automatically migrated or restarted. The Mac must stay
awake and connected to the network. Logs are in `/tmp/serve_research.log` and `/tmp/cf_tunnel.log`.

The reader browses repository folders, including `experiments/`, `scripts/`, `jevdrive/`, `research/`,
`docs/` and `tmp/`. The directory tree loads each folder when expanded. Search matches filenames and paths
across the repository, with a limit of 200 results. Enter a local absolute path or a relative path in the
same search box and press Enter to open it. Relative paths are tried from the repository root first,
then from the current folder. `./` and `../` are supported. Paths outside the repository are blocked.

Markdown renders with tables, code highlighting, KaTeX and Mermaid. Text and code files show escaped source
with syntax highlighting. HTML files have a preview and a source view. HTML previews run inside an isolated
iframe with user-activated top navigation and popups enabled (`allow-scripts allow-top-navigation-by-user-activation allow-popups allow-forms`).
Links clicked inside an HTML preview navigate the top window to the target repository file via the native reader
(stripping any `_preview/` prefix), preserving browser history so the browser Back button returns to the preview.
In-page anchor links scroll within the preview, and external links open in a new tab. Direct requests to `_preview/`
for repository documents or directories automatically redirect (HTTP 302) to their clean reader routes.
Relative CSS, JavaScript, media and data files load through the same access checks. Images and PDFs have previews.
Other files can be downloaded. Text previews are limited to 2 MB. The UI uses a fixed white theme, with
adjustable width and font size.

One access policy protects directory listings, search, path jumps, previews, raw files, downloads and HEAD
requests. It excludes hidden files/directories (including `.git` and `.env`), common credential filenames
and key formats, `config.yaml`, `config.yml`, `clash/`, runtime environments and caches. Symbolic links are
blocked, including links within the repository. File contents are checked for private keys, common token
formats, credential URLs and literal credential assignments. Files that match are omitted and return 404.
Environment-variable references and common placeholders remain readable. Checks are repeated when file
metadata changes. Pattern matching cannot identify every possible secret; keep credentials outside the
public repository or in the excluded paths.

Run `python3 scripts/test_research_reader.py` to verify access checks and preview routes.

Last verified: 2026-10-02
