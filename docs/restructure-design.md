# Repo restructure by experiment topic (design)

Read this when you review or apply the restructure, add an experiment topic, or need the layout rules behind
`experiments/`.

Status: draft on branch `restructure-draft`, planned against `bcbdde4`. Nothing is applied to `main` until the
manifest is reviewed. Tooling: `tools/restructure_plan.py` (rules -> manifest), `tools/restructure.py`
(check / apply / verify / compare), `tools/restructure_docs.py` (READMEs, INDEX), `tools/split_decisions.py`,
`tools/token_bench.py`. Reviewed artifact: `restructure/manifest.tsv` (+ `manual_edits.tsv`, `acknowledged.tsv`).

## Goal and design criterion

Minimise the tokens the next agent spends to find something. Every choice below is a retrieval-cost choice:

1. **Three reads to anything.** CLAUDE.md (read every session, kept small) carries a task -> location routing table.
   Below it: `experiments/INDEX.md` (one line per topic, with aliases for grep) -> `experiments/<topic>/README.md`
   (question, status, conclusion with decision numbers, file list with one-line descriptions) -> the file.
2. **No big blobs on the hot path.** `research/decisions.md` (4228 lines, ~120k tokens) becomes a one-read index
   (~3k tokens, one line per entry: number, status, topic, truncated claim) plus `research/decisions/<NNN>.md`, one
   file per entry, copied verbatim. One fact, one home: READMEs cite entry numbers instead of restating the log.
3. **Greppable names and descriptions.** Directory names are topics; the README file index lifts every file's first
   docstring line, so `grep <word> experiments/*/README.md` locates code without opening it. Status words are fixed
   (below), so `grep "status: live" experiments/*/README.md` lists open work.
4. **Cheap listings.** `scripts/` 517 -> 48 entries, `jevdrive/` 150 -> ~70 (incl. its subpackages), research/ 39.
   Exceptions, never listed by an agent in normal work: `research/decisions/` (89 entry files, addressed by number)
   and the inside of old data packages under `experiments/*/results/` (kept byte-identical).
5. **Archived code is separated.** One-off code of a concluded experiment sits in `<topic>/archive/`; it is reachable
   from its README but not mixed with live entry points.

Measured effect: see "Token benchmark" at the end.

## Layout

```
CLAUDE.md                  rules + routing table (task -> where to look)
README.md                  top-level doc index
experiments/INDEX.md       one line per topic: status, decision entries, headline, aliases
experiments/<topic>/
  README.md                <= ~60 lines: question, status, conclusion pointer, file index (generated part)
  scripts/                 entry points of a live topic
  lib/                     modules another topic (one or two) imports, and all modules of a live topic
  archive/                 one-off code of a concluded topic: reproduce from it, do not extend it
  results/                 small result files (from research/results/<x>/ and todos/<data package>/)
  figs/                    the topic's figures (from research/figs/)
  plans/                   live lanes only: the Chinese plan notes, until the lane closes
jevdrive/                  shared Python library (+ jevdrive/cl, openpilot, alpamayo; next round: data/, run/, ...)
scripts/                   shared box entry points: tmux, downloads, CARLA server, Bench2Drive harness, policy servers
research/                  narrative layer (Chinese): decisions index + decisions/, topic essays, articles, roadmap, lit
docs/                      how-to docs (English); docs/path-map.tsv maps every old path to its new one
tests/                     shared tests (jevdrive.cl); topic tests sit next to the code they test
tools/                     repo tooling (restructure, README/INDEX generator, token benchmark)
```

Directory names are snake_case so `experiments.<topic>.lib.<module>` imports work (namespace packages, no
`__init__.py` needed). A scripts/ subdir named after the topic (scripts/p3, scripts/hugsim, scripts/bench_baselines,
scripts/lanes) is flattened into the topic; other subdirs (nq3_d, top10_t2, top10_smoke, sch_gpu_helpers,
zeroshot_b2d_alp_stall_probes) are kept as subdirs, and a subdir with an exported member moves to lib/ as a whole.

## Shared vs topic: decided by the import graph

`tools/restructure_plan.py` builds the Python import graph (absolute `jevdrive.x`, relative `from .x` inside jevdrive,
bare script-module imports resolved like `python file.py` resolves them) and iterates to a fixed point:

- seed: infrastructure that is shared by nature (data paths, run dirs, openpilot/Alpamayo runners, `jevdrive/cl`,
  the Bench2Drive harness `b2d_run`/`b2d_route`/`b2d_hooks`/agents/policy servers, box ops, downloads);
- a module that shared code imports becomes shared (shared code never depends on a topic);
- a module imported by **three or more** other topics becomes shared (stays in `jevdrive/` or `scripts/`);
- a module imported by **one or two** other topics is **exported**: it stays with its owner topic but in `lib/`;
- everything else of a concluded topic goes to `archive/`, of a live topic to `scripts/` or `lib/`.

Result: 133 shared files (70 of them promoted by the graph, e.g. `wod_zeroshot`, `navsim_zs`, `p5_openpilot`,
`drive_backbones`, `p4_carla`, `b2d_controller`), 47 exported modules, 558 archived and 81 live topic files.

## Topics

| topic | status | code in scripts/+lib/ | code in archive/ | result files | figures | plans kept | todos/tmp files deleted |
|---|---|---|---|---|---|---|---|
| op_adapt_l | live | 21 | 0 | 53 | 7 | 5 | 1 |
| b2d_privileged | live | 6 | 0 | 40 | 0 | 1 | 2 |
| cl_infra | concluded | 0 | 23 | 38 | 3 | 0 | 3 |
| b2d_controller | concluded | 2 | 36 | 709 | 2 | 0 | 1 |
| b2d_controller_eval | concluded | 0 | 23 | 0 | 0 | 0 | 0 |
| b2d_tcp | concluded | 1 | 7 | 282 | 0 | 0 | 0 |
| b2d_tfv6 | concluded | 2 | 40 | 200 | 0 | 0 | 0 |
| tfv6_rules | concluded | 1 | 4 | 4 | 0 | 0 | 1 |
| simlingo_catalogue | concluded | 0 | 4 | 0 | 0 | 0 | 1 |
| zeroshot_b2d | concluded | 1 | 24 | 16 | 5 | 0 | 0 |
| zeroshot_openloop | concluded | 4 | 15 | 21 | 13 | 0 | 7 |
| model_smoke | concluded | 2 | 10 | 13 | 9 | 0 | 1 |
| hugsim | concluded | 0 | 31 | 18 | 7 | 0 | 0 |
| leaderboard_audit | concluded | 0 | 1 | 123 | 6 | 0 | 3 |
| probe_planner_v0 | concluded | 0 | 8 | 0 | 3 | 0 | 4 |
| prediag | concluded | 0 | 6 | 130 | 12 | 0 | 6 |
| driving_backbones | concluded | 1 | 3 | 48 | 1 | 0 | 1 |
| reactivity | concluded | 1 | 14 | 66 | 8 | 0 | 10 |
| fusion_diag | concluded | 0 | 11 | 110 | 2 | 0 | 1 |
| fastperc | concluded | 0 | 3 | 2 | 1 | 0 | 1 |
| elicitation | concluded | 2 | 8 | 76 | 5 | 0 | 2 |
| real_transfer | concluded | 0 | 7 | 82 | 5 | 0 | 1 |
| night_queue_2 | concluded | 2 | 16 | 45 | 6 | 0 | 1 |
| night_queue_3 | concluded | 6 | 44 | 39 | 1 | 0 | 2 |
| p3_ped_exam | concluded | 0 | 24 | 1 | 57 | 0 | 5 |
| world_model | concluded | 3 | 26 | 48 | 5 | 0 | 9 |
| night_queue_4 | concluded | 1 | 37 | 169 | 0 | 0 | 6 |
| carla_rewind | concluded | 1 | 5 | 4 | 0 | 0 | 2 |
| top10 | concluded | 9 | 19 | 60 | 3 | 0 | 1 |
| statepol | concluded | 0 | 4 | 0 | 0 | 0 | 1 |
| cosmos | concluded | 4 | 12 | 47 | 13 | 0 | 2 |
| feature_adapter | concluded | 1 | 1 | 12 | 0 | 0 | 2 |
| controlnet_pair | concluded | 0 | 4 | 1 | 46 | 0 | 1 |
| op_adapt_r1 | concluded | 2 | 6 | 21 | 0 | 0 | 1 |
| op_adapt_r2 | superseded-by op_adapt_l | 4 | 23 | 37 | 0 | 0 | 2 |
| op_closed_loop | concluded | 2 | 15 | 40 | 2 | 0 | 2 |
| op_openloop | concluded | 1 | 13 | 48 | 5 | 0 | 2 |
| skill_pack | concluded | 0 | 19 | 27 | 0 | 0 | 4 |
| log_expert_audit | concluded | 1 | 1 | 3 | 1 | 0 | 1 |
| baselines_latency | concluded | 0 | 11 | 0 | 0 | 0 | 0 |
| **total (40 topics)** | | 81 | 558 | 2633 | 228 | 6 | 90 |

`b2d_*` is split into harness (shared), controller, controller_eval (Task 10), tcp, tfv6, tfv6_rules, privileged and
zeroshot_b2d; `op_*` into op_adapt_r1 / r2 / l, op_closed_loop (op-arb, op-drive), op_openloop (op-interp, op-lb,
standing, navhard), feature_adapter (E0/E1 layer probes). Classification is by reading docstrings, plans and the
decision log, not by prefix alone (e.g. `wod_openpilot_rigs.py` is the rig study, `model_smoke`; `navsim_heads` is
skill_pack; `p6.py` is night queue 2 N1).

## Naming rules

- Topic dir: snake_case noun phrase of the experiment, <= 3 words; the decision log's alias goes into INDEX.md, not
  into the dir name (`night_queue_3 (nq3, lanes A-D, Q1-Q6)`).
- Files keep their names (history and muscle memory); the topic dir carries the topic. Module names do not change,
  only their package path, so importers change mechanically.
- Results subdir = the old `research/results/<x>` name (or the todo slug without date); a topic with one results
  source has it flattened into `results/`.

## Status vocabulary

Topic READMEs and INDEX.md use exactly: `live` (work continues; code in scripts/ + lib/), `concluded` (result in the
decision log; code in archive/), `superseded-by <topic>` (a later topic replaced the method; say why in one clause).
Manifest rows use `live | one-off | superseded` per file (one-off = code of a concluded experiment, kept, archived).

## README template (<= 60 lines; the file index is generated)

```
# <topic>: <short title>

status: live | concluded | superseded-by <topic>
decisions: 78, 79, 80
headline: <one line, <= 120 chars, the key number and what it means>

**Question.** <1-2 sentences>

**Conclusion.** <2-4 sentences with the numbers as the decision entries state them, "(decisions 78)">

**Next.** <live topics only>

**Read more.** <research docs, docs/, `git show bcbdde4:todos/<plan>.md` for removed plans, plans/ for live ones>

<!-- files:begin -->   (generated by tools/restructure_docs.py: scripts/, lib/, archive/ with docstring lines;
<!-- files:end -->       results/, figs/, plans/ summaries; figures whose plan note was removed are linked here)
```

## todos/ and tmp/

- todos/ plan files of concluded topics and all tracked tmp/ notes are deleted (`git rm`; 96 files). Their essentials
  are the README's question / conclusion / decision numbers; the full text stays readable with
  `git show bcbdde4:<path>`, and every link to them is rewritten to a GitHub permalink at `bcbdde4` (inline paths
  become `bcbdde4:<path>`).
- todos/ data packages (dirs with results, figures, agent reports) move whole into `experiments/<topic>/results/<slug>/`,
  byte-identical except Markdown path rewrites.
- Plans of the two live topics move to `experiments/<topic>/plans/` (Chinese working notes) until the lane closes.
- tmp/ itself stays as gitignored scratch space (`.gitignore` already lists it); paths into it are left alone.
- No `prompts/` folder. 63 figures were described only by deleted plans; the README file index links them.

## Decision log split

`tools/split_decisions.py`: `research/decisions.md` keeps its path (all links stay valid) and becomes the index
(maintenance rules + one row per entry: number, status word from the heading, topics from the READMEs' `decisions:`
lines, claim truncated to 50 characters). Each entry goes verbatim to `research/decisions/<NNN>.md` (headings promoted
one level, relative links re-based). The two early duplicate numbers 8/9/10 keep both versions (`008.md`, `008-2.md`).
Prose references ("decisions 55", "第 55 条") stay valid because numbers do not change.

## Routing table carried by CLAUDE.md

| Task | Look in |
|---|---|
| What did we conclude, is it still open | `research/decisions.md` -> `research/decisions/<N>.md` |
| One experiment: code, results, figures, status | `experiments/INDEX.md` -> `experiments/<topic>/README.md` |
| Shared Python code | `jevdrive/` |
| Shared box entry points | `scripts/`, `docs/README.md` |
| A CARLA / Bench2Drive run | `docs/closed-loop-runbook.md`, `jevdrive/cl/` |
| Narrative, literature, roadmap (Chinese) | `research/README.md` |
| An old path from a note or a commit | `docs/path-map.tsv`, then `git show bcbdde4:<old path>` |

## experiments/INDEX.md format

Grouped by area (live first); one table row per topic:
`| [<topic>](<topic>/README.md) (<aliases>) | <status> | <decision numbers> | <headline> |`. Generated from the READMEs
by `tools/restructure_docs.py`; `--check` exits 1 when an index or README file list is stale.

## The tool

`restructure/manifest.tsv` columns: `old_path, new_path, topic, kind, status, note`. kind is
`shared | topic | archive | data | doc | delete`; a row whose old_path ends in `/` covers every tracked file below it.
Notes carry tags: `exported: imported by ...`, `shared by import graph: ...`, `vendored third-party`, `unsafe-proc`.

`tools/restructure.py apply` (requires a clean tracked tree; `--dry-run` writes only the report):

1. Validate the manifest: every row matches tracked files, no file listed twice, no two files land on one path, no
   untracked file left inside a directory that moves (untracked files are refused; ignored files next to a moved file
   or inside a moved dir travel with it, listed in `report/companions.tsv`).
2. Rewrite, deterministically, in every tracked text file except data and vendored code (data Markdown is rewritten):
   - Markdown links (incl. link text, `<img src>`, reference definitions), recomputed relative to the doc's new place;
     links to removed files become permalinks at `bcbdde4`; already-broken links keep their (missing) target;
   - repo-root path tokens in any text (backticks, prose, string literals, shell, JSON, tmux/systemd snippets),
     including the literal prefix of a glob (`research/results/nq3/q1/*.csv`);
   - `jevdrive.<mod>` and `scripts.<mod>` dotted names (imports, `python -m`, importlib strings, prose);
   - `from jevdrive import a as A, b` split per destination package; relative imports inside the old jevdrive package
     made absolute when the importer or the target moved;
   - bare imports of script modules: a two-line `sys.path` block naming the new dirs of exactly the modules imported;
   - `Path(__file__)...parent(s)` / `os.path.dirname(...__file__)` anchors and literal joins after them or after names
     bound to them (`REPO / "research" / "results" / "x"`), re-expressed from the new location; a name bound to the
     file's own dir is re-bound to its new dir when everything it reaches moved with it;
   - shell `$(dirname "$0")/..` anchors, variables bound to them (`$here/x.py`) and `$repo/scripts/x` style paths;
   - `restructure/manual_edits.tsv` (22 exact find/replace edits: CLAUDE.md, README.md, research/README.md,
     research/midterm-inventory.md and four code files that index `research/results/<x>` at run time or bind a name
     twice) is applied first; `restructure/summaries/<path>` inserts a front summary into the four longest how-to docs.
   Anything it cannot resolve (a path into a directory that split, a non-literal join on a moved base, an ambiguous
   module name) is reported in `report/unresolved.tsv` and **stops the apply** unless listed with a reason in
   `restructure/acknowledged.tsv` (16 entries, all prose or one-off code that is not rerun).
3. `git mv` every move, `git rm` every deletion, write rewritten files, write `docs/path-map.tsv`, generate READMEs,
   INDEX.md and the decision split, stage everything. A second run finds nothing to do.

`verify` (run on the moved tree and on an unmoved copy) and `compare` report pass/fail per check: in-memory compile of
every .py; static resolution of every repo import against the sys.path each file sets; import of every import-safe
module and `--help` of every argparse script, each in a sandbox (`sandbox-exec`: no network, no writes under /Users,
DATA_DIR unset, no GPU), 60 s / 30 s timeouts; `bash -n` on every .sh; every `test_*.py` (180 s); Markdown links and
repo-path tokens across the repo; figures without a referencing doc; and conservation (every tracked file present at
its manifest destination; non-text files byte-identical; deletions limited to the listed todos/ and tmp/ files).

## Shared-code candidates (next library round)

From the same import graph and a scan for functions re-implemented across topics. The parallel `lib-core` branch adds
`jevdrive/run/`, `jevdrive/data/`, `jevdrive/cache`, `jevdrive/par`, `jevdrive/stats`; these paths stay free here.

| what | today | used / re-implemented in | proposed home |
|---|---|---|---|
| run dir + logging | `jevdrive.runlog.RunLog`, `jevdrive.common` | 17 topics import them | `jevdrive/run/` (lib-core) |
| openpilot model + frames | `jevdrive.openpilot.model.OPModel/decode`, `frames` | 14 + 5 topics; 27 files build ONNX sessions / OPModel | stays `jevdrive/openpilot/` |
| parallel map / pools | `bounded_map` in `scripts/drive_backbones_openpilot.py`; ad-hoc pools | 5 topics import `bounded_map`; 86 files open thread/process pools | `jevdrive/par` |
| bootstrap / paired CIs, AUC, logreg | `boot`, `boot_mean`, `paired` (20 copies), `p4_carla.auc/logreg` | 7+ topics each | `jevdrive/stats` |
| WOD-E2E frames, rater frames | `jevdrive/waymo.py`, `wod_zeroshot` helpers | 65 files touch WOD-E2E paths | `jevdrive/data/wod/` |
| NAVSIM / OpenScene tokens, PDM scoring | `jevdrive/navsim_zs.py`, per-topic score scripts | 15 files read OpenScene, 19 call PDM scoring | `jevdrive/data/navsim/` |
| nuScenes index | `jevdrive/nuscenes_index.py`, `nuscenes_zs.py` | 4 topics | `jevdrive/data/nuscenes/` |
| model download with proxy / mirror fallback | `jevdrive/hfdl.py`, `openpilot/dl.py`, shell `proxy_on` | 13 files hand-roll proxy / mirror logic | `jevdrive/hfdl.py` (extend) |
| paper figure save | `jevdrive/plots.save`, `research/plot_style.py` | 6 topics import plots; 18 files call savefig directly | one style module (merge the two) |
| B2D controller + adapter | `scripts/b2d_controller.py`, `b2d_controller_adapter.py` | 7 + 4 topics | `jevdrive/cl/` or a `drive_runtime` package (py3.8) |
| CARLA agent boilerplate | `get_entry_point`, `root`, `project` re-defined | 19 / 30 / 13 files | one agent base in the harness |

## Open items for the reviewer

See the restructure report: live-plan handling, figures whose captions lived in deleted plans, untracked files on the
Mac (research/articles, research/roadmap, docs_site, mkdocs.yml, scripts/serve_research.py, ...), and the exact apply
sequence.

## Token benchmark

`python tools/token_bench.py --old <unmoved> --new <moved>`; 15 lookup tasks, minimal path in each tree, tokens =
bytes / 4 of what the agent reads (files, line ranges after a grep, listings, grep output).

| task | old steps | old tokens | new steps | new tokens | ratio |
|---|---|---|---|---|---|
| op-adapt L: which script produced the result, what was the conclusion | 4 | 8,537 | 2 | 3,783 | 2.3x |
| default CARLA worker profile and its evidence | 2 | 7,815 | 1 | 4,552 | 1.7x |
| where is the nq3 Q1 table | 4 | 3,537 | 2 | 66 | 53.6x |
| which controller experiments exist, which are superseded | 3 | 5,511 | 1 | 2,201 | 2.5x |
| status and claim of decision 55 | 2 | 1,277 | 1 | 1,208 | 1.1x |
| list every decision still pending (待定) | 1 | 5,467 | 1 | 3,149 | 1.7x |
| which file implements the S_jev rule scorer, what is it | 2 | 312 | 2 | 222 | 1.4x |
| WL-2 verdict, results and figures | 4 | 7,355 | 3 | 3,607 | 2.0x |
| how to run the HUGSIM zero-shot exam | 3 | 7,423 | 3 | 2,097 | 3.5x |
| what did the zero-shot Bench2Drive exam conclude | 3 | 8,576 | 2 | 3,873 | 2.2x |
| which shared module reads WOD-E2E frames | 2 | 850 | 2 | 553 | 1.5x |
| night queue 4 G (ghost test): conclusion and code | 5 | 9,113 | 2 | 3,791 | 2.4x |
| NAVSIM skill pack N3 score and its run script | 3 | 3,455 | 3 | 726 | 4.8x |
| TFv6 controller campaign: conclusion and report | 3 | 6,349 | 2 | 4,019 | 1.6x |
| which experiments are live right now | 3 | 5,529 | 1 | 190 | 29.1x |
| **total (15 tasks)** | | **81,106** | | **34,037** | **2.4x** |

CLAUDE.md, read every session: 1,760 -> 2,151 tokens. Directory listings: scripts/ 517 -> 48 entries, jevdrive/ 150 -> 67; research/decisions.md 121,852 -> 3,149 tokens.

The tasks that gain least (decision 55: 1.1x; the S_jev lookup: 1.4x) were already one grep in the old tree; every
task is cheaper, none got more expensive. The gains come from three design features, each of which an earlier draft
lacked and the benchmark exposed: topic aliases in INDEX.md (a grep for `nq3` hits one line), the per-entry decision
files (read `decisions/055.md` directly instead of grep + line range), and file descriptions in the README index (a
grep over READMEs locates code without opening it). CLAUDE.md grows by ~390 tokens per session for the routing table.

## Dry-run result (scratch copy of bcbdde4)

`compare` of `verify` on the moved tree against the unmoved tree: **PASS**.

| check | unmoved | moved | regressions |
|---|---|---|---|
| compile (all .py outside data) | 0 fail | 0 fail | 0 |
| static import resolution | 4 files with an unresolved repo import | 2 | 0 new |
| import of import-safe modules (351) | 290 ok | 290 ok | 0 |
| `--help` of argparse scripts (339) | 190 ok | 242 ok | 0 (54 fixed: relative imports made absolute, paths injected) |
| `test_*.py` (46) | 29 ok | 29 ok | 0 (the 17 non-ok fail identically on the Mac: Linux-only APIs, box paths, float goldens) |
| `bash -n` (all .sh) | 0 fail | 0 fail | 0 |
| broken Markdown links / repo-path tokens | 282 | 192 | 0 new (2 acknowledged: links into the Mac's untracked research/articles/) |
| figures with no referencing doc | 0 | 0 | 0 (63 figures described only by deleted plans are linked from their README) |
| tracked files | 3815 | 3868 | 0 lost; 96 deleted on purpose (todos/ 73, tmp/ 23); non-text files byte-identical |

A second `apply` on the applied tree finds nothing to move and nothing to rewrite (idempotent). Rewrites: 4557 in 848
files (repo paths 2024, Markdown links 742, imports 577 + module names 388, `__file__` anchors 295 + chains 119,
bare-import path blocks 198, shell anchors 134 + chains 30, globs 24, manual 22, summaries 4). Every rewritten
`__file__` expression and path chain (413) was evaluated statically: 409 point at an existing path, the other 4 at
nq3 q2/q3 result dirs that are still untracked on the Mac (see restructure/untracked.tsv). Frozen code inside data
packages (e.g. `results/controller-next/lateral_v2/run_campaign.sh`) is deliberately not rewritten.

Last verified: 2026-10-01
