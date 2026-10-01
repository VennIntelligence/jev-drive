# Repo restructure by experiment topic (design)

Read this when you review or apply the restructure, or need the reasons behind the `experiments/` layout. After the
apply this file lives at `tools/restructure/design.md`, off the hot path.

## Criterion

Minimise what the next agent reads. One router (CLAUDE.md, read every session) points to four places only:
`experiments/INDEX.md`, `research/decisions.md`, `docs/lib.md`, `docs/closed-loop-runbook.md`. Everything else is
one hop below them. Budgets (tokens = bytes / 4): CLAUDE.md <= 1300, README.md <= 900, docs/README.md <= 500,
research/README.md <= 600, INDEX.md <= 2500 (<= 60 per topic line), decisions index <= 25 per entry, topic README soft
300 / hard 60 lines. Measured results at the end.

## Layout

`experiments/<topic>/{README.md, scripts/, lib/, archive/, results/, figs/, plans/}` (template:
`experiments/TEMPLATE.md`); `jevdrive/` shared library (incl. lib-core `run/ data/ cache par stats`); `scripts/` shared box
entry points (49 entries); `research/` Chinese narrative layer + `decisions.md` + `decisions/`; `docs/` how-to;
`tools/topic_index.py` keeps the README file lists and INDEX.md current. Topic dirs are snake_case (namespace packages:
`experiments.<topic>.lib.<module>`). A scripts/ subdir named after the topic is flattened; other subdirs move whole.

Shared vs topic comes from the import graph (absolute, relative and bare script-module imports), iterated to a fixed
point: imported by shared code or by >= 3 other topics -> shared; by 1-2 topics -> exported into the owner's `lib/`;
the rest of a concluded topic -> `archive/`, of a live topic -> `scripts/` / `lib/`.

| topic | status | scripts+lib | archive | results | figs | plans | deleted notes |
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
| **total** | | 81 | 558 | 2633 | 228 | 6 | 90 |

shared 180 (graph-promoted 70), exported 47, rows 1218

## Hot path documents

- **CLAUDE.md**: rules (language, workflow, secrets, no Artifact pages, code, shared-library rule, closed loop, results)
  and a five-row routing table. Detail moved out: Tokyo box -> docs/tokyo-box.md (already there), box size and the
  pre-run checklist -> docs/long-runs.md (appended verbatim), web presentation (the user's pending hunk) ->
  docs/web-reader.md (verbatim). Written as a whole-file overlay guarded by the blob of the committed CLAUDE.md
  (main + the user's pending hunk, blob 88e0f44); any other version stops the apply.
- **README.md**: seven places, one line each; every doc is reachable from it (verified by a link walk).
- **docs/README.md**: one line per how-to doc; the only link to `docs/path-map.tsv`.
- **research/README.md**: writing rules + the eight cross-cutting essays; topic essays are linked from their topic README.
- **experiments/INDEX.md**: one list item per topic: name (aliases for grep): status; key finding [d decision numbers].
- **research/decisions.md**: the 72 standing entries, one row each (number, claim <= 12 words, evidence 强/中/弱, status);
  17 withdrawn / superseded / closed entries in `research/decisions/ARCHIVE.md`; full text in `research/decisions/<NNN>.md`
  (numbers unchanged). Tiers and claims: `restructure/decisions_index.tsv` (curated from the headings).
- **Topic README**: status, decisions, index (the INDEX line), one-sentence question, two-sentence conclusion,
  next (live only), read more; generated list of the 8-12 key files with a one-clause docstring each; archive/,
  results/, figs/, plans/ as counts with a link.

## todos/ and tmp/

Concluded plans and tracked tmp notes are deleted (96 files; links become permalinks / `<sha>:<path>`); data packages move
whole into `<topic>/results/<slug>/`; live plans go to `<topic>/plans/`; tmp/ stays as gitignored scratch. Figures that
only deleted plans described are collected by their topic README's `figs/` link (user decision).

## The tool

`restructure/manifest.tsv` (1218 rows: old_path, new_path, topic, kind, status, note) drives `tools/restructure.py apply`:
validate (tracked only; untracked files inside moving dirs refuse), then rewrite every tracked text file except data and
vendored code: Markdown links and link text, repo-root path tokens (also `$repo/...`), globs, `jevdrive.x` /
`scripts.x` names, `from jevdrive import` and relative imports, bare-import sys.path blocks, `__file__` anchors and
literal joins on names bound to them (two-pass, ambiguity-checked), shell `$(dirname "$0")` anchors and `$here/x`.
Before that: overlays (`restructure/overlay.tsv`, blob-guarded whole files), appends (`restructure/append/`), front
summaries for the four longest how-to docs, and `manual_edits.tsv` (7 code / prose edits). Unresolvable references stop
the apply unless listed with a reason in `acknowledged.tsv`. Then `git mv` / `git rm`, path-map, READMEs + INDEX +
decision split, and the staging move: manifest, edits, acks, report, design doc and the one-off tools go to
`tools/restructure/`; overlays, drafts and summaries are removed. A second apply changes nothing.

`verify` / `compare` (moved tree vs unmoved tree): compile, static import resolution, sandboxed import and `--help`,
`test_*.py`, `bash -n`, Markdown links and paths, figures without a doc, docs unreachable from README.md, file
conservation (non-text files byte-identical).

## Apply sequence (Mac, repo root)

```
git add CLAUDE.md docs/remote-box.md && git commit -m "..."                 # the user's pending hunks first (overlay guard)
git add <files listed "commit first" in restructure/untracked.tsv> && git commit -m "..."
git merge --no-ff origin/restructure-draft
python3 tools/restructure_plan.py && git diff --stat restructure/manifest.tsv   # regenerate on current main, review
python3 tools/restructure.py check
.venv/bin/python tools/restructure.py verify --out /tmp/base.json
git push                                                                   # permalinks point at this commit
python3 tools/restructure.py apply --dry-run && python3 tools/restructure.py apply
.venv/bin/python tools/restructure/restructure.py verify --out /tmp/after.json
.venv/bin/python tools/restructure/restructure.py compare --baseline /tmp/base.json --after /tmp/after.json   # PASS
git commit -m "restructure: experiments/<topic> layout" && git push
ssh autodl 'cd ~/data/jev-drive && git status --short && git pull'
```

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

## Results

Dry run on a scratch copy of current main (250c6a5) + the user's pending CLAUDE.md / remote-box.md hunks:
`compare` **PASS**; a second apply changes nothing.

```
import: baseline 297/358 ok, after 297/358 ok; regressions 0, fixed 0, unchanged non-ok 61, new entries 0
help: baseline 191/340 ok, after 243/340 ok; regressions 0, fixed 54, unchanged non-ok 95, new entries 0
tests: baseline 30/47 ok, after 30/47 ok; regressions 0, fixed 0, unchanged non-ok 17, new entries 0
compile_fail: baseline 0, after 0, new []
bash_n_fail: baseline 0, after 0, new []
static_import_missing: baseline 3 files, after 1, new 0
md_broken: baseline 283, after 184, new 0
fig_unreferenced: baseline 0, after 0 (new: [])
md_unreachable from README.md: baseline 29, after 0, new 0
files: baseline 3863, after 3916, deleted on purpose 96 (todos/ 73, tmp/ 23), lost 0, binary changed 0
```

Pass 1 (first draft) on the same 15 lookup tasks and hop definitions: 34,037 tokens; pass 2 with the pass-1 hops: 19,549;
with grep hops into INDEX.md (aliases make that one line): 9,239. Benchmark (`tools/token_bench.py`):

| task | old steps | old tokens | new steps | new tokens | ratio |
|---|---|---|---|---|---|
| *lookup* | | | | | |
| op-adapt L: which script produced the result, what was the conclusion | 4 | 8,540 | 2 | 405 | 21.1x |
| default CARLA worker profile and its evidence | 2 | 7,815 | 2 | 806 | 9.7x |
| where is the nq3 Q1 table | 4 | 3,537 | 2 | 45 | 78.6x |
| which controller experiments exist, which are superseded | 3 | 5,511 | 1 | 1,322 | 4.2x |
| status and claim of decision 55 | 2 | 1,277 | 1 | 1,208 | 1.1x |
| list every decision still pending (待定) | 1 | 5,467 | 1 | 1,730 | 3.2x |
| which file implements the S_jev rule scorer, what is it | 2 | 312 | 2 | 89 | 3.5x |
| WL-2 verdict, results and figures | 4 | 7,355 | 3 | 429 | 17.1x |
| how to run the HUGSIM zero-shot exam | 3 | 7,426 | 3 | 750 | 9.9x |
| what did the zero-shot Bench2Drive exam conclude | 3 | 8,576 | 2 | 373 | 23.0x |
| which shared module reads WOD-E2E frames | 2 | 859 | 2 | 562 | 1.5x |
| night queue 4 G (ghost test): conclusion and code | 5 | 9,124 | 2 | 396 | 23.0x |
| NAVSIM skill pack N3 score and its run script | 3 | 3,458 | 3 | 663 | 5.2x |
| TFv6 controller campaign: conclusion and report | 3 | 6,349 | 2 | 373 | 17.0x |
| which experiments are live right now | 3 | 5,529 | 1 | 88 | 62.8x |
| **lookup: 15 tasks** | | **81,135** | | **9,239** | **8.8x** |
| *cold start (CLAUDE.md included)* | | | | | |
| what rules apply when I add an experiment | 4 | 10,071 | 2 | 1,134 | 8.9x |
| where do I put a new result and its figure | 2 | 2,958 | 2 | 1,134 | 2.6x |
| which experiments are live | 3 | 7,256 | 2 | 1,003 | 7.2x |
| what is the CARLA default worker profile | 2 | 6,399 | 2 | 942 | 6.8x |
| how do I start a long job on the box | 2 | 3,594 | 2 | 3,138 | 1.1x |
| which shared library gives run dirs and splits | 5 | 8,311 | 2 | 2,751 | 3.0x |
| open the op-adapt L experiment page | 3 | 11,994 | 3 | 1,320 | 9.1x |
| how do I log in to the GPU box | 2 | 2,982 | 2 | 2,004 | 1.5x |
| what are the writing rules for research notes | 2 | 2,958 | 2 | 1,441 | 2.1x |
| which experiments touched NAVSIM | 2 | 3,627 | 2 | 1,066 | 3.4x |
| **cold start (CLAUDE.md included): 10 tasks** | | **60,150** | | **15,933** | **3.8x** |
| **all 25 tasks** | | **141,285** | | **25,172** | **5.6x** |

CLAUDE.md, read every session: 1,898 -> 915 tokens. Directory listings: scripts/ 518 -> 49 entries, jevdrive/ 155 -> 72; research/decisions.md 121,852 -> 1,730 tokens.

| file | old tokens | new tokens |
|---|---|---|
| CLAUDE.md | 1,898 | 915 |
| README.md | 3,314 | 293 |
| docs/README.md | 746 | 525 |
| research/README.md | 1,060 | 526 |
| experiments/INDEX.md | 0 | 1,322 |
| research/decisions.md | 121,852 | 1,730 |
| experiments/TEMPLATE.md | 0 | 218 |
| experiments/*/README.md (40) | - | avg 318, max 377 |

Cold start to any topic README (CLAUDE.md + one INDEX.md grep line <= 39 tok + README): <= 1,331 tok; reading the whole INDEX.md instead: <= 2,614 tok.

Budgets not met, and why: topic READMEs average 318 tok (soft cap 300; 22 of 40 above, max 377) because the 8-12
key files with a clause each cost ~150 tok on top of a ~160 tok header; cutting to 6 files would meet it but drop
entry points agents open. Reading the whole INDEX.md on a cold start costs 2,614 tok (target 2.5k); grepping it costs
1,331. The 25-task total is 25,172 (target ~25k): the cold-start tasks are dominated by the how-to docs themselves
(long-runs.md 2.2k, lib.md 1.8k), not by the router.

Last verified: 2026-10-02
