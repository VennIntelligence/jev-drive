# PAI-PCS-LOCAL pre-registration (2026-10-19, written before any garage / go-straight PAI read)

Goal: a local PAI PCS ruler (the PAI analogue of `pcs_local.py`, decision 382) that predicts private PAI-board PCS from PAI 281 reads.

## Reads

- Same 8-chunk table as every stored PAI 281 read (`CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks`, hash 28800209).
- **garage 0014** (`DRV=garage`, `GARAGE_RIG=sensor_rig_2.yaml`) and **go-straight** (`DRV=starter`, AlpaSim's starter kit as it ships).
  Both: `ALPASIM_FAST=0` (the official camera set; fast mode is single-camera and the garage reads three) with the eval video off
  (`eval.video.render_video=false`, drawn after scoring), controller gains at the preset default (no `WIZ` gain: third-party entries
  have no reason to set the CLI gain our probes set to 10), CONC 4.
- Smoke first: 2 scenes (the first two of c000) for garage with rig 2, garage with rig 3, and go-straight.

## Rig rule (stated before the smoke)

- Rig 2 is primary: the checkpoint README builds the PAI-track image with `sensor_rig_2.yaml`, so the host's `014-pai-0014-v1` most
  likely used it. The garage driver replaces the rig's camera calibration by the simulator's announced f-theta calibration whenever
  one is announced (driver_service.py `start_session`), so rig 2 and rig 3 are expected to drive identically.
- If the rig-2 and rig-3 smoke scores are identical: the rig does not matter, run rig 2.
- If they differ: run rig 2 (README) for the full read; do not pick by comparing full-read means to 0.436 (that would tune the anchor).
- Stop and report (no full read) if either garage smoke has driver input / inference errors on most drive calls, a plan count of zero, or
  scores that are all 0 with route-fail / RPC failure reasons.

## Sanity check of the full garage read (before it enters the ruler)

The private board orders garage 0014 (mean 0.436) far below ours RSG + ACC + GAP (0.632) and above go-straight (0.140). The local
garage read is accepted if its PAI 281 mean is below the stored RSG + ACC + GAP read's and above go-straight's, and fewer than 5 % of its
sessions report driver input / inference errors. Otherwise stop and report.

## Ruler

zoib 2PL with the official settings (`pcs_local.py`'s `fit`), pool rows weighted to mimic the PAI board's family composition (counted
from `leaderboard_pai_20261019.json`), ability mapped to board PCS by least squares on three anchors (garage 1819, go-straight 1446,
RSG + ACC + GAP 2089), leave-one-anchor-out reported. Predictions above 2089 are extrapolation and are reported as such.
