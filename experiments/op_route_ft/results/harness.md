# Route-choice fine-tune: serving and closed-loop harness

How a route fine-tune checkpoint (scripts/rft.py: `ckpt-final.pt` + `adapter.npz`) is served in CARLA and measured on decision 127's
25 B2D junction turns. Code: [scripts/route_onnx.py](../scripts/route_onnx.py), [scripts/route_turns.py](../scripts/route_turns.py),
`lib/op_arb_agent.py` (`route_adapter`), `experiments/op_closed_loop/archive/op_arb_server.py` (`_route_bias`),
`jevdrive/openpilot/interface.py` (`command.route_geometry = nav-polyline`), guard plumbing in `experiments/op_guard/scripts/{guardlib,cllib}.py`.

## Path of the route command

1. Agent (each 20 Hz plan step, config key `"route_adapter": <adapter.npz>` at the top level): the dense route from the ego's projection on
   (rear axle, `self.poses`, the same reference point as the plan transform), up to 170 m of arc, to the rear-axle frame (x forward, y left;
   `world_to_local`), then `route_adapter.route_poly_from_path` -> 16 vertices at 10 m, mask. No noise at test time. Sent as request meta
   `route_poly`, `route_mask`, `route_adapter`.
2. Server: `NumpyAdapter(adapter.npz)` (cached per path) -> `features(enc, poly, mask)` -> bias (1, 32, 512) fp16 -> the ONNX input
   `intent_bias` (added to the hidden tokens of all 9 context frames, as in training: op_l_onnx.py `--bias-input`). Returns the features as
   `info["ra_f"]`.
3. Log: plans.jsonl gets `"ra": {"f": features, "poly": vertices}` next to the existing `act_k` (desired curvature) per step;
   `interface.json` records `command.route_geometry: nav-polyline` (declared for b2d, privilege semi: the dense route is lane-exact, a
   navigation route is road-level) and the config with the adapter path.
4. Without `route_adapter` nothing changes (agent sends the same meta, server does not touch `intent_bias`; an ONNX with the input and no
   adapter gets zeros from `OPModel`, i.e. the unconditioned fine-tuned model: arm rc-ctl).

Desire: kept as in decision 127's zones-off arm (`DESIRE=true`): the route turn desire (turnLeft / turnRight from 20 m before a LEFT /
RIGHT command) still reaches the model. Note the trainer feeds desire = 0 (`A.policy_feeds`), so in closed loop the fine-tuned model sees a
desire pulse it never saw in training; shipped sees the same pulse, so the pairing is clean, but a no-desire arm is the cleaner test of the
adapter alone (not wired in the guard line, which fixes DESIRE=true to match the cached shipped run).

## Equivalence (synthetic checkpoint)

Synthetic checkpoint: shipped weights with 3 initializers x 1.01 (vision head fc, a plan-pathway bias) + adapters with random out layers.
Port = `LModel` (fp16) + torch `RouteAdapter`, bias added before the valid mask (as rft.py `RModel.policy`); served = onnxruntime
(cuda-iob) + `NumpyAdapter`. WOD reference streams (2 x 134 frames, first 16 excluded), command cycled every 3 frames over
none / straight / left in 40 m / right in 15 m. "Bias dropped" = the served ONNX with zero bias against the biased port: what the check
reads if the bias path were broken.

| adapter out std | enc | numpy vs torch bias max | served: plan xy max / mean (m) | bias dropped: plan xy max / mean (m) | command effect in the port: max / mean (m) |
|---|---|--:|--:|--:|--:|
| 0.01 | bear | 3.0e-5 | 0.125 / 0.010, 0.125 / 0.011 | 0.19 / 0.017, 0.25 / 0.020 | 0.06 / 0.008, 0.09 / 0.008 |
| 0.01 | poly | 3.0e-5 | 0.125 / 0.010, 0.125 / 0.011 | 0.25 / 0.019, 0.25 / 0.020 | 0.02 / 0.005, 0.04 / 0.005 |
| 0.3 | bear | 9.7e-4 | 0.125 / 0.010, 0.125 / 0.012 | 4.38 / 0.302, 2.81 / 0.185 | 2.28 / 0.257, 4.05 / 0.287 |
| 0.3 | poly | 9.8e-4 | 0.125 / 0.011, 0.125 / 0.011 | 4.00 / 0.413, 6.25 / 0.555 | 2.28 / 0.257, 2.88 / 0.265 |

(two numbers per cell = the two streams.) Served vs port: mean 0.010-0.012 m, max 0.125 m = fp16 rounding of the plan output, the same
for every adapter; with std 0.3 (command moves the plan up to 2-4 m) dropping the bias raises the mean 17-50x. The std 0.01 adapter the
brief asked for moves the plan less than the fp16 noise, so it does not discriminate; std 0.3 does. Logs:
`$DATA_DIR/runs/op_route_ft/synth/eq2.log`.

Server in process (`route_onnx.py serve`, `op_arb_server.ArbModel` as op_arb.sh starts it, random frames, std 0.3 adapters): features
arrive (bear: `[1, 1, 0.71, 1.0, -0.99, 0.99]` for left in 40 m; poly: vertices / 50), the plan's y at 3 s moves with the command
(poly: 0.34 m no command, 0.07 straight, 0.24 left, 0.23 right).

## Commands

All on the box from `~/data/jev-drive`; `$R = $DATA_DIR/runs/op_route_ft`.

(a) Build the serving ONNX for an arm (op-train env, ~1 min, CPU):

    $DATA_DIR/envs/op-train/bin/python experiments/op_route_ft/scripts/route_onnx.py build \
        --ckpt $R/runs/<arm>-s0/ckpt-final.pt --adapter $R/runs/<arm>-s0/adapter.npz --out $R/onnx/<arm>-s0.onnx
    # rc-ctl (adapter only ever saw "no command"): --adapter none  -> no .adapter.npz, zero bias

Optional equivalence on the real checkpoint (one card, ~3 min):

    CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/op-train/bin/python experiments/op_route_ft/scripts/route_onnx.py ref \
        --ckpt $R/runs/<arm>-s0/ckpt-final.pt --adapter $R/runs/<arm>-s0/adapter.npz --out $R/onnx/<arm>-s0.ref.npz
    CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/openpilot/bin/python experiments/op_route_ft/scripts/route_onnx.py check \
        --onnx $R/onnx/<arm>-s0.onnx --ref $R/onnx/<arm>-s0.ref.npz          # pass: served plan xy mean ~0.01 m, max <= ~0.25 m

(b) Run the 25 turns for a list of arms (leases `op-route-ft-turns`, retries every 5 min up to `--wait-h` while no card is free; ~30-40 min
per arm on one card, arms one after another; run in tmux):

    scripts/tmux_run.sh rft-turns .venv/bin/python experiments/op_route_ft/scripts/route_turns.py run \
        $R/onnx/rc-bear-s0.onnx $R/onnx/rc-poly-s0.onnx $R/onnx/rc-ctl-s0.onnx --cards 1

`--cards 2` halves the wall time when two cards are free. Shipped is not rerun: decision 127's olnz units (`$DATA_DIR/runs/rig122/arms/olnz-s2-k*`)
are the guard's checked cache (same agent config; a candidate differs only by `SRV_ONNX` and `route_adapter`). The same units are what
`experiments/op_guard/scripts/line_b2d_turns.py --candidate $R/onnx/<arm>-s0.onnx` runs (the guard picks up `<stem>.adapter.npz` next to
the ONNX, or a `route_adapter` key in candidates.json), so the guard set and this lane share runs.

(c) Results:
- per arm: `$R/turns/<arm>-s0.{json,md}` (+ `shipped.{json,md}`): summary (choice / forced / all: took exit, shipped took, gained / lost,
  leaves lane of entered, collisions in turn windows; paired took-exit difference with a route-cluster bootstrap CI), per-turn table
  (forced, kind, angle, R_min, entered, branch, leaves, peak cross-track, head peak vs needed curvature, collisions), and in the json a
  5 Hz per-route step series (t, v, route index, desired curvature `act_k`, desire, lateral owner, adapter features) for BEV panels / GIFs.
- guard line file: `$DATA_DIR/runs/op_guard/<arm>-s0/subset/lines/b2d_turns.json`; units `.../b2d/turns-s2-k*/attempts/<route>/<n>/`
  (plans.jsonl with `ra` + `act_k`, ticks.jsonl, route.json, interface.json).
- GIF of one turn (chase camera + model input frames): the rig122 lane's `stage=gif` unit with `SRV_ONNX` + `TOP_ARGS='"route_adapter": "..."'`
  added to its env (junction_rig122_lane.py `unit(..., extra=...)`); not wired as a command.

Reports only (finished units): `.venv/bin/python experiments/op_route_ft/scripts/route_turns.py report <onnx> [...]`.

## Smoke

Shipped report from the cache reproduces decision 127: choice 0 / 13, forced 1 / 12, leaves lane 11 / 11 entered choice turns.

CARLA smoke (synthetic std 0.3 bear and poly candidates, `--stage smoke` = route 10255, one choice turn each): see below.

## Open-loop guard lines (drift, negatives) and registration

`scripts/guard_adapter.py:RouteCmdAdapter` implements op_guard's CommandAdapter: the candidate ONNX on the training port (portcand.load)
+ the torch RouteAdapter from the adapter.npz, bias before the valid mask; Command none = zero bias, correct / negative = `Command.poly`
padded to 16 vertices -> `route_adapter.features(enc)`, no noise. Frame sources: `op_img_cmd/ft/bank/<bank>#<row>` (line_drift) and
`op_lb/<data>#<i>` (line_negatives). Checked on the synthetic std 0.3 bear candidate: command none on 8 `dist` bank rows equals
portcand.plans_from_trunks bit for bit (max |d| 0); a left-turn polyline moves the plan end y by -0.5..+2.3 m; the nav path runs on
op_lb/g_navtest (negative vs none: -4.0..+0.4 m). lb_guardneg's frames are built by line_negatives itself (ensure_frames).

Register an arm (on the Mac, then commit candidates.json, push, pull on the box):

    python3 experiments/op_route_ft/scripts/guard_adapter.py register rc-bear-s0        # onnx + .adapter.npz under $DATA_DIR/runs/op_route_ft/onnx
    python3 experiments/op_route_ft/scripts/guard_adapter.py register rc-ctl-s0 --adapter none   # command_adapter null, zero bias

Entry: `{"onnx": "$DATA_DIR/runs/op_route_ft/onnx/<name>.onnx", "route_adapter": ".../<name>.adapter.npz", "command_adapter":
"experiments.op_route_ft.scripts.guard_adapter:RouteCmdAdapter", "note": ...}`; guardlib.resolve passes `route_adapter` to the B2D units
(b2d_turns, b2d_ds). HUGSIM has no route plumbing: a route candidate runs there with zero bias (no command).
