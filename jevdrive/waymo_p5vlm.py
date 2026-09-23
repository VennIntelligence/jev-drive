"""P5 prototype without CARLA: does a frozen Qwen3-VL, asked zero-shot, name the right discrete meta-action?

Decision 25 wants a "VLM zero-shot meta-action" column in P5 to decide whether VLM answers can be a label source
for the reaction decoder. The CARLA pair generator waits for P4, so this prototype asks the same question on
Waymo E2E val, judged by the logged future instead of an expert re-run (todos/2026-09-23-p5-vlm-metaaction-proto.md).

  frames    from the frozen P2/P3 subset (`waymo_ladder.SUBSET_FILE`), strictly complete 4-frame stride-2 clips:
            every pre_onset frame, a speed-matched straight_yaw control of the same size, a turn_yaw sample and
            every rater-scored frame. Frozen to `frames.parquet` with a sha256 of the frame names.
  input     the P3(d'') clip -- three cameras, 4 frames, stride 2, oldest first, through Qwen3-VL's own video
            path at native resolution -- plus a compact text of ego speed history and the route command.
  output    one line of strict JSON {"reason", "longitudinal", "lateral"} from a fixed label set; parsed strictly
            (no repair), invalid answers count as wrong and are reported as a rate.
  judge     `meta_labels`, a closed-form labelling of a 5 s trajectory: the same function labels the logged
            future (the judge), the rater_best trajectory (rater agreement) and the CTRA extrapolation of the
            ego history (a baseline), so all of them are read with one ruler.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import traj, waymo, waymo_ladder as ladder
from .common import data_dir, get_logger

log = get_logger(__name__)
QWEN4B = "Qwen/Qwen3-VL-4B-Instruct"            # bf16
QWEN32B = "Qwen/Qwen3-VL-32B-Instruct-FP8"   # the official block-wise FP8 checkpoint: ~35 GB, fits beside other jobs
FRAMES, STRIDE = 4, 2                 # the P3(d'') clip: 4 frames, 0.2 s apart, the last one is "now"
FRAME_SET = "p5vlm_v1"
TURN_N = 400                          # turn_yaw is only a sanity control: already turning, the answer is visible

LONG = ("keep", "slow", "stop", "accelerate")
LAT = ("keep_lane", "nudge_left", "nudge_right", "lane_change_left", "lane_change_right", "turn_left", "turn_right")
LAT3 = ("keep", "left", "right")                                   # the primary lateral read: which way
LAT3_OF = {"keep_lane": "keep", "nudge_left": "left", "lane_change_left": "left", "turn_left": "left",
           "nudge_right": "right", "lane_change_right": "right", "turn_right": "right"}
GROUPS = ("pre_onset", "straight_yaw", "turn_yaw", "rater")

# Judge thresholds, fixed in the todo before any VLM answer was read. Longitudinal over 3 s, lateral over 5 s.
STOP_V = 1.0                          # m/s; mean speed over 2.5-3.0 s below this is a standstill
DV_ABS, DV_REL = 1.5, 0.15            # |v3 - v0| above max(DV_ABS, DV_REL * v0) is a speed change
MIN_CHORD5 = 3.0                      # m; below this the car does not move enough to have a lateral action
TURN_DEG = 30.0                       # end heading beyond this is a turn
LC_M, NUDGE_M = 2.0, 0.75             # lateral offset at 5 s: >= LC_M lane change, >= NUDGE_M nudge
CURVE_DEG = 10.0                      # |end heading| in [CURVE_DEG, TURN_DEG): curve-or-manoeuvre, flagged ambiguous


# ---------------------------------------------------------------- the judge

def meta_labels(xy: np.ndarray, v0: np.ndarray) -> pd.DataFrame:
    """Meta-action labels of trajectories `xy` (n, 20, 2) at 4 Hz in the current ego frame (+x forward, +y left).

    Longitudinal (3 s): v3 is the mean segment speed over 2.5-3.0 s. stop if v3 < STOP_V (this includes staying
    stopped), else slow / accelerate if v3 moves by more than max(DV_ABS, DV_REL * v0), else keep.
    Lateral (5 s): no lateral action if the 5 s chord is under MIN_CHORD5. Otherwise the end heading is the
    direction of the last 0.5 s of motion (or twice the chord bearing, the constant-curvature value, if that
    segment is under 1 m): turn if it exceeds TURN_DEG, else lane change / nudge / keep by the lateral offset
    at 5 s. `ambiguous` marks non-turn frames whose end heading is CURVE_DEG-TURN_DEG: without a map, a road
    curve and a lane change look the same there.
    """
    xy = xy.astype(np.float64)
    seg = np.linalg.norm(np.diff(np.pad(xy, ((0, 0), (1, 0), (0, 0))), axis=1), axis=-1) / waymo.DT
    v3 = seg[:, 9:12].mean(1)                                   # segments ending at 2.5, 2.75, 3.0 s
    dv = np.maximum(DV_ABS, DV_REL * v0)
    lng = np.where(v3 < STOP_V, "stop", np.where(v3 < v0 - dv, "slow", np.where(v3 > v0 + dv, "accelerate", "keep")))
    end = xy[:, -1] - xy[:, -3]
    chord = np.linalg.norm(xy[:, -1], axis=-1)
    head = np.where(np.linalg.norm(end, axis=-1) >= 1.0, np.degrees(np.arctan2(end[:, 1], end[:, 0])),
                    2 * np.degrees(np.arctan2(xy[:, -1, 1], xy[:, -1, 0])))
    y5, side = xy[:, -1, 1], np.where(xy[:, -1, 1] >= 0, "left", "right")
    turn = np.abs(head) >= TURN_DEG
    lat = np.where(turn, np.where(head > 0, "turn_left", "turn_right"),
                   np.where(np.abs(y5) >= LC_M, np.char.add("lane_change_", side),
                            np.where(np.abs(y5) >= NUDGE_M, np.char.add("nudge_", side), "keep_lane")))
    still = chord < MIN_CHORD5
    lat = np.where(still, "keep_lane", lat)
    amb = ~still & ~turn & (np.abs(head) >= CURVE_DEG)
    return pd.DataFrame({"long": lng, "lat": lat, "lat3": pd.Series(lat).map(LAT3_OF).to_numpy(),
                         "v3": v3, "head5": head, "y5": y5, "chord5": chord, "ambiguous": amb})


# ---------------------------------------------------------------- frames

def frames_path(name: str = FRAME_SET) -> Path:
    return ladder.subset_path(name)


def select_frames(seed: int = 0, straight_bins: int = 10) -> pd.DataFrame:
    """Freeze the evaluation frames. Only strictly complete clips (decision 13): the VLM must see what it is
    credited with. The straight control is drawn to match pre_onset's speed distribution decile by decile,
    because every longitudinal label depends on speed and an unmatched control would compare speeds, not
    situations."""
    ctx = ladder.base_context(seed)
    df, keep = ctx["df"], ladder.load_subset(ctx)
    full = waymo.history_set(df, FRAMES - 1, STRIDE, targets=ctx["rows"])
    ok = keep & full
    past, _ = waymo.load_ego()
    v0 = waymo.init_speed(past[ctx["rows"]])
    rng = np.random.default_rng(seed)
    pre = ok & ctx["sub"]["pre_onset"]
    grp = {"pre_onset": pre, "rater": ok & ctx["rater"]}
    edges = np.quantile(v0[pre], np.linspace(0, 1, straight_bins + 1)[1:-1])
    pool = np.flatnonzero(ok & ctx["sub"]["straight_yaw"])
    b_pre, b_pool = np.searchsorted(edges, v0[pre]), np.searchsorted(edges, v0[pool])
    take = np.concatenate([rng.choice(pool[b_pool == b], min(int((b_pre == b).sum()), int((b_pool == b).sum())),
                                      replace=False) for b in range(straight_bins)])
    grp["straight_yaw"] = np.isin(np.arange(len(ok)), take)
    tpool = np.flatnonzero(ok & ctx["sub"]["turn_yaw"])
    grp["turn_yaw"] = np.isin(np.arange(len(ok)), rng.choice(tpool, min(TURN_N, len(tpool)), replace=False))
    anyg = np.logical_or.reduce([grp[g] for g in GROUPS])
    r = ctx["rows"][anyg]
    t = pd.DataFrame({"frame_name": ctx["fname"][anyg], "row": r, "sequence": ctx["seq"][anyg],
                      "intent": df.intent.to_numpy()[r], "v0": v0[anyg], **{g: grp[g][anyg] for g in GROUPS}})
    t = t.sort_values("frame_name").reset_index(drop=True)
    t.to_parquet(frames_path(), index=False)
    sha = hashlib.sha256("\n".join(t.frame_name).encode()).hexdigest()
    log.info("frames: %d (%d sequences) -> %s, sha256 %s; %s", len(t), t.sequence.nunique(), frames_path(), sha,
             {g: int(t[g].sum()) for g in GROUPS})
    return t


def load_frames(name: str = FRAME_SET) -> pd.DataFrame:
    """The frozen frames, with their rows re-resolved by frame name against the current index."""
    t = pd.read_parquet(frames_path(name))
    df = waymo.load_index()
    at = pd.Series(np.arange(len(df)), index=waymo.frame_names(df))
    t["row"] = at.reindex(t.frame_name.to_numpy()).to_numpy().astype(int)
    return t


def judge_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per frame: log labels, CTRA-baseline labels, rater_best labels (rater frames only) and the rater set."""
    df = waymo.load_index()
    past, fut = waymo.load_ego()
    p, v0 = past[t.row], t.v0.to_numpy()
    out = t.copy()
    for pre, xy in (("log", waymo.future_xy(fut[t.row])), ("ctra", waymo.baselines(p)["ctra"])):
        lab = meta_labels(xy, v0)
        out[[f"{pre}_{c}" for c in lab.columns]] = lab.to_numpy()
    rrow, rtraj, rscore = waymo.load_rater(df)
    pos = pd.Series(np.arange(len(t)), index=t.row.to_numpy()).reindex(rrow).to_numpy()
    ok = ~np.isnan(pos)
    pos, rtraj, rscore = pos[ok].astype(int), rtraj[ok], rscore[ok]
    for c in ("long", "lat", "lat3"):
        out[f"rbest_{c}"] = None
    best = meta_labels(rtraj[np.arange(len(pos)), rscore.argmax(1)], v0[pos])
    for c in ("long", "lat", "lat3"):
        out.loc[pos, f"rbest_{c}"] = best[c].to_numpy()
    # every rater trajectory scored >= 7: the set of answers a rater called good, not just the best one
    each = [meta_labels(rtraj[:, k], v0[pos]) for k in range(rtraj.shape[1])]
    out["rgood"] = None
    out.loc[pos, "rgood"] = pd.Series([json.dumps(sorted({(e.long[i], e.lat3[i]) for k, e in enumerate(each)
                                                          if rscore[i, k] >= 7}))
                                       for i in range(len(pos))], index=pos)
    return out


# ---------------------------------------------------------------- prompt

INTENT_TEXT = {0: "unknown", 1: "go straight", 2: "turn left", 3: "turn right"}
TASK = """Decide the ego vehicle's meta-action from now on.
longitudinal (next 3 s), one of:
- keep: hold roughly the current speed
- slow: decelerate noticeably but do not stop
- stop: come to, or stay at, a standstill within 3 s
- accelerate: speed up noticeably, including starting from standstill
lateral (next 5 s), one of:
- keep_lane: stay in the current lane
- nudge_left / nudge_right: a small sideways shift (under about 2 m), e.g. to pass an obstacle
- lane_change_left / lane_change_right: move into the adjacent lane
- turn_left / turn_right: turn at an intersection or into another road or driveway (heading change over 30 degrees)
Answer with one line of strict JSON and nothing else:
{"reason": "<one short sentence>", "longitudinal": "<label>", "lateral": "<label>"}"""
CAM_TEXT = {"front": "Front camera", "front_left": "Front-left camera", "front_right": "Front-right camera"}


def ego_text(past: np.ndarray, intent: int) -> str:
    """Speed now and 1/2/3 s ago, acceleration, yaw rate and heading change, and the route command: the ego
    readout's own input, in words. `past` is (16, 6) at 4 Hz, oldest first, in the current ego frame."""
    kin = waymo.past_kinematics(past[None])
    sp = np.linalg.norm(past[:, 2:4], axis=-1)
    h3 = np.degrees(np.arctan2(*(past[-1, :2] - past[-13, :2])[::-1])) if np.linalg.norm(
        past[-1, :2] - past[-13, :2]) > 1.0 else 0.0
    return (f"Ego state: speed now {sp[-1]:.1f} m/s; 1 s / 2 s / 3 s ago {sp[-5]:.1f} / {sp[-9]:.1f} / "
            f"{sp[-13]:.1f} m/s; acceleration {float(kin['a'][0]):+.1f} m/s^2; yaw rate "
            f"{float(np.degrees(kin['w'][0])):+.1f} deg/s (positive = turning left); direction of travel over "
            f"the last 3 s {h3:+.0f} deg from the current heading. Route command: {INTENT_TEXT[int(intent)]}.")


def build_content(ego: str, cams, video: bool = True) -> list:
    """The user turn. With `video`, one captioned clip per camera; without, the text-only control."""
    if not video:
        return [{"type": "text", "text": "You are driving the ego vehicle. You cannot see the cameras; decide from "
                                         "the ego state and route command alone.\n" + ego + "\n" + TASK}]
    c = [{"type": "text", "text": f"You are driving the ego vehicle. Below are {len(cams)} synchronized camera "
                                  f"clips, each {FRAMES} frames over the last {(FRAMES - 1) * STRIDE * waymo.FRAME_DT:.1f} s; "
                                  "the last frame of each clip is now.\n"}]
    for cam in cams:
        c += [{"type": "text", "text": f"{CAM_TEXT[cam]}:"}, {"type": "video"}, {"type": "text", "text": "\n"}]
    return c + [{"type": "text", "text": ego + "\n" + TASK}]


def prompt_hash(proc, cams, video: bool) -> str:
    """sha256 of the rendered template with a fixed placeholder ego text: what the run was asked, minus the frame."""
    txt = proc.apply_chat_template([{"role": "user", "content": build_content("<EGO>", cams, video)}],
                                   add_generation_prompt=True, tokenize=False)
    return hashlib.sha256(txt.encode()).hexdigest()


# ---------------------------------------------------------------- parsing

def parse(text: str) -> dict:
    """Strict: the answer must be a JSON object with both labels spelled exactly as listed. No repair, no
    fuzzy matching; anything else is invalid and counts as wrong."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        o = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        o = None
    if not isinstance(o, dict):
        return {"valid": False, "why": "not json"}
    lg, lt = o.get("longitudinal"), o.get("lateral")
    ok = lg in LONG and lt in LAT
    return {"valid": bool(ok), "why": "" if ok else "label", "long": lg if ok else None, "lat": lt if ok else None,
            "reason": str(o.get("reason", ""))[:300]}


# ---------------------------------------------------------------- generation

class Clips:
    """Dataset: one frame -> (frame_name, ego text, clips per camera as PIL lists, oldest first)."""

    def __init__(self, items, names, texts, n_cams):
        self.shards = waymo.Shards(items, lambda imgs: imgs)
        self.names, self.texts, self.n_cams = names, texts, n_cams

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        imgs = self.shards[i] if self.n_cams else []
        return self.names[i], self.texts[i], [imgs[k * FRAMES:(k + 1) * FRAMES] for k in range(self.n_cams)]


def run(rl, model_id: str = QWEN4B, variant: str = "main", groups=GROUPS, batch_size: int = 4,
        limit: int | None = None, max_new_tokens: int = 96, out: Path | None = None, workers: int = 6) -> dict:
    """Generate answers for the frozen frames; append to `answers.jsonl` in `out` (resumable by frame name).

    variant: main (3 cameras, 4-frame clip ending now), text (no video), front (front camera only),
             shift1 (the 3-camera clip ending one frame, 0.1 s, earlier; ego text unchanged).
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t = load_frames()
    t = t[t[list(groups)].any(axis=1)].reset_index(drop=True)
    if limit:
        t = t.iloc[np.random.default_rng(0).permutation(len(t))[:limit]].reset_index(drop=True)
    out = Path(out or rl.dir)
    ans = out / "answers.jsonl"
    prev = pd.read_json(ans, lines=True) if ans.exists() and ans.stat().st_size else None
    done = set() if prev is None else set(prev.frame_name[(prev.variant == variant) & (prev.model == model_id)])
    t = t[~t.frame_name.isin(done)].reset_index(drop=True)
    cams = {"main": waymo.CAMS, "shift1": waymo.CAMS, "front": ("front",), "text": ()}[variant]
    df = waymo.load_index()
    past, _ = waymo.load_ego()
    texts = [ego_text(past[r], i) for r, i in zip(t.row, t.intent)]
    rows = t.row.to_numpy()
    if variant == "shift1":                          # the same frames, each clip ending at frame - 1
        at = pd.Series(np.arange(len(df)), index=df.sequence.astype(str) + "/" + df.frame.astype(str))
        prev = at.reindex((df.sequence.astype(str) + "/" + (df.frame - 1).astype(str)).to_numpy()[rows]).to_numpy()
        okp = ~np.isnan(prev)
        t, rows, texts = t[okp].reset_index(drop=True), prev[okp].astype(int), [x for x, o in zip(texts, okp) if o]
    items = []
    if cams:
        items, idx, full = waymo.multicam_clip_items(df, rows, FRAMES - 1, STRIDE, cams)
        t, texts = t[full].reset_index(drop=True), [x for x, o in zip(texts, full) if o]
    log.info("%s / %s: %d frames to answer (%d already done), cameras %s", model_id, variant, len(t), len(done),
             cams)

    proc = AutoProcessor.from_pretrained(model_id)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
    phash = prompt_hash(proc, cams, bool(cams))
    meta = {"model": model_id, "variant": variant, "cams": list(cams), "frames": FRAMES, "stride": STRIDE,
            "prompt_sha256": phash, "decoding": "greedy", "max_new_tokens": max_new_tokens, "batch_size": batch_size,
            "n_todo": len(t), "n_done_before": len(done), "frame_set": FRAME_SET}
    (out / f"meta_{variant}.json").write_text(json.dumps(meta, indent=2))
    rl.event("gen_start", **meta)
    vmeta = {"total_num_frames": FRAMES, "fps": 1 / (STRIDE * waymo.FRAME_DT), "frames_indices": list(range(FRAMES))}

    def collate(batch):
        names, txts, clips = zip(*batch)
        msgs = [proc.apply_chat_template([{"role": "user", "content": build_content(x, cams, bool(cams))}],
                                         add_generation_prompt=True, tokenize=False) for x in txts]
        kw = {}
        if cams:
            kw = {"videos": [c for cl in clips for c in cl], "video_metadata": [vmeta] * (len(cams) * len(batch)),
                  "do_sample_frames": False, "cap_pixels_per_frame": False}
        return names, proc(text=list(msgs), padding=True, return_tensors="pt", **kw)

    loader = DataLoader(Clips(items or [None] * len(t), list(t.frame_name), texts, len(cams)), batch_size=batch_size,
                        num_workers=workers, collate_fn=collate, prefetch_factor=4 if workers else None)
    torch.cuda.reset_peak_memory_stats()
    n, n_tok, t_first, n_first, t0 = 0, 0, None, 0, time.perf_counter()
    bar = tqdm(total=len(t), desc=f"p5vlm/{variant}", unit="frame", dynamic_ncols=True)
    with open(ans, "a", buffering=1) as f, torch.inference_mode():
        for names, b in loader:
            b = b.to("cuda")
            L = b["input_ids"].shape[1]
            g = model.generate(**b, do_sample=False, max_new_tokens=max_new_tokens, temperature=None, top_p=None,
                               top_k=None)
            outs = proc.batch_decode(g[:, L:], skip_special_tokens=True)
            for name, o in zip(names, outs):
                f.write(json.dumps({"frame_name": name, "variant": variant, "model": model_id, "raw": o,
                                    **parse(o)}) + "\n")
            n += len(names)
            n_tok += int((g[:, L:] != proc.tokenizer.pad_token_id).sum())
            bar.update(len(names))
            if t_first is None:
                t_first, n_first = time.perf_counter(), n
            rl.scalar(f"p5vlm/{variant}/prompt_tokens", L, n)
    bar.close()
    wall = time.perf_counter() - t0
    steady = (time.perf_counter() - t_first) / max(n - n_first, 1) if t_first else float("nan")
    stats = {**meta, "n": n, "wall_s": wall, "s_per_frame": steady, "gen_tokens_per_frame": n_tok / max(n, 1),
             "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
             "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
    (out / f"timing_{variant}.json").write_text(json.dumps(stats, indent=2, default=float))
    log.info("%s: %d frames, %.2f s/frame steady, peak VRAM %.1f GB (reserved %.1f)", variant, n, steady,
             stats["peak_vram_gb"], stats["peak_vram_reserved_gb"])
    rl.event("gen_end", **stats)
    return stats


# ---------------------------------------------------------------- report

def _acc(correct: np.ndarray, seq: np.ndarray, m: np.ndarray) -> dict:
    if not m.any():
        return {"n": 0, "acc": np.nan, "lo": np.nan, "hi": np.nan}
    v = correct[m].astype(float)
    lo, hi = traj.boot_ci(v, seq[m])
    return {"n": int(m.sum()), "acc": float(v.mean()), "lo": lo, "hi": hi}


def _paired(a: np.ndarray, b: np.ndarray, seq: np.ndarray, m: np.ndarray) -> dict:
    d = (a.astype(float) - b.astype(float))[m]
    lo, hi = traj.boot_ci(d, seq[m]) if m.any() else (np.nan, np.nan)
    return {"delta": float(d.mean()) if m.any() else np.nan, "d_lo": lo, "d_hi": hi}


def baselines(j: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Answers of the trivial and ego-only baselines, in the VLM's label space (long, lat, lat3)."""
    stopped = j.v0.to_numpy() < STOP_V
    cont_long = np.where(stopped, "stop", "keep")
    intent_lat = pd.Series(j.intent).map({2: "turn_left", 3: "turn_right"}).fillna("keep_lane").to_numpy()
    mk = lambda lg, lt: pd.DataFrame({"long": lg, "lat": lt, "lat3": pd.Series(lt).map(LAT3_OF).to_numpy()})
    return {"continuation (stop if stopped, else keep; keep_lane)": mk(cont_long, np.full(len(j), "keep_lane")),
            "ego kinematic (CTRA extrapolation, same judge)": mk(j.ctra_long.to_numpy(), j.ctra_lat.to_numpy()),
            "route command (CTRA long; lateral = intent)": mk(j.ctra_long.to_numpy(), intent_lat)}


def _joint(lng, lat3) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(lng).astype(str), "|"), np.asarray(lat3).astype(str))


def group_masks(j: pd.DataFrame) -> dict[str, np.ndarray]:
    """The four frame groups, plus pre_onset split by route command: where the command says go straight, the
    route-command baseline cannot know the lateral answer and anything right has to come from the cameras."""
    m = {g: j[g].to_numpy().astype(bool) for g in GROUPS}
    it = j.intent.to_numpy()
    m["pre_onset | route straight"] = m["pre_onset"] & (it == 1)
    m["pre_onset | route turn"] = m["pre_onset"] & np.isin(it, (2, 3))
    return m


def load_answers(run_dirs) -> pd.DataFrame:
    """Answers from one or more run directories (comma list), with `arm` = "<model> <variant>"."""
    a = pd.concat([pd.read_json(Path(d) / "answers.jsonl", lines=True) for d in str(run_dirs).split(",")])
    return a.assign(arm=a.model.str.split("/").str[-1].str.replace("-Instruct", "") + " " + a.variant)


def report(run_dir, variants=None) -> dict[str, pd.DataFrame]:
    """Accuracy per group x axis for every VLM variant and baseline (sequence-bootstrap CIs), the paired delta of
    each VLM variant against the best baseline on the same frames, invalid rate, confusions, rater agreement."""
    j = judge_table(load_frames())
    a = load_answers(run_dir)
    seq = j.sequence.to_numpy()
    arms = baselines(j)
    for k in arms.values():
        k["answered"], k["valid"] = True, True
    for v in variants or list(pd.unique(a.arm)):
        x = a[a.arm == v].drop_duplicates("frame_name", keep="last").set_index("frame_name").reindex(j.frame_name)
        valid = x.valid.fillna(False).astype(bool).to_numpy()
        lat = np.where(valid, x.lat.to_numpy(), "invalid")
        arms[f"VLM {v}"] = pd.DataFrame({"long": np.where(valid, x.long.to_numpy(), "invalid"), "lat": lat,
                                         "lat3": pd.Series(lat).map(LAT3_OF).fillna("invalid").to_numpy(),
                                         "answered": x.raw.notna().to_numpy(), "valid": valid})
    truth = {"long": j.log_long.to_numpy(), "lat3": j.log_lat3.to_numpy(), "lat": j.log_lat.to_numpy()}
    truth["joint"] = _joint(truth["long"], truth["lat3"])
    masks = group_masks(j)
    rows, delta, conf, inv = [], [], [], []
    for g, m0 in masks.items():
        for ax, y in truth.items():
            maj = pd.Series(y[m0]).mode().iloc[0]
            cand = {f"majority ({maj})": (np.full(len(j), maj), np.ones(len(j), bool))}
            for k, v in arms.items():
                p = _joint(v.long.to_numpy(), v.lat3.to_numpy()) if ax == "joint" else v[ax].to_numpy()
                cand[k] = (p, v.answered.to_numpy())
            for k, (p, ans) in cand.items():
                rows.append({"group": g, "axis": ax, "arm": k, **_acc(p == y, seq, m0 & ans)})
            base = max((k for k in cand if not k.startswith("VLM")), key=lambda k: np.mean(cand[k][0][m0] == y[m0]))
            for k in (k for k in cand if k.startswith("VLM")):
                m = m0 & cand[k][1]
                delta.append({"group": g, "axis": ax, "vlm": k, "best_baseline": base,
                              "baseline_acc": float((cand[base][0] == y)[m].mean()) if m.any() else np.nan,
                              "vlm_acc": float((cand[k][0] == y)[m].mean()) if m.any() else np.nan,
                              **_paired(cand[k][0] == y, cand[base][0] == y, seq, m)})
    for k, v in arms.items():
        if not k.startswith("VLM"):
            continue
        ans, bad = v.answered.to_numpy(), v.answered.to_numpy() & ~v.valid.to_numpy()
        inv.append({"variant": k, "answered": int(ans.sum()), "invalid": int(bad.sum()),
                    "invalid_rate": float(bad.sum() / max(ans.sum(), 1))})
        for g, mg in masks.items():
            m = mg & ans
            for ax in ("long", "lat"):
                c = pd.crosstab(pd.Series(truth[ax][m], name="judge"), pd.Series(v[ax].to_numpy()[m], name="vlm"))
                conf.append(c.stack().rename("n").reset_index().assign(variant=k, group=g, axis=ax))
    # rater frames: agreement with rater_best, with the log as the ceiling, and "matches any answer scored >= 7"
    rr = j.rbest_long.notna().to_numpy()
    good = j.rgood.map(lambda s: {tuple(p) for p in json.loads(s)} if isinstance(s, str) else set())
    has_good = good.map(len).gt(0).to_numpy()
    ragree = []
    judge_arm = pd.DataFrame({"long": j.log_long, "lat3": j.log_lat3, "answered": True})
    for k, v in {"log (the judge itself)": judge_arm, **arms}.items():
        m = rr & v.answered.to_numpy().astype(bool)
        for ax in ("long", "lat3"):
            ragree.append({"arm": k, "vs": "rater_best", "axis": ax,
                           **_acc(v[ax].to_numpy() == j[f"rbest_{ax}"].to_numpy(), seq, m)})
        hit = np.array([(lg, lt) in s for lg, lt, s in zip(v.long.to_numpy(), v.lat3.to_numpy(), good)])
        ragree.append({"arm": k, "vs": "any rater traj scored >= 7", "axis": "joint", **_acc(hit, seq, m & has_good)})
    # sensitivity: pre_onset without the curve-or-manoeuvre band the judge cannot resolve
    unamb = j.pre_onset.to_numpy().astype(bool) & ~j.log_ambiguous.astype(bool).to_numpy()
    sens = [{"arm": k, "subset": "pre_onset, unambiguous lateral", "axis": "lat3",
             **_acc(v.lat3.to_numpy() == truth["lat3"], seq, unamb & v.answered.to_numpy())} for k, v in arms.items()]
    dist = pd.concat([j.loc[j[g].astype(bool), c].value_counts().rename("n").reset_index()
                      .rename(columns={c: "label"}).assign(group=g, axis=c)
                      for g in GROUPS for c in ("log_long", "log_lat", "log_lat3")])
    return {"accuracy": pd.DataFrame(rows), "delta_vs_best_baseline": pd.DataFrame(delta),
            "invalid": pd.DataFrame(inv), "confusion": pd.concat(conf) if conf else pd.DataFrame(),
            "rater_agreement": pd.DataFrame(ragree), "sensitivity": pd.DataFrame(sens), "label_distribution": dist}


# ---------------------------------------------------------------- consistency (flip analogue)

def consistency(run_dir, base: str = "main", others=("front", "shift1")) -> pd.DataFrame:
    """Share of frames whose (long, lat) answer is unchanged under a small input perturbation, per group
    (one model per run directory)."""
    a = load_answers(run_dir)
    j = load_frames()
    b = a[a.variant == base].drop_duplicates("frame_name", keep="last").set_index("frame_name")
    rows = []
    for o in others:
        x = a[a.variant == o].drop_duplicates("frame_name", keep="last").set_index("frame_name")
        common = b.index.intersection(x.index)
        if not len(common):
            continue
        jj = j.set_index("frame_name").loc[common]
        same_lat = (b.loc[common, "lat"].to_numpy() == x.loc[common, "lat"].to_numpy())
        same_both = same_lat & (b.loc[common, "long"].to_numpy() == x.loc[common, "long"].to_numpy())
        for g in GROUPS:
            m = jj[g].to_numpy().astype(bool)
            for name, v in (("lat", same_lat), ("long & lat", same_both)):
                rows.append({"perturbation": o, "group": g, "same": name,
                             **_acc(v, jj.sequence.to_numpy(), m)})
    return pd.DataFrame(rows)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="labels", help="comma list of select,labels,gen,report,consistency")
    ap.add_argument("--model", default=QWEN4B)
    ap.add_argument("--variant", default="main", choices=("main", "text", "front", "shift1"))
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="profiling: answer only this many random frames")
    ap.add_argument("--out", default=None, help="gen: append into this run directory (resume / add a variant)")
    ap.add_argument("--run", default=None, help="report: the run directory holding answers.jsonl")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    steps = a.steps.split(",")
    rl = RunLog("waymo_p5vlm", a.tag or a.steps.replace(",", "-"))
    rl.log.info("args %s -> %s", vars(a), rl.dir)
    rl.event("start", args=vars(a))
    if "select" in steps:
        t = select_frames()
        rl.event("select", n=len(t), sha256=hashlib.sha256("\n".join(t.frame_name).encode()).hexdigest(),
                 **{g: int(t[g].sum()) for g in GROUPS})
    if "labels" in steps:
        j = judge_table(load_frames())
        j.to_parquet(rl.dir / "judge.parquet", index=False)
        for g in GROUPS:
            s = j[j[g].astype(bool)]
            rl.log.info("%s n=%d v0 median %.1f\n long %s\n lat %s\n lat3 %s\n ctra long %s\n ambiguous %.3f\n intent %s",
                        g, len(s), s.v0.median(), s.log_long.value_counts().to_dict(),
                        s.log_lat.value_counts().to_dict(), s.log_lat3.value_counts().to_dict(),
                        s.ctra_long.value_counts().to_dict(), s.log_ambiguous.astype(bool).mean(),
                        s.intent.value_counts().to_dict())
    if "gen" in steps:
        run(rl, a.model, a.variant, a.groups.split(","), a.batch_size, a.limit, out=a.out)
    if "report" in steps or "consistency" in steps:
        src = Path(a.run or a.out or rl.dir)
        if "report" in steps:
            for k, t in report(src).items():
                t.to_csv(rl.dir / f"{k}.csv", index=False)
                rl.log.info("%s\n%s", k, t.to_markdown(index=False, floatfmt=".3f") if k != "confusion" else len(t))
        if "consistency" in steps:
            c = consistency(src)
            c.to_csv(rl.dir / "consistency.csv", index=False)
            rl.log.info("consistency\n%s", c.to_markdown(index=False, floatfmt=".3f"))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
