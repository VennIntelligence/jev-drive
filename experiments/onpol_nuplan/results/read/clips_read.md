# ONPOL-NUPLAN clip read: is ON-IMIT's extra at-fault offroad imitation at the visited states, or the imitation target overshooting logged stops?

Lane R-NIGHT, 2026-10-12 night (CPU, Mac mini). Descriptive read of the 46 clips indexed in [clips.md](clips.md) (box copies under
`$DATA_DIR/runs/onpol_nuplan/clips`, frames extracted to `tmp/onpol_clips/` on the Mac, never in the repo). The numbers it starts from:
ON-IMIT GT distance 0.677 m (lowest of every arm; control P2H10S-F 1.009 m), at-fault zeros 20 against 12.75 (4 control seeds), offroad
18 against 12 ([on_report.md](on_report.md), decision 259 items 5 and 7). The imitation target is Amendment 2 of the
[prereg](../../plans/2026-10-11-onpol-prereg.md): lateral critically damped return to the logged path, speed starting at the ego's own
speed and blending linearly into the log's over 2 s; the on-policy rows sit on average 0.92 m (median) past the logged stop point when
the log is stopped.

## Pre-registration (written before any frame was looked at)

Two readings and what tells them apart in a clip (layout of col1_clip.py: bird's-eye view with blue ego trail, green served plan, dashed
logged path, grey road area; speed panel ego against log; gap and lead-prob panels):

- **Reading A, imitation at the visited states.** The zeros are lateral: the ego sits beside or behind the logged path, the plan pulls it
  back onto the dashed line (or the training on return-to-log rows makes it cut toward the line), and the ego clips the road edge, an
  island or a corner while the speed panel shows ego and log at about the same speed. The log is not stopping at the moment of the zero.
  Expected signature: offroad at a turn or lane bend, ego lateral offset to the dashed line visible before the zero, ego speed within
  about 1 m/s of the log's at the zero.
- **Reading B, the modified target overshooting logged stop points.** The zeros are longitudinal: the log decelerates toward a stop (log
  speed falls under about 1 m/s inside the 5 s window, or is falling steeply) while the ego, whose target speed started at its own speed
  and only reaches the log's after 2 s, keeps rolling and passes the point where the log stops: past an intersection stop line, a lane
  end, a kerb or into a stopped actor. Expected signature: ego speed clearly above the log's (more than about 1 m/s) in the last second
  before the zero, the zero located ahead of where the dashed path ends or turns, no lateral offset before it.
- **Neither.** The zero looks like the control's own failure class on the same road (same corner cut, same island), or an actor or map
  artefact unrelated to the label: then the extra offroads are not explained by either reading and the note says so.

Per clip the reader records: scene, tag, zero kind, time of the zero (first frame the ego box is red or outside the grey road area),
ego and log speed at that time and whether the log is stopping, lateral offset to the dashed path before the zero, road geometry
(straight, turn, intersection), any actor involved, and the reading it supports (A / B / neither). Decision rule for the answer: B is
supported if at least 4 of the 6 ON-IMIT offroad clips (3 broken scenes x 2 seeds) show the B signature; A if at least 4 show the A
signature; otherwise mixed or neither. The fixed scenes (control clips only: the arm's passing rollouts were pruned) say what ON-IMIT
avoided; the SSELFnoL-ON fixed scenes are read the same way to see whether the S-only arm avoids the same control failures by a different
route. Per-scene numbers (score / zero class / at-fault events / progress per arm and seed) come from on_per_scene.csv and are quoted
where they bear on the reading; they were not looked at scene by scene before this section was written beyond the aggregate above.

Known limit of the material: a broken scene has only the arm's clips, a fixed scene only the control's (ot2_loop keeps rollout.asl for
failed rollouts only), so "control counterpart" of a broken ON-IMIT scene means the control's per-scene numbers and, where another arm
also failed there, that arm's clip. Scenes are 5 s (10 decisions); a stop the log reaches after the window cannot be seen.

