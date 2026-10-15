# others_turn: do third-party policies turn in our closed loop

status: live
decisions: none
index: held 24 turns: RSG 16, rule expert 22, WA-JEPA 7, garage 1, LTF 0, DD 0

**Question.** Do policies trained without turn-specific work (WA-JEPA, the AlpaSim LTF / DiffusionDrive samples, the host's py123d-garage
TransFuser) complete the 24 held-460 turns at 18 decisions that RSG-L18-s0 completes 16 of, run as shipped and read with the same ruler (`rf_follow.py`)?

**Conclusion.** No. Completed turns of 24: WA-JEPA 7 (paired vs RSG -37.5 pp [-68.8, 0.0]), garage 1, LTF 0, DiffusionDrive 0 (CI upper < 0 for all
three). Every failure falls inside the turn after about half the route's heading (median 0.45-0.62), at entry speeds like RSG's (4.2-4.8 m/s). On the
24 turn scenes they score 0.43 / 0.25 / 0.02 / 0.02 against RSG's 0.92; on held 452 they all score below RSG (WA-JEPA -0.115, 85 vs 26 zeros). The
reruns for the page reproduce the counts exactly (WA-JEPA 7, RSG 16, rule expert 22). Impressions from the page: WA-JEPA either
turns well or goes straight with the correct left / right command on every decision; LTF / DD start every turn and drift outward plan by plan
(~1 m per decision at 2 s) until they leave the corridor; garage turns early and cuts the inside. PAI side read (score only): of the organisers' references only
Alpamayo 1 passes more strict-turn rollouts than RSG (52 % vs 39 %).

**Next.** Look at the comparison page (Mac ~/Desktop/others_turn/index.html, Tokyo /home/ujs/Desktop/others_turn/index.html; static, 52 MB). Candidates borrowed from the literature (geometric target points from the
route memory, decoupled curve speed, a lateral-acceleration cap in the controller) are listed in research/others-turn.md section 4. None is registered yet.

**Read more.** [research/others-turn.md](../../research/others-turn.md) (Chinese: inventory, literature, difference list, answer);
[plans/2026-10-15-others-turn-prereg.md](plans/2026-10-15-others-turn-prereg.md).

## Results

- [results/follow.md](results/follow.md): rf_follow on the 24 turns (WA-JEPA from the stored alpasim_long run, LTF / DD / garage reruns), pairs vs RSG.
- [results/turn_mech.md](results/turn_mech.md): where the zero falls, heading covered, entry speed.
- [results/held18.md](results/held18.md): held 452 mean / zeros vs RSG (stored 460 runs).
- [results/follow_page.md](results/follow_page.md): the page's reruns (WAJEPA, RSGr, XPr) next to the stored reads.
- [results/pai_ref.md](results/pai_ref.md): the organisers' PAI reference subjects on the 36 strict turns (score only).
- Gates: G1 [results/g1.json](results/g1.json) (asl anchors = the driver's own anchors on 1 527 decisions, 0.0 m), G2 bridge YR10m10 9 / 24 (as in
  follow_ref), G3 [results/g3.json](results/g3.json) (reruns agree with the stored 460 runs on zero / non-zero 24 / 23 / 24).

## How to run (box, repo root)

```bash
bash experiments/others_turn/scripts/ot_chain.sh check    # G1 (CPU)
bash experiments/others_turn/scripts/ot_chain.sh rerun    # pool: ltf / dd / garage on results/turn24.txt (as shipped, asl kept)
bash experiments/others_turn/scripts/ot_chain.sh read     # views, rf_follow, ol_confirm, turn_mech, G3 -> $DATA_DIR/runs/others_turn
bash experiments/others_turn/scripts/ot_chain.sh rerun2   # pool PRIO 10: WA-JEPA, RSG-L18-s0, XP-L12-E46 on the 24 turns (asl kept)
bash experiments/others_turn/scripts/ot_chain.sh page     # rf_follow over the page labels + ot_viz extract (records, CAM_F0 frames)
# Mac: rsync $DATA_DIR/runs/others_turn/page, then ot_viz.py render (see its docstring) -> ~/Desktop/others_turn/
python3 experiments/others_turn/scripts/ot_pai_ref.py --ref <dir with the subject jsons> --out experiments/others_turn/results/pai_ref.md
```

The camera clips are the CAM_F0 frames the driver received, read from rollout.asl (`driver_camera_image`). No driver was changed. The stored RSG and
rule-expert runs pruned their asl, so their clips come from the reruns, and each cell states the rerun's outcome.

<!-- files:begin -->
## Files


[results/](results/) 12 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 5 entry points
<!-- files:end -->
