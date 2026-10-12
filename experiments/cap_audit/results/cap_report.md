# CAP-AUDIT read: where our driver falls short, on every stored decision

Pre-registration: [../plans/2026-10-12-cap-audit-prereg.md](../plans/2026-10-12-cap-audit-prereg.md) (f9981841, before any read; amendment 1 changed
only the G-lat ruler, before any read). Full tables: [cap_table.md](cap_table.md) (every cell, every driver), `cells_{pub,held}.csv`, `flags.csv`,
`pairwise_worse.csv` (exploratory), `gates_{pub,held}.json`, `tags.csv`, `percep_held.csv.gz`. Box: `$DATA_DIR/runs/cap_audit/collect-pub/20261012-182318`,
`collect-held/20261012-182423` (per-decision parquet, bev.pkl). Cost: CPU only, 45 s + 20 s collect at 64 workers, 45 s read on the Mac.

## Data and gates

Primary = the stored 10-decision public rollouts, 1479 scenes / 136 logs, ten labels (YR10m10-F-s0, YS2G-L18-s0/s1, YS-L18-s0/s1, APY10m10-AB-s0,
WA-JEPA, LTF, DD, G0014), every label 1479 / 1479 rollouts with metrics and controller CSV. Secondary = 18-decision held, 452 scenes / 39 logs, the
same ten labels (references one run each). All gates pass: G-cov 1479 / 452 per label; G-pose 1.000; G-lat 0.967 / 0.971 (line 0.95; first run 0.856
on the rear axle, see amendment 1); G-gap Spearman 0.990; G-dir log same-direction 0.995, oncoming 0.004; G-tag 1457 / 1491 scenes tagged.
Our anchor = controller pose within 5 cm in 99.6 % of decisions.

## Registered read: 5 flagged cells (of 287 testable per row), none strong

m* = min over the four references of (ours - reference), paired per scene; flag = 2.5 % log-bootstrap quantile of m* > 0, i.e. worse than every reference.

| row | group | measure | m* [95 % CI] | z | n / logs | survived-only m* | held 18 m* | class | lever |
|:--|:--|:--|--:|--:|--:|--:|--:|:--|:--|
| YR10m10-F-s0 | start 5-10 m/s | behind (m lag behind the human at the last kept decision) | +0.774 [+0.377, +1.190] | 3.7 | 714 / 129 | +0.675 (flag) | +0.710 | longitudinal | new label course L |
| YS2G | start 5-10 m/s | behind | +0.593 [+0.217, +0.972] | 3.1 | 714 / 129 | +0.554 (flag) | +1.244 | longitudinal | new label course L |
| YS-L18 | start 5-10 m/s | behind | +0.590 [+0.214, +0.971] | 3.0 | 714 / 129 | +0.531 (flag) | +1.064 | longitudinal | new label course L |
| YS-L18 | kerb-side turn | behind | +1.210 [+0.308, +2.031] | 2.7 | 113 / 57 | +0.343 (no) | +1.181 | longitudinal | new label course L |
| YS-L18 | straight | head_path (deg vs the logged path) | +0.263 [+0.104, +0.362] | 3.6 | 1070 / 136 | +0.253 (flag) | +0.232 | perception | perception-side fix |

Classification evidence: the straight head_path cell has no serving signature (plan vs executed at +1 s 0.075 m against 0.074 m on all scenes;
|plan offset 10 m ahead| 0.46 m against 0.58 m) and the shipped P0 heads on the held straight decisions of those scenes miss the 20 m ego-lane
boundary by a median 0.62 m and show both ego lines in 0.15 of decisions (decision 272's lines 0.30 m / 0.80), so the registered rule says perception.
The rule is permissive there: decision 272 found the eyes bad on every straight decision, so any flagged straight lateral cell would be classed this way.
YS2G does not carry this cell (+0.125 [-0.017, +0.262] against G0014), YR neither (+0.106).

What the lag is: on cruise starts our three rows run 0.87-0.92 m/s slower than the human (signed mean v - v_log at 5-10 m/s: YR -0.91, YS2G -0.87;
WA-JEPA -0.73, G0014 -0.53, LTF +0.23, DD +0.12) and end the 5 s about 4.1-4.2 m behind the human (WA-JEPA 3.5, G0014 2.8, LTF 1.7, DD 1.8).
Over all scenes we are 3.1 m behind (G0014 3.1, WA-JEPA 2.7, APY 2.2, LTF 2.0, DD 2.1), so the all-scene cell is not flagged only because the garage
model is as slow. Our own family's APY10m10-AB-s0 is 0.9 m less behind (-0.23 m/s).

## Our profile against the human log (all scenes, YS2G)

lat_log 0.27 m (refs 0.32-0.45), head_path 2.0 deg (2.2-2.8), drift 0.073 m / decision (0.10-0.14), cor_half 7.8 % of scenes use half the corridor
(10-18 %), stop_speed 0.51 m/s where the human is stopped (0.49-0.93), brake_miss 0.22 m/s (0.36-0.77), speed_err 0.89 m/s (0.87-1.04), behind 3.1 m.
On every lateral, legality and interaction measure we are at or better than the best reference in nearly every group (heatmap mostly blue); the largest
remaining absolute deficits are in turns (across-traffic turns: lat_log 0.45 m, cor_half 13 %, head_path 3.2 deg; kerb-side 0.39 m / 13 % / 3.4 deg).

## Exploratory (not registered): worse than three of four references

`pairwise_worse.csv`. Two families, both for YR and the S rows:
- **Lag behind the human** (all scenes, intersection, straight, plain, lead; worse than WA-JEPA, LTF, DD; G0014 equally slow).
- **Oncoming lane and closeness in across-traffic turns**: oncoming-lane share above the log YS2G +0.031 / YS-L18 +0.041 / YR +0.018 (WA-JEPA -0.009,
  LTF -0.006, DD -0.005; G0014 +0.117 drives there far more, which is why no flag); YS2G - WA-JEPA +0.039 [+0.018, +0.064]. The same for stop_tl,
  turn_start, v<0.5 and v5-10 groups and closer minimum gaps in across turns (YS2G - WA-JEPA +0.196 m). The S rows raise it over YR, which is decision
  268's H3 seen on all decisions, not only zeros; the label arms that target it failed (274, 276, 277).

## Caveats

Ten-decision public scenes are this family's development scenes; the reference set mixes much weaker drivers (LTF / DD / garage score 0.75-0.83 public),
so "worse than every reference" is strict and only catches capabilities where even weak drivers beat us; the private-best reference (garage 0.848)
is slow and drives in the oncoming lane, so neither family of deficits explains the private ranking by itself. Simulated traffic is vehicles only and
replays the log (no pedestrians, no reaction), so yielding is measured as "brake where the human braked". Perception covers only the held YS-L18-s1
rerun (d275). One run per reference.
