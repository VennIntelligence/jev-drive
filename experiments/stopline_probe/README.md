# stopline_probe: do frozen openpilot tokens see red / green

status: concluded
decisions: (inputs: 089, 238, 256, 286)
index: nuPlan red vs non-red from t0 view_39 tokens 0.936 (base 0.875); PAI: 4 red events only

**Question.** Does the frozen openpilot (Cinque) `view_39` token stream of one frame carry a readable red / non-red signal for the ego
lane's light, beyond what distance to the line and speed already say?

**Conclusion.** Yes on nuPlan: on 16 146 navtrain tokens with a traffic-light stop line 0-40 m ahead and a known ego-lane light at t0
(744 logs, log-grouped 5-fold), a logistic probe on the t0 frame's tokens reaches AUC 0.936 [0.925, 0.945] against 0.875 for the
position-only baseline (gain +0.061 [+0.047, +0.076]); on moving rows with no lead within 15 m 0.920 [0.900, 0.938] against 0.780 (+0.140
[+0.104, +0.177]); on the subset whose labels agree with the logged driver 0.942 / 0.934. PAI cannot be read the same way: the PAI scenes carry
no light state (map `traffic_light_status` is None in all 441 scenes, nothing in the usdz), and the behaviour rule finds only 4 red-light
stop events in train160 (the only scenes with PAI_DUMP frames) and 24 scenes in test281 (no frames); the nuPlan probe applied unchanged to
the PAI frames gives 0.78 [0.66, 0.90] for red vs green on 4 red scenes (descriptive). The PAI renderer does switch the lamp colour over time
(`figs/slp_render_switch.jpg`: 4 of 6 test281 stop-at-red scenes show red while waiting and green after the human departs).

**Read more.** [research/next-round/traffic-light-probe.md](../../research/next-round/traffic-light-probe.md) (Chinese note: verdict, what a
B term would look like); `results/nuplan_fields.json` (field check: raw DB, navtrain logs, ego-lane resolution).

Figures: `figs/slp_auc.png` (AUC per stratum, tokens vs baseline; look at the gap between blue and grey on the moving / no-lead rows, and at
the PAI rows' width), `figs/slp_render_switch.jpg` (left: 1 s into a test281 scene where the human waits at a red light; right: 1.5 s after
the human departs; RSG-L18-s0 rollout video inset).

Method notes. nuPlan labels: LONG-PROBE's context (lc_ctx.py, decision 238 matching: route -> stop polygon type 2 -> lane connector -> the
log's `traffic_lights` (lane_connector_id, is_red)); red at t0 vs non-red (green / yellow). Tokens: slot 7 (t0) of the op_parity
`navtrain_full@warp` caches (fp16, warp protocol, the t0 slot is a real keyframe). Models fitted inside each fold (PCA-256 + logistic C 0.05;
MLP 256; baseline = gradient-boosted trees on d, v, a); CIs = 2 000 log resamples. 11 % of red rows are crossed on red by the logged driver
(label noise or turn arrows); `nuplan_elig` drops them (LONG-PROBE ln_elig). Caveat: the probe does not show the tokens read the lamp itself;
cross traffic and brake lights also differ between red and green (the no-lead stratum removes only the lead). PAI labels (slp_labels.py RULES):
the human stops >= 1 s at a map wait line crossing its lane (STOP / UNKNOWN type, no lead within 15 m) -> red, or stop_sign when a mapped
R1 stop sign is within 20 m; passing such a line at >= 2 m/s -> green (many "green" lines have no light at all).

<!-- files:begin -->
## Files

- `slp_probe.py` (scripts): is the light of the ego lane readable …
- `slp_labels.py` (scripts): red / green behaviour labels of the …
- `slp_facts.py` (scripts): per-scene facts for the
- `slp_feats.py` (scripts): STOPLINE-PROBE PAI features
- `slp_nuplan_fields.py` (scripts): STOPLINE-PROBE nuPlan field check
- `nuplan.json` (results): 
- `nuplan_elig.json` (results): 
- `pai.json` (results): 
- `slp_auc.png` (figs): 

[results/](results/) 7 result files · [figs/](figs/) 2 figures · [scripts/](scripts/) 6 entry points
<!-- files:end -->

Layout: `scripts/` entry points, `results/` small result files, `figs/` figures. Run dirs: box `$DATA_DIR/runs/stopline_probe/`,
Tokyo `/data/runs/stopline_probe/` (facts, PAI features).
