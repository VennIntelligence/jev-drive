# onpol_long: closed-loop training at 18 decisions, round 1

status: live
decisions: (pending; inputs 259, 262)
index: round 1 queued: S + corridor hinge rows at 18 decisions vs ON-NONE, held 460
key: experiments/onpol_long/plans/2026-10-12-ol1-prereg.md, experiments/onpol_long/scripts/ol_chain.sh, experiments/onpol_long/scripts/ol_labels.py, experiments/onpol_long/lib/ol_kc.py, experiments/onpol_long/scripts/ol_report.py, experiments/onpol_nuplan/scripts/on_rows.py, experiments/onpol_nuplan/scripts/on_train.py, experiments/alpasim_long/scripts/long_read.py, research/next-round/closed-loop.md

**Question.** At 18 decisions (token + 8 s) in local AlpaSim nuPlan, do on-policy rows of the base's own rollouts on the 1031 train scenes,
carrying only consequence hinges on the model's own plan (S: swept footprint against vehicles, road-and-lane raster; K: 1.5 m dead band to
the logged path; C: lane-centre offset if it passes its label gate), give fewer zeros than the same continued training without rows
(ON-NONE) on the 460 held scenes, with at-fault zeros and GT distance not worse?

**Conclusion.** None yet; the chain is queued (bases YR10m10-F-s0 and P2H10S-F-s0, two training seeds each, both from the s0 base).

**Next.** Held read at 18 decisions of `YSKC-L18` / `YNONE-L18` / `PSKC-L18` / `PNONE-L18` s0 s1 -> `report_{Y,P}.md`, registered lines in the prereg.

**Read more.** [plans/2026-10-12-ol1-prereg.md](plans/2026-10-12-ol1-prereg.md); the plan: [research/next-round/closed-loop.md](../../research/next-round/closed-loop.md);
the row and trainer code it reuses unedited: [../onpol_nuplan/README.md](../onpol_nuplan/README.md); the 18-decision driver path:
[../alpasim_long/README.md](../alpasim_long/README.md).

## Chain

```bash
bash experiments/onpol_long/scripts/ol_chain.sh submit-roll <base tag> <after ids>        # 6 rollout jobs (cl fanout), 18 decisions, frame dump
bash experiments/onpol_long/scripts/ol_chain.sh submit-rest <base tag> Y|P <rollout ids> <after ids for ON-NONE>
#   geo (on_rows.py geo + ol_labels.py build) -> tok (on_rows.py tok + ol_labels.py gate) -> strip (drop dump frames) ;
#   tok -> train <F>SKC-L18-s{0,1} -> read1 ; train <F>NONE-L18-s{0,1} -> read1 ; all reads -> report (ol_report.py)
```

State on the box: `$DATA_DIR/runs/onpol_long/{roll,rows,read,report,pool}`, job ids in `jobs.txt`; pools `runs/op_parity/cache/on1_L18{Y,P}`,
rows `runs/onpol_nuplan/rows/L18{Y,P}` (on_rows.py's own locations).

<!-- files:begin -->
<!-- files:end -->
