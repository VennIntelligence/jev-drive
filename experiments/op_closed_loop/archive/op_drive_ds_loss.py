#!/usr/bin/env python3
"""op-drive: how many DS points do red lights and start-stop collisions cost per arm? Counterfactual from the B2D penalty multipliers
(red light x0.7, vehicle collision x0.6, pedestrian x0.5, static / layout x0.65; route completion unchanged):
DS_without = min(100, DS / factor of the removed events). Uses per_route.csv of `experiments.op_closed_loop.lib.op_arb_report drive` (dev tag) and the collision /
stop-then-go tables of experiments/op_closed_loop/archive/op_drive_collisions.py --csv.
    python3 experiments/op_closed_loop/archive/op_drive_ds_loss.py --per-route P.csv --collisions C.csv --stops S.csv [--tag f]"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-route", required=True)
    ap.add_argument("--collisions", required=True)
    ap.add_argument("--stops", required=True)
    ap.add_argument("--tag", default="f")
    a = ap.parse_args()
    d = pd.read_csv(a.per_route).assign(route=lambda x: x.route.astype(str))
    c = pd.read_csv(a.collisions, dtype={"route": str})
    s = pd.read_csv(a.stops, dtype={"route": str})
    c, s = c[c.tag == a.tag], s[s.tag == a.tag]
    ss = c[(c.kind == "veh") & c.cls.isin(["junction/launch", "creep"])].groupby(["arm", "seed", "route"]).size().rename("ss_coll")
    d = d.join(ss, on=["arm", "seed", "route"]).fillna({"ss_coll": 0})
    d["ds_no_red"] = (d.DS / 0.7 ** d.red_light).clip(upper=100)
    d["ds_no_ss"] = (d.DS / 0.6 ** d.ss_coll).clip(upper=100)
    d["ds_no_both"] = (d.DS / 0.7 ** d.red_light / 0.6 ** d.ss_coll).clip(upper=100)
    g = d.groupby("arm")
    out = g.agg(runs=("route", "size"), DS=("DS", "mean"), red_light=("red_light", "sum"), ss_coll=("ss_coll", "sum"), blocked=("blocked", "sum"),
                v=("v_mean", "mean"))
    for k, col in (("lost_red", "ds_no_red"), ("lost_startstop_coll", "ds_no_ss"), ("lost_both", "ds_no_both")):
        out[k] = g[col].mean() - out.DS
    ev = s.groupby("arm").agg(start_events=("t", "size"), at_red=("red", "sum"), started_on_red=("on_red", "sum"))
    ev["by_timer"] = s[s.rel == "timeout"].groupby("arm").size()
    out = out.join(ev).fillna(0).round(2)
    print(out.to_markdown())


if __name__ == "__main__":
    main()
