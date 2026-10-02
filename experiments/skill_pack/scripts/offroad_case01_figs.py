"""Figures for case 01 (token 04d2f35c7f6db137f): (1) the plans and the PDM reference projected into the t0 camera image with the
dataset calibration (an independent check of the sign / frame conventions), (2) the perturbation battery in the ego frame
over the map polygons, (3) the five frames of the input history.

navsim2 env (nuplan for the PDM reference, cv2, matplotlib).
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offroad_lib as L  # noqa: E402

TOKEN = "04d2f35c7f6db137f"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else L.OUT / "figs")
OUT.mkdir(parents=True, exist_ok=True)


def project(cam, P):
    """Ego-frame (rear axle) points (n, 3) -> pixels with the dataset calibration (R columns = camera axes in the ego frame)."""
    R, t, K, D = (np.asarray(cam[k], np.float64) for k in ("R", "t", "K", "D"))
    pc = (P - t) @ R                                  # R^T (P - t)
    ok = pc[:, 2] > 2.5
    uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(-1, 2), ok


def main():
    idx = {e["token"]: e for e in L.index()}
    e = idx[TOKEN]
    mc = L.load_cache(L.cache_paths()[TOKEN])
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    ref = L.pdm_ref_ego(mc, TrajectorySampling(num_poses=40, interval_length=0.1))
    pn, p4 = L.poses_by_token(L.NATIVE_POSES)[TOKEN], L.poses_by_token(L.N4_POSES)[TOKEN]
    plans = {"PDM reference": ref, "native (raw plan)": L.dense_from_poses(pn), "N4 (raw plan)": L.dense_from_poses(p4)}
    col = {"PDM reference": "#1b9e77", "native (raw plan)": "#1f77b4", "N4 (raw plan)": "#ff7f0e"}
    cam = e["cams"][-1]["CAM_F0"]
    img = cv2.cvtColor(cv2.imread(cam["path"]), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(img)
    for name, d in plans.items():
        P = np.c_[d[:, 0], d[:, 1], np.zeros(len(d))]
        uv, ok = project(cam, P)
        ax.plot(uv[ok, 0], uv[ok, 1], "-", color=col[name], lw=3, label=name)
        for i in (10, 20, 30, 40):
            if ok[i]:
                ax.plot(*uv[i], "o", color=col[name], ms=7)
                ax.text(uv[i, 0] + 8, uv[i, 1] - 8, f"{i / 10:g}s", color=col[name], fontsize=9)
    ax.set_xlim(-200, img.shape[1] + 200)
    ax.set_ylim(img.shape[0] + 100, -50)
    ax.legend(loc="lower left")
    ax.set_title("case 01 (stage 2, command left): plans projected into the t0 camera image (road plane z = 0)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "case01_overlay.png", dpi=110)
    plt.close(fig)

    # perturbation battery over the map
    cases = json.load(open(L.D / "runs/openloop_visual_review_20261002/cases.json"))["cases"]
    c = next(c for c in cases if c["token"] == TOKEN)
    pert = pickle.load(open(L.OUT / "perturb_case01.pkl", "rb"))[TOKEN]
    groups = [("desire pulses (single rising edge)", ["base", "turn_right@-1.5", "turn_left@-1.5", "turn_left@-0.5", "turn_right@0", "turn_left@-1.5_len0.5"]),
              ("history (speed kept, current frame only, warped)", ["base", "warp_true_track", "warp_no_yaw_track", "warp_mirrored_track"]),
              ("mirror / traffic convention", ["base", "mirror", "mirror_tc_swap", "tc_rhd_traffic"])]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.5), sharex=True, sharey=True)
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for a, (title, names) in zip(axs, groups):
        for p in c["polygons"]:
            ex = np.array(p["exterior"])
            a.fill(ex[:, 1], ex[:, 0], fc="#dfeee6" if p["kind"] == "route" else "#e8ebef" if p["kind"] == "lane" else "#f3f5f7",
                   ec="#9aa7b2", lw=0.4)
        a.plot(ref[:, 1], ref[:, 0], "--", color="k", lw=2, label="PDM reference")
        for k, n in enumerate(names):
            p = L.dense_from_poses(np.asarray(pert[n]["pose8"], float))
            a.plot(p[:, 1], p[:, 0], "-", lw=2, color=cyc[k], label=n)
        a.plot(0, 0, "ko")
        a.set_title(title)
        a.set_aspect("equal")
        a.legend(fontsize=7, loc="upper right")
        a.set_xlabel("y left (m)")
    axs[0].set_ylabel("x forward (m)")
    axs[0].set_xlim(15, -15)          # left of the car on the left of the plot
    axs[0].set_ylim(-5, 35)
    fig.suptitle("case 01: native Cinque plan under input perturbations (ego frame at t0, 4 s; route lane green)")
    fig.tight_layout()
    fig.savefig(OUT / "case01_perturbations.png", dpi=110)
    plt.close(fig)

    # history frames + ego track
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.6))
    for k, a in enumerate(axs):
        a.imshow(cv2.cvtColor(cv2.imread(e["cams"][k]["CAM_F0"]["path"]), cv2.COLOR_BGR2RGB))
        a.set_title(f"t = {-1.5 + 0.5 * k:g} s   yaw {np.degrees(e['pose'][k, 2]):+.1f} deg   v {np.linalg.norm(e['vel'][k]):.1f} m/s")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "case01_history.png", dpi=80)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
