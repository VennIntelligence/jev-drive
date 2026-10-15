"""Schematic figures for research/base-recipe/index.html: how the current base RSG-L18-s0 is trained (no data read).

Facts drawn from: experiments/op_parity/scripts/pp_train.py, pp_prep.py, experiments/alpasim/scripts/joint_train.py, ot3_rows.py,
experiments/onpol_nuplan/scripts/on_train.py, experiments/route_mem/scripts/rm_train.py, lib/rmem.py, lib/parity_adapter.py and the
run meta.json of RB0-F-s0 / RSG-L18-s0.

  .venv/bin/python research/base-recipe/figs/make_figs.py
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import plot_style  # noqa: E402

plot_style.apply()
mpl.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'legend.fontsize': 10})
P = plot_style.PALETTE
C = dict(frozen='#BDBDBD', train=P['blue'], new=P['orange'], data='#DCEBF5', distill=P['purple'], imit=P['blue'], anchor='#9E9E9E',
         yaw=P['green'], onp=P['vermillion'], ink='#222222', mute='#666666')
DPI = 150


def canvas(w=12, h=7, xlim=(0, 12), ylim=(0, 7)):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim), ax.set_ylim(*ylim), ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, text='', fc='white', ec='#444444', tc=None, fs=10.5, lw=1.0, ls='-', hatch=None, bold=False, ha='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=ec, lw=lw, ls=ls, hatch=hatch))
    if text:
        dark = fc in (C['train'], C['distill'], C['onp'], C['yaw'])
        tx = x + w / 2 if ha == 'center' else x + 0.12
        ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs, color=tc or ('white' if dark else C['ink']),
                fontweight='bold' if bold else 'normal', linespacing=1.3)


def arrow(ax, p, q, color='#444444', lw=1.4, rad=0.0, style='-|>', ls='-', ms=14):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))


def legend_chips(ax, items, x, y, dx=2.1, fs=10):
    for i, (lab, fc, kw) in enumerate(items):
        ax.add_patch(Rectangle((x + i * dx, y), 0.3, 0.2, fc=fc, ec='#444444', lw=0.6, **kw))
        ax.text(x + i * dx + 0.4, y + 0.1, lab, va='center', fontsize=fs)


def save(fig, name):
    fig.savefig(HERE / name, dpi=DPI, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- 1. model
def fig_model():
    fig, ax = canvas()
    ax.text(0.2, 6.75, 'Base model RSG-L18-s0: what is frozen, what is trained, what is new', fontsize=13, fontweight='bold')

    # front images -> frozen encoder -> cache
    box(ax, 0.2, 2.6, 1.9, 1.9, 'front camera\nimage pairs\n(t - 0.2 s, t)\n8 slots at\n-1.4 .. 0 s', fc=C['data'])
    box(ax, 2.5, 2.6, 1.8, 1.9, 'Cinque vision\nencoder\n\nFROZEN\nrun once in prep', fc=C['frozen'], lw=1.2)
    arrow(ax, (2.1, 3.55), (2.5, 3.55))
    box(ax, 2.5, 1.75, 1.8, 0.6, 'front.npy fp16\n(N, 8, 32, 512)', fc='white', fs=9.5, ls='--')
    arrow(ax, (3.4, 2.6), (3.4, 2.35), ms=10)
    arrow(ax, (4.3, 3.55), (4.75, 3.55))

    # 9-slot context
    x0, w, h0 = 4.8, 1.25, 0.34
    for i in range(9):
        y = 1.85 + i * (h0 + 0.04)
        zero = i == 0
        ax.add_patch(Rectangle((x0, y), w, h0, fc='white' if zero else C['frozen'], ec='#444444', lw=0.8, hatch='///' if zero else None))
        lab = 'slot 0 = zero' if zero else f'{-1.6 + 0.2 * i:+.1f} s'
        ax.text(x0 + w / 2, y + h0 / 2, lab, ha='center', va='center', fontsize=8.5,
                bbox=dict(fc='white', ec='none', pad=0.5) if zero else None)
    ax.text(x0 + w + 0.12, 5.05, '9-frame context\n9 x (32 x 512)', ha='left', va='center', fontsize=9.5)

    # adapter and route branch (bias on the context)
    box(ax, 0.2, 0.25, 3.0, 1.15, 'ego vector (20-d): present, command (3),\nvx, vy, ax, ay,\n4 history poses (x, y, yaw)', fc=C['data'], fs=9.5)
    box(ax, 3.6, 0.25, 2.6, 1.15, 'P2 adapter\nbias on all 9 context frames\n(fresh at stage 1, lr 3e-4)', fc=C['new'], fs=9.5)
    arrow(ax, (3.2, 0.82), (3.6, 0.82))
    arrow(ax, (5.42, 1.4), (5.42, 1.83))
    box(ax, 0.2, 5.75, 3.0, 0.85, 'route memory (200-d):\n40 points -6 .. +72 m, validity mask', fc=C['data'], fs=9.5)
    box(ax, 3.6, 5.75, 2.6, 0.85, 'route-memory branch\noutput layer starts at 0, lr 3e-4\ndropped on 25 % non-anchor rows', fc=C['new'], fs=8.8)
    arrow(ax, (3.2, 6.17), (3.6, 6.17))
    arrow(ax, (5.42, 5.75), (5.42, 5.35 + 0.03))
    ax.text(6.2, 1.55, '+ bias  (x present;\nanchor rows: present = 0)', fontsize=8.8, color=C['mute'], va='center')

    # policy pathway
    box(ax, 6.65, 2.4, 2.0, 2.3, 'policy pathway\n\nTRAINABLE\nONNX nodes 479-665\nlr 3e-5 (fp32 masters)', fc=C['train'], fs=10)
    arrow(ax, (6.05, 3.55), (6.65, 3.55))

    # heads
    box(ax, 9.15, 3.65, 2.65, 1.65, 'plan head\n33 x 15 MDN mean\ntraining reads 8 poses\n(x, y, yaw) at 0.5 .. 4 s', fc=C['train'], fs=10)
    box(ax, 9.15, 1.75, 2.65, 1.55, 'all other heads\ndistilled to the shipped\nCinque teacher (lam_d 30)', fc=C['distill'], fs=10)
    arrow(ax, (8.65, 3.8), (9.15, 4.45))
    arrow(ax, (8.65, 3.3), (9.15, 2.55))
    ax.text(10.47, 5.45, 'imitation / hinge losses', ha='center', fontsize=9.5, color=C['mute'])

    legend_chips(ax, [('frozen', C['frozen'], {}), ('trainable (from shipped)', C['train'], {}), ('new module', C['new'], {})],
                 6.65, 0.75, dx=1.75, fs=9.5)
    legend_chips(ax, [('distilled output', C['distill'], {}), ('input / cache', C['data'], {})], 6.65, 0.3, dx=1.75, fs=9.5)
    save(fig, 'fig_model.png')


# ---------------------------------------------------------------- 2. preprocessing
def fig_prep():
    fig = plt.figure(figsize=(12, 7))
    top = fig.add_axes([0, 0.5, 1, 0.5])
    top.set_xlim(0, 12), top.set_ylim(0, 3.5), top.axis('off')
    top.text(0.15, 3.25, 'a  navtrain token -> frozen-encoder cache (frame protocol "warp")', fontsize=12, fontweight='bold')
    box(top, 0.15, 1.0, 1.5, 1.6, 'navtrain log\n103 288 tokens\n12 shards\n\ntrain 101 499\ndev 1 789', fc=C['data'], fs=9.5)

    # timeline -1.6 .. 0
    X = lambda t: 2.9 + (t + 1.6) / 1.6 * 3.6  # noqa: E731
    yk, ys = 2.45, 1.35
    top.plot([X(-1.6), X(0)], [ys, ys], color='#444444', lw=0.8)
    for t in np.arange(-1.6, 0.01, 0.2):
        top.plot([X(t)] * 2, [ys - 0.06, ys + 0.06], color='#444444', lw=0.8)
        top.text(X(t), ys - 0.25, f'{t + 0.0:.1f}'.replace('-0.0', '0.0'), ha='center', fontsize=8.5)
    top.text(X(-0.8), ys - 0.6, 'time (s), slots every 0.2 s', ha='center', fontsize=9.5)
    keys = np.array([-1.5, -1.0, -0.5, 0.0])
    for t in keys:
        top.add_patch(Rectangle((X(t) - 0.12, yk - 0.12), 0.24, 0.24, fc='#333333'))
    top.text(X(-1.6) - 0.1, yk, 'keys\n2 Hz', ha='right', va='center', fontsize=9)
    top.text(X(-1.6) - 0.25, ys + 0.3, 'policy\nslots', ha='right', va='center', fontsize=9)
    slots = np.round(np.arange(-1.4, 0.01, 0.2), 1)
    for t in slots:
        iskey = np.isclose(keys, t).any()
        top.plot(X(t), ys + 0.3, 'o', ms=9, mfc='#333333' if iskey else C['new'], mec='#333333')
        if not iskey:
            src = keys[np.argmin(np.abs(keys - t))]
            arrow(top, (X(src), yk - 0.13), (X(t), ys + 0.42), color=C['new'], lw=1.0, ms=9, rad=0.0)
    top.plot(X(-1.6), ys + 0.3, 'o', ms=9, mfc='white', mec='#333333')
    top.text(X(-1.6), ys + 0.55, 'zero', ha='center', fontsize=8.5)
    top.text(X(-0.8), 3.0, '6 lattice slots = nearest key warped along the ego motion (orange)', ha='center', fontsize=9.5, color=C['mute'])
    arrow(top, (1.65, 1.8), (2.05, 1.8))

    box(top, 7.0, 1.0, 1.4, 1.6, 'frozen\nCinque\nencoder\n(once)', fc=C['frozen'])
    arrow(top, (6.45, 1.8), (7.0, 1.8))
    outs = [('front.npy  (N, 8, 32, 512) fp16\ntokens view_39 of the 8 slots', 2.45),
            ('tab.npz  ego (N, 20); fut (N, 8, 3):\nlogged x, y, yaw at 0.5 .. 4 s', 1.6),
            ('teacher.npz  shipped Cinque\noutputs + plan (N, 33, 15)', 0.75)]
    for txt, y in outs:
        box(top, 8.9, y - 0.32, 3.0, 0.64, txt, fc='white', fs=8.8, ls='--', ha='left')
        arrow(top, (8.4, 1.8), (8.9, y), rad=0.0)
    top.text(10.4, 0.05, 'tab / teacher need no encoder; drawn together for brevity', ha='center', fontsize=8, color=C['mute'])

    # b: yaw-rate rows
    fig.text(0.013, 0.47, 'b  yaw-rate rows yr1 (69 572 rows): perturbed history, target = logged future', fontsize=12, fontweight='bold')
    a1 = fig.add_axes([0.06, 0.08, 0.19, 0.31])
    t = np.array([-2.0, -1.5, -1.0, -0.5, 0.0])
    a1.plot(t, [0, 0, 0.8, 1.4, 2.4], '-o', color=C['yaw'], ms=4, label='random walk: a1, a2, a3')
    a1.plot(t, [0, 0, 0, 0, 1.2], '--o', color=C['onp'], ms=4, label='recent (p = 0.4): a1 = a2 = 0')
    for k in (-1.5, -1.0, -0.5, 0.0):
        a1.axvline(k, color='#999999', lw=0.5, zorder=0)
    a1.set_xlabel('t (s)'), a1.set_ylabel('injected heading error psi (deg)')
    a1.set_xticks([-1.5, -1.0, -0.5, 0.0])
    a1.legend(loc='upper left', fontsize=8.5)
    a1.set_ylim(-0.3, 3.6)
    fig.text(0.06, 0.005, 'a_i ~ N(0, 1 deg) clipped at 2.5 deg; y(0) ~ U(+-1.0 m), recent U(+-0.3 m); dy/dt = v psi',
             fontsize=8.5, color=C['mute'])

    a2 = fig.add_axes([0.30, 0.17, 0.25, 0.25])
    a2.set_xlim(-18, 42), a2.set_ylim(-2.2, 4.6), a2.axis('off')
    a2.fill_between([-18, 42], -1.8, 1.8, color='#F2F2F2', zorder=0)
    a2.plot([-18, 42], [0, 0], color='#888888', lw=1.2, label='logged path')
    tp = np.linspace(-1.5, 0, 16)
    v = 10.0
    psi = np.interp(tp, [-1.5, -1.0, -0.5, 0.0], np.radians([0, 2.2, 4.2, 7.0]))
    yy = 0.6 + np.cumsum(np.r_[0, v * np.sin(psi[:-1]) * np.diff(tp)])
    a2.plot(v * tp, yy, color=C['yaw'], lw=1.6, label='presented (perturbed) history')
    ego = Polygon([[-2.5, yy[-1] - 0.45], [2.5, yy[-1] - 0.45 + 5 * np.tan(psi[-1])], [2.5, yy[-1] + 0.45 + 5 * np.tan(psi[-1])],
                   [-2.5, yy[-1] + 0.45]], closed=True, fc=C['yaw'], ec='#333333', alpha=0.85)
    a2.add_patch(ego)
    fx = np.arange(5, 41, 5.0)[:6]
    a2.plot(fx, 0 * fx, 'o', color=C['imit'], ms=5, label='target: logged future poses')
    cx = np.linspace(2, 16, 20)
    a2.plot(cx, yy[-1] + np.tan(psi[-1]) * cx, ':', color=C['onp'], lw=1.6, label='continuing the yaw rate (not taught)')
    arrow(a2, (3, yy[-1] - 0.2), (14, 0.25), color=C['imit'], lw=1.2, rad=-0.15, ms=10)
    a2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, handlelength=1.6)
    a2.text(-17, 4.3, 'BEV sketch (x forward; lateral x 5)', fontsize=8.5, color=C['mute'])
    a2.text(16, -1.4, 'plan returns to the lane', fontsize=8.5, color=C['imit'], ha='center')

    # c: route memory
    fig.text(0.58, 0.47, 'c  route memory: messages carried by the ego poses', fontsize=12, fontweight='bold')
    a3 = fig.add_axes([0.58, 0.06, 0.40, 0.36])
    a3.set_xlim(-30, 80), a3.set_ylim(-12, 24), a3.axis('off')
    s = np.linspace(-30, 80, 300)
    yr = np.where(s < 35, 0, 0.012 * (s - 35) ** 2)
    a3.plot(s, yr, color='#BBBBBB', lw=6, solid_capstyle='round', zorder=0)
    m1 = (s > -26) & (s < 18)
    m2 = (s >= 18) & (s < 64)
    a3.plot(s[m1][::12], yr[m1][::12], 's', color=C['frozen'], mec='#555555', ms=5, label='older message (carried by ego poses)')
    a3.plot(s[m2][::12], yr[m2][::12], '^', color=C['new'], mec='#555555', ms=6, label='newer message (points beyond the end)')
    a3.plot([-28, 0], [0, 0], color=C['yaw'], lw=2, label='ego track')
    a3.add_patch(Rectangle((-2.5, -0.9), 5.0, 1.8, fc=C['yaw'], ec='#333333', zorder=3))
    a3.text(0, -3.2, 'ego', ha='center', fontsize=8.5)
    offs = np.arange(-6, 73, 2)
    ok = offs <= 62
    a3.plot(offs[ok], np.full(ok.sum(), -6.5), '|', color=C['ink'], ms=9, mew=1.3)
    a3.plot(offs[~ok], np.full((~ok).sum(), -6.5), '|', color='#BBBBBB', ms=9, mew=1.3)
    a3.text(-6, -9.5, '-6 m', ha='center', fontsize=8.5), a3.text(72, -9.5, '+72 m', ha='center', fontsize=8.5)
    a3.text(33, -10.8, '40 slots every 2 m; grey = not covered (valid = 0)', ha='center', fontsize=8.5, color=C['mute'])
    a3.text(-29, 23.5, 'per slot [valid, x/20, y/20, cos h, sin h] -> 200 numbers\nmerged polyline = oldest message + newer points beyond its end',
            fontsize=8.8, va='top')
    a3.legend(loc='upper left', bbox_to_anchor=(0.0, 0.7), fontsize=8)
    save(fig, 'fig_prep.png')


# ---------------------------------------------------------------- 3. batch and losses
def fig_batch_loss():
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0, 0, 0.7, 1])
    ax.set_xlim(0, 8.4), ax.set_ylim(0, 7), ax.axis('off')
    ax.text(0.2, 6.7, 'a  one stage-2 batch of 128 rows and the losses each row type carries', fontsize=12, fontweight='bold')
    segs = [('imitation\n~77', 77, C['imit']), ('anchor\n~25', 25, C['anchor']), ('yaw-rate\n13', 13, C['yaw']), ('on-policy\n13', 13, C['onp'])]
    notes = [
        'imitation (normal rows, p = 0.75)\n'
        'Huber(delta 1) on (plan - logged) / sigma, 8 poses x, y, psi   lam_i 1\n'
        'drivable-area hinge on the plan footprint, margin 0.3 m   lam 10\n'
        'other heads -> shipped teacher   lam_d 30;  route memory dropped on 25 %',
        'anchor (normal rows, p = 0.25)\n'
        'new inputs zeroed (present = 0): plan -> shipped plan, same sigma-Huber   lam_c 3\n'
        'every head -> shipped teacher   lam_d 30',
        'yaw-rate (yr1, never anchor)\n'
        'imitation of the logged future in the perturbed frame   lam_i 1\n'
        'drivable hinge in the row\'s own frame   lam 10;  heads   lam_d 30',
        'on-policy `self` (pool on1_L18RB0, stage 2 only)\n'
        'no imitation; agent-box hinge   lam 10  +  road-and-lane raster hinge   lam 10\n'
        'shape-only, each x ho_w 3 / n_imit;  heads   lam_d 30;  memory never dropped',
    ]
    y0, H, bx, bw = 0.35, 5.95, 0.3, 1.1
    yy = y0 + H
    for (lab, n, col), note in zip(segs, notes):
        h = H * n / 128
        yy -= h
        ax.add_patch(Rectangle((bx, yy), bw, h - 0.03, fc=col, ec='white', lw=1.5))
        ax.text(bx + bw / 2, yy + h / 2, lab, ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        yc = yy + h / 2
        ty = {77: 5.15, 25: 3.25, 13: None}[n] if n != 13 else (2.05 if col == C['yaw'] else 0.75)
        ax.add_patch(FancyBboxPatch((2.0, ty - 0.5), 6.3, 1.0, boxstyle='round,pad=0.02,rounding_size=0.06', fc='white', ec=col, lw=1.4))
        ax.text(2.1, ty, note, va='center', fontsize=9.2, linespacing=1.35)
        arrow(ax, (bx + bw, yc), (2.0, ty), color=col, lw=1.0, ms=9)
    ax.text(bx + bw / 2, 0.12, '128 rows', ha='center', fontsize=10)
    ax.text(1.5, 6.38, 'stage 1 (RB0-F-s0): no on-policy rows -> 115 normal + 13 yaw-rate', fontsize=9.2, color=C['mute'])

    T = 0.5 * np.arange(1, 9)
    b1 = fig.add_axes([0.76, 0.56, 0.21, 0.33])
    b1.plot(T, 0.3 + 0.2 * T, '-o', ms=3.5, color=C['imit'], label='sigma_x (m)')
    b1.plot(T, 0.1 + 0.1 * T, '-o', ms=3.5, color=C['yaw'], label='sigma_y (m)')
    b1.set_ylabel('m'), b1.set_xticks([1, 2, 3, 4]), b1.set_ylim(0, 1.2)
    b1.legend(loc='upper left')
    b1.set_title('b  loss scale per pose', loc='left', fontweight='bold')
    b2 = fig.add_axes([0.76, 0.13, 0.21, 0.33])
    b2.plot(T, 1.0 + 1.0 * T, '-o', ms=3.5, color=C['onp'], label='sigma_psi (deg)')
    b2.set_xlabel('horizon t (s)'), b2.set_ylabel('deg'), b2.set_xticks([1, 2, 3, 4]), b2.set_ylim(0, 5.5)
    b2.legend(loc='upper left')
    fig.text(0.76, 0.025, 'error / sigma: far poses are allowed\nlarger errors before the Huber turns linear', fontsize=9, color=C['mute'])
    save(fig, 'fig_batch_loss.png')


# ---------------------------------------------------------------- 4. stages
def lr_curve(steps, warm, peak):
    s = np.arange(steps)
    return s, peak * np.minimum(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * s / steps))


def fig_stages():
    fig, ax = canvas()
    ax.text(0.2, 6.7, 'Two stages from the shipped weights to RSG-L18-s0', fontsize=13, fontweight='bold')
    box(ax, 0.2, 5.0, 1.7, 1.3, 'shipped\nopenpilot\nCinque', fc=C['frozen'], fs=11)
    box(ax, 2.4, 5.0, 3.9, 1.3, 'stage 1  RB0-F-s0\n10 000 steps, batch 128, 22.5 min (1 350 s)\nfresh P2 adapter + route branch at 0',
        fc=C['train'], fs=10)
    box(ax, 7.4, 5.0, 4.4, 1.3, 'stage 2  RSG-L18-s0  (current base)\n2 500 steps from RB0-F-s0, 6.4 min (382 s)\n'
        'same navtrain + yr1 rows (102 + 13) + 13 on-policy', fc=C['onp'], fs=10)
    arrow(ax, (1.9, 5.65), (2.4, 5.65))
    arrow(ax, (6.3, 5.65), (7.4, 5.65))
    ax.text(6.85, 5.8, 'init', ha='center', fontsize=9, color=C['mute'])

    # data row
    box(ax, 2.4, 3.3, 1.85, 1.1, 'navtrain cache\n101 499 train rows\n(imitation + anchor)', fc=C['data'], fs=9)
    box(ax, 4.45, 3.3, 1.85, 1.1, 'yaw-rate rows yr1\n69 572 rows\n13 per batch', fc=C['data'], fs=9)
    for xc in (3.32, 5.37):
        arrow(ax, (xc, 4.4), (xc, 5.0), color=C['mute'], lw=1.1, ms=10)
    box(ax, 7.4, 3.3, 2.05, 1.1, 'AlpaSim nuPlan rollouts\nof RB0-F-s0 itself\nonpol-train, 1 031 scenes', fc='white', fs=8.8, ls='--')
    box(ax, 9.75, 3.3, 2.05, 1.1, 'pool on1_L18RB0\n17 429 rows, 28 logs\n13 per batch', fc=C['data'], fs=9)
    arrow(ax, (9.45, 3.85), (9.75, 3.85), color=C['onp'], lw=1.5)
    arrow(ax, (10.77, 4.4), (10.77, 5.0), color=C['onp'], lw=1.5)
    arrow(ax, (6.3, 5.15), (7.4, 3.95), color=C['onp'], lw=1.5, rad=-0.3)
    ax.text(6.42, 3.45, 'drives in\nAlpaSim', fontsize=8.8, color=C['onp'], ha='left')
    ax.text(0.2, 3.85, 'data per batch', fontsize=10, color=C['mute'])
    ax.text(0.2, 1.6, 'learning rate', fontsize=10, color=C['mute'])

    # lr sketches
    for k, (x0, w, steps, warm, sc, title) in enumerate([(2.4, 3.9, 10000, 300, 1.0, 'stage 1: warm-up 300, cosine to 0'),
                                                          (7.4, 4.4, 2500, 1, 0.1, 'stage 2: warm-up 1, cosine; net and adapter x 0.1')]):
        a = fig.add_axes([x0 / 12, 0.08, w / 12, 0.30])
        for lab, peak, wu, col in [('policy net', 3e-5 * sc, warm, C['train']), ('P2 adapter', 3e-4 * sc, warm, C['new']),
                                   ('route branch (own warm-up 100, never scaled)', 3e-4, 100, C['ink'])]:
            s, l = lr_curve(steps, wu, peak)
            a.plot(s[:-steps // 50], l[:-steps // 50], color=col, lw=1.6, ls='--' if col == C['ink'] else '-', label=lab)
        a.set_yscale('log'), a.set_ylim(1e-7, 1e-3), a.set_xlim(0, steps)
        a.set_title(title, loc='left', fontsize=10.5)
        a.set_xlabel('step', fontsize=9.5, labelpad=1)
        if k == 0:
            a.legend(loc='lower left', fontsize=8.5)
            a.set_ylabel('lr (log scale)')
    save(fig, 'fig_stages.png')


if __name__ == '__main__':
    fig_model()
    fig_prep()
    fig_batch_loss()
    fig_stages()
    print('figures written to', HERE)
