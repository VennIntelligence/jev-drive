"""Pre-register held-out turns for lateral v2 from Bench2Drive route geometry only (no driving).

Each bench2drive220 route is densified exactly like the evaluator (GlobalRoutePlanner, 1 m hops,
leg by leg between XML waypoints) on a client-side carla.Map built from the town's OpenDRIVE, so no
simulator runs. Turns are found on the dense path's curvature; the fixed rule below picks the set;
each pick becomes a trimmed route (XML waypoints from core start - 40 m to core end + 45 m) whose
windows are stations on that trimmed route's own dense reference.

    DATA_DIR=... envs/carla/bin/python select_heldout.py --out <dir>      # on the GPU box
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))

FROZEN = {'24240', '26966', '17563'}           # development routes: never held out
HOP = 1.0                                      # interpolate_trajectory hop resolution
STEP = .5                                      # resampling step for curvature, m
CHORD = 5.                                     # heading chord (analysis uses the same 5 m centred chord)
KAPPA_ON = 1 / 40.                             # |kappa| threshold of a curve segment, 1/m
MERGE_GAP = 3.                                 # same-sign segments closer than this merge, m
TURN_MIN_DEG = 60.                             # single turn: |heading change| >= this
S_ARM_MIN_DEG = 20.                            # S: each opposite-sign arm >= this ...
S_GAP_MAX = 12.                                # ... with at most this straight between the arms, m
S_KAPPA_MIN = 1 / 15.                          # ... and a peak |kappa| >= this (excludes lane changes)
CLEAN_BEFORE = 25.                             # no other >= 30 deg curve this far before the core, m
LEAD, TAIL = 25., 35.                          # need >= LEAD before the core (start-up) and >= TAIL after
                                               # it (pad, 10 m post window, stop); longer routes are trimmed
TRIM_BEFORE, TRIM_AFTER = 40., 45.             # kept route around the core (shorter routes kept whole), m
FROZEN_WINDOWS = {'24240': [(23., 59.)], '26966': [(29., 46.)], '17563': [(32.5, 43.5), (78.5, 90.)]}
FROZEN_RADIUS = 15.                            # a core within this of a development core is not held out, m
PAD = 5.                                       # window = core padded by 5 m each side (as frozen windows)
A_LAT_MAX = 9.5                                # cruise v allowed if v^2 / R_min <= this, m/s^2
QUOTA = [('right', 3), ('left', 3), ('S', 2)]  # held-out set size per class


def dense_route(grp, carla, positions):
    xy = []
    for a, b in zip(positions, positions[1:]):
        for wp, _ in grp.trace_route(carla.Location(*a), carla.Location(*b)):
            xy.append((wp.transform.location.x, wp.transform.location.y))
    xy = np.asarray(xy, float)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-5]
    return xy[keep]


def profile(xy):
    """Stations, heading (CARLA world, rad) and left-positive curvature on a STEP-resampled path."""
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    grid = np.arange(0., s[-1], STEP)
    p = np.column_stack([np.interp(grid, s, xy[:, i]) for i in (0, 1)])
    half = int(round(CHORD / 2 / STEP))
    ahead = p[np.minimum(np.arange(len(p)) + half, len(p) - 1)]
    behind = p[np.maximum(np.arange(len(p)) - half, 0)]
    heading = np.unwrap(np.arctan2(ahead[:, 1] - behind[:, 1], ahead[:, 0] - behind[:, 0]))
    # CARLA world is y-south, so heading grows clockwise (to the right): left-positive kappa = -d(psi)/ds.
    kappa = -np.gradient(heading, grid)
    return grid, heading, kappa


def segments(grid, kappa):
    """Maximal same-sign runs with |kappa| >= KAPPA_ON, merged across gaps < MERGE_GAP."""
    sign = np.where(kappa >= KAPPA_ON, 1, np.where(kappa <= -KAPPA_ON, -1, 0))
    runs, i = [], 0
    while i < len(sign):
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(sign) and sign[j + 1] == sign[i]:
            j += 1
        if runs and runs[-1][2] == sign[i] and grid[i] - grid[runs[-1][1]] < MERGE_GAP:
            runs[-1][1] = j
        else:
            runs.append([i, j, int(sign[i])])
        i = j + 1
    out = []
    for i, j, sgn in runs:
        turn = float(np.trapz(kappa[i:j + 1], grid[i:j + 1]))
        out.append(dict(start=float(grid[i]), end=float(grid[j]), sign=sgn, turn_deg=math.degrees(turn),
                        peak_kappa=float(np.max(np.abs(kappa[i:j + 1])))))
    return out


def candidates(route_id, town, grid, kappa, xy=None, frozen_xy=()):
    segs = segments(grid, kappa)
    found = []
    for k, seg in enumerate(segs):
        if abs(seg['turn_deg']) >= TURN_MIN_DEG:
            found.append(dict(cls='left' if seg['sign'] > 0 else 'right', core=[seg['start'], seg['end']],
                              turn_deg=seg['turn_deg'], peak_kappa=seg['peak_kappa'], parts=[k]))
        if k + 1 < len(segs):
            a, b = seg, segs[k + 1]
            if (a['sign'] == -b['sign'] and min(abs(a['turn_deg']), abs(b['turn_deg'])) >= S_ARM_MIN_DEG
                    and b['start'] - a['end'] <= S_GAP_MAX
                    and max(a['peak_kappa'], b['peak_kappa']) >= S_KAPPA_MIN
                    and max(abs(a['turn_deg']), abs(b['turn_deg'])) < TURN_MIN_DEG):
                found.append(dict(cls='S', core=[a['start'], b['end']], turn_deg=[a['turn_deg'], b['turn_deg']],
                                  peak_kappa=max(a['peak_kappa'], b['peak_kappa']), parts=[k, k + 1]))
    usable = []
    for c in found:
        c.update(route_id=route_id, town=town)
        r_min = 1. / c['peak_kappa']
        c['r_min_m'] = r_min
        c['cruise_mps'] = 8. if 64. / r_min <= A_LAT_MAX else (6. if 36. / r_min <= A_LAT_MAX else None)
        before = [s for i, s in enumerate(segs) if i not in c['parts'] and abs(s['turn_deg']) >= 30.
                  and c['core'][0] - CLEAN_BEFORE <= s['end'] <= c['core'][0]]
        near_frozen = False
        if xy is not None:
            s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            core = np.column_stack([np.interp(np.linspace(*c['core'], 9), s, xy[:, i]) for i in (0, 1)])
            c['core_xy'] = core.round(2).tolist()
            near_frozen = any(np.min(np.linalg.norm(core[:, None] - f[None], axis=2)) < FROZEN_RADIUS
                              for f in frozen_xy)
        c['reject'] = ('too_tight' if c['cruise_mps'] is None else
                       'no_lead' if c['core'][0] < LEAD else
                       'no_tail' if grid[-1] - c['core'][1] < TAIL else
                       'unclean_entry' if before else
                       'overlaps_development_window' if near_frozen else None)
        usable.append(c)
    return usable


def select(pool):
    """Fixed rule: classes round-robin right, left, S; within a class the lowest numeric route id
    whose town is not yet in the set; if every such town is used, the lowest id overall.
    One turn per route; each route's earliest qualifying turn of that class. A core within
    FROZEN_RADIUS of an already chosen core (same town) is the same place and is skipped."""
    chosen, need = [], dict(QUOTA)
    ordered = sorted((c for c in pool if c['reject'] is None), key=lambda c: (int(c['route_id']), c['core'][0]))
    while any(need.values()):
        progress = False
        for cls, _ in QUOTA:
            if not need[cls]:
                continue
            used_routes = {c['route_id'] for c in chosen}
            used_towns = {c['town'] for c in chosen}
            options = [c for c in ordered if c['cls'] == cls and c['route_id'] not in used_routes
                       and not any(o['town'] == c['town'] and np.min(np.linalg.norm(
                           np.asarray(o['core_xy'])[:, None] - np.asarray(c['core_xy'])[None], axis=2)) < FROZEN_RADIUS
                           for o in chosen)]
            fresh = [c for c in options if c['town'] not in used_towns]
            pick = (fresh or options or [None])[0]
            if pick is not None:
                chosen.append(pick)
                need[cls] -= 1
                progress = True
        if not progress:
            raise RuntimeError('not enough held-out turns for %s' % need)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from b2d_run import BENCH2DRIVE, CARLA_ROOT
    from b2d_route import add_bench2drive_to_path
    add_bench2drive_to_path(str(BENCH2DRIVE))
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = BENCH2DRIVE / 'leaderboard/data/bench2drive220.xml'
    root = ET.parse(str(source)).getroot()
    maps = {}
    pool, planners, frozen_xy = [], {}, {}

    def planner(town):
        if town not in planners:
            xodr = next((p for p in [CARLA_ROOT / 'CarlaUE4/Content/Carla/Maps/OpenDrive' / (town + '.xodr'),
                                     CARLA_ROOT / 'CarlaUE4/Content/Carla/Maps' / town / 'OpenDrive' / (town + '.xodr')]
                         if p.exists()))
            maps[town] = carla.Map(town, xodr.read_text())
            planners[town] = GlobalRoutePlanner(maps[town], HOP)
            print('map', town, flush=True)
        return planners[town]

    def positions_of(route):
        return [(float(p.get('x')), float(p.get('y')), float(p.get('z'))) for p in route.findall('./waypoints/position')]

    for route in root.findall('route'):
        if route.get('id') in FROZEN:
            xy = dense_route(planner(route.get('town')), carla, positions_of(route))
            s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            for lo, hi in FROZEN_WINDOWS[route.get('id')]:
                frozen_xy.setdefault(route.get('town'), []).append(
                    np.column_stack([np.interp(np.linspace(lo, hi, 9), s, xy[:, i]) for i in (0, 1)]))
    for route in root.findall('route'):
        rid, town = route.get('id'), route.get('town')
        if rid in FROZEN:
            continue
        xy = dense_route(planner(town), carla, positions_of(route))
        grid, _, kappa = profile(xy)
        pool.extend(candidates(rid, town, grid, kappa, xy, frozen_xy.get(town, ())))
    (out / 'candidates.json').write_text(json.dumps(pool, indent=1))
    chosen = select(pool)
    # Trim each chosen route and re-derive its windows on the trimmed route's own dense reference.
    trimmed_root = ET.Element('routes')
    windows = []
    for c in chosen:
        route = next(r for r in root.findall('route') if r.get('id') == c['route_id'])
        positions = positions_of(route)
        full = dense_route(planners[c['town']], carla, positions)
        s_full = np.r_[0., np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))]
        station = [float(s_full[np.argmin(np.linalg.norm(full - p[:2], axis=1))]) for p in np.asarray(positions)]
        lo, hi = c['core'][0] - TRIM_BEFORE, c['core'][1] + TRIM_AFTER
        keep = [i for i, s in enumerate(station) if lo <= s <= hi]
        keep = list(range(max(0, keep[0] - 1), min(len(positions), keep[-1] + 2)))
        element = copy.deepcopy(route)
        wps = element.find('waypoints')
        for p in list(wps):
            wps.remove(p)
        for i in keep:
            ET.SubElement(wps, 'position', dict(x='%.1f' % positions[i][0], y='%.1f' % positions[i][1],
                                                z='%.1f' % positions[i][2]))
        scen = element.find('scenarios')
        if scen is not None:
            element.remove(scen)
        ET.SubElement(element, 'scenarios')
        trimmed_root.append(element)
        trimmed = dense_route(planners[c['town']], carla, [positions[i] for i in keep])
        grid, _, kappa = profile(trimmed)
        again = [d for d in candidates(c['route_id'], c['town'], grid, kappa) if d['cls'] == c['cls']]
        match = min(again, key=lambda d: abs(d['core'][0] - (c['core'][0] - station[keep[0]])))
        core = [round(match['core'][0], 1), round(match['core'][1], 1)]
        windows.append(dict(route_id=c['route_id'], town=c['town'], cls=c['cls'], cruise_mps=c['cruise_mps'],
                            turn_deg=match['turn_deg'], r_min_m=round(1 / match['peak_kappa'], 2),
                            core_m=core, window_m=[round(core[0] - PAD, 1), round(core[1] + PAD, 1)],
                            trimmed_length_m=round(float(np.sum(np.linalg.norm(np.diff(trimmed, axis=0), axis=1))), 1),
                            reference_xy=trimmed.round(3).tolist()))
    ET.ElementTree(trimmed_root).write(str(out / 'heldout-routes.xml'), encoding='utf-8', xml_declaration=True)
    (out / 'heldout-windows.json').write_text(json.dumps(windows, indent=1))
    (out / 'heldout-cruises.json').write_text(json.dumps({w['route_id']: w['cruise_mps'] for w in windows}, indent=1))
    for w in windows:
        print(w['route_id'], w['town'], w['cls'], w['cruise_mps'], w['turn_deg'], w['r_min_m'], w['core_m'],
              w['trimmed_length_m'], flush=True)


if __name__ == '__main__':
    main()
