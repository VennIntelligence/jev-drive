"""Contact sheet and mp4 from the lateral v2 chase-camera render pass (GPU box, envs/jevdrive).

Each render case is a separate validate run of 26966 / p00 with --chase-camera. Keyframes are the
frames nearest to fixed stations through the turn; the render's CTE trace is compared frame by frame
with the campaign's p00 case of the same arm, so a render that drove differently is reported.

    python render.py --render <render root> --campaign <campaign root> --out <dir>
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / 'research'))
import analyze  # noqa: E402
import plot_style  # noqa: E402

ARMS = [('prod-kfix', 'Production (fixed k)'), ('slipack-kfix', 'Rear-slip pursuit + Ackermann (fixed k)')]
STATIONS = [24., 30., 35., 40., 46., 51.]
ROUTE = '26966'


def case(root, arm):
    found = sorted(Path(root).glob('*/p00/%s/%s/pursuit' % (ROUTE, arm)))
    if len(found) != 1:
        raise SystemExit('expected one %s case under %s, found %d' % (arm, root, len(found)))
    return found[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--render', type=Path, required=True, help='root holding one render validate run per arm')
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_style.apply()
    sheet, parity = [], {}
    for arm, label in ARMS:
        rendered, reference = case(args.render, arm), case(args.campaign, arm)
        f, g = analyze.case_frames(rendered), analyze.case_frames(reference)
        n = min(len(f['cte']), len(g['cte']))
        parity[arm] = dict(frames=[len(f['cte']), len(g['cte'])],
                           max_abs_cte_diff_m=float(np.max(np.abs(f['cte'][:n] - g['cte'][:n]))),
                           window_cte_rms=[analyze.window_metrics(x, 24., 51.)['cte_rms'] for x in (f, g)])
        images = sorted((args.render / 'chase').glob('**/%s/%s/*.jpg' % (ROUTE, arm)))
        by_frame = {int(p.stem): p for p in images}
        row = []
        for station in STATIONS:
            k = int(np.argmin(np.abs(f['s'] - station)))
            path = by_frame.get(int(f['frame'][k]))
            row.append((station, f['cte'][k], path))
        sheet.append((label, row))
        first = cv2.imread(str(images[0]))
        writer = cv2.VideoWriter(str(args.out / ('26966-%s.mp4' % arm)), cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                 (first.shape[1], first.shape[0]))
        for p in images:
            writer.write(cv2.imread(str(p)))
        writer.release()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(STATIONS), figsize=(plot_style.DOUBLE_COLUMN_IN, 2.35))
    for r, (label, row) in enumerate(sheet):
        for c, (station, cte, path) in enumerate(row):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title('s = %.0f m, CTE %+.2f m' % (station, cte), fontsize=6.5, pad=2)
            if c == 0:
                ax.set_ylabel(label.replace(' + ', ' +\n').replace(' (', '\n('), fontsize=6.5)
    fig.subplots_adjust(left=.075, right=.995, top=.93, bottom=.01, wspace=.03, hspace=.18)
    fig.savefig(str(args.out / '26966-contact-sheet.png'), dpi=220)
    (args.out / 'render-parity.json').write_text(json.dumps(parity, indent=1))
    print(json.dumps(parity, indent=1))


if __name__ == '__main__':
    main()
