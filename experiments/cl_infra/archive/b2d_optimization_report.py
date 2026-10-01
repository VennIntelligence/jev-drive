"""Summarize short, controlled profiling runs (not driving benchmark scores)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--json', help='Save measured records')
    parser.add_argument('--markdown', help='Save concise timing table')
    args = parser.parse_args()
    records = []
    for run in sorted(Path(args.root).glob('*-*')):
        if not run.is_dir():
            continue
        profiles = list(run.glob('attempts/*/*/route_result.json'))
        phases = list(run.glob('live/performance-*.json'))
        traces = list(run.glob('live/sensor-trace-*.json'))
        if not profiles or not phases or not traces:
            print(run.name, 'pending')
            continue
        profile = json.loads(profiles[-1].read_text())['profile']
        phase = json.loads(phases[-1].read_text())['phases']
        rows = json.loads(traces[-1].read_text())[20:]
        last = Counter()
        delay, lag = {}, {}
        for row in rows:
            sensors = row['sensors']
            if sensors:
                last[max(sensors, key=lambda k: sensors[k]['arrival_ms'])] += 1
            for tag, times in sensors.items():
                delay.setdefault(tag, []).append(times['arrival_ms'])
                lag.setdefault(tag, []).append(times['queue_ms'])
        record = dict(run=run.name, ticks=profile['ticks'],
                              tick_ms=profile['total_ms_mean'],
                              world_ms=profile['world_tick_ms_mean'],
                              copy_ms=profile['copy_ms_mean'],
                              phases=phase, last_sensor=dict(last),
                              arrival_ms={k: statistics.mean(v) for k, v in delay.items()},
                              queue_ms={k: statistics.mean(v) for k, v in lag.items()})
        records.append(record)
        print(json.dumps(record, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2))
    if args.markdown:
        lines = ['# Python-only TCP profiling', '',
                 'Town12 route 1711, 350 ticks per run, first 20 excluded. Windowed GPU 1,',
                 'unchanged four cameras and model policy. Capped profiling, not driving scores.', '',
                 '| Run | ms/tick | ticks/s | sensor wait ms | exposed preprocess ms | GPU ms | preview ms |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for row in records:
            p = row['phases']
            lines.append('| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (
                row['run'], row['tick_ms'], 1000 / row['tick_ms'], p['sensor_wait_ms']['mean'],
                p['preprocess_ms']['mean'], p['gpu_ms']['mean'], p['preview_ms']['mean']))
        lines += ['', 'Exposed preprocessing excludes CPU work overlapped with sensor waiting.']
        if any(r['run'].endswith('-async') for r in records):
            lines += ['Strict: all four cameras block control. Async: only the debug camera is optional.',
                      'Both use the same optimized model preprocessing, sensors and Epic quality.',
                      'Required model inputs remain exact-frame. Display images may lag; overlays match their frames.']
        else:
            lines += ['0/3/5: baseline (previous parallel JPEG implementation).',
                      '1/2: early JPEG + direct BGRA conversion. 4: additionally RGB debug views.',
                      '6/7: additionally prepare the complete model RGB array while waiting.',
                      'All original frame filtering/waits are retained; no decimation or old-frame reuse.']
        lines += ['Runs differ slightly in scene evolution and host timing; this is a short local estimate.', '']
        Path(args.markdown).write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
