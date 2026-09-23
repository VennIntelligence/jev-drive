#!/usr/bin/env python3
"""Read-only cost accounting from a CLOSED six-group real-TCP campaign."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base, out = args.run_root.resolve(), args.out.resolve()
    if out.exists():
        parser.error('Refusing existing output directory; choose a fresh edition')
    inputs = {}
    def capture(path):
        path = Path(path)
        raw = path.read_bytes()
        inputs[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return raw
    def read(path):
        return json.loads(capture(path))
    events = [json.loads(line) for line in capture(base / 'events.jsonl').splitlines() if line]
    ends = [r for r in events if r['kind'] == 'end']
    if len(ends) != 1 or ends[0].get('status') != 'completed' or ends[0].get('groups') != 6:
        parser.error('Campaign must have one completed end event with six groups')
    manifest = read(base / 'manifest.json')
    provenance = read(base / 'provenance/manifest.json')
    groups = [r for r in events if r['kind'] == 'group_end']
    starts = [r for r in events if r['kind'] == 'group_start']
    expected = [(r, a) for r in manifest['routes'] for a in manifest['arms']]
    if [(r['route'], r['arm']) for r in groups] != expected or len(groups) != 6:
        parser.error('Closed group sequence differs from manifest')
    gpu = capture(base / 'gpu-live.csv').decode()
    for name in ('b2d_tcp_campaign.py', 'b2d_tcp_visual_agent.py', 'b2d_hooks.py', 'b2d_route.py', 'b2d_run.py'):
        capture(base / 'provenance/source/scripts' / name)
    rows = []
    for group in groups:
        route, arm = group['route'], group['arm']
        directory = base / (route + '-' + arm)
        group_events = [json.loads(line) for line in capture(directory / 'events.jsonl').splitlines() if line]
        attempts = sorted((directory / 'attempts' / route).glob('*/attempt.json'), key=lambda p: int(p.parent.name))
        for ap in attempts:
            attempt = read(ap)
            route_starts = [x for x in group_events if x['kind'] == 'route_start' and x.get('attempt') == attempt['attempt']]
            route_ends = [x for x in group_events if x['kind'] == 'route_end' and x.get('attempt') == attempt['attempt']]
            if len(route_starts) != 1 or len(route_ends) != 1:
                raise ValueError('Need one matching route_start/end pair per attempt')
            supervised_wall = route_ends[0]['t'] - route_starts[0]['t']
            result = read(ap.with_name('route_result.json'))
            official = read(ap.with_name('results.json'))['_checkpoint']['records']
            if len(official) != 1:
                raise ValueError('Expected exactly one official route record per attempt')
            record = official[0]
            setup = read(ap.with_name('tcp-setup.json'))
            performance_path = directory / 'live' / ('performance-' + record['save_name'] + '.json')
            performance = read(performance_path)
            profile = result['profile']
            ticks, used = profile['ticks'], profile['ticks_used']
            mean = profile['total_ms_mean']
            used_s = mean * used / 1000.
            full_estimate_s = mean * ticks / 1000.
            telemetry = [json.loads(line) for line in capture(ap.with_name('tcp-control.jsonl')).splitlines() if line]
            row = dict(route=route, arm=arm, attempt=attempt['attempt'],
                       official_status=record['status'], completion=record['scores']['score_route'],
                       driving_score=record['scores']['score_composed'],
                       collision_events=sum(len(v) for k, v in record['infractions'].items() if k.startswith('collisions_')),
                       harness_status=attempt['status'], returncode=attempt.get('returncode'),
                       profile_ticks=ticks, profile_ticks_used=used, discarded_profile_ticks=ticks-used,
                       telemetry_ticks=len(telemetry), network_forward_count=sum(r.get('forward_count', 0) for r in telemetry),
                       tick_mean_ms=mean, tick_median_ms=profile['total_ms_median'], tick_p95_ms=profile['total_ms_p95'],
                       steady_profiled_seconds_approx=used_s,
                       full_ticks_at_steady_mean_seconds_estimate=full_estimate_s,
                       attempt_wall_s=attempt['wall_s'], evaluator_wall_s=result['wall_s'],
                       supervised_route_event_wall_s=supervised_wall,
                       supervised_minus_recorded_attempt_s=supervised_wall-attempt['wall_s'],
                       official_duration_system_s=record['meta']['duration_system'],
                       official_duration_game_s=record['meta']['duration_game'],
                       attempt_minus_profiled_steady_s=attempt['wall_s']-used_s,
                       evaluator_minus_profiled_steady_s=result['wall_s']-used_s,
                       attempt_minus_all_ticks_steady_estimate_s=attempt['wall_s']-full_estimate_s,
                       group_wall_s=group['wall_s'], group_attempts=len(attempts),
                       group_restart_events=sum(r['kind'] == 'server_restart' for r in group_events),
                       server_index=attempt['server_index'], server_log=attempt['server_log'],
                       model_phase_samples=performance['samples'],
                       model_samples_match_profile_used=performance['samples'] == used,
                       cuda_visible_devices=setup['environment'].get('CUDA_VISIBLE_DEVICES'),
                       source_attempt=str(ap.parent), performance_source=str(performance_path))
            for key, value in profile.items():
                if key.endswith(('_ms_mean', '_ms_median', '_ms_p95')):
                    row['profile_' + key] = value
            for phase, values in performance['phases'].items():
                for stat, value in values.items():
                    row['visual_' + phase + '_' + stat] = value
            rows.append(row)
    if len({(r['route'], r['arm']) for r in rows}) != 6:
        raise ValueError('Missing attempt evidence for a closed group')
    attempt_total = sum(r['attempt_wall_s'] for r in rows)
    supervised_total = sum(r['supervised_route_event_wall_s'] for r in rows)
    group_total = sum(g['wall_s'] for g in groups)
    steady_total = sum(r['steady_profiled_seconds_approx'] for r in rows)
    start_event = next(r for r in events if r['kind'] == 'start')
    total = ends[0]['t'] - manifest['started']
    statuses = {}
    for status in sorted(set(r['official_status'] for r in rows)):
        subset = [r for r in rows if r['official_status'] == status]
        statuses[status] = dict(attempts=len(subset), ticks=sum(r['profile_ticks'] for r in subset),
                               supervised_route_event_wall_s=sum(r['supervised_route_event_wall_s'] for r in subset),
                               attempt_wall_s=sum(r['attempt_wall_s'] for r in subset),
                               profiled_steady_s=sum(r['steady_profiled_seconds_approx'] for r in subset))
    summary = dict(schema_version=1, run_root=str(base), source_commit=manifest['git_commit'],
                   manifest_started_unix=manifest['started'], end_unix=ends[0]['t'],
                   groups=len(groups), attempts=len(rows), retries=len(rows)-6,
                   profile_ticks=sum(r['profile_ticks'] for r in rows),
                   profile_ticks_used=sum(r['profile_ticks_used'] for r in rows),
                   network_forward_count=sum(r['network_forward_count'] for r in rows),
                   manifest_to_end_wall_s=total,
                   manifest_to_server_ready_event_s=start_event['t']-manifest['started'],
                   group_wall_s_sum=group_total, attempt_wall_s_sum=attempt_total,
                   supervised_route_event_wall_s_sum=supervised_total,
                   supervised_minus_recorded_attempt_s=supervised_total-attempt_total,
                   group_minus_supervised_route_event_s=group_total-supervised_total,
                   group_minus_attempt_wall_s=group_total-attempt_total,
                   outside_groups_wall_s=total-group_total,
                   profiled_steady_s_approx=steady_total,
                   attempt_minus_profiled_steady_s=attempt_total-steady_total,
                   full_ticks_steady_estimate_s=sum(r['full_ticks_at_steady_mean_seconds_estimate'] for r in rows),
                   post_start_intergroup_accounting_s=total-group_total-(start_event['t']-manifest['started']),
                   server_pids=sorted(set(r['server_pid'] for r in starts)),
                   server_logs=sorted(set(r['server_log'] for r in rows)),
                   all_cuda_visible_devices_1=all(r['cuda_visible_devices']=='1' for r in rows),
                   gpu_snapshot='gpu-live.csv; a single in-run sample, not continuous monitoring',
                   outcomes=statuses,
                   limitations=['End event precedes finally server.stop(); final shutdown is excluded.',
                                'Recorded attempt.wall_s is overwritten by route_result.wall_s in Runner.run_once; it measures evaluator scope, not full supervised process. Event route_start/end intervals recover the larger supervised scope.',
                                'Profile means rounded to .001ms; mean*ticks_used reconstructs retained steady samples approximately.',
                                'Residual includes first20 warmup ticks, map/model/setup, teardown and unprofiled evaluator work; not pure warmup or startup.',
                                'Visual policy/preprocess/GPU event phases overlap; do not sum all phases.',
                                'No separately instrumented PI or native PID-only timing; gpu_ms measures forward CUDA event interval.',
                                'Common envelope removes official low-speed tail in both arms; historical official-tail speedup is not PI gain.',
                                'Real TCP checkpoint inference confirmed; generic attempt policy:none field is not an oracle/no-model run.',
                                'One hardware/run/seed; no full220 extrapolation or causal controller-only cost attribution.'])
    out.mkdir(parents=True)
    shutil.copyfile(__file__, str(out / 'recompute.py'))
    (out / 'gpu-live.csv').write_text(gpu)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / 'attempts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (out / 'inputs-sha256.json').write_text(json.dumps(inputs, indent=2)+'\n')
    table = ['|route|arm|官方结果|ticks|tick mean ms|GPU forward mean ms|attempt s|group s|',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        table.append('|{route}|{arm}|{official_status}|{profile_ticks}|{tick_mean_ms:.3f}|{visual_gpu_ms_mean:.3f}|{attempt_wall_s:.1f}|{group_wall_s:.3f}|'.format(**r))
    failures = [r for r in rows if r['official_status'] != 'Completed']
    failure_wall = sum(r['attempt_wall_s'] for r in failures)
    text = '''# 真实 TCP paired-v2 成本归档

六组均已闭合，官方成功/失败独立列示；harness finished和returncode=0不等于驾驶成功。本文基于实际checkpoint inference，非无模型oracle。源码版本 `{commit}`。

{table}

manifest.started→completed end共 **{total:.3f}s（{minutes:.2f}min）**；六group合计{groups:.3f}s，记录的attempt合计{attempts:.1f}s。**该attempt.wall_s被route_result覆盖，实际是evaluator区间**；独立route_start→route_end事件合计{supervised:.3f}s，才是较完整的监督执行区间，逐例CSV保留两者。profile共{ticks}tick，去掉每例前20tick后{used}tick；保留样本mean×count约{steady:.3f}s。记录attempt减此数的{residual:.3f}s包含warmup、地图/模型载入、setup/teardown和未覆盖工作，不能全叫“启动”或“warmup”。初始manifest→server-ready事件{startup:.3f}s；group外合计{outside:.3f}s。end事件发生在最终server.stop之前，因此这些总数不包括最终关闭server的耗时。

每例CSV保留profile mean/median/p95、world tick/agent/scenario tree等分项，以及visual性能文件实际提供的sensor_wait/preprocess/GPU forward/policy/preview mean和p95。visual去掉前20帧；本轮样本数是否与profile匹配逐例列出。GPU forward是CUDA event间隔，policy/preprocess/GPU范围有重叠，不能把这些均值相加当总耗时。未单独计时PI或native PID，故不编造控制器自身耗时。

本轮{failed_count}例官方失败，合计{failed_ticks}tick和{failed_wall:.1f}s记录attempt/evaluator成本（占该口径总量{failed_share:.2f}%）；这些时间全部保留，可在summary.outcomes和逐例CSV直接复算；不把碰撞后运行当可删除成本。只报告本轮同机观测，不混入paired-v1的前四例或维护中止第五例，也不投射220路线。当前两臂共同移除官方低速尾部；相对历史官方尾部的更少tick不能归为PI加速，B/C差异也含闭环轨迹/速度与场景差异。

本轮实际attempt数{count}、额外attempt数{retries}；group_start记录server PID集合{pids}，server日志集合数{logs}。各setup记录CUDA_VISIBLE_DEVICES均为1：{visible}。[运行中GPU快照](gpu-live.csv)保留物理GPU UUID及采样时renderer/TCP进程；这是一次采样，不声称全程监控。GPU身份请以该CSV为准。

- [全attempt逐例CSV](attempts.csv)
- [汇总与口径](summary.json)
- [原始输入路径/大小/SHA](inputs-sha256.json)
- [复算脚本](recompute.py)

```bash
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python recompute.py --run-root {base} --out /tmp/tcp-paired-v2-cost-replay
```

必须使用新输出目录。脚本先确认end=completed和六组顺序，再读取closed证据；不启动CARLA、不加载模型、不改运行源码。
'''.format(commit=summary['source_commit'], table='\n'.join(table), total=total, minutes=total/60,
           groups=group_total, attempts=attempt_total,supervised=supervised_total,ticks=summary['profile_ticks'],used=summary['profile_ticks_used'],
           steady=steady_total,residual=attempt_total-steady_total,startup=summary['manifest_to_server_ready_event_s'],
           outside=summary['outside_groups_wall_s'],count=len(rows),retries=len(rows)-6,pids=summary['server_pids'],
           logs=len(summary['server_logs']),visible=summary['all_cuda_visible_devices_1'],base=base,
           failed_count=len(failures), failed_ticks=sum(r['profile_ticks'] for r in failures),
           failed_wall=failure_wall, failed_share=100*failure_wall/attempt_total)
    (out / 'README.md').write_text(text)
    output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out / 'output-sha256.json').write_text(json.dumps(output_hashes, indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
