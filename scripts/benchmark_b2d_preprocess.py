"""Compare against the installed official tick source, including exact output pixels."""
import ast
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from b2d_tcp_preprocess import model_rgb


def main():
    source = Path('/data/third_party/Bench2DriveZoo/team_code/tcp_b2d_agent.py').read_text()
    tree = ast.parse(source)
    agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TCPAgent')
    tick = next(n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == 'tick')
    namespace = dict(cv2=cv2, np=np, torch=torch, math=math)
    exec(compile(ast.Module(body=[tick], type_ignores=[]), '<official TCP tick>', 'exec'), namespace)
    wrapper_tree = ast.parse(Path(__file__).with_name('b2d_tcp_visual_agent.py').read_text())
    wrapper = next(n for n in wrapper_tree.body if isinstance(n, ast.ClassDef))
    wrapper_tick = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == 'tick')
    fast_namespace = dict(namespace, time=time, islice=islice, model_rgb=model_rgb, os=os)
    exec(compile(ast.Module(body=[wrapper_tick], type_ignores=[]), '<parallel wrapper tick>', 'exec'), fast_namespace)
    fake = SimpleNamespace(step=0, gps_to_location=lambda gps: gps,
                           _route_planner=SimpleNamespace(route=[(np.array([10., 2.]), None)],
                               run_step=lambda pos: (pos + np.array([10., 2.]), SimpleNamespace(value=4))))
    torch.set_num_threads(4)
    rng = np.random.RandomState(42)
    inputs = {k: (1, rng.randint(0, 256, (900, 1600, 4), dtype=np.uint8))
              for k in ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT')}
    inputs.update(bev=(1, np.zeros((360, 640, 4), dtype=np.uint8)),
                  GPS=(1, np.zeros(3)), IMU=(1, np.zeros(7)), SPEED=(1, {'speed': 1.0}))
    samples = {'official_tick': [], 'parallel_rgb': []}
    with ThreadPoolExecutor(max_workers=3) as pool:
        fake._image_pool = pool
        for i in range(35):
            inputs['IMU'][1][-1] = float('nan') if i == 0 else i / 10
            start = time.perf_counter()
            expected = namespace['tick'](fake, inputs)
            t1 = time.perf_counter()
            actual = fast_namespace['tick'](fake, inputs)
            end = time.perf_counter()
            assert expected.keys() == actual.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
            if i >= 5:
                samples['official_tick'].append((t1 - start) * 1000)
                samples['parallel_rgb'].append((end - t1) * 1000)
        for key, values in samples.items():
            print(key, 'mean_ms', np.mean(values), 'p95_ms', np.percentile(values, 95), flush=True)
        print('PASS: 35 exact comparisons of ALL tick outputs against official source, synthetic 1600x900 inputs, varying compass including NaN', flush=True)


if __name__ == '__main__':
    main()
