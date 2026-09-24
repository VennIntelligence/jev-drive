"""D3 runner must import from the same sparse environment used in tmux."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import json

from b2d_tfv6_d3_semantic import SemanticFailure, SemanticSequence, planner_to_world


class D3LaunchTests(unittest.TestCase):
    def test_runner_help_imports_without_inherited_pythonpath(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['OPENBLAS_CORETYPE'] = 'Barcelona'
        result = subprocess.run([sys.executable, str(root/'scripts/b2d_tfv6_d3.py'), '--help'],
                                cwd=root, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('factorial', result.stdout)
        self.assertIn('kalman', result.stdout)


class D3SemanticTests(unittest.TestCase):
    def setUp(self):
        self.package={'dense':[{'xyz':[float(x),0.,0.]} for x in range(6)],
                      'agent_sparse_world_xy':[[1.,0.],[3.,0.],[5.,0.]],
                      'agent_sparse_dense_indices':[1,3,5],
                      'planner_to_world':{'scale':1.,'rotation':[[1.,0.],[0.,1.]],
                                          'translation_xy':[10.,0.]}}

    def frame(self,ego=0.,target=1.,step=0):
        point=[target-10.,0.,0.]
        return {'step':step,'truth':{'location':[ego,0.,0.]},
                'd3_nav':{'selected_target_planner_xyz':point,'filtered_state':[-10.,0.,0.,0.],
                          'noisy_state':[-10.,0.,0.],'compass_rad':0.,
                          'target_point':[target,0.],'target_point_previous':[target,0.],
                          'target_point_next':[target,0.]}}

    def test_ego_off_route_is_logged_even_as_completion_increases(self):
        checker=SemanticSequence(self.package)
        checker.check(self.frame())
        bad=self.frame(ego=2.,step=1)
        bad['truth']['location'][1]=3.1
        metric=checker.check(bad)
        self.assertTrue(metric['rc_increasing'])
        self.assertGreater(metric['ego_dense_distance_m'],3.)

    def test_target_dense_index_must_be_monotone(self):
        checker=SemanticSequence(self.package)
        checker.check(self.frame(target=3.))
        with self.assertRaisesRegex(SemanticFailure,'regressed'):
            checker.check(self.frame(target=1.,step=1))

    def test_target_must_lie_on_dense_route(self):
        checker=SemanticSequence(self.package)
        bad=self.frame()
        bad['d3_nav']['target_point']=[1.,1.1]
        with self.assertRaisesRegex(SemanticFailure,'local target'):
            checker.check(bad)

    def test_all_route_alignment_residuals(self):
        root=Path('/data/runs/b2d/tfv6-d3/dense')
        for name in ('2-2084','2-27529','1-2091'):
            package=json.loads((root/f'{name}.json').read_text())
            transform=package['planner_to_world']
            residuals=[sum((a-b)**2 for a,b in zip(planner_to_world(p,transform),w))**.5
                       for p,w in zip(package['agent_sparse_planner_xy'],package['agent_sparse_world_xy'])]
            self.assertLess(max(residuals),.1,name)
            self.assertLess(abs(transform['scale']-1),.001,name)
            self.assertLess(abs(transform['rotation_rad']),.001,name)


if __name__ == '__main__':
    unittest.main()
