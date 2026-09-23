"""W2b stage partition and immediate post-case invariant checks."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from b2d_tfv6_coordinates import rear_waypoints
from b2d_tfv6_w2b import InvariantFailure, items_for_stage, validate_attempt


class W2bTests(unittest.TestCase):
    def test_stage_partition_has_each_formal_cd_case_once(self):
        stages = [items_for_stage(stage) for stage in ('S1','S2','S3')]
        self.assertEqual([len(s) for s in stages], [7,19,72])
        cd = [x for stage in stages for x in stage if x[3] in 'CD']
        self.assertEqual(len(cd),96)
        self.assertEqual(len(set(cd)),96)

    def test_completed_case_requires_contiguous_frames_and_zero_phantom(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            run = attempt / 'run'
            official = run / 'attempts/3514/1'
            official.mkdir(parents=True)
            p = np.column_stack((np.arange(1,9,dtype=float),np.zeros(8)))
            rear = rear_waypoints(p).tolist()
            frames = [dict(step=i,arm='C',route='3514',waypoint=p.tolist(),
                           rear_waypoint=rear) for i in range(2)]
            def write_frames():
                (attempt/'frames.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in frames))
            write_frames()
            (attempt/'infractions.json').write_text(json.dumps({'infractions':[]}))
            (official/'results.json').write_text(json.dumps({'_checkpoint':{'records':[
                {'scores':{'score_composed':65},'infractions':{}}]}}))
            (run/'summary.json').write_text(json.dumps({'attempts':{'3514':[{'ticks':2}]}}))
            result=dict(level='1',route='3514',seed=1,arm='C',attempt=1,
                        status='finished',official_status='Completed')
            self.assertEqual(validate_attempt(result,attempt,run)['phantom'],0)
            frames[1]['step']=2
            write_frames()
            with self.assertRaisesRegex(InvariantFailure,'I3 frame coverage'):
                validate_attempt(result,attempt,run)
            frames[1]['step']=1
            frames[1]['waypoint']=[[0.,-.08]]*8
            frames[1]['rear_waypoint']=[[1.389,-1.309]]*8
            write_frames()
            with self.assertRaisesRegex(InvariantFailure,'I1 offline phantom'):
                validate_attempt(result,attempt,run)


if __name__=='__main__':
    unittest.main()
