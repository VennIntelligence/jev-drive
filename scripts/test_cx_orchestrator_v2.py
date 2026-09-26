"""Resource regressions from the live independent queue; no scientific mocks."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import cx_orchestrator_v2 as q

class CapacityTests(unittest.TestCase):
    def probe(self):
        return dict(cores_used=91.6, pids=16316, pids_max=20480, gpus=[
            dict(gpu=1, used_gb=30, total_gb=96, util=0, carla=4),
            dict(gpu=5, used_gb=50, total_gb=96, util=15, carla=5)])
    def rows(self):
        return [dict(workers='16', gpus='5', status='batch')]
    def test_ready_cpu_no_longer_waits_for_impossible_carla_peak(self):
        gpu, _ = q.budget(dict(pids=64, gpu_gb=0), [192,193], [], self.probe(), self.rows())
        self.assertEqual(gpu, -1)
    def test_hard_thread_headroom_still_blocks(self):
        p=self.probe();p['pids']=19500
        self.assertIsNone(q.budget(dict(pids=128,gpu_gb=0),[192,193],[],p,[])[0])
    def test_q5_launch_alongside_adopted_scoring_and_fit(self):
        active=[dict(pids=384,resident_pids=900,started=0),dict(pids=128,resident_pids=30,started=0)]
        self.assertEqual(q.budget(dict(pids=1000,gpu_gb=0),list(range(10)),active,self.probe(),self.rows())[0],-1)
    def test_gpu_resident_memory_not_reserved_twice(self):
        p=self.probe();p['gpus'][1]['used_gb']=60
        active=[dict(pids=128,resident_pids=128,started=0,gpu=5,gpu_gb=20,resident_gpu_gb=20)]
        self.assertEqual(q.budget(dict(pids=128,gpu_gb=20),[192,193],active,p,[])[0],5)
        active[0]['resident_gpu_gb']=0
        self.assertIsNone(q.budget(dict(pids=128,gpu_gb=20),[192,193],active,p,[])[0])
    def test_dynamic_core_choice_preserves_live_score_and_watcher(self):
        job=dict(cpu_pool='142-149,180-203',min_cores=10,max_cores=10)
        active=[dict(cpus='180-191',identity=dict(pgid=100))]
        procs=[dict(pgid=200,cores=[196]),dict(pgid=201,cores=list(range(134,142)))]
        with patch.object(q.os,'sched_getaffinity',create=True,return_value=set(range(208))):
            cores=q.choose_cores(job,active,procs)
        self.assertEqual(len(cores),10)
        self.assertFalse(set(cores)&set(range(180,192)))
        self.assertNotIn(196,cores)
    def test_detached_ray_threads_accounted_once(self):
        active=[dict(cpus='180-191',identity=dict(pgid=100))]
        procs=[dict(pid=1,pgid=100,threads=3,cores=list(range(208))),dict(pid=2,pgid=999,threads=200,cores=[180,181])]
        measured=q.resident(active,procs,[(2,4)])
        self.assertEqual(measured[0]['resident_pids'],203)
        self.assertEqual(measured[0]['resident_gpu_gb'],4)
    def test_successor_readopts_actual_launch_command(self):
        with tempfile.TemporaryDirectory() as temp:
            launch=Path(temp)/'launch.json'
            job=dict(id='q5-child',command='SCORE_THREADS=6 original-child')
            launch.write_text(json.dumps(dict(jobs=[job])))
            old={'q5-child':dict(command='SCORE_THREADS=14 original-child')}
            q.validate_adoption(job,dict(launch=str(launch)),old)
            with self.assertRaises(AssertionError):
                q.validate_adoption(dict(job,command='different-scientific-command'),dict(launch=str(launch)),old)
    def test_manifest_keeps_scoring_command_and_fit_gate(self):
        old={j['id']:j for j in json.loads(q.base.DEFAULT.read_text())['jobs']}
        new={j['id']:j for j in json.loads(q.DEFAULT.read_text())['jobs']}
        self.assertEqual(old['q6-rt-score']['command'],new['q6-rt-score']['command'])
        for s in range(3):self.assertEqual(new[f'q6-sf-fit-s{s}']['after'],['q6-refit-check'])
        self.assertEqual(len(new['q6-sf-report']['after']),3)
        self.assertEqual(new['q5-child']['command'],old['q5-child']['command'].replace('SCORE_THREADS=14','SCORE_THREADS=6'))

if __name__=='__main__':unittest.main()
