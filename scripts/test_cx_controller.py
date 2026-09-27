import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cx_controller as m
import sch_table as sch


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
    def controller(self, jobs, **extra):
        return m.Controller(self.data, dict(jobs=jobs, deadline='2099-01-01T00:00:00+00:00', **extra))
    def write(self, relative, value='ok'):
        p = self.data / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(value); return p
    def gpu(self, pids=()):
        return {0:dict(pids=list(pids),used_mb=0,total_mb=96000)}
    def proc(self, pid=42, argv=None, start='11'):
        return dict(pid=pid,start_ticks=start,pgid=pid,sid=pid,ppid=1,argv=argv or ['job'])
    def test_failure_releases_resources_but_does_not_complete_dependency(self):
        a = dict(id='A',process_match=[['post']],done=['$DATA_DIR/A/DONE'],artifacts=['$DATA_DIR/A/result'],errors=['$DATA_DIR/A/ERROR'])
        c = self.controller([a,dict(id='C',after=['A'],disabled='A dependent'),dict(id='independent')])
        c.tick({42:self.proc(argv=['post'])},self.gpu([42]))
        self.write('A/ERROR','post failed')
        c.tick({},self.gpu())
        self.assertTrue(c.state['jobs']['A']['resources_released'])
        self.assertFalse(c.state['jobs']['A']['complete'])
        self.assertEqual(c.state['jobs']['A']['status'],'QUARANTINED')
        self.assertEqual(c.state['jobs']['C']['status'],'BLOCKED_HUMAN')
        self.assertEqual(c.state['jobs']['independent']['status'],'PENDING')
        self.assertEqual(c.state['queue']['free_gpus'],[0])
    def test_unknown_never_reclaims_and_resets_absence(self):
        c=self.controller([dict(id='B',retired=True,schedule_lane='B')])
        c.state['jobs']['B']=dict(resources_released=True,complete=True,owner_absent_since=1)
        p=self.data/'runs/sched/table.tsv';p.parent.mkdir(parents=True,exist_ok=True);sch.TABLE=p
        sch.save([dict(zip(sch.COLS,['B','0','6','0','30','0','batch','-']))])
        with patch.object(c,'observe',side_effect=PermissionError('partial read')): c.tick({},self.gpu())
        self.assertEqual(sch.load()[0]['status'],'batch')
        self.assertFalse(c.state['jobs']['B']['complete'])
        with patch.object(m,'process_snapshot',side_effect=OSError('node unreachable')): c.tick()
        self.assertEqual(c.state['node'],'UNKNOWN')
    def test_reused_pid_is_not_alive(self):
        self.assertFalse(m.same(m.identity(self.proc()),{42:self.proc(start='12')}))
    def test_orphan_runner_is_adopted_without_parent(self):
        job=dict(id='B',output_root='$DATA_DIR/runs/nq3/b/arms')
        c=self.controller([job]);p=self.proc(argv=['python','b2d_run.py','--out',str(self.data/'runs/nq3/b/arms/blue/s0')])
        c.tick({42:p},self.gpu([42]))
        self.assertFalse(c.state['jobs']['B']['resources_released'])
    def test_partial_result_unknown_and_glob_complete(self):
        j=dict(id='combination',done=['$DATA_DIR/DONE'],result='$DATA_DIR/result.json',artifacts=['$DATA_DIR/chunk*.csv'])
        self.write('DONE');self.write('chunk1.csv');self.write('result.json','{')
        c=self.controller([j]);c.tick({},self.gpu())
        self.assertEqual(c.state['jobs']['combination']['status'],'UNKNOWN')
        self.write('result.json',json.dumps(dict(status='DONE',rc=0)));c.tick({},self.gpu())
        self.assertTrue(c.state['jobs']['combination']['complete'])
    def test_restart_state_and_duplicate_owner_lock(self):
        c=self.controller([]);c.acquire();self.addCleanup(lambda:[f.close() for f in c.locks]);c.tick({},self.gpu())
        other=self.controller([])
        with self.assertRaises(RuntimeError): other.acquire()
        self.assertEqual(other.state['queue']['status'],'NO_READY_WORK')
    def test_deadline_preserves_processes(self):
        c=self.controller([dict(id='B',process_match=[['job']])]);c.config['deadline']='2000-01-01T00:00:00+00:00'
        c.tick({42:self.proc()},self.gpu([42]))
        self.assertEqual(c.state['jobs']['B']['status'],'RUNNING')
        self.assertEqual(c.state['queue']['status'],'DEADLINE_NO_NEW_CONTROLLER_LAUNCH')
    def test_b_ack_requires_current_output_runners_and_gpu(self):
        c=self.controller([dict(id='B')],b_handoff=dict(gpus=[0],workers=6,ack_timeout_s=300))
        self.write('runs/nq3/b/CURRENT','blue 0')
        output=str(self.data/'runs/nq3/b/arms/blue/s0')
        p=self.proc(argv=['bash','nq3_b_cl10.sh','blue','0','6','300','0',output])
        c.state['jobs']['B']={'processes':[m.identity(p)]}
        c.audit_b({42:p},self.gpu())
        self.assertNotEqual(c.state['b_handoff']['status'],'RUNNER_AND_GPU_ACK')
        c.audit_b({42:p},self.gpu([42]))
        self.assertEqual(c.state['b_handoff']['status'],'RUNNER_AND_GPU_ACK')
        self.write('runs/nq3/b/CURRENT','simlingo 0');c.audit_b({42:p},self.gpu([42]))
        self.assertNotEqual(c.state['b_handoff']['status'],'RUNNER_AND_GPU_ACK')
    def test_scientific_gates_never_created(self):
        c=self.controller([dict(id='GK',disabled='human gate'),dict(id='OPL',disabled='prohibited')])
        c.tick({},self.gpu())
        self.assertTrue(all(j['status']=='BLOCKED_HUMAN' for j in c.state['jobs'].values()))
        self.assertFalse((self.data/'runs/nq4/gk/prep/DONE').exists())
    def test_ready_queue_is_fixed_and_does_not_create_pass(self):
        self.write('runs/nq3/b/QUEUE','blue 0 all 3\nsimlingo 0 all 8\n')
        self.write('runs/sched/pilot/b/arms/blue/s0/verdict.json','{"verdict":"PASS"}')
        c=self.controller([],b_handoff=dict(gpus=[0],workers=6))
        self.assertEqual(c.b_ready_queue(),['B:blue:0'])
        self.assertFalse((self.data/'runs/sched/pilot/b/arms/simlingo/s0/verdict.json').exists())

    def test_b_grant_expands_registered_old_row_and_preserves_extras(self):
        cfg=dict(gpus=[0,2],workers=6,cpus='0-3',allowed_from_cpus=['0-1','0-3'],idx0='0:300,2:360')
        c=self.controller([dict(id='A'),dict(id='B')],b_handoff=cfg)
        c.state['jobs']={'A':dict(resources_released=True,status='QUARANTINED'), 'B':dict(status='RUNNING')}
        go=self.write('runs/nq3/b/GO','GPUS="0"\nWORKERS=9\n# --- lane extras\nCUSTOM_KEEP=yes\nB_IDX="0:300 2:360"\n')
        sch.TABLE=self.data/'runs/sched/table.tsv'
        sch.save([dict(zip(sch.COLS,['nq3-b','0','9','0:300,2:360','30','0-1','batch',str(go)]))])
        with patch.object(c,'capacity',return_value=(True,{'projected_threads':4800})), patch.object(m.os,'sched_getaffinity',return_value=set(range(4)),create=True):
            c.b_grant({},self.gpu())
        self.assertEqual(sch.load()[0]['gpus'],'0,2')
        self.assertEqual(sch.load()[0]['cpus'],'0-3')
        self.assertIn('CUSTOM_KEEP=yes',go.read_text())
        self.assertIn("B_EXPAND_GPUS='0 2'",go.read_text())
        self.assertTrue(c.state['b_grant_adopted'])
    def test_unknown_resident_gpu_prevents_reclaim(self):
        c=self.controller([dict(id='A',schedule_lane='nq3-a',retired=True)])
        c.state['jobs']['A']=dict(resources_released=True,status='COMPLETE',complete=True)
        sch.TABLE=self.data/'runs/sched/table.tsv'
        sch.save([dict(zip(sch.COLS,['nq3-a','0','6','0','30','0','batch','-']))])
        c.reclaim(self.gpu([99]))
        self.assertEqual(sch.load()[0]['status'],'batch')
    def test_owner_loss_retry_budget_and_unknown_gap(self):
        c=self.controller([dict(id='B')],b_handoff=dict(gpus=[0],workers=6))
        c.state['jobs']['B']=dict(ever_adopted=True,processes=[],owner_absent_since=1,owner_loss_retries=1)
        with patch.object(m.subprocess,'run') as run: c.b_recover({},100)
        run.assert_not_called()
        self.assertIn('B:owner_loss_retry_exhausted',c.state['alerts'])
        c.state['jobs']['B']['processes']=[dict(pid=42)]
        c.b_recover({},101)
        self.assertNotIn('owner_absent_since',c.state['jobs']['B'])

    def test_authorized_p3_adopts_independent_runner_and_reserves_without_go(self):
        job=dict(id='P3',authorization='USER_AUTHORIZED_BUILD_AND_SMOKE',
                 process_match=[['scripts/p3/gpu_enable.py']],
                 output_roots=['$DATA_DIR/processed/waymo_ds','$DATA_DIR/ckpt/nq4_p3_short3000'],
                 resource_claims=dict(gpus=[1,6],cpus='180-189',prep_cpus='182-189',scope='build_scene0_smoke'))
        c=self.controller([job]);sch.TABLE=self.data/'runs/sched/table.tsv'
        sch.save([dict(zip(sch.COLS,['nq4-p3','-','-','-','-','-','waiting','$DATA_DIR/runs/nq4/p3/GO']))])
        # The independent prep output can equal its registered root exactly.
        p=self.proc(argv=['python','scripts/p3/ds.py','prep','--out',str(self.data/'processed/waymo_ds')])
        g={1:dict(pids=[],used_mb=0,total_mb=96000),6:dict(pids=[],used_mb=0,total_mb=96000)}
        c.tick({42:p},g)
        self.assertEqual(c.state['jobs']['P3']['status'],'RUNNING')
        self.assertEqual(c.state['jobs']['P3']['authorization'],'USER_AUTHORIZED_BUILD_AND_SMOKE')
        self.assertEqual(c.state['queue']['reserved_gpus'],[1,6])
        self.assertEqual(c.state['queue']['free_gpus'],[])
        self.assertEqual(c.state['queue']['physically_idle_gpus'],[1,6])
        self.assertEqual(sch.load()[0]['gpus'],'1,6')
        self.assertFalse((self.data/'runs/nq4/p3/GO').exists())
        c.tick({},g)
        self.assertEqual(c.state['jobs']['P3']['status'],'AUTHORIZED_AWAITING_OWNER')
    def test_p3_training_output_root_cannot_match_neighbor_project(self):
        job=dict(output_roots=['/data/ckpt/nq4_p3'])
        expected=self.proc(argv=['python','tools/train.py','--output_root','/data/ckpt/nq4_p3/p3/000'])
        neighbor=self.proc(pid=43,argv=['python','tools/train.py','--output_root','/data/ckpt/nq4_p3_other'])
        self.assertEqual([p['pid'] for p in m.members(job,{}, {42:expected,43:neighbor})],[42])

if __name__=='__main__': unittest.main()
