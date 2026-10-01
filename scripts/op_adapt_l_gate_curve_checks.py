"""Independent numerical checks of gate residency, paired bootstrap, weighted quantiles and frontier.

Uses training rows and synthetic statistics only. No validation outcomes are read.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from jevdrive import op_adapt_l as L  # noqa: E402
from op_adapt_l_gate_curve import ROOT, PREP, Gate, grant_check  # noqa: E402
from op_adapt_l_gate_curve_readout import Bootstrap, hull  # noqa: E402
from op_adapt_l_gate_curve_prep import dump  # noqa: E402
from op_adapt_l_train import Log  # noqa: E402


def main():
    out=ROOT/"numeric-checks";out.mkdir(parents=True,exist_ok=True)
    if (out/"DONE").exists() or (out/"ERROR").exists():raise SystemExit("Existing checks sentinel")
    log=Log(out)
    assert log.tb is not None
    try:
        grant_check(gpu=True)
        torch.set_num_threads(2)
        groups=np.asarray(["a"]*2+["b"]*3+["c"]*4+["d"]*7)
        x=np.linspace(-.4,1.2,len(groups));y=np.sin(x)
        boot=Bootstrap(groups)
        reference=L.paired_delta(x,y,groups,B=2000,seed=0)
        fast=boot.paired(x,y)
        assert max(abs(reference[k]-fast[k]) for k in ("delta","lo","hi"))<1e-12
        quantiles={}
        for q in (.5,.95):
            dd=[]
            for w in tqdm(boot.weights,desc=f"Expanded quantile check {q}",mininterval=5):
                expanded=np.repeat(x,w[boot.gi].astype(int))
                dd.append(np.quantile(expanded,q))
            lo,hi=np.percentile(dd,[2.5,97.5]);stat=boot.quantile(x,q)
            error=max(abs(lo-stat["lo"]),abs(hi-stat["hi"]))
            assert error<1e-12
            quantiles[str(q)]=error
        assert np.array_equal(hull([(0,0),(1,1),(2,1.5),(3,3)]),[[0,0],[3,3]])
        assert np.array_equal(hull([(0,0),(1,2),(2,3),(3,3.5)]),[[0,0],[1,2],[2,3],[3,3.5]])
        D=L.Data(("wod",),hstore=True)
        rr=np.load(PREP/"gate_rows.npz")["train"][:256]
        original=D.hs["wod"].gather(D.dom["wod"].ctx[rr]).reshape(len(rr),-1)
        cached=np.load(ROOT/"gate_train.npy",mmap_mode="r")[:len(rr)].copy()
        assert np.array_equal(original,cached)
        ck=torch.load(ROOT/"gate-pilot/head.pt",map_location="cpu",weights_only=False)
        h=Gate(ck["features"],ck["hidden"]).cuda();h.load_state_dict(ck["head"])
        labels=torch.as_tensor(D.tab["wod"]["s_start"][rr].astype(np.float32),device="cuda")
        criterion=torch.nn.BCEWithLogitsLoss()
        gradients=[];losses=[]
        for arr in (original,cached):
            h.zero_grad(set_to_none=True)
            loss=criterion(h(torch.as_tensor(arr,device="cuda")),labels)
            loss.backward()
            gradients.append(torch.cat([p.grad.flatten().detach().clone() for p in h.parameters()]))
            losses.append(float(loss))
        relative=float((gradients[0]-gradients[1]).norm()/gradients[0].norm())
        assert abs(losses[0]-losses[1])<=1e-6 and relative<=1e-5
        resident=torch.as_tensor(cached,device="cuda")
        timings={}
        for mode in ("reference_gather","resident"):
            elapsed=[]
            for j in tqdm(range(15),desc=f"Gate path profile {mode}",mininterval=5):
                torch.cuda.synchronize();t0=time.perf_counter()
                batch=resident if mode=="resident" else torch.as_tensor(D.hs["wod"].gather(D.dom["wod"].ctx[rr]).reshape(len(rr),-1),device="cuda")
                h.zero_grad(set_to_none=True);loss=criterion(h(batch),labels);loss.backward()
                torch.cuda.synchronize();elapsed.append(time.perf_counter()-t0)
            timings[mode] = {"seconds_per_batch":float(np.median(elapsed[3:])),"frames_per_s":float(len(rr)/np.median(elapsed[3:]))}
        result={"bootstrap_paired_maxabs":max(abs(reference[k]-fast[k]) for k in ("delta","lo","hi")),
                "quantile_ci_maxabs":quantiles,"frontier_hull_checks":True,"gate_feature_maxabs":0.,
                "gate_loss_reference":losses[0],"gate_loss_resident":losses[1],"gate_gradient_relative_error":relative,
                "gate_path_profile":timings,"training_rows_only":True}
        dump(out/"checks.json",result);log.info(json.dumps(result));log.event("end",**result)
        (out/"DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    except BaseException as e:
        (out/"ERROR").write_text(traceback.format_exc());log.event("error",error=str(e));raise
    finally:log.tb.close()


if __name__=="__main__":main()
