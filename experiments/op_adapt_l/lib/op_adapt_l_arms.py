"""op-adapt L: the named run configurations (prereg section 'Arme'). `main` resolves to the configuration the selection wave
picked ($L/selection.json), so the ablations are defined relative to it."""
from __future__ import annotations

import json
from dataclasses import replace

from experiments.op_adapt_l.lib.op_adapt_l import LCfg, lroot

# selection wave: the pre-registered set of (trainable set, intent condition), seed 0; D1 (prereg 'Abweichungen'): each is run at
# two distillation weights dw in {1, 3}, added after the 800-step pilot showed the dev drift on the line and before any selection run
_BASE = {
    "sel_s4ia": dict(s4=True, pol=False, intent="ia"),          # stage 4 (r2's set) + token-embedding intent adapter
    "sel_polia": dict(s4=False, pol=True, intent="ia"),         # off-policy plan pathway + adapter, stage 4 frozen
    "sel_s4polia": dict(s4=True, pol=True, intent="ia"),        # both + adapter
    "sel_polid": dict(s4=False, pol=True, intent="id"),         # off-policy plan pathway, intent through the native desire input
}
SELECTION = {**_BASE, **{f"{k}_dw3": {**v, "dw": 3.0} for k, v in _BASE.items()}}
# ablations: overrides applied on top of the selected main configuration (dw ones are absolute values; the one equal to main's is skipped)
ABLATIONS = {
    "noint": dict(intent="none"),
    "only_start": dict(slices=("start",)),
    "only_stop": dict(slices=("stop",)),
    "only_turn": dict(slices=("turn_onset",)),
    "dw03": dict(dw=0.3),
    "dw1": dict(dw=1.0),
    "dw3": dict(dw=3.0),
    "dw10": dict(dw=10.0),
    "nocontrast": dict(contrast=False),
    "stayheavy": dict(contrast_mix=(3.0, 1.0, 1.0)),          # D4: stay : control : straight_int = 3 : 1 : 1 (start : stay closer to the log's 1 : 1.6)
    "long": dict(steps=8000, eval_every=4000),
}
# ablations of the trainable set that are not among the selection candidates
EXTRA = {
    "tr_ad": dict(s4=False, pol=False, intent="ia"),            # adapter only
    "tr_ad_dw3": dict(s4=False, pol=False, intent="ia", dw=3.0),   # D4: the same at the distillation weight the selection may pick
}


def selected() -> str:
    p = lroot() / "selection.json"
    return json.loads(p.read_text())["config"] if p.exists() else "sel_s4ia"


def same_as_main(name: str) -> bool:
    """An ablation whose configuration is the selected main one (e.g. dw3 when main was selected at dw = 3) is not run again."""
    a, m = get(name), get("main")
    return {**a.dump(), "name": ""} == {**m.dump(), "name": ""}


def get(name: str, seed: int = 0, steps: int = 0, eval_every: int = 0) -> LCfg:
    if name in SELECTION:
        kw = SELECTION[name]
    elif name in EXTRA:
        kw = EXTRA[name]
    elif name == "main":
        kw = SELECTION[selected()]
    elif name in ABLATIONS:
        kw = {**SELECTION[selected()], **ABLATIONS[name]}
    else:
        raise KeyError(name)
    cfg = LCfg(name=name, seed=seed, **kw)
    if steps:
        cfg = replace(cfg, steps=steps)
    if eval_every:
        cfg = replace(cfg, eval_every=eval_every)
    return cfg
