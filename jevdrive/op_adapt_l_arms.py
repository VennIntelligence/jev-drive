"""op-adapt L: the named run configurations (prereg section 'Arme'). `main` resolves to the configuration the selection wave
picked ($L/selection.json), so the ablations are defined relative to it."""
from __future__ import annotations

import json
from dataclasses import replace

from .op_adapt_l import LCfg, lroot

# selection wave: the pre-registered set of (trainable set, intent condition), seed 0
SELECTION = {
    "sel_s4ia": dict(s4=True, pol=False, intent="ia"),          # stage 4 (r2's set) + token-embedding intent adapter
    "sel_polia": dict(s4=False, pol=True, intent="ia"),         # off-policy plan pathway + adapter, stage 4 frozen
    "sel_s4polia": dict(s4=True, pol=True, intent="ia"),        # both + adapter
    "sel_polid": dict(s4=False, pol=True, intent="id"),         # off-policy plan pathway, intent through the native desire input
}
# ablations: overrides applied on top of the selected main configuration
ABLATIONS = {
    "noint": dict(intent="none"),
    "only_start": dict(slices=("start",)),
    "only_stop": dict(slices=("stop",)),
    "only_turn": dict(slices=("turn_onset",)),
    "dw03": dict(dw=0.3),
    "dw3": dict(dw=3.0),
    "nocontrast": dict(contrast=False),
    "long": dict(steps=8000, eval_every=4000),
}
# ablations of the trainable set that are not among the selection candidates
EXTRA = {
    "tr_ad": dict(s4=False, pol=False, intent="ia"),            # adapter only
}


def selected() -> str:
    p = lroot() / "selection.json"
    return json.loads(p.read_text())["config"] if p.exists() else "sel_s4ia"


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
