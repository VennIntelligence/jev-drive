"""Lane leases in the box's schedule table ($DATA_DIR/runs/sched/table.tsv, the format scripts/sch_table.py keeps).

A lease is one live row: the cards a lane may use, a core slice and a block of CARLA server indices per card, and the
workers per card. `find_free` picks what is verifiably free right now: cards with no live row, no compute process and
no CARLA server on them; cores that no live row holds, NUMA-local to the card and one per physical core first; index
blocks clear of every live row's block and of its traffic-manager shadow (index i's TM ports are the RPC block of
i + 120), with every port >= 10000 and below the ephemeral range (capacity.index_bounds).

Row columns: lane, gpus "1,2", workers, idx0 "1:160,2:200", idx_span, cpus "1:52-76,2:77-101", status, go.
"""
from __future__ import annotations

import csv
import fcntl
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .box import format_cpus, parse_cpus
from .capacity import index_bounds

COLS = ["lane", "gpus", "workers", "idx0", "idx_span", "cpus", "status", "go"]
TM_SHIFT = 120                          # (TM_BASE - PORT_BASE) / PORT_STRIDE


def table_path() -> Path:
    return Path(os.environ.get("DATA_DIR", str(Path.home() / "data"))) / "runs" / "sched" / "table.tsv"


def load(path: Path = None) -> list:
    path = path or table_path()
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for c in COLS:
            r[c] = (r.get(c) or "-").strip() or "-"
    return rows


def save(rows: list, path: Path = None) -> None:
    path = path or table_path()
    tmp = path.with_suffix(".tmp.%d" % os.getpid())
    with tmp.open("w") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows({c: r.get(c, "-") for c in COLS} for r in rows)
    tmp.replace(path)


def live(rows: list) -> list:
    """Rows that hold resources: not done / revoked, with cards. Rows without cards are proposals."""
    return [r for r in rows if not r["status"].startswith(("done", "revoked")) and r["gpus"] != "-"]


def gpus(r: dict) -> list:
    return [int(g) for g in r["gpus"].replace(" ", ",").split(",") if g.strip().isdigit()]


def _map(field: str) -> dict:
    """'g:v,g:v' -> {g: v}; values may themselves be core lists ('1:52-76,2:77-101')."""
    out, key = {}, None
    for part in field.replace(" ", ",").split(","):
        if ":" in part:
            key, part = part.split(":", 1)
            key = int(key)
            out[key] = part
        elif key is not None and part:
            out[key] += "," + part
    return out


def index_blocks(r: dict) -> dict:
    """card -> range of server indices (a plain idx0 n means card k of `gpus` holds [n + k span, n + (k+1) span))."""
    if r["idx0"] == "-" or r["idx_span"] == "-":
        return {}
    span = int(r["idx_span"])
    if ":" in r["idx0"]:
        return {g: range(int(i), int(i) + span) for g, i in _map(r["idx0"]).items()}
    return {g: range(int(r["idx0"]) + k * span, int(r["idx0"]) + (k + 1) * span) for k, g in enumerate(gpus(r) or [-1])}


def indices(r: dict) -> set:
    return {i for b in index_blocks(r).values() for i in b}


def card_cpus(r: dict) -> dict:
    """card -> core list string ('-' / empty: none); a plain list is shared by every card of the row."""
    if r["cpus"] in ("-", ""):
        return {}
    if ":" in r["cpus"]:
        return _map(r["cpus"])
    return {g: r["cpus"] for g in gpus(r) or [-1]}


def cpus(r: dict) -> set:
    return {c for v in card_cpus(r).values() for c in parse_cpus(v)}


def shadow(ix: set) -> set:
    return ix | {i + TM_SHIFT for i in ix} | {i - TM_SHIFT for i in ix}


def conflicts(rows: list, ephemeral_lo: int = 32768) -> list:
    """Index blocks that collide (i, i +- 120 across rows) or reach the ephemeral range; overlapping core slices."""
    out, rs = [], live(rows)
    hi = index_bounds(ephemeral_lo)[1]
    for r in rs:
        top = max(indices(r), default=-1)
        if top > hi and not r["status"].startswith("legacy"):
            out.append("%s: index %d puts TM ports in the ephemeral range (keep every index <= %d)" % (r["lane"], top, hi))
    for a in range(len(rs)):
        for b in range(a + 1, len(rs)):
            hit = indices(rs[a]) & shadow(indices(rs[b]))
            if hit:
                out.append("index conflict %s x %s: %d..%d (%d)" % (rs[a]["lane"], rs[b]["lane"], min(hit), max(hit), len(hit)))
            both = cpus(rs[a]) & cpus(rs[b])
            if both:
                out.append("core overlap %s x %s: %s" % (rs[a]["lane"], rs[b]["lane"], format_cpus(both)))
    return out


@dataclass
class Lease:
    lane: str
    cards: dict          # card -> {"cpus": "52-76", "idx0": 160}
    span: int
    workers: int         # CARLA workers per card
    status: str = ""

    def row(self) -> dict:
        gs = sorted(self.cards)
        return dict(lane=self.lane, gpus=",".join(map(str, gs)), workers=str(self.workers),
                    idx0=",".join("%d:%d" % (g, self.cards[g]["idx0"]) for g in gs), idx_span=str(self.span),
                    cpus=",".join("%d:%s" % (g, self.cards[g]["cpus"]) for g in gs), status=self.status or "-", go="-")

    @classmethod
    def from_row(cls, r: dict) -> "Lease":
        blocks, cc = index_blocks(r), card_cpus(r)
        span = int(r["idx_span"]) if r["idx_span"] != "-" else 0
        w = int(r["workers"]) if r["workers"].isdigit() else 0
        return cls(r["lane"], {g: {"cpus": cc.get(g, cc.get(-1, "")), "idx0": blocks[g].start if g in blocks else -1}
                               for g in gpus(r)}, span, w, r["status"])

    @property
    def revoked(self) -> bool:
        return self.status.startswith(("done", "revoked")) or not self.cards


def get(lane: str, rows: list = None) -> Lease:
    r = next((x for x in (rows if rows is not None else load()) if x["lane"] == lane), None)
    return Lease.from_row(r) if r else None


def find_free(box, rows: list, lane: str, n_gpus: int = 1, want: list = None, cores_per_card: int = None,
              span: int = 24, workers: int = 6) -> Lease:
    """Free cards, NUMA-local free cores and conflict-free index blocks for a new lease (see module doc)."""
    others = [r for r in live(rows) if r["lane"] != lane]
    held_cards = {g for r in others for g in gpus(r)}
    free_cards = [c for c in box.cards if c.index not in held_cards and not c.compute_pids and not c.carla
                  and c.mem_used_mib < 1024]
    if want:
        bad = sorted(set(want) - {c.index for c in free_cards})
        if bad:
            raise RuntimeError("cards not free: %s" % bad)
        chosen = [box.card(g) for g in want]
    else:
        if len(free_cards) < n_gpus:
            raise RuntimeError("only %d free cards (%s), %d wanted" % (len(free_cards), [c.index for c in free_cards], n_gpus))
        chosen = free_cards[:n_gpus]
    cpc = int(cores_per_card or box.cores // max(len(box.cards), 1))
    taken = set().union(*[cpus(r) for r in others]) if others else set()
    lo, hi = index_bounds(box.ephemeral[0])
    used_ix = set().union(*[shadow(indices(r)) for r in others]) if others else set()
    cards = {}
    for c in chosen:
        pool = [x for x in box.primary_cpus(c.numa) if x not in taken]
        pool += [x for x in box.primary_cpus() if x not in taken and x not in pool]
        pool += [x for x in box.affinity if x not in taken and x not in pool]     # hyperthread siblings last
        if len(pool) < cpc:
            raise RuntimeError("card %d: %d free cores, %d wanted" % (c.index, len(pool), cpc))
        mine = pool[:cpc]
        taken |= set(mine)
        i0 = next((i for i in range(lo, hi - span + 2) if not set(range(i, i + span)) & used_ix), None)
        if i0 is None:
            raise RuntimeError("no free index block of %d in [%d, %d]" % (span, lo, hi))
        used_ix |= shadow(set(range(i0, i0 + span)))
        cards[c.index] = {"cpus": format_cpus(mine), "idx0": i0}
    return Lease(lane, cards, span, workers)


@contextmanager
def owner_lock(path: Path = None):
    """The table's single-writer lock (scripts/sch_table.py: one scheduler owns all mutations)."""
    p = (path or table_path()).parent / "owner.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("schedule owner lock is held (%s); do not race other table writers" % p)
        yield


def grant(lease: Lease, path: Path = None, ephemeral_lo: int = 32768) -> dict:
    """Write (or rewrite in place) the lane's row; refuse if it conflicts with another live row."""
    with owner_lock(path):
        rows = load(path)
        row = lease.row()
        row["status"] = lease.status or "running since %s" % time.strftime("%F %H:%M")
        rows = [r for r in rows if r["lane"] != lease.lane] + [row]
        bad = conflicts(rows, ephemeral_lo)
        if bad:
            raise RuntimeError("not granted: " + "; ".join(bad))
        save(rows, path)
        return row


def finish(lane: str, note: str = "", path: Path = None) -> None:
    """Move the lane's row to archive.tsv (append-only) and drop it from the live table."""
    path = path or table_path()
    with owner_lock(path):
        rows = load(path)
        r = next((x for x in rows if x["lane"] == lane), None)
        if r is None:
            raise RuntimeError("no row for lane %s" % lane)
        if note:
            r["status"] = note
        arch = path.with_name("archive.tsv")
        new = not arch.exists()
        with arch.open("a") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            if new:
                w.writerow(["archived"] + COLS)
            w.writerow([time.strftime("%F %T")] + [r[c] for c in COLS])
        save([x for x in rows if x is not r], path)
