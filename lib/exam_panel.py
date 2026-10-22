"""Desktop control panel for a human taking a driving exam item by item (tkinter, Python 3.8+, stdlib only). The default front end
is now the web page lib/exam_web.py over the same Backend and protocol; this Tk window stays as the fallback (human_panel.py --tk).

Generic over a backend. Bench2Drive: experiments/phone_wheel/lib/b2d_exam.py (README of phone_wheel, section "Desktop panel");
AlpaSim: experiments/alpasim_human/lib/alpasim_exam.py (README of alpasim_human, section "Desktop panel").

Protocol between the panel and a backend's long-lived drive loop. Two JSON files sit in the backend's live dir, and each has
exactly one writer:
  <live>/control.json  (written only by the panel, through Control.update / Control.act)
      {"paused": bool, "playlist": bool, "record": bool, "options": {name: value}, "order": [item ids in the panel's current order],
       "actions": [{"id": int, "kind": "play" | "next" | "restart" | "stop", "item": id or null, "t": unix s}]}
  <live>/loop.json     (written only by the loop, through write_json)
      {"run": dir, "item": id or null, "state": "loading" | "driving" | "paused" | "idle" | "stopped", "msg": str,
       "consumed": [action ids handled], "pid": int, "t": unix s, "scoring": rollouts waiting for a score (optional)}
The loop handles an action once (it records the action's id in consumed):
  - play: run that item next, even if it was already driven.
  - next: abort the running item and go on.
  - restart: abort the running item and drive it again.
  - stop: abort the running item and exit.
"paused" holds at the next item boundary; the agent may also hold mid-item. "playlist" makes the loop auto-advance to the next
unfinished item in "order"; without it the loop waits for a play action or the driver's own start signal.
"record" (default true; only for a backend with record = True): false = practice, the loop drives the item but scores, counts and keeps
nothing (no done/ or tried/ record, the rollout files are deleted when the item returns); it is read when an item returns, and auto-advance
skips items practised in this loop while it stays false.
Batches: a list of the backend's run dirs (runs()), newest first, with set label, created time, driven / total, mean human score,
disk size (du, cached; the running batch re-measured every 60 s) and an optional backend column (batch_extra). Selecting one shows its
items; "Continue this batch" (re)starts the loop on it, "Delete this batch" removes its whole dir after a confirmation naming it and its
size, never while the loop runs on it. Nothing is ever deleted automatically. "Reload" (web panel) starts a batch over: archive_results()
moves its results (done/, tried/, results.md) into <run>/archive/<stamp>/, kept on disk but no longer counted, then the loop restarts.
The top of the panel shows the phone page address (from `tailscale status --json`, with a Copy button) and the live input status
from the wheel daemon's status file (WHEEL_STATUS, default $DATA_DIR/tmp/wheel/status.json; experiments/phone_wheel/scripts/wheel_server.py).
"""
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".%d.%d" % (os.getpid(), threading.get_ident()))   # the web server writes from several threads
    tmp.write_text(json.dumps(obj))
    os.replace(str(tmp), str(path))


class Control(object):
    """control.json: the panel writes it, the loop and the agent read it."""
    DEFAULT = {"paused": False, "playlist": True, "record": True, "options": {}, "order": [], "actions": []}
    LOCK = threading.RLock()       # read-modify-write by several server threads of one panel must not lose an update

    def __init__(self, live):
        self.path = Path(live) / "control.json"

    def read(self):
        return dict(self.DEFAULT, **(read_json(self.path, {}) or {}))

    def update(self, **kw):
        with self.LOCK:
            c = self.read()
            c.update(kw)
            write_json(self.path, c)
            return c

    def act(self, kind, item=None):
        with self.LOCK:
            c = self.read()
            aid = max([a["id"] for a in c["actions"]] + [0]) + 1
            c["actions"] = (c["actions"] + [{"id": aid, "kind": kind, "item": item, "t": time.time()}])[-50:]
            write_json(self.path, c)
            return aid

    def pending(self, consumed):
        """Actions the loop has not handled yet, oldest first."""
        seen = set(consumed or [])
        return [a for a in self.read()["actions"] if a["id"] not in seen]


class Backend(object):
    """What the panel needs from an exam. Override every method; items are dicts with
    id, group, scenario, human (score or None), model (score or None), status (str); optional label (row text, default id) and
    model_src (short name of the model the model score comes from, shown after it)."""
    title = "exam"
    score_fmt = "%.1f"             # human / model / group-mean cells
    live = None                    # live dir (control.json / loop.json)
    options = {}                   # name -> (label, default bool) = a toggle, e.g. {"stop_on_collision": ("Stop on collision", False)},
                                   # or (label, default str, [choices]) = a choice, e.g. {"size": ("Size", "auto", ["auto", "720"])};
                                   # a choice may be [value, label, group] (the web panel shows the label, grouped)
    batch_extra = None             # title of an extra column of the batch list (batch_info "extra"), or None
    record = False                 # the loop honours control "record" (practice mode); the web panel hides the toggle otherwise
    option_help = {}               # name -> (what it does, when it applies): one plain line each for the web panel's settings
    pinned = ()                    # toggle options the web panel also shows as a big switch in its header (always visible)
    selected = None                # the batch the panel shows (set by the panel); None = default_run()

    def batch_sets(self):          # [(key, label)] offered by "New batch"
        raise NotImplementedError

    def new_batch(self, key, ids=None, name=None, start=True):  # run dir for set `key` (or the picked `ids`, key "custom"), labelled
        raise NotImplementedError                                 # `name` if given; start the loop on it if `start`; (ok, message)

    def catalog(self):             # every selectable item for a custom pick, as dicts (backend-specific fields); [] = none
        return []

    def start_loop(self, run):     # (re)start the drive loop on that run dir; return (ok, message)
        raise NotImplementedError

    def continue_batch(self, run):  # "Continue this batch"; return (ok, message)
        return self.start_loop(run)

    def runs(self):                # the batches (run dirs, str), newest first
        raise NotImplementedError

    def batch_info(self, run):     # dict(label, created, driven, total, mean (or None), extra)
        raise NotImplementedError

    def archive_results(self, run):  # "Reload": move the batch's results into <run>/archive/<stamp>/ (kept, no longer counted)
        if run not in self.runs():
            return False, "%s is not a batch of this panel" % run
        dst = Path(run, "archive", time.strftime("%m%d-%H%M%S"))
        moved = [f for f in ("done", "tried", "results.md") if Path(run, f).exists()]
        dst.mkdir(parents=True, exist_ok=True)
        for f in moved:
            shutil.move(str(Path(run, f)), str(dst / f))
        return True, "previous results moved to %s" % dst if moved else "no previous results"

    def item_label(self, item):    # short display name of an item of any batch (the web panel's now-playing line)
        return item

    def delete_batch(self, run):   # remove a batch dir (the panel has confirmed it and checked that no loop runs on it)
        if run not in self.runs():
            return False, "%s is not a batch of this panel" % run
        shutil.rmtree(run, ignore_errors=True)
        return (not os.path.exists(run)), ("deleted %s" % run if not os.path.exists(run) else "could not delete all of %s" % run)

    def busy_other(self):          # a non-panel run that must be stopped before the panel takes over, as text, or ""
        return ""

    def stop_other(self):          # stop that run cleanly; return message
        return ""

    def run_dir(self):             # the run the panel shows: the selected batch, else default_run()
        return self.selected if self.selected and os.path.isdir(self.selected) else self.default_run()

    def default_run(self):         # the loop's run, else the newest
        raise NotImplementedError

    def loop_run(self):            # the run the drive loop is working on, or None
        st = read_json(Path(self.live) / "loop.json", {}) or {}
        alive = st.get("pid") and os.path.exists("/proc/%s" % st["pid"]) and st.get("state") != "stopped"
        return st.get("run") if alive else None

    def items(self):               # items of run_dir() in run order
        raise NotImplementedError

    def official(self):            # official-format table of run_dir() as text
        return ""

    def ensure_side(self):         # start helpers if missing (wheel daemon, viewer); return message
        return ""


WHEEL_STATUS = os.environ.get("WHEEL_STATUS", os.path.join(os.environ.get("DATA_DIR", "/data"), "tmp/wheel/status.json"))


def phone_urls():
    """(https URL via tailscale serve, plain-http LAN fallback) of the phone page; fallbacks if tailscale does not answer."""
    try:
        s = json.loads(subprocess.run(["tailscale", "status", "--json"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      timeout=5).stdout)["Self"]
        dns, ip = s.get("DNSName", "").rstrip("."), next((a for a in s.get("TailscaleIPs", []) if "." in a), None)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        dns, ip = None, None
    return ("https://%s/" % dns if dns else "https://<tailscale name>/ (tailscale status failed)",
            "http://%s:8766/" % (ip or "<tailscale IP>"))


def wheel_line(path=WHEEL_STATUS):
    """(text, colour) of the phone / keyboard input status from the wheel daemon's status file."""
    st = read_json(path)
    if not st or time.time() - st.get("t", 0) > 5:
        return "input: wheel daemon not running (no fresh %s)" % path, "#c00000"
    ph, kb = st.get("phone") or {}, st.get("keyboard") or {}
    pa, ka = ph.get("age_s"), kb.get("age_s")
    p = ("phone connected, last frame %s" % ("none yet" if pa is None else "%.1f s ago" % pa if pa > 0.3 else "live")
         if ph.get("clients") else "phone not connected")
    k = "keyboard active%s" % (" (%s)" % " ".join(kb["keys"]) if kb.get("keys") else "") if ka is not None and ka < 0.3 else \
        "keyboard idle (focus a viewer window to drive with it)"
    src = st.get("source")
    return "input: %s;  %s;  driving: %s" % (p, k, src or "NOBODY (car brakes)"), "#2e7d32" if src else "#c05000"


def du(path):
    try:
        return int(subprocess.run(["du", "-sb", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60).stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def human_size(b):
    return "-" if b is None else "%.1f GB" % (b / 1e9) if b >= 1e9 else "%.0f MB" % (b / 1e6) if b >= 1e6 else "%.0f kB" % (b / 1e3)


def model_key(it):
    return (it["model"] is None, it["model"] if it["model"] is not None else 0.0)


def panel(backend):
    import tkinter as tk
    from tkinter import ttk, messagebox

    ctl = Control(backend.live)
    root = tk.Tk()
    root.title("%s - control panel" % backend.title)
    root.geometry(os.environ.get("EXAM_PANEL_GEOMETRY", "1260x1700+20+60"))
    style = ttk.Style(root)
    big = ("DejaVu Sans", 13)
    style.configure(".", font=big)
    style.configure("Treeview", rowheight=30, font=big)
    style.configure("Treeview.Heading", font=("DejaVu Sans", 13, "bold"))

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    run_var, state_var, wheel_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
    url, lan = phone_urls()
    prow = ttk.Frame(top)
    prow.pack(fill="x")
    ttk.Label(prow, text="Phone: %s" % url, font=("DejaVu Sans", 15, "bold"), foreground="#0050a0").pack(side="left")
    ttk.Label(prow, text="   (LAN fallback %s)" % lan).pack(side="left")

    def copy_url():
        root.clipboard_clear()
        root.clipboard_append(url)
    ttk.Button(prow, text="Copy", command=copy_url).pack(side="left", padx=8)
    wheel_lab = ttk.Label(top, textvariable=wheel_var)
    wheel_lab.pack(anchor="w")
    ttk.Label(top, textvariable=run_var).pack(anchor="w")
    ttk.Label(top, textvariable=state_var, foreground="#c05000", font=("DejaVu Sans", 14, "bold")).pack(anchor="w")

    def info(msg):
        if msg:
            messagebox.showinfo(backend.title, msg)

    row1 = ttk.Frame(root, padding=(8, 2))
    row1.pack(fill="x")
    ttk.Label(row1, text="New:").pack(side="left")
    sets = backend.batch_sets()
    set_var = tk.StringVar(value=sets[0][1] if sets else "")
    ttk.Combobox(row1, textvariable=set_var, values=[s[1] for s in sets], state="readonly", width=34).pack(side="left")

    def take_over():
        other = backend.busy_other()
        if other:
            if not messagebox.askyesno(backend.title, "%s\n\nStop it now (the current item is aborted and can be driven again)?" % other):
                return False
            info(backend.stop_other())
        return True

    def new_batch():
        key = dict((l, k) for k, l in sets).get(set_var.get())
        if key and take_over():
            before = set(backend.runs())
            info(backend.new_batch(key)[1])
            new = [r for r in backend.runs() if r not in before]
            backend.selected = new[0] if new else backend.selected
            refresh_batches(True)

    def chosen():
        sel = btree.selection()
        return sel[0] if sel else None

    def continue_batch():
        run = chosen()
        if run and take_over():
            backend.selected = run
            info(backend.continue_batch(run)[1])
            refresh_batches(True)

    def delete_batch():
        run = chosen()
        if not run:
            return
        if run == backend.loop_run():
            return info("The loop is running on this batch:\n%s\n\nStop it first (Stop), then delete." % run)
        b, size = backend.batch_info(run), human_size(du(run))
        if not messagebox.askyesno(backend.title, "Delete the batch \"%s\" (%s, created %s, %d / %d driven, %s on disk)?\n\nThis removes "
                                   "the whole directory\n%s\nwith every attempt, rollout and score. It cannot be undone." % (
                                       b["label"], os.path.basename(run), b["created"], b["driven"], b["total"], size, run), icon="warning"):
            return
        info(backend.delete_batch(run)[1])
        sizes.pop(run, None)
        if backend.selected == run:
            backend.selected = None
        refresh_batches(True)
    ttk.Button(row1, text="New batch", command=new_batch).pack(side="left", padx=4)
    ttk.Button(row1, text="Official table", command=lambda: show_text("Official table", backend.official())).pack(side="right")

    bcols = ("created", "driven", "mean", "size") + (("extra",) if backend.batch_extra else ())
    bframe = ttk.Frame(root, padding=(8, 2))
    bframe.pack(fill="x")
    btree = ttk.Treeview(bframe, columns=bcols, show="tree headings", selectmode="browse", height=5)
    btree.heading("#0", text="batch")
    btree.column("#0", width=470)
    for c, w, t in zip(bcols, (170, 120, 110, 110, 130), ("created", "driven / total", "mean human", "disk", backend.batch_extra)):
        btree.heading(c, text=t)
        btree.column(c, width=w, anchor="w" if c in ("created", "extra") else "e")
    bsb = ttk.Scrollbar(bframe, orient="vertical", command=btree.yview)
    btree.configure(yscrollcommand=bsb.set)
    bsb.pack(side="right", fill="y")
    btree.pack(fill="x", expand=True)
    btree.tag_configure("current", background="#ffe9b0")
    brow = ttk.Frame(root, padding=(8, 2))
    brow.pack(fill="x")
    ttk.Button(brow, text="Continue this batch", command=continue_batch).pack(side="left", padx=4)
    ttk.Button(brow, text="Delete this batch", command=delete_batch).pack(side="left", padx=4)
    ttk.Label(brow, text="select a batch to show its items; the running loop's batch is highlighted").pack(side="left", padx=8)
    sizes, bstate = {}, {"t": 0.0, "busy": False}

    def on_bselect(_):
        run = chosen()
        if run and not bstate["busy"] and run != backend.run_dir():
            backend.selected = run
            refresh(True)
    btree.bind("<<TreeviewSelect>>", on_bselect)

    def refresh_batches(force=False):
        if not force and time.time() - bstate["t"] < 5:
            return
        bstate["t"], bstate["busy"] = time.time(), True
        try:
            runs, cur, shown = backend.runs(), backend.loop_run(), backend.run_dir()
            if list(btree.get_children()) != runs:
                btree.delete(*btree.get_children())
                for r in runs:
                    btree.insert("", "end", iid=r)
            for r in runs:
                if r not in sizes or (r == cur and time.time() - sizes[r][0] > 60):
                    sizes[r] = (time.time(), du(r))
                try:
                    b = backend.batch_info(r)
                except Exception:  # noqa: BLE001  (a half-written batch must not kill the panel)
                    b = dict(label="?", created="?", driven=0, total=0, mean=None, extra="")
                btree.item(r, text="%s   %s" % (b["label"], os.path.basename(r)), tags=("current",) if r == cur else (),
                           values=(b["created"], "%d / %d" % (b["driven"], b["total"]), fmt(b["mean"]), human_size(sizes[r][1]))
                           + ((b.get("extra", ""),) if backend.batch_extra else ()))
            if shown in runs and chosen() != shown:
                btree.selection_set(shown)
                btree.see(shown)
        finally:
            bstate["busy"] = False

    row2 = ttk.Frame(root, padding=(8, 2))
    row2.pack(fill="x")
    pause_btn = ttk.Button(row2, text="Pause", width=10)

    def toggle_pause():
        ctl.update(paused=not ctl.read()["paused"])
    pause_btn.configure(command=toggle_pause)
    pause_btn.pack(side="left", padx=4)
    ttk.Button(row2, text="Restart item", command=lambda: ctl.act("restart")).pack(side="left", padx=4)
    ttk.Button(row2, text="Next item", command=lambda: ctl.act("next")).pack(side="left", padx=4)

    def stop():
        if messagebox.askyesno(backend.title, "Stop the loop? The current item is aborted (it can be driven again)."):
            ctl.act("stop")
    ttk.Button(row2, text="Stop", command=stop).pack(side="left", padx=4)

    row3 = ttk.Frame(root, padding=(8, 2))
    row3.pack(fill="x")
    c0 = ctl.read()
    play_var = tk.BooleanVar(value=c0["playlist"])
    ttk.Checkbutton(row3, text="Playlist (auto-advance)", variable=play_var,
                    command=lambda: ctl.update(playlist=play_var.get())).pack(side="left", padx=4)
    opt_vars = {}
    for name, (label, default, *choices) in backend.options.items():
        v = (tk.StringVar if choices else tk.BooleanVar)(value=c0["options"].get(name, default))
        opt_vars[name] = v

        def set_opt(*_, name=name, v=v):
            c = ctl.read()
            ctl.update(options=dict(c["options"], **{name: v.get()}))
        if choices:
            ttk.Label(row3, text=label + ":").pack(side="left", padx=(8, 2))
            ttk.OptionMenu(row3, v, v.get(), *[x[0] if isinstance(x, (list, tuple)) else x for x in choices[0]], command=set_opt).pack(side="left")
        else:
            ttk.Checkbutton(row3, text=label, variable=v, command=set_opt).pack(side="left", padx=8)
    if any(name not in c0["options"] for name in opt_vars):
        ctl.update(options=dict(c0["options"], **{n: v.get() for n, v in opt_vars.items()}))
    sort_var = tk.StringVar(value="group")
    ttk.Label(row3, text="   sort:").pack(side="left")
    ttk.Radiobutton(row3, text="by group", value="group", variable=sort_var, command=lambda: refresh(True)).pack(side="left")
    ttk.Radiobutton(row3, text="by score, low first", value="score", variable=sort_var, command=lambda: refresh(True)).pack(side="left")

    cols = ("scenario", "human", "model", "status")
    tree = ttk.Treeview(root, columns=cols, show="tree headings", selectmode="browse")
    tree.heading("#0", text="item")
    tree.column("#0", width=230)
    for c, w, t in zip(cols, (430, 110, 110, 300), ("scenario", "human", "model", "status")):
        tree.heading(c, text=t)
        tree.column(c, width=w + (50 if c == "model" else 0), anchor="w" if c in ("scenario", "status") else "e")
    sb = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True, padx=8, pady=8)
    tree.tag_configure("current", background="#ffe9b0")
    tree.tag_configure("done", foreground="#2e7d32")
    tree.tag_configure("group", font=("DejaVu Sans", 13, "bold"))
    ttk.Label(root, text="Double-click an item: drive it next (aborts nothing; with Playlist off it waits for the throttle).",
              padding=6).pack(anchor="w")

    def on_double(_):
        sel = tree.selection()
        if sel and not sel[0].startswith("group:"):
            ctl.act("play", sel[0])
    tree.bind("<Double-1>", on_double)

    def show_text(title, text):
        w = tk.Toplevel(root)
        w.title(title)
        t = tk.Text(w, width=170, height=40, font=("DejaVu Sans Mono", 11))
        t.insert("1.0", text or "(nothing yet)")
        t.pack(fill="both", expand=True)

    shape = {"key": None}
    fmt = lambda v: "" if v is None else backend.score_fmt % v  # noqa: E731

    def refresh(force=False):
        try:
            items = backend.items()
        except Exception:  # never let the panel die on a half-written file
            items = []
        st = read_json(Path(backend.live) / "loop.json", {}) or {}
        c = ctl.read()
        w, col = wheel_line()
        wheel_var.set(w)
        wheel_lab.configure(foreground=col)
        alive = st.get("pid") and os.path.exists("/proc/%s" % st.get("pid"))
        run_var.set("shown: %s%s" % (backend.run_dir() or "-", "" if not alive or st.get("run") == backend.run_dir() else
                                      "   (the loop runs on %s)" % os.path.basename(st.get("run") or "")))
        other = "" if alive else backend.busy_other()
        state_var.set("loop: %s%s%s   %s" % (st.get("state", "-") if alive else ("not running; " + other if other else "not running"),
                                            "  item %s" % st["item"] if st.get("item") and alive else "",
                                            "   PAUSED" if c["paused"] else "", st.get("msg", "") if alive else ""))
        pause_btn.configure(text="Resume" if c["paused"] else "Pause")
        if sort_var.get() == "score":
            order = sorted(items, key=lambda it: (it["human"] is None and it["model"] is None,
                                                  it["human"] if it["human"] is not None else it["model"] or 0.0))
            groups = [(None, order)]
        else:
            order, groups = items, []
            for it in items:
                if not groups or groups[-1][0] != it["group"]:
                    groups.append((it["group"], []))
                groups[-1][1].append(it)
        key = (sort_var.get(), tuple(it["id"] for it in order))
        if force or key != shape["key"]:
            open_groups = {g for g in tree.get_children() if tree.item(g, "open")}
            tree.delete(*tree.get_children())
            for g, its in groups:
                parent = ""
                if g is not None:
                    parent = "group:%s" % g
                    tree.insert("", "end", iid=parent, text="%s (%d)" % (g, len(its)), open=(parent in open_groups or not shape["key"]),
                                tags=("group",))
                for it in its:
                    tree.insert(parent, "end", iid=it["id"], text=it.get("label") or it["id"])
            shape["key"] = key
            if not alive or st.get("run") == backend.run_dir():      # viewing another batch never reorders the loop's
                ctl.update(order=[it["id"] for it in order])
        for it in order:
            tags = ("current",) if alive and st.get("run") == backend.run_dir() and st.get("item") == it["id"] else ("done",) if it["human"] is not None else ()
            model = fmt(it["model"]) + (" %s" % it["model_src"] if it.get("model_src") and it["model"] is not None else "")
            tree.item(it["id"], values=(it["scenario"], fmt(it["human"]), model, it["status"]), tags=tags)
        for g, its in groups:
            if g is not None:
                d = [it["human"] for it in its if it["human"] is not None]
                tree.item("group:%s" % g, values=("", fmt(sum(d) / len(d)) if d else "", "", "%d / %d driven" % (len(d), len(its))))

    def tick():
        try:
            refresh_batches()
            refresh()
        finally:
            root.after(1000, tick)
    info(backend.ensure_side())
    tick()
    root.mainloop()
