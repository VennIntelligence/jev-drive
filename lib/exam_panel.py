"""Desktop control panel for a human taking a driving exam item by item (tkinter, Python 3.8+, stdlib only).

Generic over a backend. Bench2Drive is the first one: experiments/phone_wheel/lib/b2d_exam.py.
experiments/phone_wheel/README.md, section "Desktop panel", documents both.

Protocol between the panel and a backend's long-lived drive loop. Two JSON files sit in the backend's live dir, and each has
exactly one writer:
  <live>/control.json  (written only by the panel, through Control.update / Control.act)
      {"paused": bool, "playlist": bool, "options": {name: value}, "order": [item ids in the panel's current order],
       "actions": [{"id": int, "kind": "play" | "next" | "restart" | "stop", "item": id or null, "t": unix s}]}
  <live>/loop.json     (written only by the loop, through write_json)
      {"run": dir, "item": id or null, "state": "loading" | "driving" | "paused" | "idle" | "stopped", "msg": str,
       "consumed": [action ids handled], "pid": int, "t": unix s}
The loop handles an action once (it records the action's id in consumed):
  - play: run that item next, even if it was already driven.
  - next: abort the running item and go on.
  - restart: abort the running item and drive it again.
  - stop: abort the running item and exit.
"paused" holds at the next item boundary; the agent may also hold mid-item. "playlist" makes the loop auto-advance to the next
unfinished item in "order"; without it the loop waits for a play action or the driver's own start signal.
"""
import json
import os
import time
from pathlib import Path


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".%d" % os.getpid())
    tmp.write_text(json.dumps(obj))
    os.replace(str(tmp), str(path))


class Control(object):
    """control.json: the panel writes it, the loop and the agent read it."""
    DEFAULT = {"paused": False, "playlist": True, "options": {}, "order": [], "actions": []}

    def __init__(self, live):
        self.path = Path(live) / "control.json"

    def read(self):
        return dict(self.DEFAULT, **(read_json(self.path, {}) or {}))

    def update(self, **kw):
        c = self.read()
        c.update(kw)
        write_json(self.path, c)
        return c

    def act(self, kind, item=None):
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
    id, group, scenario, human (score or None), model (score or None), status (str)."""
    title = "exam"
    live = None                    # live dir (control.json / loop.json)
    options = {}                   # name -> (label, default bool), e.g. {"stop_on_collision": ("Stop on collision", False)}

    def batch_sets(self):          # [(key, label)] offered by "New batch"
        raise NotImplementedError

    def new_batch(self, key):      # make a run dir for that set, start the loop on it; return (ok, message)
        raise NotImplementedError

    def resume_last(self):         # take over / restart the loop on the latest run dir; return (ok, message)
        raise NotImplementedError

    def busy_other(self):          # a non-panel run that must be stopped before the panel takes over, as text, or ""
        return ""

    def stop_other(self):          # stop that run cleanly; return message
        return ""

    def run_dir(self):             # the run the panel shows
        raise NotImplementedError

    def items(self):               # items of run_dir() in run order
        raise NotImplementedError

    def official(self):            # official-format table of run_dir() as text
        return ""

    def ensure_side(self):         # start helpers if missing (wheel daemon, viewer); return message
        return ""


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
    run_var, state_var = tk.StringVar(), tk.StringVar()
    ttk.Label(top, textvariable=run_var).pack(anchor="w")
    ttk.Label(top, textvariable=state_var, foreground="#c05000", font=("DejaVu Sans", 14, "bold")).pack(anchor="w")

    def info(msg):
        if msg:
            messagebox.showinfo(backend.title, msg)

    row1 = ttk.Frame(root, padding=(8, 2))
    row1.pack(fill="x")
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
            info(backend.new_batch(key)[1])

    def resume_last():
        if take_over():
            info(backend.resume_last()[1])
    ttk.Button(row1, text="New batch", command=new_batch).pack(side="left", padx=4)
    ttk.Button(row1, text="Resume last", command=resume_last).pack(side="left", padx=4)
    ttk.Button(row1, text="Official table", command=lambda: show_text("Official table", backend.official())).pack(side="right")

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
    for name, (label, default) in backend.options.items():
        v = tk.BooleanVar(value=c0["options"].get(name, default))
        opt_vars[name] = v

        def set_opt(name=name, v=v):
            c = ctl.read()
            ctl.update(options=dict(c["options"], **{name: v.get()}))
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
        tree.column(c, width=w, anchor="w" if c in ("scenario", "status") else "e")
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
    fmt = lambda v: "" if v is None else "%.1f" % v  # noqa: E731

    def refresh(force=False):
        try:
            items = backend.items()
        except Exception:  # never let the panel die on a half-written file
            items = []
        st = read_json(Path(backend.live) / "loop.json", {}) or {}
        c = ctl.read()
        run_var.set("run: %s" % (backend.run_dir() or "-"))
        alive = st.get("pid") and os.path.exists("/proc/%s" % st.get("pid"))
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
                    tree.insert(parent, "end", iid=it["id"], text=it["id"])
            shape["key"] = key
            ctl.update(order=[it["id"] for it in order])
        for it in order:
            tags = ("current",) if alive and st.get("item") == it["id"] else ("done",) if it["human"] is not None else ()
            tree.item(it["id"], values=(it["scenario"], fmt(it["human"]), fmt(it["model"]), it["status"]), tags=tags)
        for g, its in groups:
            if g is not None:
                d = [it["human"] for it in its if it["human"] is not None]
                tree.item("group:%s" % g, values=("", "%.1f" % (sum(d) / len(d)) if d else "", "", "%d / %d driven" % (len(d), len(its))))

    def tick():
        try:
            refresh()
        finally:
            root.after(1000, tick)
    info(backend.ensure_side())
    tick()
    root.mainloop()
