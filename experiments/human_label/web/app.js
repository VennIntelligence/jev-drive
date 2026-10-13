'use strict';
// Human-label page: video + BEV + timeline for one clip at a time; every key press is one record on the server (POST api/labels).
const $ = (id) => document.getElementById(id);
const KEYS = { '1': 'observable', '2': 'brake', '3': 'turn', '4': 'unwind', '9': 'verdict_wrong', 'n': 'note', 'd': 'done' };
const KCOL = { '1': '#2a9d8f', '2': '#d1495b', '3': '#3d6fd8', '4': '#9b5de5', '9': '#e08e0b', 'n': '#555', 'd': '#18794e' };
const CMDCOL = { S: '#d5d9df', L: '#3d6fd8', R: '#e8590c', '?': '#999' };
const RATES = [0.5, 1, 2];
const ls = { get: (k, d) => { const v = localStorage.getItem('hl.' + k); return v === null ? d : v; }, set: (k, v) => localStorage.setItem('hl.' + k, v) };

const S = {
  man: null, status: {}, list: [], cur: null, clip: null, labels: [], view: ls.get('view', 'crop'), bev: ls.get('bev', 'fixed'),
  zoom: parseFloat(ls.get('zoom', '1')), lastF: -1, dirty: true, note: null,
};
const vid = $('vid');

// ------------------------------------------------------------------ api
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (r.status === 401) { location.reload(); throw new Error('login required'); }
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

let toastT = 0;
function toast(msg, err) {
  const t = $('toast'); t.textContent = msg; t.className = 'on' + (err ? ' err' : '');
  clearTimeout(toastT); toastT = setTimeout(() => { t.className = ''; }, err ? 4000 : 1800);
}

// ------------------------------------------------------------------ time base
const fps = () => S.clip.video.fps;
const nFrames = () => S.clip.video.n;
const tFirst = () => S.clip.video.t_first;
function frame() { return Math.max(0, Math.min(nFrames() - 1, Math.floor(vid.currentTime * fps() + 1e-4))); }
const simT = (f) => tFirst() + f / fps();
function seekFrame(f) { f = Math.max(0, Math.min(nFrames() - 1, f)); vid.currentTime = (f + 0.5) / fps(); S.dirty = true; }
function seekT(t) { seekFrame(Math.round((t - tFirst()) * fps())); }
function idxLE(arr, t) { // last index with arr[i] <= t (0 if none)
  let lo = 0, hi = arr.length - 1;
  if (t < arr[0]) return 0;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (arr[m] <= t + 1e-6) lo = m; else hi = m - 1; }
  return lo;
}
function interp(ts, xs, t) {
  if (t <= ts[0]) return xs[0];
  const i = idxLE(ts, t); if (i >= ts.length - 1) return xs[ts.length - 1];
  const w = (t - ts[i]) / (ts[i + 1] - ts[i]); return xs[i] + w * (xs[i + 1] - xs[i]);
}
function stateAt(f) {
  const c = S.clip, t = simT(f), e = c.ego;
  const i = idxLE(e.t, t), j = idxLE(S.decT, t);
  return { t, f, i, dec: c.decisions[j], x: e.x[i], y: e.y[i], yaw: e.yaw[i], v: e.vx[i], d: e.d_ent[i], lat: e.lat[i] };
}

// ------------------------------------------------------------------ clip list
function buildList() {
  const mode = $('armSel').value;
  S.list = S.man.clips.filter((c) => mode === 'all' || c.primary);
  const ul = $('list'); ul.innerHTML = '';
  for (const c of S.list) {
    const li = document.createElement('li'); li.dataset.id = c.id;
    if (mode === 'all' && !c.primary) li.classList.add('sec');
    const sc = c.score === null ? '-' : c.score.toFixed(2);
    li.innerHTML = `<span class="nm">${c.scene.slice(7, 15)} &middot; ${c.arm}</span><span class="st"></span>` +
      `<span class="sub">${c.dir}${Math.abs(c.ang).toFixed(0)}&deg; &middot; ${sc}${c.kind ? ' ' + c.kind : ''} &middot; ${c.cls}</span>`;
    li.onclick = () => openClip(c.id);
    ul.appendChild(li);
  }
  paintList();
}
function paintList() {
  let done = 0;
  for (const li of $('list').children) {
    const st = S.status[li.dataset.id] || { n: 0, done: false };
    done += st.done;
    li.classList.toggle('done', st.done); li.classList.toggle('cur', li.dataset.id === S.cur);
    li.querySelector('.st').textContent = (st.done ? 'done' : 'todo') + (st.n ? ` · ${st.n}` : '');
  }
  $('progress').textContent = `${done} / ${S.list.length} done · batch ${S.man.batch}`;
}
function recomputeStatus(id) {
  const undone = new Set(S.labels.filter((r) => r.type === 'undo').map((r) => r.ref));
  const act = S.labels.filter((r) => r.type === 'label' && !undone.has(r.id));
  S.status[id] = { n: act.filter((r) => r.key !== 'd').length, done: act.some((r) => r.key === 'd') };
  return act;
}
const activeLabels = () => recomputeStatus(S.cur);

// ------------------------------------------------------------------ open a clip
async function openClip(id) {
  if (!id) return;
  if (S.cur && S.clip) ls.set('pos.' + S.cur, frame());
  vid.pause();
  S.cur = id; ls.set('cur', id);
  const [clip, labels] = await Promise.all([api('api/clip/' + encodeURIComponent(id)), api('api/labels?clip=' + encodeURIComponent(id))]);
  if (S.cur !== id) return;
  S.clip = clip; S.labels = labels; S.decT = clip.decisions.map((d) => d.t);
  recomputeStatus(id);
  vid.src = 'video/' + encodeURIComponent(id);
  vid.playbackRate = parseFloat($('rate').value);
  const f0 = parseInt(ls.get('pos.' + id, '0'), 10) || 0;
  vid.onloadedmetadata = () => { layout(); seekFrame(f0); };
  header(); layout(); paintList(); S.dirty = true;
  const li = [...$('list').children].find((x) => x.dataset.id === id); if (li) li.scrollIntoView({ block: 'nearest' });
}

function header() {
  const c = S.clip, v = c.verdict, tu = c.turn;
  $('hScene').textContent = c.scene.slice(7, 15); $('hScene').title = c.scene + '\nrollout ' + c.rollout;
  $('hArm').textContent = `${c.arm} · ${c.tag}`;
  $('hSplit').textContent = c.split + (c.primary ? '' : ' · secondary arm');
  $('hScore').innerHTML = `score <b class="${c.score > 0 ? 'good' : 'bad'}">${c.score === null ? '-' : c.score.toFixed(3)}</b>`;
  $('hZero').innerHTML = c.kind ? `zero <b class="bad">${c.kind}</b> at ${c.t_zero === null ? '?' : c.t_zero.toFixed(1) + ' s'}` : 'no zero event';
  $('hTurn').innerHTML = `turn <b>${tu.dir} ${Math.abs(tu.ang).toFixed(0)}&deg;</b> · entry ${tu.ent_s.toFixed(0)} m along log · cmd at t0 <b>${c.cmd0}</b>`;
  $('hVerdict').innerHTML = `completed <b class="${v.completed ? 'good' : 'bad'}">${v.completed ? 'yes' : 'no'}</b>` +
    ` (yaw ${v.yaw_net.toFixed(0)}&deg;) · class <b>${v.cls}</b>` +
    (v.max_in === null ? '' : ` <span class="muted">in ${v.max_in.toFixed(1)} / out ${v.max_out.toFixed(1)} m</span>`);
  $('hDone').hidden = !(S.status[S.cur] || {}).done;
}

function live(st) {
  const c = S.clip;
  $('lT').textContent = st.t.toFixed(1); $('lF').textContent = `${st.f}/${nFrames() - 1}`;
  $('lK').textContent = st.dec.k;
  const cm = $('lCmd'); cm.textContent = st.dec.cmd; cm.className = 'cmd ' + st.dec.cmd;
  $('lV').textContent = st.v.toFixed(1); $('lD').textContent = st.d.toFixed(1); $('lLat').textContent = st.lat.toFixed(1);
  const tl = c.turn.t_log_ent; $('lLog').textContent = tl === null ? '-' : (tl - st.t >= 0 ? `in ${(tl - st.t).toFixed(1)} s` : `${(st.t - tl).toFixed(1)} s ago`);
}

// ------------------------------------------------------------------ layout
function layout() {
  const st = $('stage'), H = st.clientHeight - 16, W = st.clientWidth - 16, dpr = window.devicePixelRatio || 1;
  const v = S.clip ? S.clip.video : { w: 900, h: 1000 }, crop = S.man.video_crop;
  const [cx, cy, cw, ch] = S.view === 'full' || !crop ? [0, 0, v.w, v.h] : crop;
  // the video gets the left column (timeline + label row under it), the BEV takes the rest of the width at full height
  const below = 104 + 20 + 12, minBev = Math.min(380, W * 0.34);
  const vw = Math.max(320, Math.min(W - 8 - minBev, (H - below) * cw / ch)), sc = vw / cw;
  const box = $('vidbox'); box.style.width = Math.floor(cw * sc) + 'px'; box.style.height = Math.floor(ch * sc) + 'px';
  vid.style.width = (v.w * sc) + 'px'; vid.style.height = (v.h * sc) + 'px';
  vid.style.left = (-cx * sc) + 'px'; vid.style.top = (-cy * sc) + 'px';
  $('left').style.width = Math.floor(cw * sc) + 'px';
  const tl = $('tl'); tl.style.width = Math.floor(cw * sc) + 'px'; tl.style.height = Math.max(104, Math.min(200, H - ch * sc - 32)) + 'px'; tl.width = Math.round(tl.clientWidth * dpr); tl.height = Math.round(tl.clientHeight * dpr);
  const bev = $('bev'), bw = Math.floor(W - 8 - cw * sc), bh = Math.max(200, H - 24);
  bev.style.width = bw + 'px'; bev.style.height = bh + 'px'; bev.width = Math.round(bw * dpr); bev.height = Math.round(bh * dpr);
  S.dirty = true;
}

// ------------------------------------------------------------------ BEV
function drawBev(st) {
  const cv = $('bev'), g = cv.getContext('2d'), dpr = window.devicePixelRatio || 1, Wc = cv.width / dpr, Hc = cv.height / dpr, c = S.clip;
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, Wc, Hc);
  const ref = S.bev === 'ego' ? st.yaw : c.ego.yaw[0];
  const scale = (Math.min(Wc, Hc) / 90) * S.zoom, ox = Wc / 2, oy = Hc * 0.62, ca = Math.cos(Math.PI / 2 - ref), sa = Math.sin(Math.PI / 2 - ref);
  const P = (x, y) => { const dx = x - st.x, dy = y - st.y; return [ox + scale * (ca * dx - sa * dy), oy - scale * (sa * dx + ca * dy)]; };
  const line = (xs, ys, col, w, dash, alpha) => {
    if (xs.length < 2) return;
    g.save(); g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.globalAlpha = alpha || 1; g.lineJoin = 'round';
    g.beginPath(); for (let i = 0; i < xs.length; i++) { const [a, b] = P(xs[i], ys[i]); i ? g.lineTo(a, b) : g.moveTo(a, b); } g.stroke(); g.restore();
  };
  const dot = (x, y, r, col, fill = true) => { const [a, b] = P(x, y); g.beginPath(); g.arc(a, b, r, 0, 2 * Math.PI); if (fill) { g.fillStyle = col; g.fill(); } else { g.strokeStyle = col; g.lineWidth = 2; g.stroke(); } };
  const box = (x, y, yaw, col, fill) => {
    const cs = Math.cos(yaw), sn = Math.sin(yaw), pts = [[-1.0, -1.0], [3.8, -1.0], [3.8, 1.0], [-1.0, 1.0]].map(([u, w]) => P(x + cs * u - sn * w, y + sn * u + cs * w));
    g.beginPath(); pts.forEach(([a, b], i) => (i ? g.lineTo(a, b) : g.moveTo(a, b))); g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); } g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
  };
  // route: received so far solid, the rest faint
  const R = c.route, nk = Math.min(st.dec.n_route, R.length);
  line(R.map((p) => p[0]), R.map((p) => p[1]), '#74a9e8', 2, [2, 4], 0.35);
  line(R.slice(0, nk).map((p) => p[0]), R.slice(0, nk).map((p) => p[1]), '#74a9e8', 5, [], 0.55);
  // human log path + the human's car at the same time
  line(c.log.x, c.log.y, '#7a7a7a', 1.6, [6, 5]);
  const lx = interp(c.log.t, c.log.x, st.t), ly = interp(c.log.t, c.log.y, st.t), lyaw = interp(c.log.t, c.log.yaw, st.t);
  box(lx, ly, lyaw, '#666', 'rgba(120,120,120,0.18)');
  // entry / exit
  const tu = c.turn;
  dot(tu.ent_xy[0], tu.ent_xy[1], 6, '#2b9348'); dot(tu.ext_xy[0], tu.ext_xy[1], 5, '#2b9348', false);
  g.fillStyle = '#2b9348'; g.font = '11px system-ui'; { const [a, b] = P(...tu.ent_xy); g.fillText('entry', a + 8, b + 4); }
  // ego so far
  const e = c.ego, i = st.i;
  line(e.x.slice(0, i + 1), e.y.slice(0, i + 1), '#1d2129', 2.2);
  // zero event position
  if (c.t_zero !== null && st.t >= c.t_zero) {
    const z = idxLE(e.t, c.t_zero), [a, b] = P(e.x[z], e.y[z]);
    g.strokeStyle = '#d00'; g.lineWidth = 2.5; g.beginPath(); g.moveTo(a - 6, b - 6); g.lineTo(a + 6, b + 6); g.moveTo(a + 6, b - 6); g.lineTo(a - 6, b + 6); g.stroke();
  }
  // served plan (8 poses, 0.5 .. 4 s) from the decision's anchor
  const d = st.dec, px = [st.x, ...d.plan.map((p) => p[0])], py = [st.y, ...d.plan.map((p) => p[1])];
  line(px, py, '#f08c00', 2.5);
  d.plan.forEach((p) => dot(p[0], p[1], 2.6, '#f08c00'));
  if (d.route0) { const [a, b] = P(...d.route0); g.fillStyle = '#1c64c8'; g.beginPath(); g.moveTo(a, b - 5); g.lineTo(a + 5, b); g.lineTo(a, b + 5); g.lineTo(a - 5, b); g.fill(); }
  box(st.x, st.y, st.yaw, '#1d2129', 'rgba(31,78,140,0.85)');
  // scale bar + mode
  const m = scale * 20 >= 60 ? 20 : 50, L = scale * (m === 20 ? 20 : 50);
  g.fillStyle = '#1d2129'; g.fillRect(12, Hc - 16, L, 3); g.font = '11px system-ui'; g.fillText(`${m} m`, 12, Hc - 22);
  g.fillStyle = '#6b7280'; g.fillText(S.bev === 'ego' ? 'ego-up' : 'fixed: start heading up', 10, 16);
  g.fillText(`plan k ${d.k} · cmd ${d.cmd}`, 10, 30);
}

// ------------------------------------------------------------------ timeline
function tlGeom() {
  const cv = $('tl'), dpr = window.devicePixelRatio || 1, W = cv.width / dpr, H = cv.height / dpr;
  const t0 = tFirst(), t1 = tFirst() + (nFrames() - 1) / fps(), L = 34, Rm = 10;
  return { cv, dpr, W, H, t0, t1, X: (t) => L + (W - L - Rm) * (t - t0) / (t1 - t0), T: (x) => t0 + (x - L) / (W - L - Rm) * (t1 - t0), L, Rm };
}
function drawTl(st) {
  const G = tlGeom(), g = G.cv.getContext('2d'), c = S.clip, { W, H, X } = G;
  g.setTransform(G.dpr, 0, 0, G.dpr, 0, 0); g.clearRect(0, 0, W, H); g.font = '10px system-ui';
  // command band
  const ds = c.decisions;
  for (let j = 0; j < ds.length; j++) {
    const a = X(ds[j].t), b = j + 1 < ds.length ? X(ds[j + 1].t) : X(G.t1);
    g.fillStyle = CMDCOL[ds[j].cmd] || '#999'; g.fillRect(a, 4, Math.max(1, b - a + 0.5), 10);
  }
  g.fillStyle = '#6b7280'; g.fillText('cmd', 4, 13); g.fillText('speed', 4, 44);
  // speed (ego, log dashed)
  const top = 20, bot = H - 22, vmax = Math.max(5, ...c.ego.vx, ...c.log.v.filter((_, k) => c.log.t[k] <= G.t1)) * 1.08;
  const Y = (v) => bot - (bot - top) * v / vmax;
  g.strokeStyle = '#eceef1'; g.lineWidth = 1; for (let v = 5; v < vmax; v += 5) { g.beginPath(); g.moveTo(G.L, Y(v)); g.lineTo(W - G.Rm, Y(v)); g.stroke(); }
  const curve = (ts, vs, col, dash) => { g.strokeStyle = col; g.setLineDash(dash || []); g.lineWidth = 1.5; g.beginPath(); let s = false;
    for (let k = 0; k < ts.length; k++) { if (ts[k] < G.t0 || ts[k] > G.t1) continue; const p = [X(ts[k]), Y(vs[k])]; s ? g.lineTo(...p) : g.moveTo(...p); s = true; } g.stroke(); g.setLineDash([]); };
  curve(c.log.t, c.log.v, '#a0a4ab', [4, 3]); curve(c.ego.t, c.ego.vx, '#1d2129');
  g.fillStyle = '#9aa0a6'; g.fillText(`${Math.round(vmax / 1.08)} m/s`, 4, top + 8);
  const vline = (t, col, label, dash, y = 26) => { if (t === null || t === undefined) return; const x = X(t); g.strokeStyle = col; g.lineWidth = 1.5; g.setLineDash(dash || []);
    g.beginPath(); g.moveTo(x, 4); g.lineTo(x, bot); g.stroke(); g.setLineDash([]); if (label) { g.fillStyle = col; g.fillText(label, x + 3, y); } };
  vline(c.handoff, '#9aa0a6', 'GT end', [3, 3], bot - 4);
  vline(c.turn.t_log_ent, '#2b9348', 'log entry', [5, 3], 28);
  vline(c.turn.t_ego_ent, '#2b9348', 'ego entry', [], 40);
  vline(c.t_zero, '#d00', c.kind || 'zero', [], 52);
  let lastX = -1e9, row = 0;
  for (const fl of c.flips) {
    const x = X(fl.t); row = x - lastX < 30 ? row + 1 : 0; lastX = x;
    g.fillStyle = '#1d2129'; g.beginPath(); g.moveTo(x, 15); g.lineTo(x - 4, 21); g.lineTo(x + 4, 21); g.fill(); g.fillText(`${fl.frm}>${fl.to}`, x + 4, 23 + 10 * (row % 3));
  }
  // labels
  for (const r of activeLabels()) {
    if (r.t === null || r.key === 'd') continue;
    const x = X(r.t); g.strokeStyle = KCOL[r.key]; g.lineWidth = 2; g.beginPath(); g.moveTo(x, bot + 11); g.lineTo(x, H); g.stroke();
    g.fillStyle = KCOL[r.key]; g.font = 'bold 11px system-ui'; g.fillText(r.key, x + 3, H - 1); g.font = '10px system-ui';
  }
  // seconds axis + playhead
  g.fillStyle = '#9aa0a6';
  for (let s = Math.ceil(G.t0); s <= G.t1; s += 2) g.fillText(String(s), X(s) - 3, bot + 9);
  const x = X(st.t); g.strokeStyle = '#000'; g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
}
function drawLabList() {
  const el = $('lablist'); el.innerHTML = '';
  for (const r of activeLabels()) {
    const s = document.createElement('span'); s.className = 'lb'; s.style.background = KCOL[r.key];
    s.textContent = r.key === 'd' ? 'done' : `${r.key} ${r.value} ${r.t.toFixed(1)} s${r.d_entry === null ? '' : ' / ' + r.d_entry.toFixed(0) + ' m'}${r.note ? ': ' + r.note : ''}`;
    s.title = `${r.labeller} · ${r.wall}`; if (r.t !== null) s.onclick = () => { vid.pause(); seekT(r.t); };
    el.appendChild(s);
  }
}

// ------------------------------------------------------------------ render loop
function tick() {
  if (S.clip && vid.readyState >= 1) {
    const f = frame();
    if (f !== S.lastF || S.dirty) { const st = stateAt(f); live(st); drawBev(st); drawTl(st); S.lastF = f; if (S.dirty) drawLabList(); S.dirty = false; }
  }
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ labels
async function post(key, note) {
  const st = stateAt(frame());
  try {
    const r = await api('api/labels', { clip: S.cur, key, t: st.t, frame: st.f, k: st.dec.k, d_entry: st.d, speed: st.v, cmd: st.dec.cmd, note, labeller: who() });
    S.labels.push(r); recomputeStatus(S.cur); paintList(); header(); S.dirty = true;
    toast(key === 'd' ? 'clip marked done' : `${key} ${KEYS[key]} at t ${st.t.toFixed(1)} s, ${st.d.toFixed(1)} m to entry${note ? ' - ' + note : ''}`);
  } catch (e) { toast('not saved: ' + e.message, true); }
}
async function undo(ref) {
  const act = activeLabels(); const r = ref || act[act.length - 1];
  if (!r) { toast('nothing to undo in this clip'); return; }
  try {
    const u = await api('api/labels', { undo: r.id, labeller: who() });
    S.labels.push(u); recomputeStatus(S.cur); paintList(); header(); S.dirty = true;
    toast(`undone: ${r.key === 'd' ? 'done' : r.key + ' ' + r.value + ' at ' + r.t.toFixed(1) + ' s'}`);
  } catch (e) { toast('undo failed: ' + e.message, true); }
}
function openNote(key) {
  vid.pause(); S.note = { key, f: frame() };
  $('noteLabel').textContent = key === '9' ? `9 rule verdict wrong at t ${simT(S.note.f).toFixed(1)} s - what is wrong?` : `note at t ${simT(S.note.f).toFixed(1)} s`;
  $('notebar').hidden = false; $('noteIn').value = ''; $('noteIn').focus();
}
function closeNote(save) {
  const n = S.note; S.note = null; $('notebar').hidden = true; $('noteIn').blur();
  if (!save || !n) return;
  const text = $('noteIn').value.trim();
  if (n.key === 'n' && !text) { toast('empty note, not saved'); return; }
  seekFrame(n.f); post(n.key, text || null);
}
const who = () => ($('who').value.trim() || 'anon');

// ------------------------------------------------------------------ keys
function step(df) { vid.pause(); seekFrame(frame() + df); }
function nextClip(d) { const i = S.list.findIndex((c) => c.id === S.cur); const j = Math.max(0, Math.min(S.list.length - 1, i + d)); if (S.list[j]) openClip(S.list[j].id); }
document.addEventListener('keydown', (ev) => {
  if (S.note) { if (ev.key === 'Enter') { ev.preventDefault(); closeNote(true); } else if (ev.key === 'Escape') { ev.preventDefault(); closeNote(false); } return; }
  if (ev.target.tagName === 'SELECT') ev.target.blur();
  if (ev.target.tagName === 'INPUT' || ev.metaKey || ev.ctrlKey || ev.altKey || !S.clip) return;
  const k = ev.key;
  if (k === ' ') { ev.preventDefault(); if (vid.paused) { if (frame() >= nFrames() - 1) seekFrame(0); vid.play(); } else vid.pause(); }
  else if (k === 'ArrowLeft' || k === 'ArrowRight') { ev.preventDefault(); step((k === 'ArrowLeft' ? -1 : 1) * (ev.shiftKey ? Math.round(fps()) : 1)); }
  else if (k === '[') nextClip(-1);
  else if (k === ']') nextClip(1);
  else if ('1234'.includes(k) && k.length === 1) post(k);
  else if (k === '9' || k === 'n') { ev.preventDefault(); openNote(k); }
  else if (k === 'u') undo();
  else if (k === 'd') { const d = activeLabels().filter((r) => r.key === 'd'); d.length ? undo(d[d.length - 1]) : post('d'); }
  else if (k === 'v') { S.view = S.view === 'full' ? 'crop' : 'full'; ls.set('view', S.view); layout(); }
  else if (k === 'b') { S.bev = S.bev === 'ego' ? 'fixed' : 'ego'; ls.set('bev', S.bev); S.dirty = true; }
  else if (k === '+' || k === '=') { S.zoom = Math.min(6, S.zoom * 1.25); ls.set('zoom', S.zoom); S.dirty = true; }
  else if (k === '-' || k === '_') { S.zoom = Math.max(0.25, S.zoom / 1.25); ls.set('zoom', S.zoom); S.dirty = true; }
  else if (k === 's') { const r = $('rate'); r.value = String(RATES[(RATES.indexOf(parseFloat(r.value)) + 1) % RATES.length]); vid.playbackRate = parseFloat(r.value); toast(`speed ${r.value}x`); }
});

// timeline scrubbing
(() => {
  const cv = $('tl'); let drag = false;
  const at = (ev) => { const G = tlGeom(), r = cv.getBoundingClientRect(); seekT(G.T(ev.clientX - r.left)); };
  cv.addEventListener('mousedown', (ev) => { if (!S.clip) return; drag = true; vid.pause(); at(ev); });
  window.addEventListener('mousemove', (ev) => { if (drag) at(ev); });
  window.addEventListener('mouseup', () => { drag = false; });
})();

// ------------------------------------------------------------------ boot
(async () => {
  S.man = await api('api/clips');
  S.status = S.man.status;
  $('commit').textContent = `server ${S.man.commit}`;
  $('who').value = ls.get('who', ''); $('who').onchange = () => { ls.set('who', $('who').value.trim()); $('who').blur(); };
  $('armSel').value = ls.get('arm', 'primary');
  $('armSel').onchange = () => { ls.set('arm', $('armSel').value); buildList(); $('armSel').blur(); };
  $('rate').onchange = () => { vid.playbackRate = parseFloat($('rate').value); $('rate').blur(); };
  buildList();
  window.addEventListener('resize', layout);
  window.addEventListener('beforeunload', () => { if (S.cur && S.clip) ls.set('pos.' + S.cur, frame()); });
  vid.addEventListener('seeked', () => { S.dirty = true; });
  const cur = ls.get('cur', null);
  await openClip(S.man.clips.some((c) => c.id === cur) ? cur : (S.list[0] || {}).id);
  requestAnimationFrame(tick);
})().catch((e) => toast('load failed: ' + e.message, true));
