"use strict";
/* Chess Twin front-end: setup -> training progress -> play. No build step, no external dependencies. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const store = {
  get(k, d) { try { const v = localStorage.getItem("ct." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("ct." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
async function api(path, opts = {}) {
  const r = await fetch("/api" + path, {
    method: opts.method || (opts.body ? "POST" : "GET"),
    headers: opts.body ? { "Content-Type": "application/json" } : undefined,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  let data = null;
  try { data = await r.json(); } catch { /* empty */ }
  if (!r.ok) throw new Error((data && (typeof data.detail === "string" ? data.detail : data.detail?.[0]?.msg)) || `Request failed (${r.status})`);
  return data;
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 3500);
}
function show(view) {
  for (const v of $$(".view")) v.hidden = v.id !== "view-" + view;
  window.scrollTo(0, 0);
}

/* ------------------------------------------------------------------ setup */
let profiles = [];
const initials = (n) => (n || "?").slice(0, 2).toUpperCase();

async function loadStatus() {
  try {
    const s = await api("/status");
    $("#device-badge").textContent = s.device_name;
    $("#device-badge").title = s.device === "cpu" ? "Running on the CPU. A GPU is used automatically when available." : "Training is accelerated by your GPU";
    if (s.busy) resumeTraining();
  } catch { $("#device-badge").textContent = "offline"; }
}

async function loadProfiles() {
  profiles = await api("/profiles");
  const box = $("#profiles"); box.replaceChildren();
  $("#profiles-wrap").hidden = profiles.length === 0;
  for (const p of profiles) {
    const c = document.createElement("div"); c.className = "card pcard";
    const h = document.createElement("h3");
    const av = document.createElement("span"); av.className = "avatar"; av.textContent = p.kind === "base" ? "AI" : initials(p.name);
    h.append(av, p.name);
    const info = document.createElement("p");
    if (p.kind === "base") info.textContent = p.description;
    else {
      const acc = p.val_top1 != null ? ` · predicts ${(p.val_top1 * 100).toFixed(0)}% of your moves` : "";
      info.textContent = `${p.games} games · ${p.positions.toLocaleString()} moves${acc}`;
    }
    const row = document.createElement("div"); row.className = "row wrap";
    const play = btn("Play", "btn primary", () => startGame(p.id));
    row.append(play);
    if (p.kind === "twin") {
      row.append(btn("Retrain", "btn", () => { $("#username").value = p.name; $("#username").focus(); window.scrollTo(0, 0); }));
      row.append(btn("Delete", "btn", async () => {
        if (!confirm(`Delete the twin "${p.name}"? You can always recreate it.`)) return;
        try { await api("/profiles/" + encodeURIComponent(p.id), { method: "DELETE" }); } catch (e) { toast(e.message); }
        loadProfiles();
      }));
    }
    c.append(h, info, row); box.append(c);
  }
}
function btn(label, cls, fn) { const b = document.createElement("button"); b.type = "button"; b.className = cls; b.textContent = label; b.onclick = fn; return b; }

function restoreForm() {
  const f = store.get("form", {});
  if (f.username) $("#username").value = f.username;
  if (f.max_games) { $("#max-games").value = f.max_games; }
  if (f.time_controls) for (const c of $$("#tc-chips input")) c.checked = f.time_controls.includes(c.value);
  if (f.color) $(`input[name=color][value=${f.color}]`).checked = true;
  if (f.mode) $(`input[name=mode][value=${f.mode}]`).checked = true;
  $("#rated-only").checked = !!f.rated_only;
  $("#max-games-val").textContent = $("#max-games").value;
}
$("#max-games").oninput = (e) => ($("#max-games-val").textContent = e.target.value);

$("#setup-form").onsubmit = async (e) => {
  e.preventDefault();
  const err = $("#setup-error"); err.hidden = true;
  const body = {
    username: $("#username").value.trim(),
    max_games: +$("#max-games").value,
    time_controls: $$("#tc-chips input:checked").map((c) => c.value),
    color: $("input[name=color]:checked").value,
    mode: $("input[name=mode]:checked").value,
    rated_only: $("#rated-only").checked,
  };
  if (!body.time_controls.length) { err.textContent = "Pick at least one time control."; err.hidden = false; $(".advanced").open = true; return; }
  store.set("form", body);
  $("#create-btn").disabled = true;
  try {
    await api("/train", { body });
    showTraining(body.username);
  } catch (ex) { err.textContent = ex.message; err.hidden = false; }
  finally { $("#create-btn").disabled = false; }
};

/* ------------------------------------------------------------------ training */
let pollTimer = null;
function showTraining(name) {
  show("train");
  $("#train-title").textContent = `Building ${name || "your"}'s twin…`;
  $("#train-error").hidden = true; $("#train-done").hidden = true; $("#train-back").hidden = true;
  $("#train-cancel").hidden = false; $("#train-cancel").disabled = false;
  render({ status: "running", phase: "download", progress: 0, message: "Starting…", detail: {} });
  clearInterval(pollTimer); pollTimer = setInterval(poll, 700); poll();
}
async function resumeTraining() {
  const j = await api("/train").catch(() => null);
  if (j && j.status === "running") showTraining(j.params?.username);
}
const PHASE_SPAN = { download: [0, 0.35], train: [0.35, 0.95], save: [0.95, 1] };
function render(j) {
  const [a, b] = PHASE_SPAN[j.phase] || [0, 0];
  const total = j.status === "done" ? 1 : a + (b - a) * (j.progress || 0);
  $("#bar-fill").style.width = (total * 100).toFixed(1) + "%";
  $(".bar").setAttribute("aria-valuenow", Math.round(total * 100));
  $("#train-msg").textContent = j.message;
  const order = ["download", "train", "save"];
  for (const li of $$("#steps li")) {
    const i = order.indexOf(li.dataset.phase), cur = order.indexOf(j.phase);
    li.className = j.status === "done" || i < cur ? "done" : i === cur ? "active" : "";
  }
  const d = j.detail || {};
  $("#train-sub").textContent = j.phase === "download" ? "Downloading from Lichess - this is limited by Lichess, roughly 20-60 games per second."
    : j.phase === "train" ? "Training runs on your machine. A GPU is much faster than a CPU, but CPU is fine for a few hundred games." : "";
}
async function poll() {
  let j;
  try { j = await api("/train"); } catch { return; }
  if (j.status === "running") return render(j);
  clearInterval(pollTimer);
  $("#train-cancel").hidden = true;
  if (j.status === "done") {
    render(j);
    $("#train-title").textContent = "Your twin is ready!";
    const r = j.result, st = $("#train-stats"); st.replaceChildren();
    const add = (v, l) => { const d = document.createElement("div"); d.className = "stat"; const b = document.createElement("b"); b.textContent = v; const s = document.createElement("span"); s.textContent = l; d.append(b, s); st.append(d); };
    add(r.games, "games learned"); add(r.positions.toLocaleString(), "of your moves");
    if (r.val_top1 != null) add((r.val_top1 * 100).toFixed(0) + "%", "of your held-out moves predicted" + (r.base_val_top1 != null ? ` (generic: ${(r.base_val_top1 * 100).toFixed(0)}%)` : ""));
    $("#train-done").hidden = false; $("#train-sub").textContent = "";
    $("#play-new").onclick = () => startGame(r.profile);
    loadProfiles();
  } else {
    $("#train-title").textContent = j.status === "cancelled" ? "Cancelled" : "Something went wrong";
    if (j.status === "error") { $("#train-error").textContent = j.error; $("#train-error").hidden = false; }
    $("#train-msg").textContent = ""; $("#train-sub").textContent = "";
    $("#train-back").hidden = false;
  }
}
$("#train-cancel").onclick = async () => { $("#train-cancel").disabled = true; $("#train-msg").textContent = "Cancelling…"; await api("/train/cancel", { method: "POST" }).catch(() => {}); };
$("#train-back").onclick = () => { show("setup"); loadProfiles(); };

/* ------------------------------------------------------------------ board */
const PIECE = { P: "pawn", N: "knight", B: "bishop", R: "rook", Q: "queen", K: "king" };
const FILES = "abcdefgh";
let game = null, flipped = false, selected = null, busy = false, pendingPromo = null, drag = null, viewOver = false;

function parseFen(fen) {
  const map = {}; const rows = fen.split(" ")[0].split("/");
  rows.forEach((row, i) => { let f = 0; for (const ch of row) { if (/\d/.test(ch)) f += +ch; else { map[FILES[f] + (8 - i)] = ch; f++; } } });
  return map;
}
const pieceSrc = (ch) => `/static/pieces/${ch === ch.toUpperCase() ? "white" : "black"}-${PIECE[ch.toUpperCase()]}.png`;
const myTurn = () => game && !game.result && game.turn === game.human;

function drawBoard() {
  const board = $("#board"); board.replaceChildren();
  const pieces = parseFen(game.fen);
  const legalTargets = selected && game.legal[selected] ? game.legal[selected] : [];
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const file = flipped ? 7 - c : c, rank = flipped ? r + 1 : 8 - r, sq = FILES[file] + rank;
    const d = document.createElement("div");
    d.className = "sq" + ((file + rank) % 2 === 0 ? " dark" : "");
    d.dataset.sq = sq;
    if (game.last && game.last.includes(sq)) d.classList.add("last");
    if (sq === selected) d.classList.add("sel");
    if (game.check === sq) d.classList.add("check");
    if (c === 0) { const t = document.createElement("span"); t.className = "coord r"; t.textContent = rank; d.append(t); }
    if (r === 7) { const t = document.createElement("span"); t.className = "coord f"; t.textContent = FILES[file]; d.append(t); }
    const p = pieces[sq];
    if (p) { const im = document.createElement("img"); im.src = pieceSrc(p); im.alt = ""; im.draggable = false; d.append(im); }
    if (legalTargets.includes(sq)) { d.classList.add("target"); if (p) d.classList.add("cap"); const dot = document.createElement("span"); dot.className = "dot"; d.append(dot); }
    board.append(d);
  }
}

function drawSide() {
  const moves = $("#moves"); moves.replaceChildren();
  game.moves.forEach((san, i) => {
    if (i % 2 === 0) { const n = document.createElement("span"); n.className = "n"; n.textContent = i / 2 + 1 + "."; moves.append(n); }
    const m = document.createElement("span"); m.className = "m" + (i === game.moves.length - 1 ? " cur" : ""); m.textContent = san; moves.append(m);
  });
  moves.scrollTop = moves.scrollHeight;
  const r = game.result;
  let s;
  if (r) s = resultText().title;
  else if (busy || game.bot_to_move) s = "Opponent is thinking…";
  else s = "Your move" + (game.check ? " - you're in check" : "");
  $("#status").textContent = s;
  $("#think").hidden = r || !(busy || game.bot_to_move);
  const prof = profiles.find((p) => p.id === game.profile);
  $("#opp-name").textContent = prof ? prof.name + (prof.kind === "twin" ? " (twin)" : "") : game.profile;
  $("#opp-avatar").textContent = prof && prof.kind === "twin" ? initials(prof.name) : "AI";
  $("#btn-undo").disabled = busy || game.moves.length === 0;
  $("#btn-resign").disabled = !!r;
  const go = $("#gameover");
  go.hidden = !r || viewOver;
  if (r) { const t = resultText(); $("#go-title").textContent = t.title; $("#go-sub").textContent = t.sub; }
}
function resultText() {
  const { score, reason } = game.result;
  const iWhite = game.human === "white";
  if (score === "1/2-1/2") return { title: "Draw", sub: reason[0].toUpperCase() + reason.slice(1) };
  const won = (score === "1-0") === iWhite;
  return { title: won ? "You won!" : "You lost", sub: (reason === "checkmate" ? "Checkmate" : reason[0].toUpperCase() + reason.slice(1)) + (won ? " - well played." : " - your twin got you.") };
}
function render_() { drawBoard(); drawSide(); }

function squareAt(x, y) { const el = document.elementFromPoint(x, y); const s = el && el.closest(".sq"); return s ? s.dataset.sq : null; }

function onPointerDown(e) {
  if (!game || busy || pendingPromo || e.button > 0) return;
  const el = e.target.closest(".sq"); if (!el) return;
  const sq = el.dataset.sq;
  if (!myTurn()) return;
  if (selected && game.legal[selected]?.includes(sq)) { attempt(selected, sq); return; }
  if (game.legal[sq]) {
    selected = sq; render_();
    const img = $(`.sq[data-sq=${sq}] img`);
    if (img) drag = { from: sq, x: e.clientX, y: e.clientY, ghost: null, img };
  } else { selected = null; render_(); }
}
function onPointerMove(e) {
  if (!drag) return;
  if (!drag.ghost && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 5) {
    const g = drag.img.cloneNode(); g.className = "ghost";
    const w = $(".sq").getBoundingClientRect().width; g.style.setProperty("--sqpx", w + "px");
    document.body.append(g); drag.ghost = g; drag.img.closest(".sq").classList.add("dragging");
  }
  if (drag.ghost) { drag.ghost.style.left = e.clientX + "px"; drag.ghost.style.top = e.clientY + "px"; }
}
function onPointerUp(e) {
  if (!drag) return;
  const d = drag; drag = null;
  if (d.ghost) {
    d.ghost.remove();
    const to = squareAt(e.clientX, e.clientY);
    if (to && to !== d.from && game.legal[d.from]?.includes(to)) attempt(d.from, to);
    else render_();
  }
}
$("#board").addEventListener("pointerdown", onPointerDown);
window.addEventListener("pointermove", onPointerMove);
window.addEventListener("pointerup", onPointerUp);
window.addEventListener("pointercancel", () => { if (drag?.ghost) drag.ghost.remove(); drag = null; });

function attempt(from, to) {
  if (game.promotions.includes(from + to)) {
    pendingPromo = { from, to };
    const box = $("#promo"), row = document.createElement("div");
    for (const [k, name] of [["q", "queen"], ["r", "rook"], ["b", "bishop"], ["n", "knight"]]) {
      const b = document.createElement("button"); b.setAttribute("aria-label", name);
      const im = document.createElement("img"); im.src = `/static/pieces/${game.human}-${name}.png`; im.alt = name; b.append(im);
      b.onclick = () => { box.hidden = true; pendingPromo = null; submit(from + to + k); };
      row.append(b);
    }
    box.replaceChildren(row); box.hidden = false;
    box.onclick = (e) => { if (e.target === box) { box.hidden = true; pendingPromo = null; selected = null; render_(); } };
    return;
  }
  submit(from + to);
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
async function submit(uci) {
  busy = true; selected = null;
  try {
    game = await api(`/games/${game.id}/move`, { body: { uci } });
    render_();
    await botTurn();
  } catch (e) { toast(e.message); try { game = await api(`/games/${game.id}`); } catch { /* ignore */ } }
  busy = false; render_();
}
async function botTurn() {
  if (!game.bot_to_move) return;
  busy = true; render_();
  const t0 = performance.now();
  const g = await api(`/games/${game.id}/bot-move`, { body: { temperature: temperature() } });
  const left = 450 - (performance.now() - t0);
  if (left > 0) await wait(left);        // a tiny pause so moves don't feel instantaneous
  game = g; busy = false; render_();
}

async function startGame(profile, color) {
  try {
    viewOver = false; selected = null; busy = false;
    color = color || store.get("color", "white");
    game = await api("/games", { body: { profile, color } });
    store.set("profile", profile);
    flipped = game.human === "black";
    show("play"); render_();
    await botTurn();
  } catch (e) { toast(e.message); show("setup"); loadProfiles(); }
}

/* style slider: 0 = always the best move, 50 = sample like the player (temperature 1), 100 = looser */
const temperature = () => +$("#style").value / 50;
function styleText() {
  const v = +$("#style").value;
  const [label, hint] = v < 10 ? ["Strongest", "Always plays its top choice. Least human, hardest to beat."]
    : v < 40 ? ["Sharper than you", "Favours its best ideas, so it plays more precisely than the real you."]
    : v <= 60 ? ["Like you", "Picks moves with the same odds you would - habits and occasional mistakes included."]
    : v <= 85 ? ["Looser than you", "More variety and more slips than you usually make."]
    : ["Chaotic", "Near-random wandering. Expect blunders."];
  $("#style-label").textContent = label; $("#style-hint").textContent = hint;
}
$("#style").oninput = () => { styleText(); store.set("style", +$("#style").value); };

$("#btn-flip").onclick = () => { flipped = !flipped; render_(); };
$("#btn-undo").onclick = async () => { if (busy) return; viewOver = false; selected = null; game = await api(`/games/${game.id}/undo`, { method: "POST" }); render_(); };
$("#btn-resign").onclick = async () => { if (confirm("Resign this game?")) { game = await api(`/games/${game.id}/resign`, { method: "POST" }); viewOver = false; render_(); } };
$("#btn-menu").onclick = () => { show("setup"); loadProfiles(); };
$("#brand").onclick = (e) => { e.preventDefault(); show("setup"); loadProfiles(); };
$("#go-close").onclick = () => { viewOver = true; render_(); };
function askNewGame() {
  const dlg = $("#newgame-dialog");
  dlg.onclose = () => { if (dlg.returnValue && dlg.returnValue !== "cancel") { store.set("color", dlg.returnValue); startGame(game.profile, dlg.returnValue); } };
  dlg.returnValue = ""; dlg.showModal();
}
$("#btn-new").onclick = askNewGame; $("#go-new").onclick = askNewGame;

/* ------------------------------------------------------------------ boot */
(async function init() {
  $("#style").value = store.get("style", 50); styleText();
  restoreForm();
  await loadStatus();
  try { await loadProfiles(); } catch { toast("Could not reach the Chess Twin server."); }
})();
