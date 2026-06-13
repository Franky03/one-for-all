/* One for All — custom frontend wired to the gr.Server backend.
 * Visual design is unchanged from the static mock; the mock answer engine and
 * the random thought-map are replaced by real calls to /probe, /arena, /viz
 * via @gradio/client (browser-side calls are required for ZeroGPU). */
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/+esm";

const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const cssv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ── teacher lineup (order matches the backend gate vector) ──
const TEACHERS = [
  { id: "qwen",     label: "qwen",     varc: "--t-qwen" },
  { id: "smollm",   label: "smollm",   varc: "--t-smollm" },
  { id: "phi",      label: "phi",      varc: "--t-phi" },
  { id: "gemma",    label: "gemma",    varc: "--t-gemma" },
  { id: "minicpm",  label: "minicpm",  varc: "--t-minicpm" },
  { id: "nemotron", label: "nemotron", varc: "--t-nemo" },
];
const colorFor = (name) =>
  name === "student" ? cssv("--spark")
    : cssv((TEACHERS.find((t) => t.id === name) || {}).varc || "--text-dim");

// ── 1. CONVERGENCE DIAGRAM (unchanged) ──────────────────────────────────
(function buildConverge() {
  const cx = 300, cy = 300, R = 232;
  const threads = document.getElementById("threads");
  const flows = document.getElementById("flows");
  const tnodes = document.getElementById("tnodes");
  TEACHERS.forEach((t, i) => {
    const a = (-90 + i * 60) * Math.PI / 180;
    const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
    const mx = (x + cx) / 2, my = (y + cy) / 2;
    const perp = a + Math.PI / 2;
    const ctrlX = mx + 46 * Math.cos(perp), ctrlY = my + 46 * Math.sin(perp);
    const d = `M ${x} ${y} Q ${ctrlX} ${ctrlY} ${cx} ${cy}`;
    const col = cssv(t.varc);

    const base = document.createElementNS("http://www.w3.org/2000/svg", "path");
    base.setAttribute("d", d); base.setAttribute("class", "thread");
    base.setAttribute("stroke", col);
    threads.appendChild(base);

    const flow = document.createElementNS("http://www.w3.org/2000/svg", "path");
    flow.setAttribute("d", d); flow.setAttribute("class", "thread-flow");
    flow.setAttribute("stroke", col);
    if (!reduce) flow.style.animation = `flow ${2.4 + i * 0.25}s linear infinite`;
    flows.appendChild(flow);

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const halo = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    halo.setAttribute("cx", x); halo.setAttribute("cy", y); halo.setAttribute("r", 13);
    halo.setAttribute("fill", col); halo.setAttribute("opacity", "0.22"); halo.setAttribute("filter", "url(#soft)");
    const node = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    node.setAttribute("cx", x); node.setAttribute("cy", y); node.setAttribute("r", 7);
    node.setAttribute("fill", col); node.setAttribute("class", "tnode");
    const lx = cx + (R + 30) * Math.cos(a), ly = cy + (R + 30) * Math.sin(a);
    const isRight = Math.cos(a) > 0.2, isLeft = Math.cos(a) < -0.2;
    const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lbl.setAttribute("x", lx); lbl.setAttribute("y", ly + 4);
    lbl.setAttribute("text-anchor", isRight ? "start" : isLeft ? "end" : "middle");
    lbl.setAttribute("class", "tlabel"); lbl.textContent = t.label;
    g.appendChild(halo); g.appendChild(node); g.appendChild(lbl);
    tnodes.appendChild(g);
  });
})();

// ── 2. EXAMPLES + INFLUENCE ROWS (unchanged scaffold) ───────────────────
const EXAMPLES = [
  "Natalia sold clips to 48 friends in April, then half as many in May. How many altogether?",
  "Which property of a mineral can be determined just by looking at it? (A) luster (B) mass (C) weight (D) hardness",
  "Write a Python function that checks if a word is a palindrome.",
  "Explain why the sky is blue in two sentences.",
];
const exWrap = document.getElementById("examples");
EXAMPLES.forEach((ex, i) => {
  const li = document.createElement("li");
  const b = document.createElement("button");
  b.className = "chip"; b.type = "button";
  b.innerHTML = `<span class="ix">0${i + 1}</span><span class="qt"></span><span class="go">↵</span>`;
  b.querySelector(".qt").textContent = ex;
  b.addEventListener("click", () => {
    document.getElementById("ask").value = ex;
    document.getElementById("ask").focus();
  });
  li.appendChild(b); exWrap.appendChild(li);
});

const infRows = document.getElementById("infRows");
const fills = {};
TEACHERS.forEach((t) => {
  const row = document.createElement("div"); row.className = "inf-row";
  row.style.color = cssv(t.varc);
  row.innerHTML = `<span class="nm">${t.label}</span>
    <span class="track"><span class="fill"></span></span>
    <span class="pc">—</span>`;
  infRows.appendChild(row);
  fills[t.id] = { fill: row.querySelector(".fill"), pc: row.querySelector(".pc") };
});

function setInfluence(gates) {
  if (!gates) return;
  TEACHERS.forEach((t, i) => {
    const val = gates[i] || 0;
    fills[t.id].fill.style.width = (val * 100).toFixed(0) + "%";
    fills[t.id].pc.textContent = val > 0 ? val.toFixed(2).slice(1) : "—";
  });
}
function resetInfluence() {
  TEACHERS.forEach((t) => { fills[t.id].fill.style.width = "0%"; fills[t.id].pc.textContent = "—"; });
}
function topIdx(g) { let m = 0; for (let i = 1; i < g.length; i++) if (g[i] > g[m]) m = i; return m; }

// ── 3. BACKEND-WIRED ASK / COMPARE ──────────────────────────────────────
let mode = "ask";
let busy = false;
let client = null;
const answers = document.getElementById("answers");
const beforeCard = document.getElementById("beforeCard");
const bodyAfter = document.getElementById("bodyAfter");
const bodyBefore = document.getElementById("bodyBefore");
const askBtn = document.getElementById("askBtn");
const infNote = document.getElementById("infNote");

document.querySelectorAll(".seg button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", "false"));
    b.setAttribute("aria-pressed", "true");
    mode = b.dataset.mode;
    answers.classList.toggle("compare", mode === "compare");
    beforeCard.hidden = (mode !== "compare");
  });
});

function render(el, text, streaming) {
  el.textContent = text || "";
  if (streaming) {
    const caret = document.createElement("span");
    caret.className = "caret"; caret.textContent = "▋";
    el.appendChild(caret);
  }
}

async function ask() {
  const text = document.getElementById("ask").value.trim();
  if (!text || busy) return;
  if (!client) { render(bodyAfter, "⚠ backend not connected", false); return; }

  busy = true;
  askBtn.classList.add("busy"); askBtn.firstChild.textContent = "Thinking ";
  infNote.textContent = "routing through teachers…";
  resetInfluence();
  bodyAfter.innerHTML = '<span class="placeholder-note">distilling…</span>';
  if (mode === "compare") bodyBefore.textContent = "";

  try {
    let gates = null;
    if (mode === "compare") {
      const job = client.submit("/arena", { text });
      for await (const msg of job) {
        if (msg.type !== "data" || !msg.data?.[0]) continue;
        const d = msg.data[0];
        if (d.error) { render(bodyAfter, "⚠ " + d.error, false); break; }
        gates = d.gates; setInfluence(gates);
        render(bodyBefore, d.base, !d.done);
        render(bodyAfter, d.deku, !d.done);
        if (d.done && gates) dropProbeByGates(gates);
      }
    } else {
      const job = client.submit("/probe", { text });
      for await (const msg of job) {
        if (msg.type !== "data" || !msg.data?.[0]) continue;
        const d = msg.data[0];
        if (d.error) { render(bodyAfter, "⚠ " + d.error, false); break; }
        gates = d.gates; setInfluence(gates);
        render(bodyAfter, d.text, !d.done);
        if (d.done) { if (d.point) dropProbePoint(d.point); else if (gates) dropProbeByGates(gates); }
      }
    }
    if (gates) {
      const t = TEACHERS[topIdx(gates)].label;
      infNote.textContent = `leaning on ${t} · ${(gates[topIdx(gates)] * 100).toFixed(0)}%`;
    } else {
      infNote.textContent = "idle — waiting for a prompt";
    }
  } catch (e) {
    console.error(e);
    render(bodyAfter, "⚠ " + (e.message || e), false);
    infNote.textContent = "error";
  } finally {
    busy = false;
    askBtn.classList.remove("busy"); askBtn.firstChild.textContent = "Ask ";
  }
}
askBtn.addEventListener("click", ask);
document.getElementById("ask").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask();
});

// ── 4. THOUGHT MAP (real coords from /viz, procedural fallback offline) ──
const stage = document.getElementById("mapStage");
const layers = {
  far: document.getElementById("mapFar"),
  mid: document.getElementById("mapMid"),
  near: document.getElementById("mapNear"),
};
const legend = document.getElementById("mapLegend");

let VIZ = null;
let PROJ = null;       // {minX,maxX,minY,maxY} bbox of all model points
let CENTROIDS = {};    // teacher id → [nx, ny] in 0..1, for gate-based probe drop

// fallback cluster centers (used only when the backend has no coords)
const CENTERS = {
  qwen: [0.30, 0.32], smollm: [0.70, 0.26], phi: [0.80, 0.58],
  gemma: [0.58, 0.78], minicpm: [0.30, 0.72], nemotron: [0.46, 0.50],
};
const rnd = (n) => (Math.random() - 0.5) * n;

function clearLayers() { Object.values(layers).forEach((l) => (l.innerHTML = "")); }

function addDot(layer, leftPct, topPct, size, color, opacity) {
  const d = document.createElement("div"); d.className = "dot";
  d.style.left = leftPct + "%"; d.style.top = topPct + "%";
  d.style.width = d.style.height = size + "px";
  d.style.color = color; d.style.background = color; d.style.opacity = opacity;
  layer.appendChild(d);
  return d;
}

function computeProjection(models) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const m of models) for (const p of m.points) {
    if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
  }
  return { minX, maxX, minY, maxY };
}
function project(p) {  // 3D point → [leftPct, topPct] with 8% padding
  const { minX, maxX, minY, maxY } = PROJ;
  const nx = (p[0] - minX) / ((maxX - minX) || 1);
  const ny = (p[1] - minY) / ((maxY - minY) || 1);
  return [
    Math.max(0, Math.min(1, 0.08 + nx * 0.84)) * 100,
    Math.max(0, Math.min(1, 0.08 + ny * 0.84)) * 100,
  ];
}

function placeReal() {
  clearLayers();
  PROJ = computeProjection(VIZ.models);
  CENTROIDS = {};
  for (const m of VIZ.models) {
    const col = colorFor(m.name);
    const isStudent = m.name === "student";
    let sx = 0, sy = 0;
    m.points.forEach((p, k) => {
      const [lp, tp] = project(p);
      sx += lp; sy += tp;
      const depth = isStudent ? "near" : (k % 3 === 0 ? "far" : k % 3 === 1 ? "mid" : "near");
      const sz = isStudent ? 7 : depth === "near" ? 6 : depth === "mid" ? 4.5 : 3;
      const op = isStudent ? 1 : depth === "near" ? 1 : depth === "mid" ? 0.8 : 0.55;
      addDot(layers[depth], lp, tp, sz, col, op);
    });
    if (!isStudent && m.points.length) CENTROIDS[m.name] = [sx / m.points.length / 100, sy / m.points.length / 100];
  }
}

function placeFallback() {
  clearLayers();
  PROJ = null;
  CENTROIDS = {};
  TEACHERS.forEach((t) => {
    const c = CENTERS[t.id]; const col = cssv(t.varc);
    CENTROIDS[t.id] = c;
    for (let k = 0; k < 11; k++) {
      const depth = k % 3 === 0 ? "far" : k % 3 === 1 ? "mid" : "near";
      const sz = depth === "near" ? 6 : depth === "mid" ? 4.5 : 3;
      const op = depth === "near" ? 1 : depth === "mid" ? 0.8 : 0.55;
      addDot(layers[depth], (c[0] + rnd(0.13)) * 100, (c[1] + rnd(0.13)) * 100, sz, col, op);
    }
  });
  for (let k = 0; k < 9; k++) {
    addDot(layers.near, (0.5 + rnd(0.22)) * 100, (0.52 + rnd(0.2)) * 100, 7, cssv("--spark"), 1);
  }
}

function placeMap() {
  if (VIZ && VIZ.models && VIZ.models.length) placeReal();
  else placeFallback();
}

function dropProbePoint(point) {
  let lp, tp;
  if (PROJ) { [lp, tp] = project([point.x, point.y, point.z]); }
  else { lp = 50 + rnd(20) * 2; tp = 52 + rnd(18) * 2; }
  const d = addDot(layers.near, lp, tp, 9, cssv("--text"), 1);
  d.classList.add("probe");
}
function dropProbeByGates(gates) {
  const id = TEACHERS[topIdx(gates)].id;
  const c = CENTROIDS[id] || [0.5, 0.5];
  const lp = (c[0] * 0.6 + 0.5 * 0.4 + rnd(0.06)) * 100;
  const tp = (c[1] * 0.6 + 0.52 * 0.4 + rnd(0.06)) * 100;
  const d = addDot(layers.near, lp, tp, 9, cssv("--text"), 1);
  d.classList.add("probe");
}

function buildLegend() {
  legend.innerHTML = "";
  [{ label: "student", c: "--spark" }, ...TEACHERS.map((t) => ({ label: t.label, c: t.varc })), { label: "probe", c: "--text" }]
    .forEach((it) => {
      const s = document.createElement("div"); s.className = "lg"; s.style.color = cssv(it.c);
      s.innerHTML = `<span class="s"></span><span style="color:var(--text-dim)">${it.label}</span>`;
      legend.appendChild(s);
    });
}
buildLegend();
placeMap();
window.addEventListener("resize", () => { clearTimeout(window.__rt); window.__rt = setTimeout(placeMap, 250); });

// ── 5. PARALLAX (unchanged) ─────────────────────────────────────────────
const ambient = document.getElementById("ambient");
const converge = document.getElementById("converge");
let sy = 0, ticking = false;
function onScroll() { sy = window.scrollY || 0; if (!ticking) { requestAnimationFrame(applyScroll); ticking = true; } }
function applyScroll() {
  if (!reduce) {
    ambient.style.transform = `translate3d(0, ${sy * 0.18}px, 0)`;
    converge.style.transform = `translate3d(0, ${sy * -0.04}px, 0)`;
  }
  ticking = false;
}
window.addEventListener("scroll", onScroll, { passive: true });

let mx = 0, my = 0, cmx = 0, cmy = 0;
document.addEventListener("mousemove", (e) => {
  mx = (e.clientX / window.innerWidth - 0.5);
  my = (e.clientY / window.innerHeight - 0.5);
});
function rafMouse() {
  cmx += (mx - cmx) * 0.06; cmy += (my - cmy) * 0.06;
  if (!reduce) {
    const tn = document.getElementById("tnodes");
    const fl = document.getElementById("flows");
    if (tn) tn.style.transform = `translate(${cmx * 22}px, ${cmy * 22}px)`;
    if (fl) fl.style.transform = `translate(${cmx * 10}px, ${cmy * 10}px)`;
    const r = stage.getBoundingClientRect();
    if (r.top < window.innerHeight && r.bottom > 0) {
      Object.values(layers).forEach((l) => {
        const dep = parseFloat(l.dataset.depth) * 1;
        l.style.transform = `translate(${cmx * dep * 900}px, ${cmy * dep * 700}px)`;
      });
    }
  }
  requestAnimationFrame(rafMouse);
}
requestAnimationFrame(rafMouse);

// ── 6. SCROLL REVEAL (unchanged) ────────────────────────────────────────
const io = new IntersectionObserver((ents) => {
  ents.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
}, { threshold: 0.18 });
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

// ── 7. smooth scroll buttons (unchanged) ────────────────────────────────
document.querySelectorAll("[data-scroll]").forEach((b) => {
  b.addEventListener("click", () => {
    const el = document.querySelector(b.dataset.scroll);
    if (el) el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  });
});

// ── 0. connect + load viz ───────────────────────────────────────────────
(async () => {
  try {
    client = await Client.connect(location.origin);
    const res = await client.predict("/viz", {});
    VIZ = res.data[0];
    const n = (VIZ.teachers && VIZ.teachers.length) || 6;
    document.getElementById("statTeachers").innerHTML =
      `<span class="d t"></span>${n} Teachers <b>· 1.5–4B</b>`;
    if (!VIZ.model_ready) infNote.textContent = "model offline — UI preview only";
    placeMap();
  } catch (e) {
    console.error("boot failed", e);
    infNote.textContent = "backend unreachable — UI preview only";
  }
})();
