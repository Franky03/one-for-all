"""
ofa/viz/render.py
-----------------
Turn a viz_data.json (produced by modal_app.export_viz) into a single
self-contained interactive HTML report using Plotly (loaded from CDN).

Four panels:
  1. Loss components over steps (task / geo / total)
  2. Gate routing over steps (stacked area, one band per teacher)
  3. CKA heatmap between all models
  4. UMAP scatter — teachers projected into the student space

Usage:
    python -m ofa.viz.render viz_data.json ofa_report.html
    # then open ofa_report.html in a browser
"""

from __future__ import annotations

import json
import sys


# NOTE: braces here are LITERAL — the only placeholder is the token __DATA__,
# substituted via str.replace (not str.format), so JS/CSS braces stay intact.
_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>One for All — soul transfer report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 24px; background:#0d1117; color:#e6edf3; }
  h1 { font-weight: 600; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:8px; }
  .empty { color:#8b949e; padding:24px; text-align:center; }
</style>
</head>
<body>
<h1>One for All — soul transfer report</h1>
<div class="grid">
  <div class="card"><div id="curves" style="height:360px"></div></div>
  <div class="card"><div id="gate" style="height:360px"></div></div>
  <div class="card"><div id="cka" style="height:420px"></div></div>
  <div class="card"><div id="umap" style="height:420px"></div></div>
</div>
<script>
const D = __DATA__;
const DARK = { paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
               font:{color:'#e6edf3'}, margin:{t:48,r:16,b:48,l:56} };
function empty(id, msg) { document.getElementById(id).innerHTML =
  '<div class="empty">'+msg+'</div>'; }

// 1. loss curves
const c = D.curves || {};
if (c.steps && c.steps.length) {
  Plotly.newPlot('curves', [
    {x:c.steps, y:c.task,  name:'task',  mode:'lines+markers'},
    {x:c.steps, y:c.geo,   name:'geo',   mode:'lines+markers'},
    {x:c.steps, y:c.total, name:'total', mode:'lines+markers'},
  ], Object.assign({title:'Loss components', xaxis:{title:'step'}}, DARK));
} else empty('curves', 'no training curves (history.json missing)');

// 2. gate routing (stacked area, sums to 1)
if (c.gate && c.gate.length && c.teacher_names) {
  const traces = c.teacher_names.map((nm, i) => ({
    x:c.steps, y:c.gate.map(g => g[i]), name:nm,
    mode:'lines', stackgroup:'g'
  }));
  Plotly.newPlot('gate', traces,
    Object.assign({title:'Gate (teacher routing) over steps',
                   yaxis:{title:'weight', range:[0,1]}, xaxis:{title:'step'}}, DARK));
} else empty('gate', 'no gate telemetry');

// 3. CKA heatmap
const k = D.cka;
if (k && k.matrix) {
  Plotly.newPlot('cka', [{
    z:k.matrix, x:k.models, y:k.models, type:'heatmap',
    colorscale:'Viridis', zmin:0, zmax:1, colorbar:{title:'CKA'}
  }], Object.assign({title:'CKA between models'}, DARK));
} else empty('cka', 'no CKA data');

// 4. UMAP scatter
const u = D.umap;
if (u && u.traces && u.traces.length) {
  Plotly.newPlot('umap', u.traces.map(t => ({
    x:t.x, y:t.y, name:t.name, mode:'markers', type:'scatter',
    marker:{size:7, opacity:0.8}
  })), Object.assign({title:'UMAP — teachers projected into student space'}, DARK));
} else empty('umap', 'no UMAP data');
</script>
</body>
</html>
"""


def build_html(viz: dict) -> str:
    return _TEMPLATE.replace("__DATA__", json.dumps(viz))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "viz_data.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "ofa_report.html"
    with open(src) as f:
        viz = json.load(f)
    with open(out, "w") as f:
        f.write(build_html(viz))
    print(f"wrote {out} — open it in a browser")


if __name__ == "__main__":
    main()
