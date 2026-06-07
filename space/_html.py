from __future__ import annotations

# index 0-4 = teachers (aligns with MODEL_COLORS[1:] in _fig.py)
_TEACHER_COLORS = ["#7c3aed", "#06b6d4", "#f59e0b", "#34d399", "#f472b6"]


def gate_html(gate_weights: list[float], teacher_names: list[str]) -> str:
    """Horizontal bar chart sorted by weight descending."""
    ranked = sorted(
        enumerate(zip(gate_weights, teacher_names)),
        key=lambda x: x[1][0],
        reverse=True,
    )
    rows = []
    for orig_idx, (w, name) in ranked:
        color = _TEACHER_COLORS[orig_idx % len(_TEACHER_COLORS)]
        bar_pct = int(w * 100)
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
            f'<span style="width:80px;font-family:monospace;font-size:11px;color:{color};'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</span>'
            f'<div style="flex:1;height:5px;background:#21262d;border-radius:3px;overflow:hidden;">'
            f'<div style="width:{bar_pct}%;height:100%;background:{color};border-radius:3px;"></div>'
            f'</div>'
            f'<span style="width:36px;font-family:monospace;font-size:11px;'
            f'color:#8b949e;text-align:right;">{w:.2f}</span>'
            f'</div>'
        )
    return (
        '<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:14px;margin-top:8px;">'
        '<div style="font-size:10px;color:#8b949e;font-family:monospace;margin-bottom:10px;letter-spacing:0.04em;">'
        'TEACHER INFLUENCE</div>'
        + "".join(rows)
        + "</div>"
    )


def task_html(gate_weights: list[float], teacher_names: list[str]) -> str:
    """Badge for argmax teacher + runner-up."""
    best_idx = max(range(len(gate_weights)), key=lambda i: gate_weights[i])
    second_idx = sorted(range(len(gate_weights)), key=lambda i: gate_weights[i])[-2]
    color = _TEACHER_COLORS[best_idx % len(_TEACHER_COLORS)]
    best = teacher_names[best_idx].upper()
    second = teacher_names[second_idx].upper()
    w2 = gate_weights[second_idx]
    n = len(teacher_names)
    return (
        '<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:14px;margin-top:8px;">'
        '<div style="font-size:10px;color:#8b949e;font-family:monospace;margin-bottom:10px;letter-spacing:0.04em;">'
        'DOMINANT TEACHER</div>'
        '<div style="display:flex;align-items:center;gap:12px;">'
        f'<span style="display:inline-flex;align-items:center;gap:6px;background:{color}22;'
        f'border:1px solid {color}55;border-radius:999px;padding:4px 14px;'
        f'font-family:monospace;font-weight:600;color:{color};font-size:13px;">'
        f'<span style="width:6px;height:6px;background:{color};border-radius:50%;'
        f'box-shadow:0 0 6px {color};"></span>{best}</span>'
        f'<span style="font-size:11px;color:#8b949e;font-family:monospace;">'
        f'runner-up <b style="color:#e6edf3;">{second}</b> · {w2:.2f}</span>'
        f'</div>'
        f'<div style="font-size:10px;color:#6e7681;margin-top:8px;font-family:monospace;">'
        f'gate picked this teacher for your prompt ({n} total)</div>'
        '</div>'
    )


def header_html() -> str:
    """App header with ∀ logo, wordmark, and status pills."""
    def pill(color: str, dot_color: str, text: str) -> str:
        return (
            f'<span style="display:inline-flex;align-items:center;gap:8px;height:30px;'
            f'padding:0 12px;border-radius:999px;font-family:monospace;font-size:11px;'
            f'font-weight:500;letter-spacing:0.04em;color:{color};'
            f'background:#161b22;border:1px solid #30363d;">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:{dot_color};'
            f'box-shadow:0 0 8px {dot_color};"></span>{text}</span>'
        )
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'padding:16px 4px 12px;gap:16px;">'
        '<div style="display:flex;align-items:center;gap:14px;">'
        '<div style="width:44px;height:44px;display:grid;place-items:center;'
        'border-radius:6px;background:radial-gradient(60% 60% at 50% 40%,'
        'rgba(124,58,237,0.30),transparent 70%),#14181f;border:1px solid #2a2333;">'
        '<span style="font-family:monospace;font-weight:700;font-size:28px;color:#7c3aed;">∀</span>'
        '</div>'
        '<div>'
        '<div style="font-weight:600;font-size:17px;color:#e6edf3;">One for All</div>'
        '<div style="font-size:12px;color:#8b949e;margin-top:2px;">multi-teacher soul distillation</div>'
        '</div></div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
        + pill("#06b6d4", "#06b6d4", "STUDENT Qwen2.5-0.5B")
        + pill("#7c3aed", "#7c3aed", "5 TEACHERS")
        + pill("#f59e0b", "#f59e0b", "PATH B · geometry-only")
        + '</div></div>'
        '<div style="height:1px;margin-bottom:24px;background:linear-gradient('
        '90deg,transparent 0%,rgba(124,58,237,0.55) 50%,transparent 100%);"></div>'
    )
