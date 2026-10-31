"""Generate clean, scalable SVG visual charts for README from results.csv without external dependencies."""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = Path(__file__).parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_FILE = DATA_DIR / "results.csv"


def generate_regime_chart(df: pd.DataFrame, out_path: Path):
    """Generate a grouped bar chart comparing QLIKE by market regime."""
    regimes = ["low", "mid", "high"]
    models = ["EWMA", "HAR (ridge)", "LightGBM", "naive (rv5)"]
    colors = {
        "EWMA": "#f59e0b",         # Amber
        "HAR (ridge)": "#3b82f6",   # Blue
        "LightGBM": "#10b981",      # Emerald Green
        "naive (rv5)": "#ef4444",   # Red
    }
    
    # Filter regime data
    sub = df[df["regime"].isin(regimes)]
    means = sub.groupby(["regime", "model"])["qlike"].mean().unstack()
    
    width = 760
    height = 420
    pad_left = 70
    pad_right = 30
    pad_top = 70
    pad_bottom = 60
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    max_val = 1.7  # naive peaks around 1.53
    y_ticks = [0.0, 0.4, 0.8, 1.2, 1.6]
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 18px; font-weight: 700; fill: #1e293b; }',
        '  .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #64748b; }',
        '  .axis-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 600; fill: #334155; }',
        '  .tick { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #64748b; }',
        '  .legend { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; font-weight: 500; fill: #334155; }',
        '  .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '  .bar-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: 600; fill: #1e293b; text-anchor: middle; }',
        '</style>',
        f'<rect width="{width}" height="{height}" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>',
        f'<text x="{pad_left}" y="36" class="title">Loss by Volatility Regime (Mean QLIKE)</text>',
        f'<text x="{pad_left}" y="54" class="subtitle">Lower is better. Asymmetric loss heavily penalizes underestimating volatility risk.</text>',
    ]
    
    # Legend
    leg_x = width - 420
    for i, m in enumerate(models):
        x = leg_x + i * 100
        svg.append(f'<rect x="{x}" y="24" width="12" height="12" rx="3" fill="{colors[m]}"/>')
        svg.append(f'<text x="{x + 18}" y="35" class="legend">{m.split()[0]}</text>')
        
    # Y Grid & Ticks
    for yv in y_ticks:
        y_pos = pad_top + chart_h - (yv / max_val) * chart_h
        svg.append(f'<line x1="{pad_left}" y1="{y_pos}" x2="{width - pad_right}" y2="{y_pos}" class="grid"/>')
        svg.append(f'<text x="{pad_left - 12}" y="{y_pos + 4}" class="tick" text-anchor="end">{yv:.1f}</text>')

    # Regime Groups
    regime_labels = {"low": "Low Volatility", "mid": "Mid Volatility", "high": "High Volatility"}
    group_w = chart_w / len(regimes)
    bar_w = 26
    gap = 6
    total_group_bars_w = len(models) * bar_w + (len(models) - 1) * gap
    
    for g_idx, reg in enumerate(regimes):
        group_center = pad_left + g_idx * group_w + group_w / 2
        start_x = group_center - total_group_bars_w / 2
        
        # Group background highlight
        svg.append(f'<rect x="{pad_left + g_idx * group_w + 8}" y="{pad_top}" width="{group_w - 16}" height="{chart_h}" rx="6" fill="#f8fafc" opacity="0.6"/>')
        
        for m_idx, m in enumerate(models):
            val = float(means.loc[reg, m])
            bx = start_x + m_idx * (bar_w + gap)
            bar_h = (val / max_val) * chart_h
            by = pad_top + chart_h - bar_h
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{colors[m]}"/>')
            svg.append(f'<text x="{bx + bar_w/2:.1f}" y="{by - 6:.1f}" class="bar-text">{val:.3f}</text>')
            
        svg.append(f'<text x="{group_center:.1f}" y="{pad_top + chart_h + 30}" class="axis-label" text-anchor="middle">{regime_labels[reg]}</text>')
        
    svg.append('</svg>')
    out_path.write_text("\n".join(svg), encoding="utf-8")
    print(f"Generated {out_path}")


def generate_fold_comparison_chart(df: pd.DataFrame, out_path: Path):
    """Generate time-series walk-forward plot across 33 folds."""
    allr = df[df["regime"] == "all"]
    pv = allr.pivot(index="fold", columns="model", values="qlike").sort_index()
    
    width = 760
    height = 380
    pad_left = 65
    pad_right = 35
    pad_top = 70
    pad_bottom = 50
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    models = ["EWMA", "HAR (ridge)", "LightGBM"]
    colors = {
        "EWMA": "#f59e0b",
        "HAR (ridge)": "#3b82f6",
        "LightGBM": "#10b981",
    }
    
    max_val = 1.6
    min_val = 0.2
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 18px; font-weight: 700; fill: #1e293b; }',
        '  .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #64748b; }',
        '  .axis-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; font-weight: 600; fill: #475569; }',
        '  .tick { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #64748b; }',
        '  .legend { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; font-weight: 500; fill: #334155; }',
        '  .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '</style>',
        f'<rect width="{width}" height="{height}" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>',
        f'<text x="{pad_left}" y="36" class="title">Walk-Forward QLIKE Loss Across 33 Test Folds</text>',
        f'<text x="{pad_left}" y="54" class="subtitle">Quarterly out-of-sample folds (~8.25 years). Lower is better.</text>',
    ]

    # Legend
    leg_x = width - 360
    for i, m in enumerate(models):
        x = leg_x + i * 115
        svg.append(f'<circle cx="{x + 6}" cy="28" r="5" fill="{colors[m]}"/>')
        svg.append(f'<line x1="{x}" y1="28" x2="{x + 12}" y2="28" stroke="{colors[m]}" stroke-width="2.5"/>')
        svg.append(f'<text x="{x + 20}" y="32" class="legend">{m.split()[0]}</text>')

    # Grid
    y_ticks = [0.3, 0.6, 0.9, 1.2, 1.5]
    for yv in y_ticks:
        y_pos = pad_top + chart_h - ((yv - min_val) / (max_val - min_val)) * chart_h
        svg.append(f'<line x1="{pad_left}" y1="{y_pos:.1f}" x2="{width - pad_right}" y2="{y_pos:.1f}" class="grid"/>')
        svg.append(f'<text x="{pad_left - 10}" y="{y_pos + 4:.1f}" class="tick" text-anchor="end">{yv:.1f}</text>')

    folds = pv.index.tolist()
    n_folds = len(folds)
    
    # X axis ticks
    for f in range(0, n_folds, 4):
        x = pad_left + (f / (n_folds - 1)) * chart_w
        svg.append(f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{pad_top + chart_h}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{pad_top + chart_h + 20}" class="tick" text-anchor="middle">F{f}</text>')
        
    svg.append(f'<text x="{pad_left + chart_w / 2}" y="{pad_top + chart_h + 38}" class="axis-label" text-anchor="middle">Walk-Forward Test Fold (Quarterly Steps)</text>')

    # Plot Lines
    for m in models:
        pts = []
        for f_idx, f in enumerate(folds):
            v = float(pv.loc[f, m])
            clamped_v = min(max_val, max(min_val, v))
            x = pad_left + (f_idx / (n_folds - 1)) * chart_w
            y = pad_top + chart_h - ((clamped_v - min_val) / (max_val - min_val)) * chart_h
            pts.append((x, y))
            
        path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        svg.append(f'<path d="{path_d}" fill="none" stroke="{colors[m]}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>')
        for px, py in pts:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{colors[m]}" stroke="#ffffff" stroke-width="1"/>')

    svg.append('</svg>')
    out_path.write_text("\n".join(svg), encoding="utf-8")
    print(f"Generated {out_path}")


def generate_relative_improvement_chart(df: pd.DataFrame, out_path: Path):
    """Generate bar chart showing MAE vs EWMA % improvement."""
    allr = df[df["regime"] == "all"]
    tab = allr.groupby("model")[["mae", "qlike"]].mean()
    base = tab.loc["EWMA"]
    mae_diff = 100 * (tab["mae"] / base["mae"] - 1)
    
    models = ["HAR (ridge)", "LightGBM", "naive (rv5)"]
    labels = ["HAR (Ridge)", "LightGBM", "Naive (RV5)"]
    vals = [mae_diff.loc[m] for m in models]
    
    width = 760
    height = 260
    pad_left = 130
    pad_right = 50
    pad_top = 60
    pad_bottom = 40
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    zero_x = pad_left + chart_w * 0.45  # Zero line shifted to accommodate positive & negative

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 16px; font-weight: 700; fill: #1e293b; }',
        '  .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #64748b; }',
        '  .model-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; font-weight: 600; fill: #334155; }',
        '  .val-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; font-weight: 700; }',
        '  .ref-line { stroke: #94a3b8; stroke-width: 1.5; stroke-dasharray: 3,3; }',
        '</style>',
        f'<rect width="{width}" height="{height}" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>',
        f'<text x="{pad_left - 80}" y="30" class="title">MAE Forecast Error Relative to EWMA Baseline (%)</text>',
        f'<text x="{pad_left - 80}" y="46" class="subtitle">Negative % = Outperformed EWMA benchmark (Lower Error is Better)</text>',
        f'<line x1="{zero_x}" y1="{pad_top}" x2="{zero_x}" y2="{pad_top + chart_h}" class="ref-line"/>',
        f'<text x="{zero_x}" y="{pad_top - 8}" font-family="sans-serif" font-size="10" fill="#64748b" text-anchor="middle">EWMA Baseline (0%)</text>',
    ]

    bar_h = 28
    gap = 18
    scale = 12.0  # pixels per percent
    
    for i, (lbl, val) in enumerate(zip(labels, vals)):
        y = pad_top + i * (bar_h + gap) + 10
        svg.append(f'<text x="{pad_left - 15}" y="{y + bar_h/2 + 4}" class="model-label" text-anchor="end">{lbl}</text>')
        
        if val < 0:
            bw = abs(val) * scale
            bx = zero_x - bw
            fill = "#10b981"  # green for better
            txt_anchor = "end"
            txt_x = bx - 8
        else:
            bw = val * scale
            bx = zero_x
            fill = "#ef4444"  # red for worse
            txt_anchor = "start"
            txt_x = bx + bw + 8
            
        svg.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h}" rx="4" fill="{fill}"/>')
        sign = "+" if val > 0 else ""
        svg.append(f'<text x="{txt_x:.1f}" y="{y + bar_h/2 + 4}" class="val-text" fill="{fill}" text-anchor="{txt_anchor}">{sign}{val:.2f}%</text>')

    svg.append('</svg>')
    out_path.write_text("\n".join(svg), encoding="utf-8")
    print(f"Generated {out_path}")


if __name__ == "__main__":
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found. Run walkforward.py first.")
    else:
        df = pd.read_csv(RESULTS_FILE)
        generate_regime_chart(df, ASSETS_DIR / "regime_performance.svg")
        generate_fold_comparison_chart(df, ASSETS_DIR / "walkforward_folds.svg")
        generate_relative_improvement_chart(df, ASSETS_DIR / "relative_improvement.svg")
        print("All SVG visualizations generated successfully.")

