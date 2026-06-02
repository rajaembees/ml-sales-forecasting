import os
import base64
from datetime import datetime
import pandas as pd

IND_VARS1 = ["Advertising Expenditure"]
IND_VARS2 = ["Advertising Expenditure", "Discount Percentage", "Product Price"]

MODEL_NAMES = [
    "Simple LR  (Adv. Expenditure only)",
    "Multiple LR  (Adv. Exp. + Discount % + Price)",
    "Multiple LR  (All Features)",
]

_EDA_PLOTS = [
    ("histograms.png",         "Histograms – All Numerical Features"),
    ("boxplots.png",           "Box Plots – Outlier Detection"),
    ("categorical_counts.png", "Categorical Feature Counts"),
    ("correlation_heatmap.png","Correlation Heatmap"),
    ("sales_by_popularity.png","Sales by Popularity"),
    ("sales_by_region.png",    "Sales by Region"),
]

_MODEL_PLOTS = [
    ("simple_lr_fit.png",  "Simple LR – Best-Fit Line"),
    ("model_comparison.png","Model Performance Comparison (Test Set)"),
    ("residual_plot.png",  "Residual Plot – Best Model"),
]

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa; color: #333; }
.container { max-width: 1200px; margin: 0 auto; padding: 30px 24px; }
h1 { color: #1a2e4a; font-size: 1.9rem; border-bottom: 4px solid #2980b9; padding-bottom: 12px; margin-bottom: 6px; }
h2 { color: #2471a3; font-size: 1.3rem; margin: 40px 0 14px; border-left: 5px solid #2980b9; padding-left: 12px; }
h3 { color: #555; font-size: 1rem; margin: 18px 0 6px; }
.meta { background: #eaf4fb; border: 1px solid #aed6f1; border-radius: 6px; padding: 12px 18px; margin-bottom: 28px; font-size: 0.88rem; color: #1a5276; }
.meta span { margin-right: 24px; }
.plot-grid { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 12px; }
.plot-item { flex: 1 1 46%; min-width: 280px; background: #fff; border: 1px solid #dde; border-radius: 6px; padding: 10px; }
.plot-item.full { flex: 1 1 96%; }
.plot-item img { width: 100%; border-radius: 4px; }
.plot-item .caption { text-align: center; font-size: 0.8rem; color: #666; margin-top: 6px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 4px; font-size: 0.88rem; background: #fff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
thead tr { background: #2471a3; color: #fff; }
th { padding: 11px 14px; text-align: left; font-weight: 600; }
td { padding: 10px 14px; border-bottom: 1px solid #eef; }
tbody tr:nth-child(even) { background: #f0f8ff; }
tbody tr.best-row td { background: #d5f5e3; font-weight: 600; }
.model-name { max-width: 380px; }
.eq-box { background: #f8f9fa; border-left: 4px solid #27ae60; padding: 10px 16px; margin: 8px 0 14px; font-family: Consolas, monospace; font-size: 0.82rem; color: #1e4d2b; border-radius: 0 4px 4px 0; word-break: break-word; line-height: 1.6; }
.summary-box { background: #eafaf1; border: 1px solid #a9dfbf; border-radius: 8px; padding: 18px 24px; margin: 28px 0 10px; }
.summary-box h3 { color: #1a5631; font-size: 1.05rem; margin-bottom: 8px; }
.summary-box .metric { display: inline-block; margin-right: 32px; font-size: 0.95rem; }
.summary-box .metric strong { font-size: 1.2rem; color: #1e8449; }
footer { text-align: center; font-size: 0.78rem; color: #aaa; margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd; }
"""


def _b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _html_table(df, best_name=None):
    hdr = "<th>Model</th>" + "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for idx, row in df.iterrows():
        cls = " class='best-row'" if idx == best_name else ""
        cells = f"<td class='model-name'>{idx}</td>"
        for v in row:
            cells += f"<td>{v:.4f}</td>"
        rows += f"<tr{cls}>{cells}</tr>"
    return f"<table><thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table>"


def _md_table(df):
    header = "| Model | " + " | ".join(df.columns) + " |"
    sep    = "|:------|" + "|".join(["------:"] * len(df.columns)) + "|"
    rows = []
    for idx, row in df.iterrows():
        vals = " | ".join(f"{v:.4f}" for v in row)
        rows.append(f"| {idx} | {vals} |")
    return "\n".join([header, sep] + rows)


def _stat_html_table(desc_T):
    cols = desc_T.columns.tolist()
    hdr = "<th>Feature</th>" + "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for i, (idx, row) in enumerate(desc_T.iterrows()):
        cells = f"<td><strong>{idx}</strong></td>"
        for v in row:
            cells += f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
        cls = " style='background:#f0f8ff'" if i % 2 == 0 else ""
        rows += f"<tr{cls}>{cells}</tr>"
    return f"<table><thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table>"


def _stat_md_table(desc_T):
    cols = desc_T.columns.tolist()
    header = "| Feature | " + " | ".join(cols) + " |"
    sep    = "|:--------|" + "|".join(["------:"] * len(cols)) + "|"
    rows = []
    for idx, row in desc_T.iterrows():
        vals = " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row)
        rows.append(f"| {idx} | {vals} |")
    return "\n".join([header, sep] + rows)


def _equations(models, X_train):
    lr1, lr2, lr3 = models
    eq1 = f"Sales = ({lr1.coef_[0]:.4f}) x Advertising Expenditure  +  {lr1.intercept_:.4f}"
    terms2 = "  +  ".join(f"({c:.4f}) x {v}" for c, v in zip(lr2.coef_, IND_VARS2))
    eq2 = f"Sales = {terms2}  +  {lr2.intercept_:.4f}"
    terms3 = "  +  ".join(f"({c:.4f}) x {v}" for c, v in zip(lr3.coef_, X_train.columns))
    eq3 = f"Sales = {terms3}  +  {lr3.intercept_:.4f}"
    return [
        ("Model 1 – Simple LR", eq1),
        ("Model 2 – Multiple LR (3 features)", eq2),
        ("Model 3 – Multiple LR (All Features)", eq3),
    ]


# ── HTML ──────────────────────────────────────────────────────────────────────

def _build_html(data, train_df, test_df, eqs, plot_dir, display_ts):
    best = test_df["RMSE"].idxmin()
    best_row = test_df.loc[best]

    # EDA plot grid
    eda_grid = ""
    for fname, caption in _EDA_PLOTS:
        b64 = _b64(os.path.join(plot_dir, fname))
        if b64:
            eda_grid += f"""
            <div class='plot-item'>
              <img src='data:image/png;base64,{b64}' alt='{caption}'/>
              <div class='caption'>{caption}</div>
            </div>"""

    # Model plots
    model_plots_html = ""
    for fname, caption in _MODEL_PLOTS:
        b64 = _b64(os.path.join(plot_dir, fname))
        if b64:
            css_class = "plot-item full" if "comparison" in fname else "plot-item"
            model_plots_html += f"""
            <div class='{css_class}'>
              <img src='data:image/png;base64,{b64}' alt='{caption}'/>
              <div class='caption'>{caption}</div>
            </div>"""

    # Equations
    eq_html = ""
    for name, eq in eqs:
        eq_html += f"<h3>{name}</h3><div class='eq-box'>{eq}</div>"

    # Stats table
    desc_T = data.describe().T.round(4)
    stats_html = _stat_html_table(desc_T)

    # Performance tables
    train_table = _html_table(train_df, best_name=None)
    test_table  = _html_table(test_df,  best_name=best)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Sales Forecasting Report – {display_ts}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>Linear Regression – Sales Forecasting Report</h1>
  <div class="meta">
    <span><strong>Generated:</strong> {display_ts}</span>
    <span><strong>Rows:</strong> {data.shape[0]:,}</span>
    <span><strong>Features:</strong> {data.shape[1] - 1}</span>
    <span><strong>Target:</strong> Sales ($)</span>
  </div>

  <h2>1. Dataset Overview</h2>
  {stats_html}

  <h2>2. Exploratory Data Analysis</h2>
  <div class="plot-grid">{eda_grid}</div>

  <h2>3. Model Equations</h2>
  {eq_html}

  <h2>4. Training Performance</h2>
  {train_table}

  <h2>5. Test Performance</h2>
  <p style="font-size:0.82rem;color:#888;margin-bottom:6px">Highlighted row = best model</p>
  {test_table}

  <h2>6. Model Comparison &amp; Diagnostics</h2>
  <div class="plot-grid">{model_plots_html}</div>

  <div class="summary-box">
    <h3>Best Model: {best}</h3>
    <div class="metric">RMSE <strong>${best_row['RMSE']:,.2f}</strong></div>
    <div class="metric">MAE <strong>${best_row['MAE']:,.2f}</strong></div>
    <div class="metric">MAPE <strong>{best_row['MAPE (%)']:.2f}%</strong></div>
  </div>

  <footer>Auto-generated by sales_forecasting/main.py &nbsp;|&nbsp; {display_ts}</footer>
</div>
</body>
</html>"""


# ── Markdown ──────────────────────────────────────────────────────────────────

def _build_md(data, train_df, test_df, eqs, display_ts):
    best = test_df["RMSE"].idxmin()
    best_row = test_df.loc[best]
    p = "plots"  # relative path from outputs/

    lines = [
        "# Linear Regression – Sales Forecasting Report",
        "",
        f"> **Generated:** {display_ts}  ",
        f"> **Dataset:** {data.shape[0]:,} rows × {data.shape[1] - 1} features  ",
        f"> **Target:** Sales ($)",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        _stat_md_table(data.describe().T.round(4)),
        "",
        "---",
        "",
        "## 2. Exploratory Data Analysis",
        "",
    ]

    for fname, caption in _EDA_PLOTS:
        lines += [f"### {caption}", "", f"![{caption}]({p}/{fname})", ""]

    lines += [
        "---",
        "",
        "## 3. Model Equations",
        "",
    ]
    for name, eq in eqs:
        lines += [f"### {name}", "", f"```", eq, "```", ""]

    lines += [
        "---",
        "",
        "## 4. Training Performance",
        "",
        _md_table(train_df),
        "",
        "---",
        "",
        "## 5. Test Performance",
        "",
        _md_table(test_df),
        "",
        "---",
        "",
        "## 6. Model Comparison & Diagnostics",
        "",
    ]
    for fname, caption in _MODEL_PLOTS:
        lines += [f"### {caption}", "", f"![{caption}]({p}/{fname})", ""]

    lines += [
        "---",
        "",
        "## Summary",
        "",
        f"**Best Model:** {best}",
        "",
        "| Metric | Value |",
        "|:-------|------:|",
        f"| RMSE   | ${best_row['RMSE']:,.2f} |",
        f"| MAE    | ${best_row['MAE']:,.2f} |",
        f"| MAPE   | {best_row['MAPE (%)']:.2f}% |",
        "",
        "---",
        f"*Auto-generated by `sales_forecasting/main.py` on {display_ts}*",
    ]
    return "\n".join(lines)


# ── Public entry point ────────────────────────────────────────────────────────

def generate_reports(data, models, X_train, train_df, test_df,
                     plot_dir="outputs/plots", output_dir="outputs"):
    now = datetime.now()
    ts          = now.strftime("%Y-%m-%d_%H-%M-%S")
    display_ts  = now.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(output_dir, exist_ok=True)

    eqs = _equations(models, X_train)

    html_path = os.path.join(output_dir, f"report_{ts}.html")
    md_path   = os.path.join(output_dir, f"report_{ts}.md")

    html_content = _build_html(data, train_df, test_df, eqs, plot_dir, display_ts)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    md_content = _build_md(data, train_df, test_df, eqs, display_ts)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"HTML report  -->  {html_path}")
    print(f"MD report    -->  {md_path}")
