"""Build a static HTML results dashboard from experiments/results (no app UI).

  python -m experiments.build_results_page
  → experiments/results/index.html
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table_to_html(md: str) -> str:
    """Very small markdown table → HTML (first table only)."""
    lines = [ln.strip() for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return f"<pre>{_esc(md[:4000])}</pre>"
    rows = []
    for i, ln in enumerate(lines):
        if re.match(r"^\|\s*-+", ln):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        tag = "th" if i == 0 else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{_esc(c)}</{tag}>" for c in cells) + "</tr>")
    return "<table>\n" + "\n".join(rows) + "\n</table>"


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build() -> Path:
    stats_md = _read(RESULTS / "stats.md")
    distill = _load_json(RESULTS / "distill" / "distill_report.json")
    advanced = _load_json(RESULTS / "advanced" / "advanced_summary.json")
    adv_md = _read(RESULTS / "advanced" / "advanced_report.md")
    findings42 = _read(RESULTS / "findings_seed42.md")

    distill_block = "<p>No distill report yet.</p>"
    if distill:
        distill_block = f"""
        <ul>
          <li>Ensemble: <strong>{distill.get('ensemble_val_acc', 0):.2%}</strong></li>
          <li>Hard-only deep: <strong>{distill.get('hard_only_deep_val_acc', distill.get('hard_only_shallow_val_acc', 0)):.2%}</strong></li>
          <li>Distilled: <strong>{distill.get('distilled_val_acc', 0):.2%}</strong></li>
          <li>Shipped: <code>{_esc(distill.get('shipped', 'n/a'))}</code>
              ({distill.get('shipped_val_acc', distill.get('distilled_val_acc', 0)):.2%})</li>
        </ul>
        """

    robust_block = "<p>Run advanced suite to populate.</p>"
    if advanced and advanced.get("robustness"):
        rows = ["<tr><th>Rung</th><th>Accuracy</th><th>Macro-F1</th></tr>"]
        for r in advanced["robustness"].get("rungs", []):
            rows.append(
                f"<tr><td><code>{_esc(r['rung'])}</code></td>"
                f"<td>{r['accuracy']:.2%}</td><td>{r['macro_f1']:.4f}</td></tr>"
            )
        robust_block = "<table>" + "".join(rows) + "</table>"

    cal_block = ""
    if advanced and advanced.get("calibration_baseline"):
        c = advanced["calibration_baseline"]
        cal_block = f"""
        <p>ECE {c['before']['ece']:.4f} → {c['after']['ece']:.4f}
           (T={c['after']['temperature']:.3f});
           NLL {c['before']['nll']:.4f} → {c['after']['nll']:.4f}</p>
        """

    models_block = ""
    if advanced:
        models_block = f"""
        <ul>
          <li>Baseline: <strong>{advanced.get('baseline', {}).get('accuracy', 0):.2%}</strong></li>
          <li>Cost-sensitive: <strong>{(advanced.get('cost_sensitive') or {}).get('accuracy', 0):.2%}</strong>
              (swaps={(advanced.get('cost_sensitive') or {}).get('weevil_borer_swap_count')})</li>
          <li>Hierarchical: <strong>{(advanced.get('hierarchical') or {}).get('accuracy', 0):.2%}</strong>
              (pair={(advanced.get('hierarchical') or {}).get('pair_subset_accuracy')})</li>
          <li>SSL fine-tune: <strong>{(advanced.get('ssl') or {}).get('accuracy', 0):.2%}</strong></li>
        </ul>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Project Kaan — Experiment Results</title>
<style>
  :root {{
    --bg: #f3efe6;
    --ink: #1c1917;
    --muted: #57534e;
    --accent: #0f766e;
    --card: #fffcf7;
    --line: #d6d3d1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background:
      radial-gradient(900px 400px at 10% -10%, #c7ebe3 0%, transparent 55%),
      radial-gradient(700px 360px at 100% 0%, #fde68a55 0%, transparent 50%),
      var(--bg);
    color: var(--ink); line-height: 1.5;
  }}
  header {{
    padding: 2.5rem 1.5rem 1rem; max-width: 960px; margin: 0 auto;
  }}
  header h1 {{
    font-family: "IBM Plex Serif", Georgia, serif;
    font-weight: 600; font-size: clamp(1.8rem, 4vw, 2.6rem);
    margin: 0 0 0.4rem; letter-spacing: -0.02em;
  }}
  header p {{ color: var(--muted); margin: 0; max-width: 40rem; }}
  main {{ max-width: 960px; margin: 0 auto; padding: 0 1.5rem 3rem; }}
  section {{
    background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; padding: 1.25rem 1.4rem; margin: 1rem 0;
  }}
  h2 {{ margin: 0 0 0.75rem; font-size: 1.15rem; color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{ text-align: left; padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 600; }}
  code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.88em; }}
  pre {{
    overflow: auto; background: #1c1917; color: #f5f5f4;
    padding: 0.9rem; border-radius: 8px; font-size: 0.8rem;
  }}
  .grid {{ display: grid; gap: 1rem; }}
  @media (min-width: 720px) {{ .grid.two {{ grid-template-columns: 1fr 1fr; }} }}
  footer {{ max-width: 960px; margin: 0 auto; padding: 0 1.5rem 2rem; color: var(--muted); font-size: 0.85rem; }}
</style>
</head>
<body>
<header>
  <h1>Kaan experiment results</h1>
  <p>Desk-bound acoustic pest detection: bake-off, distillation, robustness ladder,
     weevil↔borer heads, calibration, and SSL — without field recordings.</p>
</header>
<main>
  <div class="grid two">
    <section>
      <h2>v3 Distilled production</h2>
      {distill_block}
    </section>
    <section>
      <h2>Advanced heads + SSL</h2>
      {models_block or "<p>Pending advanced run.</p>"}
      {cal_block}
    </section>
  </div>
  <section>
    <h2>Robustness ladder</h2>
    {robust_block}
  </section>
  <section>
    <h2>Multi-seed bake-off (v2)</h2>
    {_md_table_to_html(stats_md)}
  </section>
  <section>
    <h2>Advanced report</h2>
    <pre>{_esc(adv_md[:6000] if adv_md else "Run: python -m experiments.run_advanced --copy-results")}</pre>
  </section>
  <section>
    <h2>Findings seed 42 (excerpt)</h2>
    <pre>{_esc(findings42[:4500])}</pre>
  </section>
</main>
<footer>
  Generated by <code>python -m experiments.build_results_page</code> ·
  Apache-2.0 · Arnav Dhiman
</footer>
</body>
</html>
"""
    out = RESULTS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return out


if __name__ == "__main__":
    build()
