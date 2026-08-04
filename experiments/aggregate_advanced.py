"""Aggregate multi-seed advanced-suite JSON into means, stds, and bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stats import bootstrap_ci, _mean_std


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect_metric(summaries: list[dict], path: tuple[str, ...]) -> np.ndarray:
    vals = []
    for s in summaries:
        v = _get(s, *path)
        if v is None:
            continue
        vals.append(float(v))
    return np.asarray(vals, dtype=float)


def summarize_values(vals: np.ndarray, name: str) -> dict[str, Any]:
    if len(vals) == 0:
        return {"name": name, "n": 0}
    mean, std = _mean_std(vals)
    lo, hi = bootstrap_ci(vals, seed=0)
    return {
        "name": name,
        "n": int(len(vals)),
        "mean": mean,
        "std": std,
        "ci95": [lo, hi],
        "values": vals.tolist(),
    }


def aggregate_advanced(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        ("baseline_acc", ("baseline", "accuracy")),
        ("baseline_macro_f1", ("baseline", "macro_f1")),
        ("cost_sensitive_acc", ("cost_sensitive", "accuracy")),
        ("cost_sensitive_swaps", ("cost_sensitive", "weevil_borer_swap_count")),
        ("hierarchical_acc", ("hierarchical", "accuracy")),
        ("hierarchical_swaps", ("hierarchical", "weevil_borer_swap_count")),
        ("hierarchical_pair_subset", ("hierarchical", "pair_subset_accuracy")),
        ("hierarchical_ssl_acc", ("hierarchical_from_ssl", "accuracy")),
        ("ssl_acc", ("ssl", "accuracy")),
        ("ssl_macro_f1", ("ssl", "macro_f1")),
        ("ece_before", ("calibration_baseline", "before", "ece")),
        ("ece_after", ("calibration_baseline", "after", "ece")),
        ("nll_before", ("calibration_baseline", "before", "nll")),
        ("nll_after", ("calibration_baseline", "after", "nll")),
        ("temperature", ("calibration_baseline", "after", "temperature")),
    ]
    out_metrics = {}
    for name, path in metrics:
        out_metrics[name] = summarize_values(collect_metric(summaries, path), name)

    rung_vals: dict[str, list[float]] = {}
    for s in summaries:
        for r in _get(s, "robustness", "rungs", default=[]) or []:
            rung_vals.setdefault(r["rung"], []).append(float(r["accuracy"]))
    robustness = {
        rung: summarize_values(np.asarray(vs, dtype=float), rung) for rung, vs in sorted(rung_vals.items())
    }

    return {
        "n_seeds": len(summaries),
        "seeds": [s.get("seed") for s in summaries],
        "metrics": out_metrics,
        "robustness": robustness,
    }


def to_markdown(agg: dict[str, Any]) -> str:
    lines = [
        "# Advanced suite - multi-seed aggregate",
        "",
        f"Seeds: `{agg.get('seeds')}` (n={agg.get('n_seeds')})",
        "",
        "Bootstrap 95% CI of the mean (10k resamples).",
        "",
        "| Metric | Mean ± std | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, row in (agg.get("metrics") or {}).items():
        if row.get("n", 0) == 0:
            continue
        lines.append(
            f"| `{name}` | {row['mean']:.4f} ± {row['std']:.4f} | "
            f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] |"
        )
    lines.extend(["", "## Robustness ladder", "", "| Rung | Acc mean ± std | 95% CI |", "|---|---:|---:|"])
    for rung, row in (agg.get("robustness") or {}).items():
        if row.get("n", 0) == 0:
            continue
        lines.append(
            f"| `{rung}` | {row['mean']:.2%} ± {row['std']:.2%} | "
            f"[{row['ci95'][0]:.2%}, {row['ci95'][1]:.2%}] |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Comma-separated advanced_summary.json paths, or a directory of seed_*/advanced_summary.json",
    )
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    raw = args.inputs
    cand = Path(raw)
    if cand.is_dir():
        paths = sorted(cand.glob("seed_*/advanced_summary.json"))
        if not paths:
            paths = sorted(cand.rglob("advanced_summary.json"))
    else:
        paths = [Path(x.strip()) for x in raw.split(",") if x.strip()]

    summaries = []
    for path in paths:
        summaries.append(json.loads(path.read_text()))
    if not summaries:
        raise SystemExit(f"No summaries found from {args.inputs}")

    agg = aggregate_advanced(summaries)
    (out / "advanced_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    md = to_markdown(agg)
    (out / "advanced_aggregate.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
