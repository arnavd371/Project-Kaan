"""Statistical summaries for multi-seed benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from experiments.common import REFERENCE_PAPER_VAL_ACC


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(x.std(ddof=1))


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def one_sided_ttest_gt(values: np.ndarray, null: float) -> dict[str, float]:
    """H1: mean(values) > null. Returns statistic and p-value."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"), "df": float("nan")}
    t_stat, p_two = stats.ttest_1samp(values, null)
    # one-sided greater
    if np.isnan(t_stat):
        p_one = float("nan")
    elif t_stat > 0:
        p_one = float(p_two / 2)
    else:
        p_one = float(1 - p_two / 2)
    return {"t_stat": float(t_stat), "p_value": p_one, "df": float(n - 1)}


def wilcoxon_gt(values: np.ndarray, null: float) -> dict[str, float]:
    """One-sided Wilcoxon signed-rank: H1 median(values - null) > 0."""
    values = np.asarray(values, dtype=float)
    diffs = values - null
    if len(diffs) < 2 or np.allclose(diffs, 0):
        return {"statistic": float("nan"), "p_value": float("nan")}
    try:
        res = stats.wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan")}


def binomial_ci_wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def summarize_approach(
    approach_id: str,
    accuracies: list[float],
    macro_f1s: list[float],
    reference: float = REFERENCE_PAPER_VAL_ACC,
    n_boot: int = 10000,
) -> dict[str, Any]:
    acc = np.asarray(accuracies, dtype=float)
    f1 = np.asarray(macro_f1s, dtype=float)
    acc_mean, acc_std = _mean_std(acc)
    f1_mean, f1_std = _mean_std(f1)
    acc_ci = bootstrap_ci(acc, n_boot=n_boot, seed=0)
    f1_ci = bootstrap_ci(f1, n_boot=n_boot, seed=1)
    ttest = one_sided_ttest_gt(acc, reference)
    wilcox = wilcoxon_gt(acc, reference)
    n_beat = int(np.sum(acc > reference))
    n = int(len(acc))
    # exact one-sided binomial: P(X >= n_beat) under p=0.5 as a crude seed-win test,
    # plus Wilson CI on fraction of seeds beating reference
    if n > 0:
        p_binom = float(stats.binomtest(n_beat, n, p=0.5, alternative="greater").pvalue)
    else:
        p_binom = float("nan")
    frac_ci = binomial_ci_wilson(n_beat, n)
    return {
        "approach_id": approach_id,
        "n_seeds": n,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "accuracy_ci95": {"low": acc_ci[0], "high": acc_ci[1]},
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "macro_f1_ci95": {"low": f1_ci[0], "high": f1_ci[1]},
        "reference_acc": reference,
        "n_seeds_beat_reference": n_beat,
        "fraction_seeds_beat_reference": float(n_beat / n) if n else float("nan"),
        "fraction_seeds_beat_ci95": {"low": frac_ci[0], "high": frac_ci[1]},
        "ttest_gt_reference": ttest,
        "wilcoxon_gt_reference": wilcox,
        "binom_seeds_beat_p05": p_binom,
        "mean_beats_reference": bool(acc_mean > reference),
        "ci_excludes_reference_below": bool(acc_ci[0] > reference),
        "per_seed_accuracy": acc.tolist(),
        "per_seed_macro_f1": f1.tolist(),
    }


def summarize_all(
    rows: list[dict[str, Any]],
    reference: float = REFERENCE_PAPER_VAL_ACC,
) -> list[dict[str, Any]]:
    by_app: dict[str, list[dict]] = {}
    for row in rows:
        by_app.setdefault(row["approach_id"], []).append(row)
    out = []
    for app, app_rows in by_app.items():
        out.append(
            summarize_approach(
                app,
                [r["accuracy"] for r in app_rows],
                [r["macro_f1"] for r in app_rows],
                reference=reference,
            )
        )
    out.sort(key=lambda d: (-d["accuracy_mean"], d["approach_id"]))
    return out


def write_stats_report(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Statistical summary",
        "",
        f"Reference accuracy (Balingbing et al. 2024): **{REFERENCE_PAPER_VAL_ACC * 100:.2f}%**",
        "",
        "Tests (per approach, across seeds):",
        "- Mean ± std accuracy / macro-F1",
        "- Bootstrap 95% CI of the mean (10k resamples)",
        "- One-sided one-sample t-test: H1 mean accuracy > reference",
        "- One-sided Wilcoxon signed-rank on (acc − reference)",
        "- Fraction of seeds beating reference (Wilson 95% CI)",
        "",
        "| Approach | Acc mean±std | Acc 95% CI | Macro-F1 mean±std | Seeds > ref | t p (>) | Wilcoxon p (>) | CI > ref |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for s in summaries:
        ci = s["accuracy_ci95"]
        lines.append(
            f"| `{s['approach_id']}` | "
            f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f} | "
            f"[{ci['low']:.4f}, {ci['high']:.4f}] | "
            f"{s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f} | "
            f"{s['n_seeds_beat_reference']}/{s['n_seeds']} | "
            f"{s['ttest_gt_reference']['p_value']:.4g} | "
            f"{s['wilcoxon_gt_reference']['p_value']:.4g} | "
            f"{'yes' if s['ci_excludes_reference_below'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- With few seeds, p-values are underpowered; report CIs alongside tests.",
            "- `CI > ref` means the bootstrap 95% CI lower bound exceeds the reference.",
            "- Comparisons to Balingbing et al. use the cited accuracy number, not a shared reimplementation.",
            "",
        ]
    )
    (out_dir / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stats] wrote {out_dir / 'stats.md'}", flush=True)
    for s in summaries:
        print(
            f"  {s['approach_id']}: acc={s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f} "
            f"ci=[{s['accuracy_ci95']['low']:.4f},{s['accuracy_ci95']['high']:.4f}] "
            f"t_p={s['ttest_gt_reference']['p_value']:.4g} "
            f"beat={s['n_seeds_beat_reference']}/{s['n_seeds']}",
            flush=True,
        )
