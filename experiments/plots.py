"""Paper figures for multi-approach pest-detection benchmarks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import CLASS_NAMES, REFERENCE_PAPER_VAL_ACC
from experiments.models import ApproachResult


def _style(ax) -> None:
    ax.set_facecolor("#f7f5f2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_accuracy_f1(results: list[ApproachResult], out_path: Path) -> None:
    active = [r for r in results if not r.skipped]
    if not active:
        return
    ids = [r.approach_id for r in active]
    acc = [r.accuracy for r in active]
    f1 = [r.macro_f1 for r in active]
    x = np.arange(len(ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    _style(ax)
    ax.bar(x - width / 2, acc, width, label="Accuracy", color="#1f6f5f")
    ax.bar(x + width / 2, f1, width, label="Macro F1", color="#c46b3a")
    ax.axhline(
        REFERENCE_PAPER_VAL_ACC,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        label=f"Reference paper ({REFERENCE_PAPER_VAL_ACC * 100:.2f}%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Same-split approaches vs Balingbing et al. reference")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_class_f1(results: list[ApproachResult], out_path: Path) -> None:
    active = [r for r in results if not r.skipped]
    if not active:
        return
    n_cls = len(CLASS_NAMES)
    n_app = len(active)
    x = np.arange(n_cls)
    width = min(0.8 / n_app, 0.18)
    colors = ["#1f6f5f", "#2a9d8f", "#c46b3a", "#6b4f3a", "#4a6fa5", "#8b5a7a"]

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
    _style(ax)
    for i, r in enumerate(active):
        vals = [r.per_class_f1[c] for c in CLASS_NAMES]
        ax.bar(x + (i - n_app / 2) * width + width / 2, vals, width, label=r.approach_id, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 (shared file-level split)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(result: ApproachResult, out_path: Path) -> None:
    if result.skipped:
        return
    cm = np.array(result.confusion, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor="white")
    im = ax.imshow(cm, cmap="YlGn", aspect="equal")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion: {result.approach_id}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_size_vs_f1(results: list[ApproachResult], out_path: Path) -> None:
    pts = [(r.approach_id, r.n_params, r.macro_f1) for r in results if not r.skipped and r.n_params]
    if len(pts) < 1:
        # Fall back: plot accuracy markers without size when params unknown
        active = [r for r in results if not r.skipped]
        if not active:
            return
        fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
        _style(ax)
        for i, r in enumerate(active):
            ax.scatter(i + 1, r.macro_f1, s=80, label=r.approach_id)
        ax.axhline(REFERENCE_PAPER_VAL_ACC, color="#333", linestyle="--", linewidth=1.2)
        ax.set_xticks(range(1, len(active) + 1))
        ax.set_xticklabels([r.approach_id for r in active], rotation=20, ha="right")
        ax.set_ylabel("Macro F1")
        ax.set_title("Approach capacity proxy vs macro F1")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    _style(ax)
    for name, n_params, f1 in pts:
        ax.scatter(n_params, f1, s=90)
        ax.annotate(name, (n_params, f1), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.axhline(REFERENCE_PAPER_VAL_ACC, color="#333", linestyle="--", linewidth=1.2, label="Reference acc")
    ax.set_xlabel("Parameter count (log)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Model size vs macro F1")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_all_plots(results: list[ApproachResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_f1(results, out_dir / "fig_accuracy_f1.png")
    plot_per_class_f1(results, out_dir / "fig_per_class_f1.png")
    plot_size_vs_f1(results, out_dir / "fig_size_vs_f1.png")
    for r in results:
        if not r.skipped:
            plot_confusion(r, out_dir / f"confusion_{r.approach_id}.png")
