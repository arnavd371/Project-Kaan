"""Same-split multi-approach benchmark for Project Kaan.

Approaches: cnn_shallow, cnn_deep, cnn1d, yamnet_probe, svm_rbf, mlp, gbdt, rf,
extratrees, knn, logreg. Also writes findings.md.

  python -m experiments.run_benchmark --smoke
  python -m experiments.run_benchmark --seeds 42,43,44 --out experiments/outputs/multi
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit import (
    assert_no_fatal,
    audit_after_training,
    audit_before_training,
    write_audit,
)
from experiments.common import CLASS_NAMES, REFERENCE_PAPER_VAL_ACC
from experiments.data_utils import (
    collect_wav_paths,
    dedupe_by_file_bytes,
    generate_smoke_paths,
    make_split,
    write_split_manifest,
)
from experiments.features import build_feature_matrices
from experiments.models import (
    APPROACH_ORDER,
    format_classification_report,
    run_all_approaches,
    tensorflow_available,
)
from experiments.findings import write_findings
from experiments.plots import write_all_plots
from experiments.stats import summarize_all, write_stats_report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project Kaan multi-approach benchmark")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--out", type=str, default="experiments/outputs/latest")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--models", type=str, default=",".join(APPROACH_ORDER))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cnn-epochs", type=int, default=60)
    p.add_argument("--cnn-baseline", action="store_true")
    p.add_argument(
        "--cnn-recipe",
        type=str,
        default="",
        help='JSON recipe flags, e.g. \'{"spec_augment":false}\'',
    )
    p.add_argument("--smoke-n", type=int, default=24)
    p.add_argument("--audit-soft", action="store_true")
    return p.parse_args()


def _write_metrics(results, out_dir: Path) -> None:
    rows = []
    for r in results:
        d = r.to_dict()
        rows.append(
            {
                "approach_id": d["approach_id"],
                "accuracy": d["accuracy"],
                "macro_f1": d["macro_f1"],
                "beats_reference": d["beats_reference"],
                "reference_acc": d["reference_acc"],
                "n_params": d["n_params"],
                "train_seconds": d["train_seconds"],
                "skipped": d["skipped"],
                "notes": d["notes"],
                **{f"f1_{k}": v for k, v in d["per_class_f1"].items()},
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _write_report(results, out_dir: Path, seed: int, smoke: bool) -> None:
    lines = [
        "# Benchmark report",
        "",
        f"- Seed: `{seed}`",
        f"- Smoke mode: `{smoke}`",
        f"- Reference accuracy (Balingbing et al. 2024): **{REFERENCE_PAPER_VAL_ACC * 100:.2f}%**",
        f"- TensorFlow available: `{tensorflow_available()}`",
        f"- Classes: {', '.join(CLASS_NAMES)}",
        "",
        "## Summary",
        "",
        "| Approach | Acc | Macro F1 | Beats ref? | Notes |",
        "|---|---:|---:|:---:|---|",
    ]
    for r in results:
        if r.skipped:
            lines.append(f"| `{r.approach_id}` | - | - | - | skipped: {r.notes} |")
        else:
            flag = "yes" if r.beats_reference else "no"
            lines.append(
                f"| `{r.approach_id}` | {r.accuracy:.4f} | {r.macro_f1:.4f} | {flag} | {r.notes} |"
            )
    lines.extend(["", "## Classification reports", ""])
    for r in results:
        lines.append(f"### {r.approach_id}")
        lines.append("```")
        lines.append(format_classification_report(r).rstrip())
        lines.append("```")
        lines.append("")
    if smoke:
        lines.extend(
            [
                "## Warning",
                "",
                "Smoke run on synthetic waveforms. Do not cite in the paper.",
                "",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_seed(
    seed: int,
    out_dir: Path,
    smoke: bool,
    approaches: list[str],
    test_size: float,
    cnn_epochs: int,
    smoke_n: int,
    data_root: Path | None = None,
    cnn_strong: bool = True,
    audit_soft: bool = False,
    cnn_recipe: dict | None = None,
) -> list:
    if smoke:
        paths, labels = generate_smoke_paths(n_per_class=smoke_n, seed=seed)
        print(f"[smoke] synthetic paths: {len(paths)} ({smoke_n}/class)")
    else:
        paths, labels = collect_wav_paths(root=data_root)
        if len(paths) == 0:
            raise SystemExit(
                "No WAV files under data/{clean,rice_weevil,lesser_grain_borer,red_flour_beetle}/.\n"
                "See data/HOW_TO_GET_DATA.md, run experiments.prepare_kaggle_data, or re-run with --smoke."
            )
        print(f"[data] loaded {len(paths)} WAV paths from {data_root or 'default data/'}")
        paths, labels, dedupe_info = dedupe_by_file_bytes(paths, labels)
        print(
            f"[data] byte-dedupe: {dedupe_info['n_before']} → {dedupe_info['n_after']} "
            f"(removed {dedupe_info['n_removed']})",
            flush=True,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dedupe_report.json").write_text(
            json.dumps(dedupe_info, indent=2) + "\n",
            encoding="utf-8",
        )
        counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))}
        print(f"[data] class counts after dedupe: {counts}")

    split = make_split(paths, labels, test_size=test_size, seed=seed)
    write_split_manifest(split, out_dir / "split_manifest.json")

    need_waves = "yamnet_probe" in approaches
    print("[features] extracting handcrafted + mel" + (" + waveforms" if need_waves else "") + " …")
    if need_waves:
        X_hand_tr, X_mel_tr, wave_tr = build_feature_matrices(
            split["train_paths"], split["y_train"], smoke_seed=seed, return_waveforms=True
        )
        X_hand_va, X_mel_va, wave_va = build_feature_matrices(
            split["val_paths"], split["y_val"], smoke_seed=seed, return_waveforms=True
        )
    else:
        X_hand_tr, X_mel_tr = build_feature_matrices(split["train_paths"], split["y_train"], smoke_seed=seed)
        X_hand_va, X_mel_va = build_feature_matrices(split["val_paths"], split["y_val"], smoke_seed=seed)
        wave_tr = wave_va = None

    print(f"[features] handcrafted dim={X_hand_tr.shape[1]}, mel={X_mel_tr.shape[1:]}")
    print(f"[train] approaches={approaches}  tf={tensorflow_available()}  cnn_strong={cnn_strong}")

    print("\n=== AUDIT (before training) ===", flush=True)
    before = audit_before_training(
        split["train_paths"],
        split["val_paths"],
        split["y_train"],
        split["y_val"],
        X_mel_tr,
        X_mel_va,
        X_hand_train=X_hand_tr,
        smoke=smoke,
    )
    write_audit(before, out_dir)
    assert_no_fatal(before, hard_fail=not audit_soft)

    results = run_all_approaches(
        X_hand_tr,
        X_hand_va,
        X_mel_tr,
        X_mel_va,
        split["y_train"],
        split["y_val"],
        approaches=approaches,
        smoke=smoke,
        cnn_epochs=cnn_epochs,
        cnn_strong=cnn_strong,
        cnn_recipe=cnn_recipe,
        wave_train=wave_tr,
        wave_val=wave_va,
    )

    history_by = {}
    for r in results:
        hist = getattr(r, "_history", None)
        if hist:
            history_by[r.approach_id] = hist

    print("\n=== AUDIT (after training) ===", flush=True)
    after = audit_after_training(results, history_by_approach=history_by)
    write_audit(after, out_dir)

    _write_metrics(results, out_dir)
    write_all_plots(results, out_dir)
    _write_report(results, out_dir, seed=seed, smoke=smoke)
    write_findings(
        results,
        out_dir,
        X_mel_val=X_mel_va,
        y_val=split["y_val"],
        X_hand_val=X_hand_va,
        seed=seed,
    )

    for r in results:
        if r.skipped:
            print(f"  {r.approach_id}: SKIPPED ({r.notes})")
        else:
            beat = "BEATS ref" if r.beats_reference else "below ref"
            print(
                f"  {r.approach_id}: acc={r.accuracy:.4f}  macro_f1={r.macro_f1:.4f}  ({beat})"
            )
    print(f"[done] wrote {out_dir}")
    return results


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    approaches = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] or [args.seed]
    data_root = Path(args.data_dir).resolve() if args.data_dir else None
    if data_root is not None:
        os.environ["PROJECT_KAAN_DATA_DIR"] = str(data_root)

    cnn_recipe = None
    if args.cnn_recipe.strip():
        cnn_recipe = json.loads(args.cnn_recipe)

    all_summaries = []
    for seed in seeds:
        seed_dir = out_root if len(seeds) == 1 else out_root / f"seed_{seed}"
        print(f"\n=== seed {seed} → {seed_dir} ===")
        results = run_one_seed(
            seed=seed,
            out_dir=seed_dir,
            smoke=args.smoke,
            approaches=approaches,
            test_size=args.test_size,
            cnn_epochs=args.cnn_epochs,
            smoke_n=args.smoke_n,
            data_root=data_root,
            cnn_strong=not args.cnn_baseline,
            audit_soft=args.audit_soft or args.smoke,
            cnn_recipe=cnn_recipe,
        )
        for r in results:
            if not r.skipped:
                all_summaries.append(
                    {
                        "seed": seed,
                        "approach_id": r.approach_id,
                        "accuracy": r.accuracy,
                        "macro_f1": r.macro_f1,
                    }
                )

    if all_summaries:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "per_seed_metrics.json").write_text(
            json.dumps(all_summaries, indent=2) + "\n",
            encoding="utf-8",
        )
        if len(seeds) > 1:
            stats_rows = summarize_all(all_summaries)
            (out_root / "aggregate_metrics.json").write_text(
                json.dumps(stats_rows, indent=2) + "\n",
                encoding="utf-8",
            )
            write_stats_report(stats_rows, out_root)
            print(f"\n[aggregate] wrote {out_root / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
