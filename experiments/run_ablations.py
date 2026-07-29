"""Run the named CNN ablation grid (one folder per version).

  python -m experiments.run_ablations --data-dir /path/to/data --out experiments/outputs/ablations
  ABLATION=all  (default) or ABLATION=full,no_specaugment
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ablations import ABLATIONS, ablation_by_id, ablation_table_rows
from experiments.common import REFERENCE_PAPER_VAL_ACC
from experiments.run_benchmark import run_one_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project Kaan CNN ablation grid")
    p.add_argument("--out", type=str, default="experiments/outputs/ablations")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--cnn-epochs", type=int, default=60)
    p.add_argument(
        "--versions",
        type=str,
        default="",
        help="Comma-separated version_ids (default: all)",
    )
    p.add_argument("--audit-soft", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    data_root = Path(args.data_dir).resolve() if args.data_dir else None
    if data_root is not None:
        os.environ["PROJECT_KAAN_DATA_DIR"] = str(data_root)

    wanted = [v.strip() for v in args.versions.split(",") if v.strip()]
    if not wanted:
        env = os.environ.get("ABLATION", "all").strip()
        if env and env != "all":
            wanted = [v.strip() for v in env.split(",") if v.strip()]
    abl_list = [ablation_by_id(v) for v in wanted] if wanted else list(ABLATIONS)

    metrics_by_version: dict[str, list] = {}
    for abl in abl_list:
        print(f"\n######## ablation {abl.version_id} → {abl.tag} ########", flush=True)
        version_dir = out_root / abl.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "version.json").write_text(
            json.dumps(
                {
                    "version_id": abl.version_id,
                    "tag": abl.tag,
                    "title": abl.title,
                    "description": abl.description,
                    "cnn_strong": abl.cnn_strong,
                    "recipe": abl.recipe,
                    "models": list(abl.models),
                    "seeds": list(abl.seeds),
                    "reference_acc": REFERENCE_PAPER_VAL_ACC,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        for seed in abl.seeds:
            seed_dir = version_dir if len(abl.seeds) == 1 else version_dir / f"seed_{seed}"
            results = run_one_seed(
                seed=seed,
                out_dir=seed_dir,
                smoke=False,
                approaches=list(abl.models),
                test_size=0.2,
                cnn_epochs=args.cnn_epochs,
                smoke_n=24,
                data_root=data_root,
                cnn_strong=abl.cnn_strong,
                audit_soft=args.audit_soft,
                cnn_recipe=dict(abl.recipe),
            )
            rows = [r.to_dict() for r in results if not r.skipped]
            metrics_by_version.setdefault(abl.version_id, []).extend(rows)

    table = ablation_table_rows(metrics_by_version)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ablation_table.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")

    csv_path = out_root / "ablation_table.csv"
    if table:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fields = ["version_id", "tag", "approach_id", "accuracy", "macro_f1", "beats_reference"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(table)

    md = [
        "# CNN ablation comparison",
        "",
        f"Reference: **{REFERENCE_PAPER_VAL_ACC * 100:.2f}%**",
        "",
        "| Version | Tag | Approach | Acc | Macro F1 | Beats ref? |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for row in table:
        md.append(
            f"| `{row['version_id']}` | `{row['tag']}` | `{row['approach_id']}` | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{'yes' if row['beats_reference'] else 'no'} |"
        )
    (out_root / "ablation_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[ablations] wrote {out_root / 'ablation_table.md'}", flush=True)


if __name__ == "__main__":
    main()
