#!/usr/bin/env python3
"""Create one GitHub release per ablation version."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def release_exists(repo: str, tag: str) -> bool:
    return subprocess.run(["gh", "release", "view", tag, "-R", repo], capture_output=True).returncode == 0


def main() -> None:
    root = Path(os.environ.get("ABLATION_ROOT", "experiments/outputs/ablations")).resolve()
    repo = os.environ.get("GITHUB_REPO", "arnavd371/Project-Kaan")
    table = root / "ablation_table.md"

    if not root.is_dir():
        print(f"No ablation root at {root}; skip ablation releases", file=sys.stderr)
        return

    versions = sorted(p for p in root.iterdir() if p.is_dir() and (p / "version.json").exists())
    for vdir in versions:
        meta = json.loads((vdir / "version.json").read_text(encoding="utf-8"))
        tag = meta["tag"]
        if release_exists(repo, tag):
            print(f"skip existing {tag}")
            continue

        notes = [
            f"# {meta['title']}",
            "",
            meta["description"],
            "",
            f"- version_id: `{meta['version_id']}`",
            f"- recipe: `{json.dumps(meta['recipe'])}`",
            f"- models: {', '.join(meta['models'])}",
            f"- seeds: {meta['seeds']}",
            "",
        ]
        metrics = vdir / "metrics.csv"
        if metrics.exists():
            notes.extend(["## Metrics", "```", metrics.read_text(encoding="utf-8").strip(), "```", ""])
        if table.exists():
            notes.extend(["## Full ablation table", "", table.read_text(encoding="utf-8")])

        notes_path = vdir / "RELEASE_NOTES.md"
        notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

        assets: list[str] = []
        for name in (
            "metrics.csv",
            "metrics.json",
            "report.md",
            "version.json",
            "audit_before_training.md",
            "audit_after_training.md",
            "dedupe_report.json",
            "split_manifest.json",
        ):
            p = vdir / name
            if p.exists():
                assets.append(str(p))
        assets.extend(str(p) for p in sorted(vdir.glob("fig_*.png")))
        assets.extend(str(p) for p in sorted(vdir.glob("confusion_*.png")))

        cmd = [
            "gh",
            "release",
            "create",
            tag,
            "-R",
            repo,
            "--title",
            meta["title"],
            "--notes-file",
            str(notes_path),
            *assets,
        ]
        print("creating", tag, "assets", len(assets))
        subprocess.run(cmd, check=True)

    summary_tag = "v1.1.6-ablation-summary"
    if not release_exists(repo, summary_tag) and table.exists():
        assets = [
            str(root / n)
            for n in ("ablation_table.md", "ablation_table.csv", "ablation_table.json")
            if (root / n).exists()
        ]
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                summary_tag,
                "-R",
                repo,
                "--title",
                "CNN ablation summary table",
                "--notes-file",
                str(table),
                *assets,
            ],
            check=True,
        )
        print("created", summary_tag)


if __name__ == "__main__":
    main()
