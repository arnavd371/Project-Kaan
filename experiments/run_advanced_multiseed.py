"""Run advanced suite across multiple seeds, then aggregate CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aggregate_advanced import aggregate_advanced, to_markdown  # noqa: E402
from experiments.run_advanced import run as run_advanced  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--ssl-pretrain-epochs", type=int, default=30)
    p.add_argument("--out", type=str, default=str(ROOT / "experiments" / "outputs" / "advanced_multiseed"))
    p.add_argument("--copy-results", action="store_true")
    p.add_argument("--hier-scratch", action="store_true")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = []

    for seed in seeds:
        seed_out = out_root / f"seed_{seed}"
        print(f"\n======== ADVANCED SEED {seed} → {seed_out} ========", flush=True)
        ns = argparse.Namespace(
            seed=seed,
            smoke=args.smoke,
            epochs=args.epochs,
            ssl_pretrain_epochs=args.ssl_pretrain_epochs,
            pair_boost=1.8,
            hier_scratch=args.hier_scratch,
            out=str(seed_out),
            no_cache=False,
            copy_results=False,
        )
        summary = run_advanced(ns)
        summaries.append(summary)
        (seed_out / "advanced_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    agg = aggregate_advanced(summaries)
    (out_root / "advanced_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    md = to_markdown(agg)
    (out_root / "advanced_aggregate.md").write_text(md)
    print(md, flush=True)

    if args.copy_results and not args.smoke:
        dest = ROOT / "experiments" / "results" / "advanced_multiseed"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "advanced_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
        (dest / "advanced_aggregate.md").write_text(md)
        for seed, summary in zip(seeds, summaries):
            (dest / f"seed_{seed}_summary.json").write_text(
                json.dumps(summary, indent=2, default=str) + "\n"
            )
        print(f"Copied aggregate → {dest}", flush=True)


if __name__ == "__main__":
    main()
