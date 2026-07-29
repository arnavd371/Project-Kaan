#!/usr/bin/env bash
# Create GitHub releases for experiments package, multi-seed stats, and ablations.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO="${GITHUB_REPO:-arnavd371/Project-Kaan}"
export ABLATION_ROOT="${ABLATION_ROOT:-$ROOT/experiments/outputs/ablations}"
STATS_ROOT="${STATS_ROOT:-$ROOT/experiments/outputs/kaggle_run_stats}"
export GITHUB_REPO="$REPO"

cd "$ROOT"
echo "Publishing releases to $REPO"

if ! gh release view v1.1.0-experiments -R "$REPO" >/dev/null 2>&1; then
  gh release create v1.1.0-experiments \
    -R "$REPO" \
    --title "Experiments suite v1.1.0" \
    --notes "Multi-approach benchmark, audits, multi-seed stats, and named CNN ablations for the Project Kaan workshop track."
fi

if [ -f "$STATS_ROOT/stats.md" ] && ! gh release view v1.1.0-multiseed -R "$REPO" >/dev/null 2>&1; then
  assets=("$STATS_ROOT/stats.md" "$STATS_ROOT/stats.json" "$STATS_ROOT/per_seed_metrics.json")
  if [ -f "$STATS_ROOT/aggregate_metrics.json" ]; then
    assets+=("$STATS_ROOT/aggregate_metrics.json")
  fi
  gh release create v1.1.0-multiseed \
    -R "$REPO" \
    --title "Multi-seed benchmark (seeds 42/43/44)" \
    --notes-file "$STATS_ROOT/stats.md" \
    "${assets[@]}"
fi

python3 "$ROOT/experiments/kaggle/publish_ablation_releases.py"
