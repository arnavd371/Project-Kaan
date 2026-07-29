"""Named CNN ablation configs and GitHub release version map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AblationVersion:
    """One workshop ablation = one GitHub release tag."""

    version_id: str
    tag: str
    title: str
    description: str
    cnn_strong: bool
    recipe: dict[str, bool]
    models: tuple[str, ...]
    seeds: tuple[int, ...]


# Each entry becomes a separate GitHub release after the Kaggle run.
ABLATIONS: list[AblationVersion] = [
    AblationVersion(
        version_id="full",
        tag="v1.1.1-ablate-full",
        title="Ablation: full CNN recipe",
        description=(
            "cnn_deep + cnn_shallow with SpecAugment, class weights, "
            "label smoothing, and cosine LR. Classical baselines included."
        ),
        cnn_strong=True,
        recipe={
            "spec_augment": True,
            "class_weights": True,
            "label_smoothing": True,
            "cosine_lr": True,
        },
        models=("cnn_shallow", "cnn_deep", "svm_rbf", "mlp", "gbdt", "logreg"),
        seeds=(42,),
    ),
    AblationVersion(
        version_id="no_specaugment",
        tag="v1.1.2-ablate-no-specaugment",
        title="Ablation: without SpecAugment",
        description="Full recipe minus SpecAugment (class weights + label smooth + cosine kept).",
        cnn_strong=True,
        recipe={
            "spec_augment": False,
            "class_weights": True,
            "label_smoothing": True,
            "cosine_lr": True,
        },
        models=("cnn_shallow", "cnn_deep"),
        seeds=(42,),
    ),
    AblationVersion(
        version_id="no_class_weights",
        tag="v1.1.3-ablate-no-class-weights",
        title="Ablation: without class weights",
        description="Full recipe minus class weights.",
        cnn_strong=True,
        recipe={
            "spec_augment": True,
            "class_weights": False,
            "label_smoothing": True,
            "cosine_lr": True,
        },
        models=("cnn_shallow", "cnn_deep"),
        seeds=(42,),
    ),
    AblationVersion(
        version_id="no_label_smoothing",
        tag="v1.1.4-ablate-no-label-smoothing",
        title="Ablation: without label smoothing",
        description="Full recipe minus label smoothing.",
        cnn_strong=True,
        recipe={
            "spec_augment": True,
            "class_weights": True,
            "label_smoothing": False,
            "cosine_lr": True,
        },
        models=("cnn_shallow", "cnn_deep"),
        seeds=(42,),
    ),
    AblationVersion(
        version_id="baseline",
        tag="v1.1.5-ablate-baseline",
        title="Ablation: baseline CNN",
        description="Adam + sparse CE only (no SpecAugment / class weights / label smooth / cosine).",
        cnn_strong=False,
        recipe={
            "spec_augment": False,
            "class_weights": False,
            "label_smoothing": False,
            "cosine_lr": False,
        },
        models=("cnn_shallow", "cnn_deep"),
        seeds=(42,),
    ),
]


def ablation_by_id(version_id: str) -> AblationVersion:
    for a in ABLATIONS:
        if a.version_id == version_id:
            return a
    raise KeyError(f"Unknown ablation version_id={version_id!r}")


def ablation_table_rows(metrics_by_version: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten per-version metrics.json rows into a comparison table."""
    rows = []
    for abl in ABLATIONS:
        for m in metrics_by_version.get(abl.version_id, []):
            if m.get("skipped"):
                continue
            rows.append(
                {
                    "version_id": abl.version_id,
                    "tag": abl.tag,
                    "approach_id": m["approach_id"],
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "beats_reference": m["beats_reference"],
                    "recipe": abl.recipe,
                }
            )
    return rows
