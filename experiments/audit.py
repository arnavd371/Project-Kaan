"""Before/after training audits for the multi-approach benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import CLASS_NAMES, REFERENCE_PAPER_VAL_ACC


def _spec_hash(spec: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(spec).tobytes()).hexdigest()


def _file_bytes_hash(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.md5(p.read_bytes()).hexdigest()


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    phase: str
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, **metrics: Any) -> None:
        self.checks.append(CheckResult(name=name, status=status, detail=detail, metrics=metrics))

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "checks": [asdict(c) for c in self.checks]}

    def summary_line(self) -> str:
        counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
        for c in self.checks:
            counts[c.status] = counts.get(c.status, 0) + 1
        return (
            f"{self.phase}: PASS={counts['PASS']} WARN={counts['WARN']} "
            f"FAIL={counts['FAIL']} INFO={counts['INFO']}"
        )


def audit_before_training(
    train_paths: np.ndarray,
    val_paths: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_mel_train: np.ndarray,
    X_mel_val: np.ndarray,
    X_hand_train: np.ndarray | None = None,
    smoke: bool = False,
) -> AuditReport:
    report = AuditReport(phase="before_training")

    train_counts = {CLASS_NAMES[i]: int((y_train == i).sum()) for i in range(len(CLASS_NAMES))}
    val_counts = {CLASS_NAMES[i]: int((y_val == i).sum()) for i in range(len(CLASS_NAMES))}
    empty_train = [c for c, n in train_counts.items() if n == 0]
    empty_val = [c for c, n in val_counts.items() if n == 0]
    status = "FAIL" if empty_train or empty_val else "PASS"
    detail = (
        f"empty classes train={empty_train} val={empty_val}"
        if empty_train or empty_val
        else f"train={train_counts} val={val_counts}"
    )
    report.add("class_counts", status, detail, train=train_counts, val=val_counts)

    train_set = set(map(str, train_paths.tolist()))
    val_set = set(map(str, val_paths.tolist()))
    overlap = train_set & val_set
    report.add(
        "file_level_leakage",
        "PASS" if len(overlap) == 0 else "FAIL",
        f"train={len(train_set)} val={len(val_set)} overlap={len(overlap)}",
        n_train=len(train_set),
        n_val=len(val_set),
        n_overlap=len(overlap),
        overlap_sample=sorted(overlap)[:5],
    )

    if smoke:
        report.add("byte_identical_wav_cross_split", "INFO", "skipped in smoke mode")
    else:
        train_hashes: dict[str, list[str]] = {}
        for p in train_paths:
            h = _file_bytes_hash(str(p))
            if h:
                train_hashes.setdefault(h, []).append(str(p))
        cross = 0
        examples = []
        for p in val_paths:
            h = _file_bytes_hash(str(p))
            if h and h in train_hashes:
                cross += 1
                if len(examples) < 3:
                    examples.append({"val": str(p), "train": train_hashes[h][0]})
        report.add(
            "byte_identical_wav_cross_split",
            "PASS" if cross == 0 else "WARN",
            f"val files with byte-identical train twin: {cross}",
            n_cross=cross,
            examples=examples,
        )

    val_hashes = {_spec_hash(s) for s in X_mel_val}
    dupe = sum(1 for s in X_mel_train if _spec_hash(s) in val_hashes)
    report.add(
        "exact_mel_duplicate_train_val",
        "PASS" if dupe == 0 else "WARN",
        f"exact mel duplicates shared train↔val: {dupe}",
        n_dupes=dupe,
    )

    mel_ok = (
        X_mel_train.ndim == 4
        and X_mel_train.shape[1:] == (128, 128, 1)
        and X_mel_val.shape[1:] == (128, 128, 1)
        and len(X_mel_train) == len(y_train)
        and len(X_mel_val) == len(y_val)
    )
    report.add(
        "mel_feature_shapes",
        "PASS" if mel_ok else "FAIL",
        f"train={tuple(X_mel_train.shape)} val={tuple(X_mel_val.shape)}",
        train_shape=list(X_mel_train.shape),
        val_shape=list(X_mel_val.shape),
    )
    if X_hand_train is not None:
        hand_ok = X_hand_train.ndim == 2 and len(X_hand_train) == len(y_train) and X_hand_train.shape[1] > 0
        report.add(
            "handcrafted_feature_shapes",
            "PASS" if hand_ok else "FAIL",
            f"train={tuple(X_hand_train.shape)}",
            dim=int(X_hand_train.shape[1]) if X_hand_train.ndim == 2 else None,
        )

    try:
        import tensorflow as tf

        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
        report.add(
            "tensorflow_runtime",
            "INFO",
            f"tf={tf.__version__} gpus={gpus or 'none'}",
            tf_version=tf.__version__,
            gpus=gpus,
        )
    except Exception as e:
        report.add("tensorflow_runtime", "WARN", f"TensorFlow unavailable: {e}")

    report.add(
        "reference_line",
        "INFO",
        f"Balingbing et al. reference accuracy = {REFERENCE_PAPER_VAL_ACC * 100:.2f}%",
        reference_acc=REFERENCE_PAPER_VAL_ACC,
    )
    return report


def audit_after_training(results: list, history_by_approach: dict[str, dict] | None = None) -> AuditReport:
    report = AuditReport(phase="after_training")
    history_by_approach = history_by_approach or {}

    if not results:
        report.add("results_present", "FAIL", "no approach results")
        return report

    report.add("results_present", "PASS", f"n_approaches={len(results)}")

    for r in results:
        if getattr(r, "skipped", False):
            report.add(f"{r.approach_id}_skipped", "WARN", r.notes or "skipped")
            continue

        preds = np.asarray(r.y_pred)
        n_unique = int(len(np.unique(preds))) if preds.size else 0
        collapsed = n_unique <= 1
        report.add(
            f"{r.approach_id}_prediction_diversity",
            "FAIL" if collapsed else "PASS",
            f"unique predicted classes={n_unique} / {len(CLASS_NAMES)}",
            n_unique_pred=n_unique,
        )

        beat = bool(r.beats_reference)
        report.add(
            f"{r.approach_id}_vs_reference",
            "PASS" if beat else "WARN",
            f"acc={r.accuracy:.4f} macro_f1={r.macro_f1:.4f} "
            f"ref={REFERENCE_PAPER_VAL_ACC:.4f} beats={beat}",
            accuracy=r.accuracy,
            macro_f1=r.macro_f1,
            beats_reference=beat,
            per_class_f1=r.per_class_f1,
        )

        cm = np.asarray(r.confusion, dtype=int)
        if cm.size:
            diag = np.diag(cm)
            zero_diag = [CLASS_NAMES[i] for i, v in enumerate(diag) if v == 0]
            report.add(
                f"{r.approach_id}_confusion_diag",
                "WARN" if zero_diag else "PASS",
                f"zero-diagonal classes={zero_diag or 'none'}",
                confusion=cm.tolist(),
            )

        hist = history_by_approach.get(r.approach_id)
        if hist:
            best_val = max(hist.get("val_accuracy", [0.0]) or [0.0])
            report.add(
                f"{r.approach_id}_training_curve",
                "INFO",
                f"epochs_ran={len(hist.get('loss', []))} best_val_acc={best_val:.4f}",
                best_val_accuracy=float(best_val),
                final_val_accuracy=float((hist.get("val_accuracy") or [0.0])[-1]),
                final_train_accuracy=float((hist.get("accuracy") or [0.0])[-1]),
            )

    beating = [r.approach_id for r in results if not r.skipped and r.beats_reference]
    report.add(
        "hypothesis_summary",
        "PASS" if beating else "WARN",
        f"approaches beating {REFERENCE_PAPER_VAL_ACC * 100:.2f}%: {beating or 'none'}",
        beating=beating,
    )
    return report


def write_audit(report: AuditReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"audit_{report.phase}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# Audit: {report.phase}",
        "",
        report.summary_line(),
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in report.checks:
        detail = c.detail.replace("|", "\\|")
        md_lines.append(f"| `{c.name}` | **{c.status}** | {detail} |")
    md_path = out_dir / f"audit_{report.phase}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[audit] {report.summary_line()}", flush=True)
    for c in report.checks:
        print(f"  [{c.status}] {c.name}: {c.detail}", flush=True)
    return path


def assert_no_fatal(report: AuditReport, *, hard_fail: bool = True) -> None:
    fails = [c for c in report.checks if c.status == "FAIL"]
    if fails and hard_fail:
        names = ", ".join(c.name for c in fails)
        raise SystemExit(f"Audit FAIL ({report.phase}): {names}. See audit_{report.phase}.json")
