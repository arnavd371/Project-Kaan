"""Post-benchmark analyses: McNemar, error tables, SNR proxy, efficiency, INT8 parity."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from experiments.common import CLASS_NAMES, REFERENCE_PAPER_VAL_ACC
from experiments.models import ApproachResult, tensorflow_available


def mcnemar_test(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> dict[str, Any]:
    """Exact McNemar on paired predictions."""
    y_true = np.asarray(y_true)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    correct_a = y_a == y_true
    correct_b = y_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "note": "no discordant pairs"}
    p = float(stats.binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p, "note": "exact McNemar"}


def pairwise_mcnemar(results: list[ApproachResult]) -> list[dict[str, Any]]:
    active = [r for r in results if not r.skipped and r.y_true and r.y_pred]
    out: list[dict[str, Any]] = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            if len(a.y_true) != len(b.y_true):
                continue
            row = mcnemar_test(np.asarray(a.y_true), np.asarray(a.y_pred), np.asarray(b.y_pred))
            row.update(
                {
                    "approach_a": a.approach_id,
                    "approach_b": b.approach_id,
                    "acc_a": a.accuracy,
                    "acc_b": b.accuracy,
                }
            )
            out.append(row)
    out.sort(key=lambda d: (d["p_value"], -abs(d["acc_a"] - d["acc_b"])))
    return out


def error_analysis(results: list[ApproachResult]) -> dict[str, Any]:
    by_app: dict[str, Any] = {}
    for r in results:
        if r.skipped or not r.y_true:
            continue
        y_true = np.asarray(r.y_true)
        y_pred = np.asarray(r.y_pred)
        cm = np.asarray(r.confusion, dtype=int)
        pairs = []
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                if i == j or cm[i, j] == 0:
                    continue
                pairs.append({"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "count": int(cm[i, j])})
        pairs.sort(key=lambda d: -d["count"])
        by_app[r.approach_id] = {
            "n_errors": int(np.sum(y_true != y_pred)),
            "n_total": int(len(y_true)),
            "top_confusions": pairs[:8],
            "per_class_f1": r.per_class_f1,
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
        }
    return by_app


def snr_robustness_on_mel(
    X_mel_val: np.ndarray,
    y_val: np.ndarray,
    snr_db_levels: list[float] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Gaussian noise on mel; logistic probe on a held-out slice of the val set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    snr_db_levels = snr_db_levels or [20.0, 10.0, 5.0, 0.0]
    rng = np.random.default_rng(seed)
    X = np.asarray(X_mel_val, dtype=np.float32)
    y = np.asarray(y_val)
    flat = X.reshape(len(X), -1)
    n = len(flat)
    if n < 8:
        return {"skipped": True, "reason": "too few val samples for SNR probe"}

    cut = max(4, int(0.7 * n))
    idx = rng.permutation(n)
    tr, va = idx[:cut], idx[cut:]
    probe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    probe.fit(flat[tr], y[tr])
    clean_acc = float(accuracy_score(y[va], probe.predict(flat[va])))

    curve = []
    for snr in snr_db_levels:
        noisy = X.copy()
        for i in range(len(noisy)):
            sig = noisy[i]
            power = float(np.mean(sig**2)) + 1e-12
            noise_power = power / (10 ** (snr / 10.0))
            noisy[i] = sig + rng.normal(0.0, np.sqrt(noise_power), size=sig.shape).astype(np.float32)
        acc = float(accuracy_score(y[va], probe.predict(noisy.reshape(len(noisy), -1)[va])))
        curve.append({"snr_db": snr, "probe_acc": acc, "drop_vs_clean": clean_acc - acc})

    return {
        "skipped": False,
        "method": "logistic probe on flattened mel; split inside val set",
        "clean_probe_acc": clean_acc,
        "curve": curve,
        "note": "Proxy only; not a phone-mic field test.",
    }


def latency_and_size(results: list[ApproachResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        if r.skipped:
            continue
        rows.append(
            {
                "approach_id": r.approach_id,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "n_params": r.n_params,
                "train_seconds": r.train_seconds,
                "beats_reference": r.beats_reference,
            }
        )
    rows.sort(key=lambda d: (-d["macro_f1"], d["train_seconds"]))
    return rows


def int8_parity(
    X_mel_val: np.ndarray,
    y_val: np.ndarray,
    h5_path: Path | None = None,
    tflite_path: Path | None = None,
) -> dict[str, Any]:
    """Compare production float Keras H5 vs INT8 TFLite on the same mel batch."""
    root = Path(__file__).resolve().parent.parent
    h5_path = h5_path or (root / "model" / "project-kaan_model.h5")
    tflite_path = tflite_path or (root / "model" / "project-kaan.tflite")
    out: dict[str, Any] = {
        "h5_path": str(h5_path),
        "tflite_path": str(tflite_path),
        "h5_exists": h5_path.exists(),
        "tflite_exists": tflite_path.exists(),
    }
    if not (h5_path.exists() and tflite_path.exists()):
        out["skipped"] = True
        out["reason"] = "production H5/TFLite missing"
        return out
    if not tensorflow_available():
        out["skipped"] = True
        out["reason"] = "TensorFlow unavailable"
        return out

    import tensorflow as tf
    from tensorflow import keras

    X = np.asarray(X_mel_val, dtype=np.float32)
    y = np.asarray(y_val)
    if X.ndim == 3:
        X = X[..., None]

    model = keras.models.load_model(h5_path)
    t0 = time.perf_counter()
    probs_f = model.predict(X, verbose=0)
    float_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(X))
    pred_f = np.argmax(probs_f, axis=1)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    preds_q = []
    t0 = time.perf_counter()
    for i in range(len(X)):
        sample = X[i : i + 1]
        if inp["dtype"] == np.uint8:
            scale, zp = inp["quantization"]
            scale = scale or 1.0
            interpreter.set_tensor(inp["index"], (sample / scale + zp).astype(np.uint8))
        else:
            interpreter.set_tensor(inp["index"], sample.astype(inp["dtype"]))
        interpreter.invoke()
        raw = interpreter.get_tensor(out_d["index"])
        if out_d["dtype"] == np.uint8:
            scale, zp = out_d["quantization"]
            raw = scale * (raw.astype(np.float32) - zp)
        preds_q.append(int(np.argmax(raw)))
    int8_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(X))
    pred_q = np.asarray(preds_q)

    out.update(
        {
            "skipped": False,
            "n": int(len(y)),
            "float_acc": float(accuracy_score(y, pred_f)),
            "int8_acc": float(accuracy_score(y, pred_q)),
            "float_macro_f1": float(f1_score(y, pred_f, average="macro", zero_division=0)),
            "int8_macro_f1": float(f1_score(y, pred_q, average="macro", zero_division=0)),
            "prediction_agreement": float(np.mean(pred_f == pred_q)),
            "acc_gap_float_minus_int8": float(accuracy_score(y, pred_f) - accuracy_score(y, pred_q)),
            "latency_ms_per_sample_float": float_ms,
            "latency_ms_per_sample_int8": int8_ms,
            "h5_kb": h5_path.stat().st_size / 1024.0,
            "tflite_kb": tflite_path.stat().st_size / 1024.0,
            "reference_acc": REFERENCE_PAPER_VAL_ACC,
        }
    )
    return out


def write_findings(
    results: list[ApproachResult],
    out_dir: Path,
    X_mel_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    X_hand_val: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    del X_hand_val  # reserved for future live latency probes
    out_dir.mkdir(parents=True, exist_ok=True)
    active = [r for r in results if not r.skipped]
    best = max(active, key=lambda r: r.accuracy) if active else None
    classical_ids = {"svm_rbf", "mlp", "gbdt", "rf", "extratrees", "knn", "logreg"}
    classical = [r for r in active if r.approach_id in classical_ids]
    cnn = [r for r in active if r.approach_id.startswith("cnn")]
    best_classical = max(classical, key=lambda r: r.accuracy) if classical else None
    best_cnn = max(cnn, key=lambda r: r.accuracy) if cnn else None

    headline = []
    if best:
        headline.append(
            f"Best same-split accuracy: `{best.approach_id}` at {best.accuracy:.4f} "
            f"(macro-F1 {best.macro_f1:.4f}); beats ref={best.beats_reference}."
        )
    if best_classical and best_cnn:
        delta = best_classical.accuracy - best_cnn.accuracy
        headline.append(
            f"Best classical `{best_classical.approach_id}` ({best_classical.accuracy:.4f}) vs "
            f"best CNN `{best_cnn.approach_id}` ({best_cnn.accuracy:.4f}); delta={delta:+.4f}."
        )
        if best_classical.y_true and best_cnn.y_true and len(best_classical.y_true) == len(best_cnn.y_true):
            m = mcnemar_test(
                np.asarray(best_classical.y_true),
                np.asarray(best_classical.y_pred),
                np.asarray(best_cnn.y_pred),
            )
            headline.append(
                f"McNemar classical vs CNN: discordant={m['n_discordant']}, p={m['p_value']:.4g}."
            )

    payload: dict[str, Any] = {
        "headline": headline,
        "mcnemar_pairwise": pairwise_mcnemar(results),
        "error_analysis": error_analysis(results),
        "efficiency": latency_and_size(results),
        "snr_proxy": (
            snr_robustness_on_mel(X_mel_val, y_val, seed=seed)
            if X_mel_val is not None and y_val is not None
            else {"skipped": True, "reason": "no mel val"}
        ),
        "int8_parity": (
            int8_parity(X_mel_val, y_val)
            if X_mel_val is not None and y_val is not None
            else {"skipped": True, "reason": "no mel val"}
        ),
    }

    (out_dir / "findings.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# Findings", "", "## Headlines", ""]
    if headline:
        for h in headline:
            lines.append(f"- {h}")
    else:
        lines.append("- No active approaches.")

    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| Approach | Acc | Macro F1 | Params | Train s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["efficiency"]:
        params = row["n_params"] if row["n_params"] is not None else "-"
        lines.append(
            f"| `{row['approach_id']}` | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{params} | {row['train_seconds']:.1f} |"
        )

    lines.extend(["", "## McNemar (lowest p)", ""])
    for row in payload["mcnemar_pairwise"][:12]:
        lines.append(
            f"- `{row['approach_a']}` vs `{row['approach_b']}`: p={row['p_value']:.4g}, "
            f"discordant={row['n_discordant']} (b={row['b']}, c={row['c']})"
        )

    lines.extend(["", "## Top confusions", ""])
    for app, info in payload["error_analysis"].items():
        top = ", ".join(f"{c['true']}→{c['pred']} ({c['count']})" for c in info["top_confusions"][:4]) or "none"
        lines.append(f"- `{app}`: {info['n_errors']}/{info['n_total']} errors; {top}")

    snr = payload["snr_proxy"]
    lines.extend(["", "## SNR proxy", ""])
    if snr.get("skipped"):
        lines.append(f"- Skipped: {snr.get('reason')}")
    else:
        lines.append(f"- Clean probe acc: {snr['clean_probe_acc']:.4f}")
        for c in snr["curve"]:
            lines.append(
                f"- SNR {c['snr_db']} dB: probe_acc={c['probe_acc']:.4f} (drop={c['drop_vs_clean']:+.4f})"
            )
        lines.append(f"- {snr.get('note', '')}")

    parity = payload["int8_parity"]
    lines.extend(["", "## INT8 vs float", ""])
    if parity.get("skipped"):
        lines.append(f"- Skipped: {parity.get('reason')}")
    else:
        lines.append(
            f"- Float acc={parity['float_acc']:.4f}, INT8 acc={parity['int8_acc']:.4f}, "
            f"agreement={parity['prediction_agreement']:.4f}, gap={parity['acc_gap_float_minus_int8']:+.4f}"
        )
        lines.append(
            f"- Size: H5 {parity['h5_kb']:.1f} KB → TFLite {parity['tflite_kb']:.1f} KB; "
            f"ms/sample float={parity['latency_ms_per_sample_float']:.2f}, "
            f"int8={parity['latency_ms_per_sample_int8']:.2f}"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- McNemar is single-fold; report multi-seed CIs with it.",
            "- SNR curve is a proxy, not a phone-mic field study.",
            "- Reference 84.51% is the cited Balingbing number on this protocol, not a locked reimplementation.",
            "",
        ]
    )
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[findings] wrote {out_dir / 'findings.md'}", flush=True)
    for h in headline:
        print(f"  · {h}", flush=True)
    return payload
