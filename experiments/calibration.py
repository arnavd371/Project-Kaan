"""Calibration (temperature scaling) and selective prediction / abstain curves."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probs_from_model_outputs(outputs: np.ndarray) -> np.ndarray:
    """Accept either probabilities or logits."""
    outputs = np.asarray(outputs, dtype=np.float64)
    if outputs.ndim != 2:
        raise ValueError("outputs must be (N, C)")
    row_sums = outputs.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-2) and np.all(outputs >= -1e-6):
        p = np.clip(outputs, 1e-12, 1.0)
        return p / p.sum(axis=1, keepdims=True)
    return softmax_np(outputs, 1.0)


def negative_log_likelihood(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(log_loss(y_true, probs, labels=list(range(probs.shape[1]))))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(m):
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Grid-search temperature minimizing NLL on held-out logits."""
    if grid is None:
        grid = np.concatenate(
            [np.linspace(0.5, 5.0, 46), np.array([0.1, 0.25, 7.0, 10.0])]
        )
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        p = softmax_np(logits, float(t))
        nll = negative_log_likelihood(y_true, p)
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def reliability_bins(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == y_true
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(m):
            rows.append(
                {
                    "bin": i,
                    "lo": float(bins[i]),
                    "hi": float(bins[i + 1]),
                    "count": 0,
                    "acc": None,
                    "conf": None,
                }
            )
            continue
        rows.append(
            {
                "bin": i,
                "lo": float(bins[i]),
                "hi": float(bins[i + 1]),
                "count": int(m.sum()),
                "acc": float(correct[m].mean()),
                "conf": float(conf[m].mean()),
            }
        )
    return rows


def abstain_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """For each confidence threshold: coverage, accuracy on kept, macro-F1 on kept."""
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.99, 28)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    rows = []
    n = len(y_true)
    for t in thresholds:
        keep = conf >= float(t)
        cov = float(keep.mean())
        if keep.sum() == 0:
            rows.append(
                {
                    "threshold": float(t),
                    "coverage": 0.0,
                    "n_kept": 0,
                    "accuracy": None,
                    "macro_f1": None,
                }
            )
            continue
        yt, yp = y_true[keep], pred[keep]
        rows.append(
            {
                "threshold": float(t),
                "coverage": cov,
                "n_kept": int(keep.sum()),
                "n_total": n,
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            }
        )
    return rows


def calibration_report(
    y_true: np.ndarray,
    probs_or_logits: np.ndarray,
    *,
    treat_as_logits: bool = False,
    val_split_for_t: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit T on a slice of val, report ECE/NLL before/after + abstain curve."""
    y_true = np.asarray(y_true, dtype=np.int32)
    raw = np.asarray(probs_or_logits, dtype=np.float64)
    if treat_as_logits:
        logits = raw
        probs = softmax_np(logits, 1.0)
    else:
        probs = probs_from_model_outputs(raw)
        logits = np.log(np.clip(probs, 1e-12, 1.0))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * val_split_for_t))
    fit_idx, eval_idx = idx[:cut], idx[cut:]
    if len(eval_idx) < 5:
        fit_idx, eval_idx = idx, idx

    t_star = fit_temperature(logits[fit_idx], y_true[fit_idx])
    probs_t = softmax_np(logits[eval_idx], t_star)
    probs_u = probs[eval_idx]
    y_e = y_true[eval_idx]

    before = {
        "nll": negative_log_likelihood(y_e, probs_u),
        "ece": expected_calibration_error(y_e, probs_u),
        "accuracy": float(accuracy_score(y_e, probs_u.argmax(1))),
    }
    after = {
        "nll": negative_log_likelihood(y_e, probs_t),
        "ece": expected_calibration_error(y_e, probs_t),
        "accuracy": float(accuracy_score(y_e, probs_t.argmax(1))),
        "temperature": t_star,
    }
    full_scaled = softmax_np(logits, t_star)
    return {
        "before": before,
        "after": after,
        "reliability_before": reliability_bins(y_e, probs_u),
        "reliability_after": reliability_bins(y_e, probs_t),
        "abstain_curve": abstain_curve(y_true, full_scaled),
        "n_fit": int(len(fit_idx)),
        "n_eval": int(len(eval_idx)),
    }
