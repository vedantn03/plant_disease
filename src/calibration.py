"""
src/calibration.py
==================
Calibration metrics and post-hoc temperature scaling.

Temperature scaling divides logits by a scalar T (Guo et al., 2017).
T is fitted on VAL by minimising NLL. T=1 is always feasible, so a correct
fit cannot increase VAL NLL relative to the unscaled model.

Serving does not apply T unless a documented refit shows VAL improvement
(see docs/DECISIONS.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width confidence bins on [0, 1]."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = float(in_bin.mean())
        if prop > 0.0:
            acc = float(accuracies[in_bin].mean())
            conf = float(confidences[in_bin].mean())
            ece += abs(acc - conf) * prop
    return float(ece)


def compute_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true class."""
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(p_true + eps)))


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error vs one-hot labels."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibration_bundle(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    return {
        "ece": round(compute_ece(probs, labels, n_bins=n_bins), 5),
        "nll": round(compute_nll(probs, labels), 5),
        "brier": round(compute_brier(probs, labels), 5),
    }


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def nll_at_temperature(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    return compute_nll(softmax_np(logits, temperature), labels)


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling: scaled_logits = logits / T.

    Default init is T=1.0 (the unscaled, always-feasible point). The previous
    evaluate.py scaler initialised T=1.5, which let LBFGS stop at T≈1.32 with
    *higher* VAL NLL than T=1 — i.e. a failed minimisation, not a real fit.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(init_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = self.temperature.clamp(min=1e-4)
        return logits / t

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )
        criterion = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


def fit_temperature_lbfgs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init_temperature: float = 1.0,
    max_iter: int = 100,
) -> float:
    device = logits.device
    scaler = TemperatureScaler(init_temperature=init_temperature).to(device)
    return scaler.fit(logits, labels, max_iter=max_iter)


def fit_temperature_grid(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.50,
    t_max: float = 5.00,
    n_grid: int = 451,
) -> tuple[float, float, list[dict[str, float]]]:
    """
    1-D grid search over T. Returns (best_T, best_NLL, curve samples).
    Grid includes T=1.0 exactly.
    """
    temperatures = np.unique(np.concatenate([
        np.linspace(t_min, t_max, n_grid),
        np.array([1.0]),
    ]))
    best_t = 1.0
    best_nll = nll_at_temperature(logits, labels, 1.0)
    curve: list[dict[str, float]] = []
    for t in temperatures:
        nll = nll_at_temperature(logits, labels, float(t))
        curve.append({"T": round(float(t), 4), "nll": round(float(nll), 6)})
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return float(best_t), float(best_nll), curve


def fit_temperature(
    logits_t: torch.Tensor,
    labels_t: torch.Tensor,
) -> dict[str, Any]:
    """
    Fit T on VAL with LBFGS (inits 1.0 and 1.5) and a 1-D grid.
    Select the T with lowest VAL NLL; never worse than T=1.
    """
    logits_np = logits_t.detach().cpu().numpy()
    labels_np = labels_t.detach().cpu().numpy().astype(np.int64)

    nll_t1 = nll_at_temperature(logits_np, labels_np, 1.0)
    t_grid, nll_grid, _curve = fit_temperature_grid(logits_np, labels_np)

    t_lbfgs_1 = fit_temperature_lbfgs(logits_t, labels_t, init_temperature=1.0)
    t_lbfgs_15 = fit_temperature_lbfgs(logits_t, labels_t, init_temperature=1.5)
    nll_lbfgs_1 = nll_at_temperature(logits_np, labels_np, t_lbfgs_1)
    nll_lbfgs_15 = nll_at_temperature(logits_np, labels_np, t_lbfgs_15)

    candidates = [
        ("T=1", 1.0, nll_t1),
        ("grid", t_grid, nll_grid),
        ("lbfgs_init_1.0", t_lbfgs_1, nll_lbfgs_1),
        ("lbfgs_init_1.5", t_lbfgs_15, nll_lbfgs_15),
    ]
    source, best_t, best_nll = min(candidates, key=lambda x: x[2])

    # Guard: never accept a T that raises VAL NLL vs T=1
    if best_nll > nll_t1 + 1e-12:
        source, best_t, best_nll = "T=1_guard", 1.0, nll_t1

    return {
        "optimal_T": round(float(best_t), 4),
        "optimal_source": source,
        "val_nll_at_T1": round(float(nll_t1), 5),
        "val_nll_at_optimal_T": round(float(best_nll), 5),
        "lbfgs_init_1.0": {"T": round(float(t_lbfgs_1), 4), "nll": round(float(nll_lbfgs_1), 5)},
        "lbfgs_init_1.5": {"T": round(float(t_lbfgs_15), 4), "nll": round(float(nll_lbfgs_15), 5)},
        "grid": {"T": round(float(t_grid), 4), "nll": round(float(nll_grid), 5)},
        "sign": "divide",
        "formula": "scaled_logits = logits / T",
    }


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over logits / T. Default T=1 (unscaled)."""
    return F.softmax(logits / float(temperature), dim=1)
