"""Temperature scaling: T=1 is feasible; a correct fit cannot raise NLL."""

import numpy as np
import torch

from src.calibration import (
    TemperatureScaler,
    fit_temperature,
    nll_at_temperature,
    probs_from_logits,
    softmax_np,
)


def test_divide_sign_softens_when_t_gt_1():
    logits = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    p1 = softmax_np(logits, 1.0)[0, 0]
    p2 = softmax_np(logits, 2.0)[0, 0]
    assert p2 < p1


def test_lbfgs_init_15_on_already_calibrated_does_not_beat_t1_guard():
    """
    Over-peaked random logits with matching labels still have T=1 feasible.
    The guard in fit_temperature must never return a T with worse VAL NLL.
    """
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 5)).astype(np.float32) * 0.1
    labels = rng.integers(0, 5, size=200)
    # Make the true class slightly higher so T=1 is a strong baseline
    logits[np.arange(200), labels] += 2.0
    lt = torch.tensor(logits)
    y = torch.tensor(labels, dtype=torch.long)
    result = fit_temperature(lt, y)
    nll_opt = nll_at_temperature(logits, labels, result["optimal_T"])
    nll_1 = nll_at_temperature(logits, labels, 1.0)
    assert nll_opt <= nll_1 + 1e-6


def test_temperature_scaler_default_init_is_one():
    scaler = TemperatureScaler()
    assert abs(float(scaler.temperature.item()) - 1.0) < 1e-8


def test_probs_from_logits_matches_softmax_t1():
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    p = probs_from_logits(logits, temperature=1.0)
    q = torch.softmax(logits, dim=1)
    assert torch.allclose(p, q)
