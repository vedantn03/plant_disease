import numpy as np

from scripts.evaluate_candidate_gate import expected_calibration_error


def test_expected_calibration_error_is_zero_for_perfect_confidence():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    labels = np.array([0, 1])
    assert expected_calibration_error(logits, labels) < 1e-4
