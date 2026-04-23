import numpy as np

from me26sid.metrics import sweep_thresholds


def test_sweep_thresholds_finds_separating_threshold() -> None:
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])

    result = sweep_thresholds(labels=labels, probs=probs, steps=11)

    assert 0.2 <= result.threshold <= 0.8
    assert result.f1 == 1.0
