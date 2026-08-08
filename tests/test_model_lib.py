import numpy as np
import pandas as pd
import pytest

from src import model_lib as ml


def _frame(n=800, signal=0.0, seed=0):
    """A synthetic panel where `signal` controls how much x tells you about y."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    latent = signal * x + noise
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": (latent > 0).astype(float)})


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------
def test_walk_forward_is_out_of_sample_only():
    """Predictions must cover exactly the rows after the initial training block."""
    frame = _frame(n=600)
    pred, truth = ml.walk_forward_predictions(frame, ["x"], "y", n_init=500, step=21)
    assert len(pred) == len(truth) == 100


def test_walk_forward_empty_feature_set_is_a_coin_flip():
    assert ml.walk_forward_auc(_frame(), [], "y") == 0.5


def test_walk_forward_recovers_a_real_signal():
    strong = ml.walk_forward_auc(_frame(signal=1.5, seed=1), ["x"], "y")
    assert strong > 0.7, f"a strong planted signal should be easy to detect, got {strong:.3f}"


def test_walk_forward_finds_nothing_in_pure_noise():
    null = ml.walk_forward_auc(_frame(signal=0.0, seed=2), ["x"], "y")
    assert 0.40 < null < 0.60, f"pure noise should sit near chance, got {null:.3f}"


def test_walk_forward_raises_when_sample_too_short():
    with pytest.raises(ValueError, match="too short"):
        ml.walk_forward_predictions(_frame(n=100), ["x"], "y", n_init=500)


# --------------------------------------------------------------------------
# bootstrap CI
# --------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_point_estimate():
    frame = _frame(n=900, signal=1.2, seed=3)
    pred, truth = ml.walk_forward_predictions(frame, ["x"], "y")
    from sklearn.metrics import roc_auc_score
    point = roc_auc_score(truth, pred)
    low, high = ml.block_bootstrap_ci(pred, truth, np.random.default_rng(0), n_boot=200)
    assert low < point < high
    assert low > 0.5, "a strong signal's CI should exclude chance"


def test_bootstrap_ci_of_noise_includes_half():
    frame = _frame(n=900, signal=0.0, seed=4)
    pred, truth = ml.walk_forward_predictions(frame, ["x"], "y")
    low, high = ml.block_bootstrap_ci(pred, truth, np.random.default_rng(0), n_boot=200)
    assert low <= 0.5 <= high, "noise must not produce a CI that excludes chance"


# --------------------------------------------------------------------------
# evaluate: the two references together
# --------------------------------------------------------------------------
def test_evaluate_detects_a_planted_signal():
    frame = _frame(n=900, signal=1.5, seed=5)
    got = ml.evaluate(frame, "planted", ["x"], [], "y", np.random.default_rng(0), n_perm=40)
    assert got.auc_with_news > 0.7
    assert got.p_value < 0.05
    assert got.clears_both, "a strong planted signal must clear both references"


def test_evaluate_rejects_pure_noise():
    frame = _frame(n=900, signal=0.0, seed=6)
    got = ml.evaluate(frame, "noise", ["x"], [], "y", np.random.default_rng(0), n_perm=40)
    assert not got.clears_both, f"noise cleared both references (p={got.p_value:.3f})"


def test_evaluate_reports_both_verdicts_independently():
    """clears_both must be the conjunction, never inferred from one reference alone."""
    frame = _frame(n=900, signal=0.6, seed=7)
    got = ml.evaluate(frame, "weak", ["x"], [], "y", np.random.default_rng(0), n_perm=40)
    assert got.clears_both == (got.clears_perm and got.beats_half)


def test_evaluate_baseline_only_lift_is_measured_against_the_baseline():
    frame = _frame(n=900, signal=1.2, seed=8)
    got = ml.evaluate(frame, "over-base", ["x"], ["z"], "y", np.random.default_rng(0), n_perm=30)
    assert got.auc_baseline != 0.5, "a non-empty baseline must be fitted, not assumed"
    assert got.lift == pytest.approx(got.auc_with_news - got.auc_baseline)


# --------------------------------------------------------------------------
# the control gate
# --------------------------------------------------------------------------
def test_control_passed_reads_the_control_spec():
    results = pd.DataFrame([
        {"spec": "1. something", "clears_perm": True},
        {"spec": "4. Trailing volatility (control)", "clears_perm": False},
    ])
    assert ml.control_passed(results) is False
    results.loc[1, "clears_perm"] = True
    assert ml.control_passed(results) is True


def test_control_passed_raises_when_control_missing():
    with pytest.raises(ValueError, match="no positive-control"):
        ml.control_passed(pd.DataFrame([{"spec": "1. only", "clears_perm": True}]))


# --------------------------------------------------------------------------
# harness self-test: the gate
# --------------------------------------------------------------------------
def test_self_test_passes_on_a_working_harness():
    frame = _frame(n=900)
    ok, auc = ml.harness_self_test(frame, "y", seed=1)
    assert ok and auc >= ml.SELF_TEST_MIN_AUC, f"planted signal not recovered (AUC {auc:.3f})"


def test_self_test_is_deterministic():
    frame = _frame(n=900)
    assert ml.harness_self_test(frame, "y", seed=1) == ml.harness_self_test(frame, "y", seed=1)


def test_self_test_fails_when_the_walk_forward_is_broken(monkeypatch):
    """Shuffling the predictions must make the gate fail — otherwise it gates nothing."""
    def broken(frame, cols, y_col, n_init=ml.N_INIT, step=ml.REFIT_STEP):
        rng = np.random.default_rng(0)
        return rng.permutation(np.arange(len(frame) - n_init)) / len(frame), \
               frame[y_col].to_numpy()[n_init:]
    monkeypatch.setattr(ml, "walk_forward_predictions", broken)
    ok, auc = ml.harness_self_test(_frame(n=900), "y", seed=1)
    assert not ok, f"gate passed on a broken walk-forward (AUC {auc:.3f})"


def test_self_test_raises_when_sample_too_short():
    with pytest.raises(ValueError, match="too short"):
        ml.harness_self_test(_frame(n=200), "y")
