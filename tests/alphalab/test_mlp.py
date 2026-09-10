"""Stage 3: the weight optimiser's leakage discipline.

The architecture is trivially checkable. What matters is that nothing from
validation or test reaches the fit -- including through the scaler, which is the
leak people commit without noticing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import MLPConfig
from alpha_lab.models.mlp import build_network, train_weight_optimizer


def make_rows(n: int, n_features: int, seed: int, signal: float = 0.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n // 20 + 1)
    index = pd.MultiIndex.from_product(
        [dates[: n // 20], [f"T{i}" for i in range(20)]], names=["date", "ticker"]
    )[:n]
    x = pd.DataFrame(
        rng.normal(size=(len(index), n_features)),
        index=index, columns=[f"a{i}" for i in range(n_features)],
    )
    y = pd.Series(signal * x["a0"].to_numpy() + rng.normal(0, 0.02, len(index)), index=index)
    return x, y


@pytest.fixture
def mlp_cfg():
    return MLPConfig(max_epochs=30, patience=10, n_restarts=1, batch_size=64)


def test_architecture_matches_the_paper():
    """Input -> 10 ReLU -> 1 linear, as section 3.4 and Table 10 specify."""
    net = build_network(n_inputs=8, hidden=10, seed=0)
    assert len(net) == 3
    assert net[0].in_features == 8 and net[0].out_features == 10
    assert net[1].__class__.__name__ == "ReLU"
    assert net[2].in_features == 10 and net[2].out_features == 1


def test_scaler_is_fitted_on_training_rows_only(mlp_cfg):
    """Changing validation data must not move the training standardisation.

    Fitting the scaler across train+validation is a small, silent leak that
    inflates results and is easy to introduce by accident.
    """
    x_train, y_train = make_rows(600, 4, seed=1, signal=0.5)
    x_val, y_val = make_rows(300, 4, seed=2, signal=0.5)

    baseline = train_weight_optimizer(x_train, y_train, x_val, y_val, mlp_cfg)
    shifted = train_weight_optimizer(x_train, y_train, x_val + 100.0, y_val, mlp_cfg)

    assert baseline is not None and shifted is not None
    np.testing.assert_allclose(baseline.feature_mean, shifted.feature_mean)
    np.testing.assert_allclose(baseline.feature_std, shifted.feature_std)
    assert baseline.target_mean == pytest.approx(shifted.target_mean)


def test_model_learns_a_real_signal(mlp_cfg):
    """Sanity floor: with a genuine linear signal the fit must beat guessing."""
    x_train, y_train = make_rows(2000, 3, seed=3, signal=1.0)
    x_val, y_val = make_rows(600, 3, seed=4, signal=1.0)
    model = train_weight_optimizer(x_train, y_train, x_val, y_val, mlp_cfg)
    assert model is not None

    predictions = model.predict(x_val)
    assert predictions.corr(y_val) > 0.5
    # The informative input should carry the largest first-layer weight.
    assert model.linear_weights.idxmax() == "a0"


def test_model_does_not_invent_signal_from_noise(mlp_cfg):
    """With no relationship, out-of-sample correlation must stay near zero."""
    x_train, y_train = make_rows(2000, 3, seed=5, signal=0.0)
    x_val, y_val = make_rows(600, 3, seed=6, signal=0.0)
    x_test, y_test = make_rows(600, 3, seed=7, signal=0.0)
    model = train_weight_optimizer(x_train, y_train, x_val, y_val, mlp_cfg)
    assert model is not None
    assert abs(model.predict(x_test).corr(y_test)) < 0.2


def test_returns_none_rather_than_fitting_on_too_few_rows(mlp_cfg):
    x_train, y_train = make_rows(40, 3, seed=8)
    x_val, y_val = make_rows(40, 3, seed=9)
    assert train_weight_optimizer(x_train, y_train, x_val, y_val, mlp_cfg) is None


def test_predictions_are_nan_where_features_are_missing(mlp_cfg):
    x_train, y_train = make_rows(800, 3, seed=10, signal=0.5)
    x_val, y_val = make_rows(400, 3, seed=11, signal=0.5)
    model = train_weight_optimizer(x_train, y_train, x_val, y_val, mlp_cfg)
    assert model is not None

    x_test, _ = make_rows(100, 3, seed=12)
    x_test.iloc[0, 0] = np.nan
    predictions = model.predict(x_test)
    assert np.isnan(predictions.iloc[0])
    assert predictions.iloc[1:].notna().all()


def test_training_is_reproducible_from_a_seed(mlp_cfg):
    x_train, y_train = make_rows(800, 3, seed=13, signal=0.4)
    x_val, y_val = make_rows(400, 3, seed=14, signal=0.4)
    a = train_weight_optimizer(x_train, y_train, x_val, y_val, mlp_cfg)
    b = train_weight_optimizer(x_train, y_train, x_val, y_val, mlp_cfg)
    assert a is not None and b is not None
    assert a.val_loss == pytest.approx(b.val_loss)
    pd.testing.assert_series_equal(a.predict(x_val), b.predict(x_val))
