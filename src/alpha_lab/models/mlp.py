"""Stage 3: MLP weight optimisation.

Architecture exactly as the paper specifies (section 3.4, Algorithm 2 line 25,
Table 10's optimal row): input layer sized to the number of selected alphas, one
hidden layer of ten ReLU units, a single linear output predicting the forward
return. Trained by gradient descent with a separate validation set for early
stopping -- the paper's own stated guard against overfitting.

The leakage discipline, which matters more than the architecture:

* Training rows come only from the fold's train window, and are further trimmed
  so that every row's forward return had completed **inside** that window. A row
  dated three days before the train/validation boundary carries a five-day
  forward return that reaches into validation, and including it would leak.
* Feature standardisation is fitted on the training rows alone and then applied
  unchanged to validation and test. Fitting the scaler on all data is a small
  but real leak that is easy to commit by accident.
* The target is standardised the same way, using training moments only.
* Early stopping selects on validation loss; the test window is never seen
  during fitting, not even for standardisation.

Runs are repeated from several seeds and the median-performing model on
validation is kept, so a result does not hinge on one lucky initialisation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainedModel:
    model: nn.Module
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    train_loss: float
    val_loss: float
    epochs_run: int
    seed: int

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict on a (date, ticker)-indexed feature frame."""
        x = features[self.feature_names].to_numpy(dtype=np.float64)
        valid = ~np.isnan(x).any(axis=1)
        out = np.full(len(features), np.nan)
        if valid.any():
            z = (x[valid] - self.feature_mean) / self.feature_std
            self.model.eval()
            with torch.no_grad():
                pred = self.model(torch.tensor(z, dtype=torch.float32)).squeeze(-1).numpy()
            out[valid] = pred * self.target_std + self.target_mean
        return pd.Series(out, index=features.index)

    @property
    def linear_weights(self) -> pd.Series:
        """First-layer weight magnitude per input, as a rough importance read.

        A three-layer network has no single weight per alpha the way the paper's
        Table 3 implies, so this is the mean absolute input weight -- indicative
        of what the network leans on, not a linear coefficient.
        """
        w = self.model[0].weight.detach().numpy()
        return pd.Series(np.abs(w).mean(axis=0), index=self.feature_names)


def build_network(n_inputs: int, hidden: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(n_inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def _fit_once(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg,
    seed: int,
) -> tuple[nn.Module, float, float, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_network(x_train.shape[1], cfg.hidden_nodes, seed)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    xv = torch.tensor(x_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

    n = len(xt)
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_train = float("inf")
    patience_left = cfg.patience
    epochs_run = 0

    generator = torch.Generator().manual_seed(seed)
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            epoch_loss += loss.detach().item() * len(idx)
        epoch_loss /= max(n, 1)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        epochs_run = epoch + 1

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_train = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, best_train, best_val, epochs_run


def train_weight_optimizer(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    val_features: pd.DataFrame,
    val_target: pd.Series,
    cfg,
) -> TrainedModel | None:
    """Fit the weight-optimisation network. Returns None if there is too little
    clean data to fit on, which is a real outcome, not an error to paper over."""
    feature_names = list(train_features.columns)

    def clean(features: pd.DataFrame, target: pd.Series):
        frame = features.copy()
        frame["__y"] = target
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        return (
            frame[feature_names].to_numpy(dtype=np.float64),
            frame["__y"].to_numpy(dtype=np.float64),
        )

    x_train, y_train = clean(train_features, train_target)
    x_val, y_val = clean(val_features[feature_names], val_target)

    if len(x_train) < 200 or len(x_val) < 50:
        logger.warning(
            "insufficient clean rows to fit MLP (train %d, val %d)", len(x_train), len(x_val)
        )
        return None

    # Standardisation fitted on TRAIN ONLY, then applied unchanged downstream.
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std[feature_std < 1e-9] = 1.0
    target_mean = float(y_train.mean()) if cfg.standardize_target else 0.0
    target_std = float(y_train.std()) if cfg.standardize_target else 1.0
    if target_std < 1e-12:
        target_std = 1.0

    zx_train = (x_train - feature_mean) / feature_std
    zx_val = (x_val - feature_mean) / feature_std
    zy_train = (y_train - target_mean) / target_std
    zy_val = (y_val - target_mean) / target_std

    candidates = []
    for offset in range(max(cfg.n_restarts, 1)):
        seed = cfg.seed + offset
        model, train_loss, val_loss, epochs = _fit_once(
            zx_train, zy_train, zx_val, zy_val, cfg, seed
        )
        candidates.append((val_loss, train_loss, epochs, seed, model))

    # Median restart by validation loss: not the luckiest seed, not the worst.
    candidates.sort(key=lambda c: c[0])
    val_loss, train_loss, epochs, seed, model = candidates[len(candidates) // 2]

    return TrainedModel(
        model=model,
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        train_loss=train_loss,
        val_loss=val_loss,
        epochs_run=epochs,
        seed=seed,
    )
