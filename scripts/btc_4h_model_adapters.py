from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSettings:
    seed: int = 42
    sequence_length: int = 24
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 0.001
    hidden_size: int = 32
    num_heads: int = 4


def make_sequence_samples(
    frame: pd.DataFrame, feature_names: tuple[str, ...], sequence_length: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Make sequences ending at each prediction row without using later feature rows."""
    if sequence_length < 1 or len(frame) < sequence_length:
        raise ValueError("frame must contain at least sequence_length rows")
    values = frame.loc[:, feature_names].to_numpy(dtype=np.float32)
    sequences = np.stack(
        [values[end - sequence_length + 1 : end + 1] for end in range(sequence_length - 1, len(frame))]
    )
    targets = frame["target_up"].iloc[sequence_length - 1 :].to_numpy(dtype=np.float32)
    times = pd.DatetimeIndex(frame["open_time"].iloc[sequence_length - 1 :])
    return sequences, targets, times


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _scaled_partition_sequences(
    train: pd.DataFrame,
    partition: pd.DataFrame,
    feature_names: tuple[str, ...],
    scaler: StandardScaler,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = pd.concat([train.tail(sequence_length - 1), partition], ignore_index=True)
    scaled = history.copy()
    scaled.loc[:, feature_names] = scaler.transform(history.loc[:, feature_names])
    sequences, targets, _ = make_sequence_samples(scaled, feature_names, sequence_length)
    return sequences, targets


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(values)
        return self.output(hidden[-1]).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_heads: int, sequence_length: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dim_feedforward=hidden_size * 2
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(values) + self.position)
        return self.output(encoded[:, -1]).squeeze(-1)


def _fit_predict_sequence_model(
    model: nn.Module,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: tuple[str, ...],
    settings: ModelSettings,
) -> tuple[np.ndarray, np.ndarray]:
    _set_seed(settings.seed)
    scaler = StandardScaler().fit(train.loc[:, feature_names])
    train_scaled = train.copy()
    train_scaled.loc[:, feature_names] = scaler.transform(train.loc[:, feature_names])
    train_x, train_y, _ = make_sequence_samples(train_scaled, feature_names, settings.sequence_length)
    validation_x, validation_y = _scaled_partition_sequences(
        train, validation, feature_names, scaler, settings.sequence_length
    )
    test_history = pd.concat([train, validation], ignore_index=True)
    test_x, _ = _scaled_partition_sequences(
        test_history, test, feature_names, scaler, settings.sequence_length
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=settings.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for _ in range(settings.max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(torch.from_numpy(validation_x)), torch.from_numpy(validation_y)
            ).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                break
    if best_state is None:
        raise RuntimeError("sequence model did not produce a validation state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probability = torch.sigmoid(model(torch.from_numpy(validation_x))).numpy()
        test_probability = torch.sigmoid(model(torch.from_numpy(test_x))).numpy()
    return validation_probability, test_probability


def fit_predict_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: tuple[str, ...],
    settings: ModelSettings,
) -> tuple[np.ndarray, np.ndarray]:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=settings.seed,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(train.loc[:, feature_names], train["target_up"])
    return (
        model.predict_proba(validation.loc[:, feature_names])[:, 1],
        model.predict_proba(test.loc[:, feature_names])[:, 1],
    )


def fit_predict_lstm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: tuple[str, ...],
    settings: ModelSettings,
) -> tuple[np.ndarray, np.ndarray]:
    _set_seed(settings.seed)
    return _fit_predict_sequence_model(
        LSTMClassifier(len(feature_names), settings.hidden_size),
        train,
        validation,
        test,
        feature_names,
        settings,
    )


def fit_predict_transformer(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: tuple[str, ...],
    settings: ModelSettings,
) -> tuple[np.ndarray, np.ndarray]:
    if settings.hidden_size % settings.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    _set_seed(settings.seed)
    return _fit_predict_sequence_model(
        TransformerClassifier(
            len(feature_names), settings.hidden_size, settings.num_heads, settings.sequence_length
        ),
        train,
        validation,
        test,
        feature_names,
        settings,
    )
