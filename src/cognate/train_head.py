"""Trained heads over cached ESM-2 features.

Two heads, both fitted on standardised features. Standardisation is not hygiene here:
mean-pooled ESM-2 embeddings have pairwise cosine similarity of 0.94-0.98 and an effective
rank near 30 out of 320-480 nominal dimensions, so an unscaled design matrix is dominated
by one shared direction.

The MLP is deliberately narrow for the same reason. Early stopping watches validation
*loss* rather than validation macro AUC0.1: the validation arm holds 533 peptides with
roughly 25 rows each, and a per-peptide partial AUC over 25 rows is too coarse to select
on. Macro AUC0.1 is still recorded every epoch as a diagnostic.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from cognate.metrics import macro_auc01

DEFAULT_SEED = 0
DEFAULT_HIDDEN = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_EPOCHS = 60
DEFAULT_PATIENCE = 8


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    max_iter: int = 1000,
    regularisation: float = 1.0,
) -> Pipeline:
    """Standardise, then fit L2 logistic regression."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter, C=regularisation, random_state=seed
        ),
    )
    model.fit(features, labels)
    return model


def logistic_scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


class MlpHead(nn.Module):
    """Two weight layers, one hidden. Narrow by design."""

    def __init__(
        self, n_features: int, hidden: int = DEFAULT_HIDDEN, dropout: float = DEFAULT_DROPOUT
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class TrainingHistory:
    """Per-epoch record, kept so the early-stopping decision is inspectable."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_auc01: list[float] = field(default_factory=list)
    best_epoch: int = -1

    @property
    def n_epochs(self) -> int:
        return len(self.train_loss)


def _to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)


@torch.no_grad()
def mlp_scores(
    model: MlpHead, scaler: StandardScaler, features: np.ndarray
) -> np.ndarray:
    """Score features with a fitted head.

    The device is read off the model rather than passed in, so a caller cannot put the
    inputs somewhere the weights are not.
    """
    model.eval()
    device = str(next(model.parameters()).device)
    logits = model(_to_tensor(scaler.transform(features), device))
    return torch.sigmoid(logits).cpu().numpy()


def fit_mlp(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_groups: np.ndarray,
    *,
    hidden: int = DEFAULT_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> tuple[MlpHead, StandardScaler, TrainingHistory]:
    """Train the MLP head with early stopping on validation loss.

    Returns the model restored to its best-validation-loss weights, the fitted scaler, and
    the full epoch history.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(train_features)
    x_train = _to_tensor(scaler.transform(train_features), device)
    y_train = _to_tensor(train_labels, device)
    x_val = _to_tensor(scaler.transform(val_features), device)
    y_val = _to_tensor(val_labels, device)

    model = MlpHead(train_features.shape[1], hidden, dropout).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    history = TrainingHistory()
    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(device)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimiser.zero_grad()
            loss = criterion(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach()) * len(batch)
        history.train_loss.append(epoch_loss / len(order))

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            history.val_loss.append(float(criterion(val_logits, y_val)))
            probabilities = torch.sigmoid(val_logits).cpu().numpy()
        history.val_macro_auc01.append(
            macro_auc01(val_labels, probabilities, val_groups).value
        )

        if history.val_loss[-1] < best_loss:
            best_loss = history.val_loss[-1]
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - history.best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, scaler, history
