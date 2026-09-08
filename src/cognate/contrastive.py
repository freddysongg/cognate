"""Fork 1 -- contrastive objectives over frozen ESM-2 embeddings.

Three sessions were spent on negative sampling because binary classification needs negatives
that positives-only data does not contain. Contrastive learning takes its negatives from the
rest of the batch, so the sampling problem dissolves instead of being managed. Nothing here
draws a negative.

Two variants, sharing one projection head design:

* **1a, TCR metric learning.** Train a projection so that TCRs binding the same peptide land
  near each other, then score with the *unchanged* k-NN -- same database, same
  max-over-database operator, same metric. Only the space is new, which makes this the
  cleanest comparison in the project against the 0.565 edit-distance baseline.
* **1b, two-tower alignment.** Separate peptide and TCR projections scored by cosine. This is
  the only construction in the project with a mechanism for unseen peptides: it reads the
  peptide's own residues rather than looking the peptide up in a database, so an unseen
  peptide is not structurally unscoreable the way it is for k-NN.

Both losses are the *supervised* form of InfoNCE. The plain CLIP form assumes the diagonal of
a batch is the only positive, which is wrong here: peptides repeat within a batch, so a naive
diagonal target would push apart two TCRs that bind the same peptide and label that progress.
Every same-peptide pair in the batch is treated as positive instead.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cognate.embed import EmbeddingCache

DEFAULT_SEED = 0
DEFAULT_HIDDEN = 256
DEFAULT_OUTPUT = 128
DEFAULT_TEMPERATURE = 0.07
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_EPOCHS = 60
DEFAULT_PATIENCE = 8
DEFAULT_PEPTIDES_PER_BATCH = 16
DEFAULT_TCRS_PER_PEPTIDE = 8
DEFAULT_BATCHES_PER_EPOCH = 64
MIN_TCRS_FOR_A_POSITIVE_PAIR = 2


class ProjectionHead(nn.Module):
    """480 -> hidden -> output, L2-normalised.

    Normalising the output is what makes the dot product a cosine, so the learned space is
    scored by the same operator the frozen space was.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = DEFAULT_HIDDEN,
        output: int = DEFAULT_OUTPUT,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(features), dim=-1)


def supervised_info_nce(
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    anchor_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
    *,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Cross-entropy over cosine similarities, with every same-label pair a positive.

    ``exclude_self`` removes the diagonal, which is required when anchors and candidates are
    the same rows: an embedding is trivially its own nearest neighbour and would otherwise
    supply almost all of the gradient.

    Rows with no positive available contribute nothing rather than a zero loss, so a batch
    that happens to contain a singleton peptide is not silently counted as solved.
    """
    logits = anchors @ candidates.T / temperature
    positives = anchor_labels[:, None] == candidate_labels[None, :]

    if exclude_self:
        diagonal = torch.eye(len(anchors), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, float("-inf"))
        positives = positives & ~diagonal

    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_positives = positives.sum(dim=1)
    usable = n_positives > 0
    if not bool(usable.any()):
        return anchors.sum() * 0.0

    # Masked entries are -inf, and -inf * False is nan rather than 0, so the non-positive
    # terms have to be dropped by selection rather than by multiplication.
    contributions = torch.where(
        positives, log_probabilities, torch.zeros_like(log_probabilities)
    )
    per_anchor = -contributions.sum(dim=1)[usable] / n_positives[usable]
    return per_anchor.mean()


def pxk_batches(
    labels: np.ndarray,
    *,
    peptides_per_batch: int = DEFAULT_PEPTIDES_PER_BATCH,
    tcrs_per_peptide: int = DEFAULT_TCRS_PER_PEPTIDE,
    batches_per_epoch: int = DEFAULT_BATCHES_PER_EPOCH,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield P-peptides x K-TCRs index batches.

    Uniform sampling over rows would put most batches at one or zero positive pairs, because
    the peptide distribution is long-tailed. Sampling peptides first and then TCRs within them
    guarantees K-1 positives per anchor, which is what makes in-batch negatives informative.

    Only peptides with at least two TCRs are eligible -- a singleton can never form a positive
    pair. Peptides with fewer than K TCRs are sampled with replacement.
    """
    by_label: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) >= MIN_TCRS_FOR_A_POSITIVE_PAIR:
            by_label[int(label)] = rows
    if not by_label:
        raise ValueError("no peptide has two or more TCRs, so no positive pair exists")

    eligible = np.array(sorted(by_label), dtype=np.int64)
    for _ in range(batches_per_epoch):
        chosen = rng.choice(
            eligible, size=min(peptides_per_batch, len(eligible)), replace=False
        )
        batch: list[int] = []
        for label in chosen:
            rows = by_label[int(label)]
            replace = len(rows) < tcrs_per_peptide
            batch.extend(rng.choice(rows, size=tcrs_per_peptide, replace=replace).tolist())
        yield np.array(batch, dtype=np.int64)


@dataclass
class ContrastiveHistory:
    """Per-epoch record, kept so the early-stopping decision is inspectable."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1

    @property
    def n_epochs(self) -> int:
        return len(self.train_loss)


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)


def encode_labels(peptides: Sequence[str]) -> np.ndarray:
    """Map peptide strings to contiguous integer ids."""
    lookup = {peptide: i for i, peptide in enumerate(sorted(set(peptides)))}
    return np.array([lookup[p] for p in peptides], dtype=np.int64)


def _standardise(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (features - mean) / scale


def fit_tcr_metric(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    hidden: int = DEFAULT_HIDDEN,
    output: int = DEFAULT_OUTPUT,
    temperature: float = DEFAULT_TEMPERATURE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    peptides_per_batch: int = DEFAULT_PEPTIDES_PER_BATCH,
    tcrs_per_peptide: int = DEFAULT_TCRS_PER_PEPTIDE,
    batches_per_epoch: int = DEFAULT_BATCHES_PER_EPOCH,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> tuple[ProjectionHead, tuple[np.ndarray, np.ndarray], ContrastiveHistory]:
    """1a -- train one projection so same-peptide TCRs cluster.

    Features are standardised for the same reason the Phase 1 heads standardise them: pooled
    ESM-2 embeddings sit at 0.94-0.98 pairwise cosine with effective rank near 30, so an
    unscaled input is dominated by a single shared direction.
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale == 0.0] = 1.0

    x_train = _tensor(_standardise(train_features, mean, scale), device)
    y_train = torch.tensor(train_labels, dtype=torch.int64, device=device)
    x_val = _tensor(_standardise(val_features, mean, scale), device)
    y_val = torch.tensor(val_labels, dtype=torch.int64, device=device)

    model = ProjectionHead(train_features.shape[1], hidden, output).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history = ContrastiveHistory()
    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(max_epochs):
        model.train()
        losses = []
        for batch in pxk_batches(
            train_labels,
            peptides_per_batch=peptides_per_batch,
            tcrs_per_peptide=tcrs_per_peptide,
            batches_per_epoch=batches_per_epoch,
            rng=rng,
        ):
            index = torch.tensor(batch, dtype=torch.int64, device=device)
            embedded = model(x_train[index])
            loss = supervised_info_nce(
                embedded, embedded, y_train[index], y_train[index],
                temperature, exclude_self=True,
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        history.train_loss.append(float(np.mean(losses)))

        model.eval()
        with torch.no_grad():
            embedded_val = model(x_val)
            history.val_loss.append(
                float(
                    supervised_info_nce(
                        embedded_val, embedded_val, y_val, y_val,
                        temperature, exclude_self=True,
                    )
                )
            )

        if history.val_loss[-1] < best_loss:
            best_loss = history.val_loss[-1]
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - history.best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, (mean, scale), history


def fit_two_tower(
    train_peptide_features: np.ndarray,
    train_tcr_features: np.ndarray,
    train_labels: np.ndarray,
    val_peptide_features: np.ndarray,
    val_tcr_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    hidden: int = DEFAULT_HIDDEN,
    output: int = DEFAULT_OUTPUT,
    temperature: float = DEFAULT_TEMPERATURE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    peptides_per_batch: int = DEFAULT_PEPTIDES_PER_BATCH,
    tcrs_per_peptide: int = DEFAULT_TCRS_PER_PEPTIDE,
    batches_per_epoch: int = DEFAULT_BATCHES_PER_EPOCH,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> tuple[ProjectionHead, ProjectionHead, tuple[np.ndarray, np.ndarray], ContrastiveHistory]:
    """1b -- two towers aligned by symmetric supervised InfoNCE.

    The peptide tower is what gives this variant a mechanism on unseen peptides: it reads the
    peptide's own embedding rather than requiring the peptide to appear in a database.
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    stacked = np.vstack([train_peptide_features, train_tcr_features])
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[scale == 0.0] = 1.0

    p_train = _tensor(_standardise(train_peptide_features, mean, scale), device)
    t_train = _tensor(_standardise(train_tcr_features, mean, scale), device)
    y_train = torch.tensor(train_labels, dtype=torch.int64, device=device)
    p_val = _tensor(_standardise(val_peptide_features, mean, scale), device)
    t_val = _tensor(_standardise(val_tcr_features, mean, scale), device)
    y_val = torch.tensor(val_labels, dtype=torch.int64, device=device)

    peptide_tower = ProjectionHead(train_peptide_features.shape[1], hidden, output).to(device)
    tcr_tower = ProjectionHead(train_tcr_features.shape[1], hidden, output).to(device)
    optimiser = torch.optim.AdamW(
        list(peptide_tower.parameters()) + list(tcr_tower.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def paired_loss(
        peptides: torch.Tensor, tcrs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        embedded_peptides = peptide_tower(peptides)
        embedded_tcrs = tcr_tower(tcrs)
        forward = supervised_info_nce(
            embedded_peptides, embedded_tcrs, labels, labels, temperature
        )
        backward = supervised_info_nce(
            embedded_tcrs, embedded_peptides, labels, labels, temperature
        )
        return 0.5 * (forward + backward)

    history = ContrastiveHistory()
    best_loss = float("inf")
    best_state = (
        {k: v.detach().clone() for k, v in peptide_tower.state_dict().items()},
        {k: v.detach().clone() for k, v in tcr_tower.state_dict().items()},
    )

    for epoch in range(max_epochs):
        peptide_tower.train()
        tcr_tower.train()
        losses = []
        for batch in pxk_batches(
            train_labels,
            peptides_per_batch=peptides_per_batch,
            tcrs_per_peptide=tcrs_per_peptide,
            batches_per_epoch=batches_per_epoch,
            rng=rng,
        ):
            index = torch.tensor(batch, dtype=torch.int64, device=device)
            loss = paired_loss(p_train[index], t_train[index], y_train[index])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        history.train_loss.append(float(np.mean(losses)))

        peptide_tower.eval()
        tcr_tower.eval()
        with torch.no_grad():
            history.val_loss.append(float(paired_loss(p_val, t_val, y_val)))

        if history.val_loss[-1] < best_loss:
            best_loss = history.val_loss[-1]
            history.best_epoch = epoch
            best_state = (
                {k: v.detach().clone() for k, v in peptide_tower.state_dict().items()},
                {k: v.detach().clone() for k, v in tcr_tower.state_dict().items()},
            )
        elif epoch - history.best_epoch >= patience:
            break

    peptide_tower.load_state_dict(best_state[0])
    tcr_tower.load_state_dict(best_state[1])
    return peptide_tower, tcr_tower, (mean, scale), history


@torch.no_grad()
def project(
    model: ProjectionHead,
    features: np.ndarray,
    standardisation: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    model.eval()
    device = str(next(model.parameters()).device)
    mean, scale = standardisation
    return model(_tensor(_standardise(features, mean, scale), device)).cpu().numpy()


def projected_cache(
    cache: EmbeddingCache,
    layer_index: int,
    model: ProjectionHead,
    standardisation: tuple[np.ndarray, np.ndarray],
) -> EmbeddingCache:
    """A cache of projected vectors, so the frozen k-NN can score the learned space unchanged.

    Returning an ``EmbeddingCache`` rather than a bare array is what lets 1a reuse
    ``cosine_similarity`` and ``score_by_nearest_positive`` with no modification at all -- the
    evaluation path is byte-identical to the baseline's apart from the numbers in the array.
    """
    sequences = [str(s) for s in cache.sequences]
    projected = project(model, cache.lookup(sequences, layer_index), standardisation)
    return EmbeddingCache(
        model_name=f"{cache.model_name}+projection",
        sequences=cache.sequences,
        layers=projected[None, :, :].astype(np.float32),
        index=dict(cache.index),
    )
