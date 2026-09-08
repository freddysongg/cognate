"""Fork 2 -- attention across residue positions instead of mean-pooling them away.

The Phase 1 head averages ~15 residue vectors into one before the model sees anything, and
binding is a question of which peptide residues contact which TCR residues. Session 4 measured
effective rank 20-33 against 480 nominal dimensions in the pooled embeddings, which is
consistent with very little surviving the pool.

Both variants share one skeleton and differ in exactly one step:

* **2a** projects each residue, lets peptide and TCR positions attend to each other, then pools.
* **2b** projects each residue and pools immediately, with no attention at all.

Everything else -- the input projections, the pooling, the interaction features, the MLP head,
the optimiser, the split, the negatives, the seeds -- is identical. That is what makes the
difference attributable to attention rather than to an incidental change in training setup, and
it is why the epic requires reporting both or neither.

A note on what 2b is. Masked mean-pooling commutes with a linear projection: for an affine map,
``mean(W x + b) == W mean(x) + b``. So 2b is not merely *similar* to the Phase 1 mean-pooled
head, it is that head with a learned input projection in front -- which is why it is the right
reference point and why it is expected to land near 0.511.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from cognate.embed import ResidueCache
from cognate.metrics import macro_auc01

DEFAULT_SEED = 0
DEFAULT_MODEL_WIDTH = 64
DEFAULT_HEADS = 2
DEFAULT_ATTENTION_LAYERS = 1
DEFAULT_MLP_HIDDEN = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 6
DEFAULT_SCORING_BATCH = 1024


@dataclass(frozen=True)
class PaddedResidues:
    """Residues for a set of distinct sequences, right-padded into one rectangle.

    Padding is materialised once over *distinct* sequences rather than per row: the evaluation
    set has 116,658 rows over 18,760 distinct CDR3b, so padding per row would cost roughly six
    times the memory for identical content.
    """

    values: np.ndarray
    mask: np.ndarray
    index: dict[str, int]

    @property
    def max_length(self) -> int:
        return self.values.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.values.shape[2]

    def rows_for(self, sequences: np.ndarray) -> np.ndarray:
        return np.array([self.index[str(s)] for s in sequences], dtype=np.int64)


def pad_residues(
    cache: ResidueCache, sequences: list[str], layer_index: int
) -> PaddedResidues:
    """Right-pad each sequence's residues to the longest in the set, with a validity mask."""
    distinct = sorted({str(s) for s in sequences})
    blocks = [cache.residues_for(s, layer_index) for s in distinct]
    longest = max(len(b) for b in blocks)

    values = np.zeros((len(distinct), longest, blocks[0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(distinct), longest), dtype=bool)
    for row, block in enumerate(blocks):
        values[row, : len(block)] = block
        mask[row, : len(block)] = True

    return PaddedResidues(
        values=values, mask=mask, index={s: i for i, s in enumerate(distinct)}
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid positions only.

    Dividing by the true length rather than the padded width is the whole point: a plain
    ``.mean(dim=1)`` would dilute a 5-residue CDR3b almost five times more than a 23-residue
    one, turning sequence length into a feature.
    """
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1)


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention between peptide and TCR positions."""

    def __init__(self, width: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.peptide_attends = nn.ModuleList(
            nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        )
        self.tcr_attends = nn.ModuleList(
            nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        )
        self.peptide_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.tcr_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))

    def forward(
        self,
        peptide: torch.Tensor,
        peptide_mask: torch.Tensor,
        tcr: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for attend_p, attend_t, norm_p, norm_t in zip(
            self.peptide_attends, self.tcr_attends, self.peptide_norms, self.tcr_norms
        ):
            attended_peptide, _ = attend_p(
                peptide, tcr, tcr, key_padding_mask=~tcr_mask, need_weights=False
            )
            attended_tcr, _ = attend_t(
                tcr, peptide, peptide, key_padding_mask=~peptide_mask, need_weights=False
            )
            peptide = norm_p(peptide + attended_peptide)
            tcr = norm_t(tcr + attended_tcr)
        return peptide, tcr


class ResidueBindingModel(nn.Module):
    """2a when ``use_attention``, 2b when not. Nothing else differs between them."""

    def __init__(
        self,
        hidden_size: int,
        *,
        use_attention: bool,
        width: int = DEFAULT_MODEL_WIDTH,
        heads: int = DEFAULT_HEADS,
        layers: int = DEFAULT_ATTENTION_LAYERS,
        mlp_hidden: int = DEFAULT_MLP_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.peptide_projection = nn.Linear(hidden_size, width)
        self.tcr_projection = nn.Linear(hidden_size, width)
        self.attention = (
            CrossAttentionBlock(width, heads, layers, dropout) if use_attention else None
        )
        self.head = nn.Sequential(
            nn.Linear(4 * width, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        peptide: torch.Tensor,
        peptide_mask: torch.Tensor,
        tcr: torch.Tensor,
        tcr_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected_peptide = self.peptide_projection(peptide)
        projected_tcr = self.tcr_projection(tcr)
        if self.attention is not None:
            projected_peptide, projected_tcr = self.attention(
                projected_peptide, peptide_mask, projected_tcr, tcr_mask
            )
        pooled_peptide = masked_mean(projected_peptide, peptide_mask)
        pooled_tcr = masked_mean(projected_tcr, tcr_mask)
        features = torch.cat(
            [
                pooled_peptide,
                pooled_tcr,
                (pooled_peptide - pooled_tcr).abs(),
                pooled_peptide * pooled_tcr,
            ],
            dim=-1,
        )
        return self.head(features).squeeze(-1)


@dataclass
class AttentionHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_auc01: list[float] = field(default_factory=list)
    best_epoch: int = -1

    @property
    def n_epochs(self) -> int:
        return len(self.train_loss)


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class RowBatches:
    """Row indices into the padded peptide and TCR tables, plus targets."""

    def __init__(
        self,
        peptide_rows: np.ndarray,
        tcr_rows: np.ndarray,
        targets: np.ndarray,
        peptides: PaddedResidues,
        tcrs: PaddedResidues,
        device: str,
    ) -> None:
        self.peptide_rows = peptide_rows
        self.tcr_rows = tcr_rows
        self.targets = targets
        self.peptide_values = torch.from_numpy(peptides.values)
        self.peptide_mask = torch.from_numpy(peptides.mask)
        self.tcr_values = torch.from_numpy(tcrs.values)
        self.tcr_mask = torch.from_numpy(tcrs.mask)
        self.device = device

    def __len__(self) -> int:
        return len(self.targets)

    def batch(self, rows: np.ndarray) -> tuple[torch.Tensor, ...]:
        peptide_index = torch.from_numpy(self.peptide_rows[rows])
        tcr_index = torch.from_numpy(self.tcr_rows[rows])
        return (
            self.peptide_values[peptide_index].to(self.device),
            self.peptide_mask[peptide_index].to(self.device),
            self.tcr_values[tcr_index].to(self.device),
            self.tcr_mask[tcr_index].to(self.device),
            torch.from_numpy(self.targets[rows]).float().to(self.device),
        )


@torch.no_grad()
def score_rows(
    model: ResidueBindingModel, data: RowBatches, batch_size: int = DEFAULT_SCORING_BATCH
) -> np.ndarray:
    model.eval()
    scores = np.zeros(len(data), dtype=np.float32)
    for start in range(0, len(data), batch_size):
        rows = np.arange(start, min(start + batch_size, len(data)))
        peptide, peptide_mask, tcr, tcr_mask, _ = data.batch(rows)
        scores[rows] = torch.sigmoid(
            model(peptide, peptide_mask, tcr, tcr_mask)
        ).cpu().numpy()
    return scores


def fit_residue_model(
    train_data: RowBatches,
    val_data: RowBatches,
    val_groups: np.ndarray,
    hidden_size: int,
    *,
    use_attention: bool,
    width: int = DEFAULT_MODEL_WIDTH,
    heads: int = DEFAULT_HEADS,
    layers: int = DEFAULT_ATTENTION_LAYERS,
    mlp_hidden: int = DEFAULT_MLP_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> tuple[ResidueBindingModel, AttentionHistory]:
    """Train one variant with early stopping on validation loss.

    Early stopping watches validation *loss* rather than validation macro AUC0.1, matching the
    Phase 1 head: the validation arm holds few rows per peptide and a per-peptide partial AUC
    over that many rows is too coarse to select on. Macro AUC0.1 is recorded as a diagnostic.
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = ResidueBindingModel(
        hidden_size,
        use_attention=use_attention,
        width=width,
        heads=heads,
        layers=layers,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    history = AttentionHistory()
    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_data), generator=generator).numpy()
        total = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            peptide, peptide_mask, tcr, tcr_mask, target = train_data.batch(rows)
            optimiser.zero_grad()
            loss = criterion(model(peptide, peptide_mask, tcr, tcr_mask), target)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(rows)
        history.train_loss.append(total / len(order))

        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for start in range(0, len(val_data), DEFAULT_SCORING_BATCH):
                rows = np.arange(start, min(start + DEFAULT_SCORING_BATCH, len(val_data)))
                peptide, peptide_mask, tcr, tcr_mask, target = val_data.batch(rows)
                validation_total += float(
                    criterion(model(peptide, peptide_mask, tcr, tcr_mask), target)
                ) * len(rows)
        history.val_loss.append(validation_total / len(val_data))
        history.val_macro_auc01.append(
            macro_auc01(val_data.targets, score_rows(model, val_data), val_groups).value
        )

        if history.val_loss[-1] < best_loss:
            best_loss = history.val_loss[-1]
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - history.best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, history
