"""ESM-2 embedding cache for peptides and CDR3b sequences.

Every unique sequence is embedded once and stored keyed by the string itself, so the
56,560-row training set costs ~10k forward passes rather than 113,120. Hidden states from
every layer are kept, because the last layer of a protein language model is tuned for
masked-token prediction rather than for downstream features and a middle layer usually
transfers better.

Two details that are easy to get wrong and are asserted in tests:

* ``add_pooling_layer=False``. The published ESM-2 checkpoints contain no pooler, so
  loading ``EsmModel`` without this flag *randomly initialises* ``pooler.dense`` and
  ``pooler_output`` is noise. Mean pooling over residues is the intended path.
* BOS and EOS are dropped before pooling. ESM-2 tokenises as
  ``<cls> residues... <eos> <pad>...``; averaging the two special tokens in would mix a
  constant offset into every embedding, diluting short sequences more than long ones.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODELS: dict[str, str] = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
}

DEFAULT_MAX_BATCH_TOKENS = 16_384
CACHE_SUFFIX = ".npz"


@dataclass(frozen=True)
class EmbeddingCache:
    """Mean-pooled residue embeddings for every layer, keyed by sequence string."""

    model_name: str
    sequences: np.ndarray
    layers: np.ndarray
    index: dict[str, int]

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    @property
    def n_layers(self) -> int:
        """Includes the embedding layer at index 0, so this is ``num_hidden_layers + 1``."""
        return self.layers.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.layers.shape[2]

    def layer(self, layer_index: int) -> np.ndarray:
        return self.layers[layer_index]

    def lookup(self, sequences: Iterable[str], layer_index: int = -1) -> np.ndarray:
        """Rows for ``sequences`` from one layer, in the order given.

        Raises on a missing sequence rather than returning zeros; a silent zero row is a
        degenerate feature that the metrics guard would only catch downstream.
        """
        wanted = [str(s) for s in sequences]
        missing = {s for s in wanted if s not in self.index}
        if missing:
            raise KeyError(
                f"{len(missing)} sequences are not in the {self.model_name} cache, "
                f"e.g. {sorted(missing)[:3]}"
            )
        return self.layers[layer_index][[self.index[s] for s in wanted]]


def _make_cache(
    model_name: str, sequences: np.ndarray, layers: np.ndarray
) -> EmbeddingCache:
    return EmbeddingCache(
        model_name=model_name,
        sequences=sequences,
        layers=layers,
        index={str(s): i for i, s in enumerate(sequences)},
    )


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _length_batches(
    sequences: list[str], max_batch_tokens: int
) -> list[list[int]]:
    """Group sequence positions into batches of similar length to limit padding waste."""
    order = sorted(range(len(sequences)), key=lambda i: (len(sequences[i]), sequences[i]))
    batches: list[list[int]] = []
    current: list[int] = []
    for position in order:
        candidate = current + [position]
        width = len(sequences[candidate[-1]]) + 2
        if current and len(candidate) * width > max_batch_tokens:
            batches.append(current)
            current = [position]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _residue_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention mask with the BOS and EOS positions cleared."""
    mask = attention_mask.clone()
    mask[:, 0] = 0
    lengths = attention_mask.sum(dim=1)
    mask[torch.arange(mask.shape[0], device=mask.device), lengths - 1] = 0
    return mask


def embed_sequences(
    sequences: Iterable[str],
    model_key: str = "8M",
    *,
    device: str | None = None,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    progress_every: int = 20,
) -> tuple[EmbeddingCache, float]:
    """Embed every distinct sequence with ESM-2, returning the cache and wall-clock seconds."""
    model_name = MODELS.get(model_key, model_key)
    unique = sorted({str(s) for s in sequences})
    device = device or pick_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval().to(device)

    n_layers = model.config.num_hidden_layers + 1
    layers = np.zeros((n_layers, len(unique), model.config.hidden_size), dtype=np.float32)

    batches = _length_batches(unique, max_batch_tokens)
    started = time.perf_counter()
    for batch_number, positions in enumerate(batches, start=1):
        encoded = tokenizer(
            [unique[i] for i in positions], return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)

        mask = _residue_mask(encoded["attention_mask"]).unsqueeze(-1)
        denominator = mask.sum(dim=1)
        for layer_index, hidden in enumerate(output.hidden_states):
            pooled = (hidden * mask).sum(dim=1) / denominator
            layers[layer_index, positions] = pooled.float().cpu().numpy()

        if progress_every and batch_number % progress_every == 0:
            print(f"  batch {batch_number}/{len(batches)}", flush=True)

    return _make_cache(model_name, np.array(unique, dtype=object), layers), (
        time.perf_counter() - started
    )


def cache_path(model_key: str, directory: Path) -> Path:
    return directory / f"esm2_{model_key}{CACHE_SUFFIX}"


def save_cache(cache: EmbeddingCache, path: Path) -> int:
    """Write the cache uncompressed and return its size in bytes.

    Float embeddings compress poorly, so compression costs minutes and saves little.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        model_name=np.array(cache.model_name),
        sequences=cache.sequences.astype(str),
        layers=cache.layers,
    )
    return path.stat().st_size


def load_cache(path: Path) -> EmbeddingCache:
    with np.load(path, allow_pickle=False) as data:
        return _make_cache(
            model_name=str(data["model_name"]),
            sequences=data["sequences"].astype(object),
            layers=data["layers"],
        )
