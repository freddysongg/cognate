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

import os
import time
from collections.abc import Iterable, Sequence
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
CACHE_DIR_ENV = "COGNATE_CACHE_DIR"
SHARED_CACHE_DIRNAME = "shared"


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


def default_cache_dir() -> Path:
    """Where embedding caches live, from ``COGNATE_CACHE_DIR`` or ``shared/`` in the repo.

    The caches run to gigabytes, so they are gitignored rather than committed. They lived in
    a sibling directory while the two fork worktrees each needed to reach one shared copy;
    with those worktrees removed the repo root is the stable anchor.
    """
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / SHARED_CACHE_DIRNAME


def cache_path(model_key: str, directory: Path | None = None) -> Path:
    return (directory or default_cache_dir()) / f"esm2_{model_key}{CACHE_SUFFIX}"


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


@dataclass(frozen=True)
class ResidueCache:
    """Per-residue embeddings for a subset of layers, stored ragged.

    Fork 2 attends across residue positions, so it cannot use the mean-pooled cache. Lengths
    vary from 5 to 23, so rows are concatenated into one array with an offsets table rather
    than padded to a rectangle -- padding to the longest CDR3b would waste roughly a third of
    the file and invite silent bugs where a pad position is treated as a residue.

    Stored float32, not float16. The acceptance check for this cache is that mean-pooling it
    reproduces the float32 pooled cache to ``atol=1e-5``; float16 carries about three decimal
    digits and would fail that for precision reasons alone, hiding whether the residues were
    actually extracted correctly.
    """

    model_name: str
    sequences: np.ndarray
    layer_indices: np.ndarray
    offsets: np.ndarray
    residues: np.ndarray
    index: dict[str, int]

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    @property
    def hidden_size(self) -> int:
        return self.residues.shape[2]

    def _layer_row(self, layer_index: int) -> int:
        matches = np.flatnonzero(self.layer_indices == layer_index)
        if not len(matches):
            raise KeyError(
                f"layer {layer_index} is not stored; have {self.layer_indices.tolist()}"
            )
        return int(matches[0])

    def residues_for(self, sequence: str, layer_index: int) -> np.ndarray:
        """The ``(length, hidden)`` block for one sequence, BOS and EOS already dropped."""
        key = str(sequence)
        if key not in self.index:
            raise KeyError(f"{key!r} is not in the {self.model_name} residue cache")
        position = self.index[key]
        start, stop = int(self.offsets[position]), int(self.offsets[position + 1])
        return self.residues[self._layer_row(layer_index), start:stop]

    def mean_pooled(self, sequences: Iterable[str], layer_index: int) -> np.ndarray:
        """Mean over residues, in the order given -- the pooled cache, recomputed."""
        return np.stack(
            [self.residues_for(s, layer_index).mean(axis=0) for s in sequences]
        )


def embed_residues(
    sequences: Iterable[str],
    model_key: str = "35M",
    *,
    layer_indices: Sequence[int] = (6, 10, 12),
    device: str | None = None,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    progress_every: int = 20,
) -> tuple[ResidueCache, float]:
    """Embed every distinct sequence and keep each residue, for the given layers."""
    model_name = MODELS.get(model_key, model_key)
    unique = sorted({str(s) for s in sequences})
    device = device or pick_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval().to(device)

    lengths = np.array([len(s) for s in unique], dtype=np.int64)
    offsets = np.zeros(len(unique) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    residues = np.zeros(
        (len(layer_indices), int(offsets[-1]), model.config.hidden_size), dtype=np.float32
    )

    batches = _length_batches(unique, max_batch_tokens)
    started = time.perf_counter()
    for batch_number, positions in enumerate(batches, start=1):
        encoded = tokenizer(
            [unique[i] for i in positions], return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)

        for row, layer_index in enumerate(layer_indices):
            hidden = output.hidden_states[layer_index].float().cpu().numpy()
            for batch_row, position in enumerate(positions):
                length = int(lengths[position])
                residues[row, offsets[position] : offsets[position + 1]] = hidden[
                    batch_row, 1 : 1 + length
                ]

        if progress_every and batch_number % progress_every == 0:
            print(f"  batch {batch_number}/{len(batches)}", flush=True)

    cache = ResidueCache(
        model_name=model_name,
        sequences=np.array(unique, dtype=object),
        layer_indices=np.array(list(layer_indices), dtype=np.int64),
        offsets=offsets,
        residues=residues,
        index={s: i for i, s in enumerate(unique)},
    )
    return cache, time.perf_counter() - started


def save_residue_cache(cache: ResidueCache, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        model_name=np.array(cache.model_name),
        sequences=cache.sequences.astype(str),
        layer_indices=cache.layer_indices,
        offsets=cache.offsets,
        residues=cache.residues,
    )
    return path.stat().st_size


def load_residue_cache(path: Path) -> ResidueCache:
    with np.load(path, allow_pickle=False) as data:
        sequences = data["sequences"].astype(object)
        return ResidueCache(
            model_name=str(data["model_name"]),
            sequences=sequences,
            layer_indices=data["layer_indices"],
            offsets=data["offsets"],
            residues=data["residues"],
            index={str(s): i for i, s in enumerate(sequences)},
        )
