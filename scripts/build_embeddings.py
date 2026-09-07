"""T5a -- embed every distinct peptide and CDR3b with ESM-2 and cache all layers.

Deduplication is the whole trick: the 56,560-row training set contains ~9k distinct CDR3b
and ~820 distinct peptides, so embedding unique strings and joining back by string key is
roughly an order of magnitude less work than embedding rows.
"""

from pathlib import Path

import numpy as np

from cognate.data import load_solutions, load_train
from cognate.embed import MODELS, cache_path, embed_sequences, pick_device, save_cache

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "embeddings"


def collect_sequences() -> np.ndarray:
    train, sol = load_train(), load_solutions()
    peptides = set(train["Peptide"]) | set(sol["Peptide"])
    cdr3b = set(train["CDR3b"]) | set(sol["CDR3b"])
    print(f"distinct peptides: {len(peptides)}")
    print(f"distinct CDR3b:    {len(cdr3b)}")
    print(f"overlap:           {len(peptides & cdr3b)}")
    print(f"rows if embedded naively: {2 * (len(train) * 6 + len(sol)):,}")
    return np.array(sorted(peptides | cdr3b), dtype=object)


def main() -> None:
    sequences = collect_sequences()
    print(f"unique strings to embed: {len(sequences):,}")
    print(f"device: {pick_device()}\n")

    for model_key in MODELS:
        print(f"=== {model_key} ({MODELS[model_key]}) ===")
        cache, seconds = embed_sequences(sequences, model_key)
        path = cache_path(model_key, CACHE_DIR)
        size = save_cache(cache, path)
        print(
            f"  layers={cache.n_layers} hidden={cache.hidden_size} "
            f"sequences={cache.n_sequences:,}\n"
            f"  wall clock: {seconds:.1f}s   "
            f"({1000 * seconds / cache.n_sequences:.2f} ms/sequence)\n"
            f"  disk: {size / 1e6:.1f} MB at {path}\n"
        )


main()
