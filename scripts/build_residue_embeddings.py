"""0.3 -- per-residue ESM-2 caches for the evaluation set and the training set.

Fork 2 attends across residue positions, so it needs the residues the pooled cache averaged
away. Built once in main and written to the shared cache directory so both worktrees read one
copy.

Layers 6, 10 and 12 are kept: the last layer, a middle layer, and layer 10 because that is the
configuration the Phase 1 head used, which the Fork 2 mean-pool control has to reproduce.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cognate.data import load_train
from cognate.embed import (
    default_cache_dir,
    embed_residues,
    pick_device,
    save_residue_cache,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
LAYERS = (6, 10, 12)
MODEL_KEY = "35M"


def residue_cache_path(model_key: str, split: str) -> Path:
    return default_cache_dir() / f"esm2_{model_key}_{split}_residues.npz"


def sequences_for(split: str) -> np.ndarray:
    if split == "eval":
        frame = pd.read_csv(EVAL_CSV)
    else:
        frame = load_train()
    return np.array(sorted(set(frame["Peptide"]) | set(frame["CDR3b"])), dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="eval", choices=["eval", "train"])
    parser.add_argument("--model", default=MODEL_KEY, choices=["8M", "35M"])
    args = parser.parse_args()

    sequences = sequences_for(args.split)
    path = residue_cache_path(args.model, args.split)
    print(f"split={args.split} model={args.model} strings={len(sequences):,}")
    print(f"layers={LAYERS} device={pick_device()}\n")

    cache, seconds = embed_residues(sequences, args.model, layer_indices=LAYERS)
    size = save_residue_cache(cache, path)
    print(
        f"\n  sequences={cache.n_sequences:,} residues={cache.residues.shape[1]:,} "
        f"hidden={cache.hidden_size}\n"
        f"  wall clock: {seconds:.1f}s\n"
        f"  disk: {size / 1e6:.1f} MB at {path}"
    )


if __name__ == "__main__":
    main()
