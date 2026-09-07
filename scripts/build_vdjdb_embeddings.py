"""Embed the Phase B evaluation set into its own ESM-2 cache.

Written to a separate file from the Phase 1 caches on purpose. Adding sequences to
`esm2_35M.npz` would change its length-sorted batching, and mean-pooled values can move in the
last float bits when batch composition changes -- which would break the byte-identical
reproduction of `data/headline.json` that Phase A established for no benefit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cognate.embed import embed_sequences, pick_device, save_cache

ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
CACHE = ROOT / "data" / "embeddings" / "esm2_35M_vdjdb.npz"
MODEL_KEY = "35M"


def main() -> None:
    evalset = pd.read_csv(EVAL_CSV)
    sequences = np.array(
        sorted(set(evalset["Peptide"]) | set(evalset["CDR3b"])), dtype=object
    )
    print(f"distinct peptides: {evalset['Peptide'].nunique()}")
    print(f"distinct CDR3b:    {evalset['CDR3b'].nunique()}")
    print(f"strings to embed:  {len(sequences):,}")
    print(f"device: {pick_device()}\n")

    cache, seconds = embed_sequences(sequences, MODEL_KEY)
    size = save_cache(cache, CACHE)
    print(
        f"\n  layers={cache.n_layers} hidden={cache.hidden_size} "
        f"sequences={cache.n_sequences:,}\n"
        f"  wall clock: {seconds:.1f}s\n"
        f"  disk: {size / 1e6:.1f} MB at {CACHE}"
    )


if __name__ == "__main__":
    main()
