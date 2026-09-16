# Phase 2 implementation evidence

## Scope

Implemented Phase 2 only in:

- `src/cognate/pmhc.py`
- `tests/test_pmhc.py`

No TCR module was edited. `docs/pmhc/src/cognate/data.py` was not read, imported,
moved, or edited. NetMHCpan was not run. Phase 3 was not started.

## Implementation

- Added one fixed-seed random-score function using
  `np.random.default_rng(0).random(n_rows)`.
- Added 21 fixed composition features: length followed by counts in
  `ACDEFGHIKLMNPQRSTVWY`, plus per-fold logistic fitting through the existing
  `fit_logistic` and `logistic_scores` functions.
- Added pMHC retrieval routing through the existing
  `score_by_nearest_positive` operator with `Allele` as the database key and
  `Peptide` as the compared sequence. The returned `KnnResult` preserves nearest
  peptide and database-size diagnostics. Both edit and cosine paths use the same
  operator call.
- Added one allele-specific PWM scorer. Each cell uses independent add-one
  smoothing over 20 residues for the positive and negative classes, and test
  scores sum the nine position log-odds values.
- Added raw BLOSUM50 encoding in fixed 20-residue order. Peptide plus 34-residue
  pseudo-sequence features have width 860; peptide plus a 47-allele one-hot block
  have width 227.
- Added deterministic 90/10 validation membership using a joint
  allele-plus-target stratification with `random_state=0` by default.
- Added the fixed MLP fold scorer using the existing `MlpHead`, hidden sizes 55
  and 66, seeds 0 through 4, dropout 0, train-partition-only scaling, Adam at
  `1e-3`, sigmoid probabilities, continuous-affinity MSE, batch size 512,
  maximum 200 epochs, patience 20, cloned best-validation weights, restored best
  weights, and a mean over all ten predictions.

Context7 was consulted before using the external APIs. The checked contracts were
scikit-learn `StratifiedShuffleSplit`/`StandardScaler`, PyTorch Adam, MSE,
generator-backed `randperm`, state-dict restoration, and Biopython substitution
matrix loading/indexing.

## Named selector red/green evidence

Every red run used a wrong-but-callable production implementation. Exit 1 and the
specific numerical or spy mismatch were observed before the implementation was
corrected.

| Selector | Red observation | Green observation |
|---|---|---|
| `random_is_seeded_uniform` | `uv run pytest tests/test_pmhc.py -q -k random_is_seeded_uniform` exited 1; zero scores mismatched all three fixed RNG values. | Same selector exited 0: `1 passed, 32 deselected`. |
| `composition_columns_are_fixed` | Selector exited 1; 10 of 21 zero columns mismatched the literal length/count vector. | Same selector exited 0: `1 passed, 33 deselected`. |
| `retrieval_is_per_allele` | Selector exited 1; wrong peptide-keyed routing scored `0.699999988079071` instead of `0.0`. | Combined corrected retrieval run exited 0: `2 passed, 34 deselected`. |
| `retrieval_operator_is_shared` | Selector exited 1; the spy observed `Peptide`/`Allele` reversed and an unwanted `similarity_fn=None`. | Combined corrected retrieval run exited 0: `2 passed, 34 deselected`. |
| `pwm_matches_hand_calculation` | Selector exited 1; callable zero scorer returned `0.0` instead of the hand-derived `3.518187904711225`. | Same selector exited 0: `1 passed, 36 deselected`. |
| `blosum50_values_are_raw` | Selector exited 1; the zero encoder mismatched 17 of 20 literal BLOSUM50 A-row values. | Same selector exited 0: `1 passed, 37 deselected`. |
| `validation_membership_is_shared` | Selector exited 1; the unstratified tail split placed all four validation rows in `("HLA-B07:02", True)`. | Same selector exited 0: `1 passed, 38 deselected`. |
| `scaler_uses_fit_partition_only` | Selector exited 1; the scaler spy observed shape `(4, 181)` instead of fit-only `(3, 181)`. | Combined corrected MLP run exited 0: `4 passed, 39 deselected`. |
| `mlp_uses_adam_mse` | Selector exited 1; spies observed `{"adam": 0, "mse": 0}` instead of one call each. | Combined corrected MLP run exited 0: `4 passed, 39 deselected`. |
| `mlp_restores_best_weights` | Selector exited 1; the model spy observed no restored weights instead of the best weight `1.0`. | Combined corrected MLP run exited 0: `4 passed, 39 deselected`; the restored prediction matched `sigmoid(1.0)`. |
| `ensemble_uses_all_ten_models` | Selector exited 1; the spy observed only `(55, 0)` instead of all ten hidden-size/seed pairs. | Combined corrected MLP run exited 0: `4 passed, 39 deselected`; the literal ensemble mean was `62.5`. |

The selector evidence is failure-proven: each named assertion was observed failing
for the intended wrong behavior and passing after the corresponding correction.

## Final verification

Fresh command after the final hygiene edits:

```text
uv run pytest tests/test_pmhc.py tests/test_baseline_knn.py tests/test_knn_cosine.py tests/test_embed.py tests/test_features.py -q
```

Observed exit code 0:

```text
........................................................................ [ 86%]
...........                                                              [100%]
83 passed in 6.90s
```

This final five-file regression result is observed. It was not independently
verified in this implementation context.
