# Phase 2 fix verification

- Composition mutation: `score_composition_fold` fitted concatenated train/test features. `uv run pytest tests/test_pmhc.py::test_composition_fold_fits_train_target_and_scores_test_only -q` failed as intended with feature shapes `(5, 21)` versus `(3, 21)`.
- Adam mutation: `_score_single_mlp` used `lr=1e-2`. `uv run pytest tests/test_pmhc.py::test_mlp_uses_adam_and_sigmoid_mse_with_affinity_targets -q` failed as intended with `[0.01] != [0.001]`.
- Sigmoid mutation: `_score_single_mlp` passed raw logits to MSE. The same focused test failed as intended with predictions `0.0` versus sigmoid outputs `0.5`.
- Target mutation: `score_mlp_fold` used binary `Target` instead of continuous `Affinity`. The same focused test failed as intended with `[0.0, 1.0, 1.0]` versus `[0.1, 0.2, 0.3]`.
- Restored focused run: `uv run pytest tests/test_pmhc.py::test_composition_fold_fits_train_target_and_scores_test_only tests/test_pmhc.py::test_mlp_uses_adam_and_sigmoid_mse_with_affinity_targets -q` passed: `2 passed in 3.53s`.
- Required regression run: `uv run pytest tests/test_pmhc.py tests/test_baseline_knn.py tests/test_knn_cosine.py tests/test_embed.py tests/test_features.py -q` passed: `84 passed in 20.58s`.
- Selector red: `uv run pytest tests/test_pmhc.py -q -k mlp_uses_adam_mse` exited `5` with `44 deselected` before the rename.
- Selector green: the same command passed with `1 passed, 43 deselected in 4.18s` after renaming the test to `test_mlp_uses_adam_mse_with_sigmoid_and_affinity_targets`.
- Post-rename regression: `uv run pytest tests/test_pmhc.py tests/test_baseline_knn.py tests/test_knn_cosine.py tests/test_embed.py tests/test_features.py -q` passed: `84 passed in 20.47s`.
