# Phase 3 completion evidence

## Real run and artifact

- The recovered real runner completed and wrote `data/pmhc/results.json` at 2026-09-11 08:56:59 PDT. The artifact is 64,994 bytes.
- Before scoring, the runner had printed and verified five BA folds, 47 alleles, 112,128 rows, 28,538 positives, 83,590 negatives, and zero cross-fold peptide overlap.
- The runner accepted the existing `shared/pmhc/esm2_35M_peptides.npz` only after verifying `facebook/esm2_t12_35M_UR50D`, availability of layer 10, finite layer values, and the exact 22,055-peptide set.
- A post-run assertion script observed exact equality between `results.json["contract"]` and `data/pmhc/source_contract.json`; exact eight top-level keys; seven score arms; 47 per-allele records; seven required paired comparisons; both fixed negative-source slices; 20,000 retained replicates for every saved interval; and no non-finite float anywhere in the JSON.
- Headline counts in the artifact are 47 alleles, 112,128 rows, 28,538 positives, 83,590 negatives, 82,448 measured non-binders, and 1,142 artificial negatives. Source counts match the frozen contract: 208,093 all-species BA rows, 170,107 HLA-A/B/C rows, and 126,375 nine-mer HLA rows.

## Observed results and decisions

- Headline macro-AUC0.1 points and 95% intervals from 20,000 draws: random 0.4998 [0.4978, 0.5023]; length/composition 0.5154 [0.5091, 0.5227]; edit retrieval 0.6088 [0.5964, 0.6231]; ESM-2 retrieval 0.5660 [0.5542, 0.5807]; PWM 0.7443 [0.7231, 0.7664]; pseudo-sequence MLP 0.8283 [0.8108, 0.8456]; one-hot MLP 0.8252 [0.8079, 0.8424].
- The mechanically derived criterion marks all six non-random arms predictive and random non-predictive.
- Pseudo-sequence MLP minus PWM is +0.0841 [+0.0688, +0.1008]. PWM is predictive but not sufficient relative to the MLP because the paired upper bound exceeds the 0.02 margin.
- On positives plus measured non-binders, edit retrieval is 0.6072 [0.5948, 0.6212] and ESM-2 retrieval is 0.5651 [0.5533, 0.5795]. Their descriptive uplifts over the slice's observed random point are +0.1074 and +0.0653, so the point ordering and above-random behavior survive removal of artificial negatives.
- On positives plus artificial negatives, edit and ESM-2 uplifts are larger at +0.1939 and +0.1231. Artificial negatives have mean/median nearest-positive edit distance 5.689/6 versus 5.272/5 for measured non-binders; both exact-duplicate counts are zero. Effect magnitude is construction-sensitive.
- The closed TCR artifact points are edit 0.5654, ESM-2 0.5358, and observed random 0.5009, giving +0.0645 and +0.0349 point uplifts. The pMHC headline uplifts are +0.1090 and +0.0663. Documents present these as within-task descriptive quantities without a fabricated cross-task interval.

## Documents

- Created `docs/pmhc/experiment.md` from the observed JSON with provenance, archive/file hashes, filters, fold/cohort counts, all seven settings and score rows, 20,000-draw intervals, paired comparisons, negative-source diagnostics, criteria, commands, and descriptive/non-causal limits.
- Appended observed point results to `docs/contrast.md` without changing the existing pre-result table or interpretation boundary.
- Created `docs/pmhc/postmortem.md` answering the five predeclared closeout questions and stopping the project without architecture or dataset expansion.

## Verification

- Final-state `uv run pytest -q` after review fixes: 251 passed, one existing expected degeneracy warning, exit 0, 106.54 seconds.
- `uv run python -m compileall -q src scripts`: exit 0, no output.
- `git diff --check`: exit 0, no findings.
- `git ls-files data/pmhc/raw data/pmhc/derived shared/pmhc`: exit 0, empty output; no raw rows, row-level derived files, or embedding cache files are tracked.
- Fresh final `reviewer-py`: 12 candidates, 10 refuted, 2 survived (0 blocker, 0 major, 2 minor). Both minors were resolved: the assigned lambda in `tests/test_pmhc.py` became a typed nested function, and documents now consistently call the seven outputs arms rather than rows. The focused retrieval test passed and `uv run ruff check tests/test_pmhc.py` reported `All checks passed!` after the code-style fix.
