# Red-team audit — Phase 1

Read-only. No training runs, no new arms, no config changes. The one execution was
re-verifying already-frozen outputs.

Classification: **(A)** invalidates a reported number · **(B)** weakens an interpretation ·
**(C)** cosmetic.

**Status: section 0 stopped at a class-(A) finding, since corrected (findings.md S10).
Section 4 was unblocked and has run; it is recorded below. Sections 1-3 were never run.**

---

## 0. Shared contract sweep

### 0a. The table

All nine arms, as read from source. `35M` = `facebook/esm2_t12_35M_UR50D`; every arm that
touches ESM-2 uses that checkpoint at layer 10, via the shared `EmbeddingCache`
(`src/cognate/embed.py`) or the residue cache verified against it.

| arm | model / ckpt | layer | pooling | normalisation | tokenisation | CDR3 convention | scope | negatives |
|---|---|---|---|---|---|---|---|---|
| `edit_retrieval` | — | — | — | — | — | `to_immrep_cdr3` | per-peptide | none |
| `esm_retrieval` | 35M | 10 | mean, BOS/EOS dropped | **L2 after pooling** | ESM-2 `<cls>…<eos>` | `to_immrep_cdr3` | per-peptide | none |
| `1a` metric | 35M | 10 | mean, BOS/EOS dropped | **L2** (`F.normalize`, projection out) | same | same | per-peptide | **none** |
| `1b` two-tower | 35M | 10 | mean, BOS/EOS dropped | **L2** (both towers) | same | same | global | **none** (in-batch) |
| `2a` cross-attn | 35M | 10 | masked mean over residues | **none** | same | same | global | `matched` ×5 |
| `2b` mean-pool | 35M | 10 | masked mean over residues | **none** | same | same | global | `matched` ×5 |
| `logistic_global` | 35M | 10 | mean, BOS/EOS dropped | **StandardScaler** | same | same | global | `matched` ×5 |
| `svc_global` | 35M | 10 | mean, BOS/EOS dropped | **StandardScaler** | same | same | global | `matched` ×5 |
| `svc_per_peptide` | 35M | 10 | mean, BOS/EOS dropped | **StandardScaler** | same | same | per-peptide | background TCRs |

Model, layer, pooling, tokenisation and CDR3 convention **agree across every arm**. The
layer-6/layer-10 error was confined to a hand-carried number in the strategy handoff; no
committed script or artifact reads a layer other than 10 outside the deliberate
`run_knn_esm_cosine.py` sweep.

### 0b. Cells where compared arms disagree — (B)

- **Normalisation.** Retrieval arms L2-normalise; parametric arms standardise per dimension;
  fork 2 does neither. `svc_per_peptide − esm_retrieval` (+0.0067, spans zero) therefore
  carries a normalisation change as well as an operator change. It is not the clean operator
  contrast §8b presents it as, though the direction of the residual is unknown. **(B)**
- **Negative construction.** Four distinct schemes across nine arms: none (retrieval, 1a),
  in-batch contrastive (1b), `matched` ×5 (2a, 2b, both global heads), and background-TCR
  sampling (`svc_per_peptide`). Cross-arm deltas in §7a and §8b span this. **(B)**
- **Training-set scope.** `svc_per_peptide` fits on the full `load_train()` database, matching
  what retrieval queries; the global arms fit on the component split's train arm. Scope and
  training-set size are confounded in the +0.0346 scope term. Already recorded in §8f. **(B)**

### 0c. Eval leakage and sampler leakage — clean

- **IMMREP23 training TCRs removed:** `observed`. Zero of the eval set's CDR3β appear in
  `load_train()`; zero exact `(Peptide, CDR3b)` pairs shared. Holds for all arms, since the
  eval set is shared and frozen.
- **Sampler peptide-identity leak:** `_sample_matched` draws negative peptides proportional to
  positive frequency with cognates excluded, which is what removes the flat-marginal shortcut
  `shuffle` leaves (0.94 pooled AUROC). Residual is the documented train-side 0.5847 pooled /
  0.4995 macro. Training-side only; the eval set is built independently. No new finding.
- `COGNATE_CACHE_DIR` is the cache env var (`embed.py:36`). The strategy handoff says
  `TCRBENCH_CACHE_DIR`, which appears nowhere in the repository and is read by nothing. The
  handoff is not a repo artifact, so there is no file to edit; the correct name is recorded
  here and in findings.md S10 for whoever writes the next one. **(C)**

### 0d. **CLASS (A) — stop condition fired**

`data/fork1_results.json:18` and `data/fork2_results.json:18` both report

    "frozen_esm_cosine_seen_layer10": 0.5364

The authoritative value is **0.5358** (`data/knn_esm_cosine.json`, key
`35M layer10 (headline)/seen`), independently reproduced this session by
`data/operator_diagnostic.json` (`esm_retrieval/seen` = 0.5358). Two computations agree on
0.5358; nothing has ever computed 0.5364.

It is a **hand-typed literal**, not a read value — `scripts/run_fork1.py:336` and
`scripts/run_fork2.py:339`. `git log -S` shows it entered at `8c7a03d`/`16a050c` and was never
the value of any artifact. `data/knn_esm_cosine.json` has held 0.5358 at every commit it has
existed.

The other three hardcoded constants in the same two `reference` blocks — `edit_knn_seen`
0.5654, `logistic_head_seen` 0.5107, `random_seen` 0.5009 — all match their sources exactly.
One of four drifted.

Magnitude is 0.0006, well inside the CI [0.5230, 0.5498], and no delta, interval, or conclusion
is computed from it — the fork comparisons use score vectors. So nothing downstream moves. It
is class (A) because the number as reported is wrong, and because it is the same failure mode
as the layer-6/layer-10 error: a reference value carried by hand instead of read from its
artifact, where drift is silent by construction.

**Stopped here at the time. The finding is corrected in findings.md S10; section 4 was
unblocked and run afterwards.**


---

## 4. SCEPTR source check

Read from the paper (arXiv:2406.06397 / *Cell Systems* 16(1):101165), not from project notes.

### 4a. Was the Fig. S6 linear SVC fit per-peptide or globally?

**Per-peptide.** Methods §III.4, verbatim:

> "To train the linear SVCs on top of PLM features, sampled 1000 random background TCRs from
> the training partition of the unlabelled Tanno et al. dataset. **For each PLM-pMHC-split
> combination, we trained a linear SVC** using the PLM embeddings of the *k* reference TCRs as
> the positives and those of the 1000 background TCRs as the negatives. The same 1000
> background TCRs were used across model-pMHC-split combinations to ensure consistency. We
> accounted for the imbalance between the number of positive and negative samples used during
> SVC fitting by weighting the penalty contributions accordingly."

One SVC per pMHC. §8b's premise — that SCEPTR's parametric arm was per-peptide and ours was
global — is correct as stated.

### 4b. Could each recorded deviation flip the operator sign?

`svc_per_peptide − esm_retrieval` = +0.0067 [−0.0006, +0.0142], p = 0.076. The interval already
touches zero, so "flip" is a low bar.

| # | deviation | flip? |
|---|---|---|
| 1 | 6 pMHCs / AUROC / k=1–200 → 48 peptides / macro AUC0.1 / full databases | **Yes.** SCEPTR scope their SVC advantage to the low-data regime explicitly. Ours is out of that regime. See §8f. |
| 2 | paired αβ full chains → CDR3β only | **Plausibly.** Both arms lose signal; a fitted classifier and a max-similarity rule need not degrade equally. Direction unknown. |
| 3 | ESM-2 T6 8M last layer → 35M layer 10 | **Unlikely.** `observed` on retrieval: 8M-last 0.5362 vs 35M-L10 0.5358. `claimed` on the SVC side — not run at 8M. |
| 4 | unlabelled background repertoire → background drawn from other peptides' binders | **Plausibly**, but biased *against* the flip: known binders are harder negatives, which should depress the SVC arm. |
| 5 | nearest-neighbour → max-over-database | **No.** Identical at `top_k=1`. |
| 6 | one shared background set → resampled per peptide | **Unlikely.** Adds per-peptide noise SCEPTR excluded; seed spread is 0.0010. Found by reading the source, absent from the original list. |
| 7 | normalisation: `svc_per_peptide` standardises, `esm_retrieval` L2-normalises | **Plausibly.** Internal to our comparison, not a SCEPTR deviation. Residual direction unknown. |

Deviations 6 and 7 were missing from `deviations_from_sceptr` and are added to the script
literal; the committed JSON is corrected by erratum (findings.md §10a).
