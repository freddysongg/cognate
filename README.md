# Cognate

Cognate compares simple binding predictors on two related tasks:

- **TCR–epitope binding:** peptide and TCR CDR3β sequences from IMMREP23 and VDJdb. The evaluation separates peptides seen during training from unseen peptides.
- **Peptide–MHC class I binding:** nine-residue peptides and HLA-A/B/C alleles from the public NetMHCpan training archive. The studies include binding replication and three leave-one-allele-out transfer designs.

The tasks use shared metrics and retrieval code, but have different data and evaluation contracts. This repository records baseline experiments, not a new binding model.

## Setup and checks

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
./scripts/fetch_data.sh
bash scripts/fetch_pmhc_data.sh
uv run pytest
```

The two fetch scripts download the public source data used by each line. The pMHC fetch verifies SHA-256 hashes recorded in the [source contract](data/pmhc/source_contract.json).

## Reproduce

For the original IMMREP23 TCR–epitope results:

```bash
uv run python scripts/build_embeddings.py
uv run python scripts/build_t6a_data.py
uv run python scripts/run_all.py
```

For the later VDJdb evaluation set, see its [construction and evaluation details](docs/eval_set_construction.md).

For the pMHC binding replication:

```bash
uv run python scripts/run_pmhc.py
```

The pMHC transfer runs have separate commands and data requirements in their study documents. Some runs take many hours.

## Results and methods

- [TCR–epitope findings](docs/findings.md) and [evaluation set construction](docs/eval_set_construction.md)
- [pMHC binding replication](docs/pmhc/experiment.md)
- pMHC transfer: [allele holdout](docs/pmhc/loao_allele_only.md), [allele and peptide holdout](docs/pmhc/loao_allele_peptide.md), [allele cluster holdout](docs/pmhc/loao_cluster.md)
- [pMHC novelty control and limitations](docs/pmhc/novelty_control.md)

Read the linked studies for result tables, uncertainty intervals, and limits of each comparison.
