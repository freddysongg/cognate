# Phase 1 repository scout

## Scope

- [observed] Inspected `.gitignore`, `scripts/fetch_data.sh`, `pyproject.toml`, the `docs/pmhc`, `data`, `scripts`, and `tests` path inventories, and relevant portions of `docs/pmhc/brief.md`.
- [observed] Did not read or modify `docs/pmhc/src/cognate/data.py`; it was explicitly excluded from file searches.
- [observed] This note is the only file created by this scout.

## Repository state and placement

- [observed] `docs/pmhc/` currently contains only `brief.md` in the directory inventory, and the whole subtree is untracked in the current working tree (`git status --short` reports `?? docs/pmhc/`).
- [observed] `data/pmhc/` does not currently exist in the directory inventory.
- [observed] `scripts/` is a flat directory: dataset builders use `build_*.py`, experiment/report entry points use `run_*.py`, and the single download entry point is `fetch_data.sh`.
- [observed] Python scripts are tracked as mode `100644`; `scripts/fetch_data.sh` is tracked as executable mode `100755`.
- [observed] Tests are flat under `tests/` and named after the unit or workflow they cover, including `tests/test_vdjdb_eval.py` for the existing dataset-construction path.

## Documentation conventions

- [observed] `docs/pmhc/brief.md:419-437` reserves `docs/pmhc/` for new pMHC work and specifies mirrored filenames: `brief.md`, `literature_gate.md`, `eval_set_construction.md`, `belief_list.md`, `findings.md`, `session_log.md`, and `postmortem.md`.
- [observed] `docs/pmhc/brief.md:424-436` keeps the cross-project contrast at `docs/contrast.md`, the new frozen-contract ADR at `docs/ADR/0002-*.md`, and says the existing top-level `docs/*.md` TCR records are read-only.
- [observed] `docs/pmhc/brief.md:231-248` requires a frozen evaluation set, an attribution file, and a SHA-256 enforced by a test; the evaluation-set construction record belongs at `docs/pmhc/eval_set_construction.md`.
- [observed] `docs/pmhc/brief.md:125-142` says NetMHCpan artifacts must be treated as non-redistributable until the shipped licence is read and requires the IEDB export's actual row count, size, and schema to be measured locally before those values enter documentation.

## Data and ignore conventions

- [observed] `.gitignore:14-17` ignores direct `data/*.npz`, direct `data/*.npy`, and `data/embeddings/` as regenerable caches.
- [observed] `.gitignore:19-22` ignores existing raw downloads by exact path: `data/IMMREP23-main/`, `data/vdjdb-2026-06-03/`, and `data/immrep23.zip`.
- [observed] `.gitignore:24-25` separately ignores regenerable submission output under `data/submissions/`.
- [observed] Existing derived CSV/JSON outputs and `data/frozen_contract_hashes.json` are tracked, as is the source attribution record `data/VDJDB_ATTRIBUTION.md`.
- [observed] `git check-ignore --no-index` matched `data/example.npy` but did not match `data/pmhc/example.npy`, `data/pmhc/netmhcpan/example.txt`, or any other tested `data/pmhc/**` path. The existing direct-file cache patterns do not protect nested pMHC artifacts.
- [claimed] Before a pMHC fetch writes raw NetMHCpan or IEDB source material under `data/pmhc/`, exact raw/archive paths should be added to `.gitignore`; a blanket `data/pmhc/` rule would also hide any intended tracked attribution, frozen derivative, or hash artifact.

## Fetch-script conventions

- [observed] `scripts/fetch_data.sh:1-4` uses `#!/usr/bin/env bash`, `set -euo pipefail`, and resolves `data/` relative to the script with `cd "$(dirname "$0")/../data"`.
- [observed] Each existing source is a separate commented section. Downloads use `curl -sL`, extraction uses `unzip -oq`, archives are removed after extraction, and the resulting directory is listed (`scripts/fetch_data.sh:5-20`).
- [observed] IMMREP23 downloads from a moving `main` branch archive, while VDJdb pins `VDJDB_RELEASE=2026-06-03` and interpolates that release into both URL and paths (`scripts/fetch_data.sh:5`, `scripts/fetch_data.sh:10-20`).
- [observed] The script assumes the repository's `data/` directory already exists; it does not create `data/` or subdirectories.
- [observed] The script does not record or verify checksums, and `curl -sL` does not request HTTP failure handling with `-f`.
- [claimed] A Phase 1 addition should preserve the script-relative path convention and source-separated sections. Pinning a source version or recording a downloaded checksum is preferable where the provider exposes a stable release; measured local metadata belongs in the pMHC construction record, not as an unverified literal in the brief.

## Python and script integration conventions

- [observed] `pyproject.toml:1-17` defines package `cognate`, requires Python `>=3.11,<3.12`, and already supplies pandas, NumPy, scikit-learn, Biopython, PyTorch, and Transformers.
- [observed] `pyproject.toml:19-26` has only `kaggle` and `pytest` in the dev group and points pytest at `tests/`; no lint, type-check, or shell-check tool is configured there.
- [observed] `pyproject.toml:28-33` packages only `src/cognate` through Hatchling.
- [observed] Existing Python entry points import reusable logic from `cognate`, derive the repository root with `Path(__file__).resolve().parents[1]`, place materialized outputs under top-level `data/`, expose `main() -> None`, and use an `if __name__ == "__main__"` guard.
- [observed] Existing documentation invokes Python scripts as `uv run python scripts/<name>.py`; `scripts/fetch_data.sh` is invoked directly or through `bash`.
- [observed] Existing build scripts use `argparse` when parameters are user-selectable and define stable defaults/constants at module scope.
- [claimed] A new pMHC construction script fits the current layout as a flat `scripts/build_*.py` entry point with reusable parsing/normalization in the packaged `src/cognate` namespace, outputs under `data/pmhc/`, and a focused `tests/test_*.py` workflow test.

## Phase 1 deltas to route

- **Ignore policy:** [observed] no `data/pmhc/**` ignore protection exists; [claimed] add narrow rules before downloading non-redistributable or raw material.
- **Fetcher:** [observed] the existing executable fetcher is the repository's sole source-download entry point; [claimed] extend it in source-separated sections unless Phase 1 explicitly requires an independent credentialed/manual fetch path.
- **Dependencies:** [observed] shell downloads need no Python dependency change; pandas/Biopython already cover common tabular and sequence parsing. Any new package would need a demonstrated parser or format requirement.
- **Docs:** [observed] new pMHC records belong under `docs/pmhc/`; existing top-level TCR documents are closed, apart from the narrowly specified forward pointer in the brief.
- **Frozen artifacts:** [observed] the repository already tracks derived evaluation CSV/JSON files, attribution, and a frozen-contract hash manifest while ignoring raw sources and caches; [claimed] the pMHC path should preserve that raw-versus-derived boundary rather than ignoring the entire subtree.
