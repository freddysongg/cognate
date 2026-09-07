# VDJdb attribution

`data/vdjdb_eval.csv` is a filtered subset derived from VDJdb, and is therefore a derivative work.
This file records the attribution that derivation requires.

## Source

- **Database:** VDJdb — <https://github.com/antigenomics/vdjdb-db>
- **Release:** `2026-06-03`
- **File used:** `vdjdb.slim.txt`
- **Retrieved by:** `scripts/fetch_data.sh` (release URL pinned in the script)

## Licence

VDJdb is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. Both the
upstream repository `LICENSE` and the `LICENSE` bundled inside the release archive are AGPL-3.0.

AGPL-3.0 permits redistribution of modified and derived works, so shipping the filtered subset as
`data/vdjdb_eval.csv` is allowed. Attribution and the citation below remain required.

The full licence text ships with the release archive and is restored locally by
`scripts/fetch_data.sh` at `data/vdjdb-2026-06-03/LICENSE`. The raw dump itself is not committed
to this repository.

## Citation

> Goncharov M, Bagaev D, Shcherbinin D, *et al.* **VDJdb in the pandemic era: a compendium of T
> cell receptors specific for SARS-CoV-2.** *Nature Methods* (2022).
> [doi:10.1038/s41592-022-01578-0](https://doi.org/10.1038/s41592-022-01578-0)

## Notes

The interaction between AGPL-3.0 and the licensing of the rest of this repository is unsettled.
The repository is private pending that decision. See `docs/eval_set_construction.md` §1 for the
licence determination and how the subset was constructed.
