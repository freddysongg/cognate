# ADR 0001 — The frozen contract

**Status:** accepted 2026-09-07, verified 2026-09-09, archived with the project.
**Supersedes:** GitHub issue #20, which held this as a tracker item with no terminal state.

## Context

Phase D ran two forks in separate git worktrees off one shared commit (`d7a62754`) so their
results could be compared. That comparison is only valid if both forks scored against the same
metric, the same split, and the same evaluation set. A tracker issue is the wrong home for a rule
that never completes; this is.

## Decision

Four files are the **frozen contract**:

| file | why it is frozen |
|---|---|
| `src/cognate/metrics.py` | defines macro AUC0.1 and every bootstrap interval |
| `src/cognate/split.py` | defines the component split both forks trained against |
| `src/cognate/features.py` | defines the `peptide_only` / `tcr_only` shortcut probes |
| `data/vdjdb_eval.csv` | the n=88 evaluation set every reported number is computed on |

Any change to one of them **lands in `main` with a test, then merges into every worktree. Never
patch one side.** A change silently reinterprets every number in `docs/findings.md`, so it must be
a deliberate act that also updates `data/frozen_contract_hashes.json`.

## Verification

Recorded in `data/frozen_contract_hashes.json`, enforced by
`tests/test_frozen_contract_hashes.py`. sha256 of all four files is identical at the shared
foundation `d7a62754`, at both fork tips `8c7a03d` / `16a050c`, at `main` `d772293`, and in all
three worktrees as they sit on disk. **Verdict: PASS.**

One limit, stated rather than glossed. The embedding caches under `COGNATE_CACHE_DIR` are not in
git. All three worktrees resolve to the one shared directory
(`/Users/freddy/Documents/repos/cognate-shared`), verified by running
`cognate.embed.default_cache_dir()` in each, so cache identity across forks holds by shared path
rather than by independent copies agreeing. Their hashes are recorded as of 2026-09-09 and attest
to current contents only — **no hash was taken at run time, so nothing retroactively proves the
bytes each fork read in September 2026.** `scripts/verify_residue_cache.py` is the only content
check that predates the record.

## Consequences

- The Phase D cross-fork comparison in `docs/findings.md` §7a is sound on this axis. `observed`.
- The rule is dormant: the project is closed and both worktrees are merged. It binds anyone who
  revives the repository, which is why it is written down instead of closed.
