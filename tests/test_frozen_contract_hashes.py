"""The frozen contract's content hashes, pinned.

Issue #21 asked for hash verification that both fork worktrees consumed identical frozen
artifacts. It was never run until closeout; `data/frozen_contract_hashes.json` is that
verification and it passed at four commits and three worktrees. This keeps it true.

The point is not to forbid ever changing `metrics.py`. It is that a change to any of these
four files silently reinterprets every number in `findings.md`, so a change must be a
deliberate act that also updates the record -- which is exactly the procedure issue #20
specified and `docs/ADR/0001-frozen-contract.md` now carries.
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "data" / "frozen_contract_hashes.json"


def recorded() -> dict[str, str]:
    manifest = json.loads(RECORD.read_text())
    head = manifest["sha256_by_commit"]["d7722937e0a037e8f8994baf1a921dfa11f674b8"]["files"]
    return head


@pytest.mark.parametrize("relative_path", json.loads(RECORD.read_text())["frozen_contract"])
def test_frozen_artifact_matches_its_recorded_hash(relative_path: str) -> None:
    actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
    assert actual == recorded()[relative_path], (
        f"{relative_path} no longer matches the hash recorded in {RECORD.name}. Every "
        "macro AUC0.1 in docs/findings.md was computed against the recorded version. Update "
        "the record deliberately, per docs/ADR/0001-frozen-contract.md, or revert the file."
    )


def test_the_verification_itself_passed() -> None:
    assert json.loads(RECORD.read_text())["verdict"] == "PASS"
