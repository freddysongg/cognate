"""The fork drivers restate other runs' headline numbers in a ``reference`` block.

Those constants are hand-typed literals, not values read from the artifacts they name, so
drift between the two is silent by construction. `frozen_esm_cosine_seen_layer10` drifted
exactly that way: both fork drivers carried 0.5364 while `data/knn_esm_cosine.json` had
0.5358 at every commit it has existed, and nothing downstream compared the two. Three of the
four constants in the same block were correct, which is what made the fourth invisible.

The assertion runs against the *source* of each driver rather than its committed output.
`data/fork1_results.json` and `data/fork2_results.json` still hold the superseded value and
are corrected by erratum (docs/findings.md S10) rather than by hand-editing an artifact that
would then be the output of no script. What has to be prevented is the next run writing a
wrong constant, and that is decided by the literal in the driver.

Each check is paired with a positive control on deliberately corrupted input, so a green run
means the check ran and could have gone red.
"""

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"

KNN_ESM_COSINE = DATA / "knn_esm_cosine.json"
B3_RESULTS = DATA / "b3_results.json"
PROBES = DATA / "vdjdb_probes.json"

REFERENCE_KEY = "reference"
SUPERSEDED_ESM_COSINE = 0.5364


def _macro_point(path: Path, slice_name: str) -> float:
    return json.loads(path.read_text())["scores"][slice_name]["macro_auc01"]["point"]


def _sources() -> dict[str, float]:
    """Each reference constant mapped to the artifact value it restates."""
    return {
        "edit_knn_seen": _macro_point(KNN_ESM_COSINE, "edit/seen"),
        "frozen_esm_cosine_seen_layer10": _macro_point(
            KNN_ESM_COSINE, "35M layer10 (headline)/seen"
        ),
        "logistic_head_seen": _macro_point(B3_RESULTS, "logistic/seen"),
        "random_seen": _macro_point(B3_RESULTS, "random/seen"),
        "inherited_train_gap": json.loads(PROBES.read_text())["full_model_train_gap"],
    }


def parse_reference_block(source: str) -> dict[str, float]:
    """The ``reference`` dict literal from a driver, without executing it.

    Reading the file rather than importing it keeps the check independent of whether the
    driver's imports resolve or its caches exist.
    """
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == REFERENCE_KEY
                and isinstance(value, ast.Dict)
            ):
                return {
                    entry.value: literal.value
                    for entry, literal in zip(value.keys, value.values)
                    if isinstance(entry, ast.Constant)
                    and isinstance(literal, ast.Constant)
                }
    raise AssertionError(f"no {REFERENCE_KEY!r} dict literal found")


def mismatches(block: dict[str, float], sources: dict[str, float]) -> dict[str, tuple]:
    """Constants whose value disagrees with the artifact they name."""
    return {
        name: (value, sources[name])
        for name, value in block.items()
        if name in sources and value != sources[name]
    }


DRIVERS = ["run_fork1.py", "run_fork2.py"]

missing_artifacts = [
    p.name for p in (KNN_ESM_COSINE, B3_RESULTS, PROBES) if not p.exists()
]
pytestmark = pytest.mark.skipif(
    bool(missing_artifacts), reason=f"missing artifacts: {missing_artifacts}"
)


@pytest.mark.parametrize("driver", DRIVERS)
def test_every_reference_constant_matches_its_source(driver: str) -> None:
    block = parse_reference_block((SCRIPTS / driver).read_text())
    sources = _sources()

    unknown = set(block) - set(sources)
    assert not unknown, (
        f"{driver} restates {sorted(unknown)} with no artifact to check them against; "
        "add the source to _sources() or stop restating the value"
    )
    assert not mismatches(block, sources), (
        f"{driver} reference constants disagree with their artifacts: "
        f"{mismatches(block, sources)}"
    )


@pytest.mark.parametrize("driver", DRIVERS)
def test_driver_carries_the_esm_cosine_constant(driver: str) -> None:
    """Guards the check above from passing vacuously if the block is ever emptied."""
    block = parse_reference_block((SCRIPTS / driver).read_text())
    assert "frozen_esm_cosine_seen_layer10" in block


def test_check_fires_on_the_superseded_value() -> None:
    """Positive control: the exact literal that drifted must be caught."""
    sources = _sources()
    corrupted = sources | {"frozen_esm_cosine_seen_layer10": SUPERSEDED_ESM_COSINE}
    found = mismatches(corrupted, sources)
    assert "frozen_esm_cosine_seen_layer10" in found
    assert found["frozen_esm_cosine_seen_layer10"] == (
        SUPERSEDED_ESM_COSINE,
        sources["frozen_esm_cosine_seen_layer10"],
    )


def test_parse_reference_block_rejects_a_file_without_one() -> None:
    with pytest.raises(AssertionError):
        parse_reference_block("x = {'config': {'layer': 10}}\n")


FORK_ARTIFACTS = ["fork1_results.json", "fork2_results.json"]


@pytest.mark.parametrize("artifact", FORK_ARTIFACTS)
def test_committed_fork_artifacts_still_carry_the_erratum(artifact: str) -> None:
    """The stale value is pinned, not fixed, and `data/errata.json` says so permanently.

    Regenerating these files means retraining; hand-editing one makes it the output of no
    script. So the divergence is left in place and nailed down here instead, which stops it
    widening silently and stops a future reader mistaking 0.5364 for a live number.
    """
    recorded = json.loads((DATA / "errata.json").read_text())["errata"][0]
    value = json.loads((DATA / artifact).read_text())[REFERENCE_KEY][
        "frozen_esm_cosine_seen_layer10"
    ]
    assert value == recorded["value_in_artifact"] == SUPERSEDED_ESM_COSINE
    assert recorded["authoritative_value"] == _sources()["frozen_esm_cosine_seen_layer10"]
