"""Loaders for the IMMREP23 TCR-epitope specificity dataset."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "IMMREP23-main" / "data"

TRAIN_CSV = DATA_DIR / "VDJdb_paired_chain.csv"
TEST_CSV = DATA_DIR / "test.csv"
SOLUTIONS_CSV = DATA_DIR / "solutions.csv"

SEQUENCE_COLUMNS = ["Peptide", "CDR3a", "CDR3b", "CDR1a", "CDR2a", "CDR1b", "CDR2b"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"IMMREP23 file missing: {path}. Run scripts/fetch_data.sh")
    return pd.read_csv(path)


def load_train() -> pd.DataFrame:
    """VDJdb paired-chain training set. Positives only, `Target` column is all 1."""
    return _read(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    """Unlabelled test set, keyed by `ID`."""
    return _read(TEST_CSV)


def load_solutions() -> pd.DataFrame:
    """Test set plus `Label` (0/1) and `Usage` (Public/Private leaderboard split)."""
    return _read(SOLUTIONS_CSV)


def train_peptides() -> set[str]:
    return set(load_train()["Peptide"].unique())


def seen_mask(test_df: pd.DataFrame) -> pd.Series:
    """True where the row's peptide also occurs in the training set."""
    return test_df["Peptide"].isin(train_peptides())


def to_immrep_cdr3(junction: str) -> str:
    """Convert a VDJdb CDR3 to the IMMREP23 convention.

    VDJdb stores the full IMGT junction (`CASSFSGNTGELFF`); IMMREP23 stores the core with the
    flanking conserved residues removed (`ASSFSGNTGELF`). Comparing the two sources without this
    yields zero shared CDR3b, which reads as clean separation and is really a units error.
    """
    core = junction.strip().upper()
    if core.startswith("C"):
        core = core[1:]
    if core.endswith(("F", "W")):
        core = core[:-1]
    return core
