"""A3 -- submit the two Phase 1 models to the IMMREP23 Kaggle scorer.

The competition closed on 2023-12-11, so nothing here enters a leaderboard. The intent was
that Kaggle's scorer is an *independent* implementation of Macro AUC0.1: a matching score
would verify this project's metric code from a context that did not write it.

That verification could not be obtained. As of 2026-09-07 every submission to this
competition fails with "Scoring session had non-zero exit code", including Kaggle's own
unmodified sample_submission.csv, which the competition README states must score 0.5. The
scoring container is broken on Kaggle's side. This script is kept because it is correct and
would work if the scorer is ever restored.

Credentials come from .env (KAGGLE_API_KEY). Nothing is written to ~/.kaggle.

Note on auth: a ``KGAT_``-prefixed value is one of Kaggle's newer API tokens and is a
**bearer** credential, which the SDK reads from ``KAGGLE_API_TOKEN``. The legacy
username+key basic-auth path returns 401 for these tokens, so the username is not used.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "tcr-specificity-prediction-challenge"
SUBMISSIONS = [
    ("knn_v1_edit_distance.csv", "k-NN baseline, normalised edit distance on CDR3b", 0.5922),
    ("esm2_logistic_matched.csv", "ESM-2 35M layer 10 + logistic head, matched negatives", 0.5474),
]


def authenticate(username: str) -> object:
    values = dotenv_values(ROOT / ".env")
    key = (values.get("KAGGLE_API_KEY") or "").strip()
    if not key:
        sys.exit("KAGGLE_API_KEY not found in .env")

    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key
    else:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", "mrfrooty"))
    parser.add_argument("--wait", type=int, default=90, help="seconds to poll for scores")
    args = parser.parse_args()

    api = authenticate(args.username)
    print("authenticated\n")

    for filename, message, predicted in SUBMISSIONS:
        path = ROOT / "data" / "submissions" / filename
        if not path.exists():
            sys.exit(f"missing {path} -- run scripts/make_kaggle_submissions.py first")
        print(f"submitting {filename}  (internal Private-split prediction: {predicted:.4f})")
        try:
            api.competition_submit(str(path), message, COMPETITION)
        except Exception as error:  # pylint: disable=broad-exception-caught
            print(f"  FAILED: {type(error).__name__}: {error}")
            if "403" in str(error) or "rules" in str(error).lower():
                print(
                    "  -> the competition rules must be accepted once, in a browser, at\n"
                    f"     https://www.kaggle.com/competitions/{COMPETITION}/rules"
                )
            continue
        print("  accepted, queued for scoring")

    print(f"\npolling for scores (up to {args.wait}s)...")
    deadline = time.time() + args.wait
    while time.time() < deadline:
        submissions = list(api.competition_submissions(COMPETITION))
        pending = [s for s in submissions[: len(SUBMISSIONS)] if "PENDING" in str(s.status)]
        if not pending:
            break
        time.sleep(10)

    print(f"\n{'file':<32} {'public':>10} {'private':>10} {'status':>24}")
    for submission in list(api.competition_submissions(COMPETITION))[:6]:
        print(
            f"{str(submission.file_name):<32} "
            f"{str(submission.public_score):>10} "
            f"{str(submission.private_score):>10} "
            f"{str(submission.status):>24}"
        )


main()
