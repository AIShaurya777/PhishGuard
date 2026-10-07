"""Loading and cleaning the PhiUSIIL phishing URL dataset.

Every notebook should get its data through `load_clean()` so the whole team
trains and evaluates on exactly the same rows. Reasons for each cleaning
step are recorded in docs/DECISIONS.md.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "DataSets" / "PhiUSIIL_Phishing_URL_Dataset.csv.zip"

# Published size of PhiUSIIL (Prasad & Chandra, 2024). Checked on every load
# because an earlier Colab upload silently truncated the file to 196,403 rows.
EXPECTED_SHAPE = (235_795, 55)

TARGET = "is_phishing"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV (zipped or not) and fail loudly if it looks incomplete."""
    df = pd.read_csv(path)

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected shape {EXPECTED_SHAPE}, got {df.shape}. "
            "The file is probably truncated or the wrong version."
        )
    if df.isna().any().any():
        raise ValueError("Raw data contains missing values; the file may be corrupted.")
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"Unexpected label values: {sorted(df['label'].unique())}")

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Replace `label` (1 = legitimate) with `is_phishing` (1 = phishing).

    Precision, recall and F1 are computed for the positive class, and the
    class we care about catching is phishing, so phishing must be 1.
    Dropping `label` stops anyone from accidentally using the old encoding.
    """
    df = df.copy()
    df[TARGET] = (df["label"] == 0).astype(int)
    return df.drop(columns="label")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise URLs and remove duplicates and label conflicts.

    Returns the cleaned frame and a report of how many rows each step removed.
    """
    report = {"rows_in": len(df)}
    df = df.copy()

    # A trailing "/" appears on ~36% of phishing URLs and on no legitimate
    # URL, so it is a collection artifact that leaks the label.
    df["URL"] = df["URL"].str.strip().str.rstrip("/")

    # The same URL with different labels is label noise; we cannot tell which
    # label is right, so every copy is dropped.
    labels_per_url = df.groupby("URL")[TARGET].transform("nunique")
    conflicting = labels_per_url > 1
    report["conflicting_label_rows_dropped"] = int(conflicting.sum())
    df = df[~conflicting]

    # Repeated URLs would let the same example sit in both train and test.
    before = len(df)
    df = df.drop_duplicates(subset="URL", keep="first")
    report["duplicate_rows_dropped"] = before - len(df)

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    """The one entry point notebooks should use."""
    df, _ = clean(add_target(load_raw(path)))
    return df
