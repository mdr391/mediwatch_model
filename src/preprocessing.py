"""Cleaning, feature engineering, and sklearn pipeline construction.

The preprocessor is NOT serialized separately — it lives inside the
full sklearn Pipeline alongside the model (see training.py).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    ALL_CAT_COLS,
    HIGH_CARDINALITY_COLS,
    ID_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)


# ── ICD-9 binning ──────────────────────────────────────────────
def _bin_icd9(val) -> str:
    if pd.isna(val):
        return "missing"
    s = str(val).strip()
    if s.startswith("V"):
        return "supplementary"
    if s.startswith("E"):
        return "external"
    try:
        code = float(s)
    except ValueError:
        return "other"

    if 390 <= code <= 459 or code == 785:
        return "circulatory"
    if 460 <= code <= 519 or code == 786:
        return "respiratory"
    if 520 <= code <= 579 or code == 787:
        return "digestive"
    if 250 <= code < 251:
        return "diabetes"
    if 800 <= code <= 999:
        return "injury"
    if 710 <= code <= 739:
        return "musculoskeletal"
    if 580 <= code <= 629 or code == 788:
        return "genitourinary"
    if 140 <= code <= 239:
        return "neoplasms"
    return "other"


# ── Cleaning / feature engineering ──────────────────────────────
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps. Call BEFORE fitting or transforming
    through the sklearn pipeline."""
    df = df.copy()

    # Replace '?' sentinel with NaN
    df.replace("?", np.nan, inplace=True)

    # Binary target
    if "readmitted" in df.columns:
        df[TARGET_COL] = (df["readmitted"] == "<30").astype(int)

    # Bin diagnosis codes
    for col in HIGH_CARDINALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(_bin_icd9)

    # Cast numeric-looking categoricals to str
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Drop IDs and original target
    drop = ID_COLS + ["readmitted"]
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

    return df


# ── sklearn ColumnTransformer ───────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer suitable for tree-based models.

    This object becomes the first step in the Pipeline serialized by
    training.py, so it does NOT need to be saved on its own.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
                ALL_CAT_COLS,
            ),
        ],
        remainder="drop",
    )


# ── Convenience split ──────────────────────────────────────────
def split_xy(df: pd.DataFrame):
    """Return (X DataFrame, y Series) after cleaning."""
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y
