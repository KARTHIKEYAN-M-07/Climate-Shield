"""
src/data/split.py
─────────────────
Chronological train / validation / test splitter:
  - 70% Train, 15% Validation, 15% Test
  - STRICTLY chronological (no random shuffling).
  - Train -> Validation -> Test in time order.
  - Imputers and scalers must be fit strictly on Train only.
"""

from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into chronological train, validation, and test subsets.

    Parameters
    ----------
    df:          Feature matrix sorted by date.
    train_frac:  Proportion for training set (default 0.70).
    val_frac:    Proportion for validation set (default 0.15).
    test_frac:   Proportion for test set (default 0.15).

    Returns
    -------
    (train_df, val_df, test_df)
    """
    total = len(df)
    if total < 10:
        raise ValueError(f"Dataset too small ({total} rows) for meaningful 70/15/15 chronological split.")

    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    # Give remaining rows to test to ensure sum equals total
    n_test = total - n_train - n_val

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    return train_df, val_df, test_df
