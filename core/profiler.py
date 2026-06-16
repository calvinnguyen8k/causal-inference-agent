"""Dataset profiler — analyses shape, types, normality, and missingness."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    numeric_cols: List[str]
    categorical_cols: List[str]
    missing_pct: float          # 0.0–100.0
    is_gaussian: bool           # True if majority of columns pass Shapiro-Wilk (p>0.05)
    gaussian_ratio: float       # Fraction of numeric cols that are Gaussian
    has_time_index: bool        # True if a datetime column is detected
    max_abs_correlation: float  # Max pairwise Pearson |r| among numeric cols


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Return a DataProfile for df."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_pct = df.isnull().mean().mean() * 100

    # Detect datetime columns
    has_time_index = any(
        pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns
    ) or isinstance(df.index, pd.DatetimeIndex)

    # Normality: Shapiro-Wilk on up to 200 samples per column (Shapiro limit)
    gaussian_count = 0
    for col in numeric_cols:
        sample = df[col].dropna().values
        if len(sample) < 3:
            continue
        sample = sample[:200]  # Shapiro-Wilk limit
        _, p = stats.shapiro(sample)
        if p > 0.05:
            gaussian_count += 1

    gaussian_ratio = gaussian_count / len(numeric_cols) if numeric_cols else 0.0
    is_gaussian = gaussian_ratio >= 0.5

    # Max pairwise correlation (absolute value)
    max_corr = 0.0
    if len(numeric_cols) >= 2:
        corr_arr = df[numeric_cols].corr().abs().values.copy()
        np.fill_diagonal(corr_arr, 0)
        max_corr = float(corr_arr.max())

    return DataProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        missing_pct=missing_pct,
        is_gaussian=is_gaussian,
        gaussian_ratio=gaussian_ratio,
        has_time_index=has_time_index,
        max_abs_correlation=max_corr,
    )
