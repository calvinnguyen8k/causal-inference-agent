"""Bootstrap confidence — edge frequency across N bootstrap resamples."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.runner import run_algorithm


@dataclass
class BootstrapResult:
    edge_frequency: np.ndarray   # Shape (n_vars, n_vars); fraction of runs where edge appeared
    column_names: list[str]
    n_resamples: int
    algorithm_key: str


def run_bootstrap(
    df: pd.DataFrame,
    algorithm_key: str,
    n_resamples: int = 100,
    alpha: float = 0.05,
) -> BootstrapResult:
    """
    Run algorithm on n_resamples bootstrap samples of df.
    Returns edge frequency matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    n_rows, n_vars = numeric_df.shape
    col_names = numeric_df.columns.tolist()

    freq = np.zeros((n_vars, n_vars))
    success_count = 0

    for _ in range(n_resamples):
        sample = numeric_df.sample(n=n_rows, replace=True)
        try:
            result = run_algorithm(sample, algorithm_key, alpha=alpha)
            adj = result.adjacency_matrix
            # Binarise: any non-zero entry counts as edge present
            freq += (np.abs(adj) > 1e-8).astype(float)
            success_count += 1
        except Exception:
            # Skip failed resamples silently; caller sees n_resamples vs success_count
            pass

    if success_count > 0:
        freq /= success_count

    return BootstrapResult(
        edge_frequency=freq,
        column_names=col_names,
        n_resamples=success_count,
        algorithm_key=algorithm_key,
    )
