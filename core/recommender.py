"""Algorithm recommender — rule-based mapping from DataProfile to best algorithm."""

from dataclasses import dataclass
from core.profiler import DataProfile


@dataclass
class Recommendation:
    algorithm_key: str   # Must match a key in ALGORITHM_REGISTRY (see runner.py)
    rationale: str
    warnings: list[str]  # Assumption violations the user should know about


def recommend(profile: DataProfile) -> Recommendation:
    """Return an algorithm recommendation and warnings based on the data profile."""
    warnings = []

    if profile.missing_pct > 5:
        warnings.append(
            f"Dataset has {profile.missing_pct:.1f}% missing values. "
            "Most causal discovery algorithms assume complete data. "
            "Consider imputing or dropping before running."
        )

    if not profile.numeric_cols:
        return Recommendation(
            algorithm_key="pc",
            rationale="No numeric columns detected. PC with chi-squared test works on discrete/categorical data.",
            warnings=warnings,
        )

    # Time-series data
    if profile.has_time_index:
        warnings.append("Time index detected. Consider Granger causality for temporal causal structure.")
        return Recommendation(
            algorithm_key="granger",
            rationale="Datetime index detected — Granger causality is the most appropriate method for temporal data.",
            warnings=warnings,
        )

    # Large datasets → scalable score-based methods
    if profile.n_rows > 5000:
        return Recommendation(
            algorithm_key="fges",
            rationale=(
                f"Large dataset ({profile.n_rows:,} rows). FGES (Fast GES) scales well to large samples "
                "and is robust under both Gaussian and non-Gaussian distributions."
            ),
            warnings=warnings,
        )

    # Small datasets
    if profile.n_rows < 100:
        warnings.append(
            f"Small sample size ({profile.n_rows} rows). Results will have high variance. "
            "Bootstrap confidence is especially important here."
        )

    # Non-Gaussian data → LiNGAM
    if not profile.is_gaussian:
        return Recommendation(
            algorithm_key="direct_lingam",
            rationale=(
                f"Only {profile.gaussian_ratio * 100:.0f}% of numeric columns are Gaussian (Shapiro-Wilk p>0.05). "
                "DirectLiNGAM exploits non-Gaussianity to identify a unique DAG without additional assumptions."
            ),
            warnings=warnings,
        )

    # Gaussian, moderate size → PC
    return Recommendation(
        algorithm_key="pc",
        rationale=(
            f"Gaussian data with {profile.n_rows} rows and {profile.n_cols} variables. "
            "PC algorithm with Fisher-Z conditional independence test is the standard choice."
        ),
        warnings=warnings,
    )
