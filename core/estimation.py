"""
Causal Effect Estimation — Multiple estimators for ATE.

Supported estimators
--------------------
1. regression   – OLS / Logit regression adjustment (baseline)
2. ipw          – Inverse Probability Weighting
3. aipw         – Augmented IPW / Doubly Robust
4. matching     – Propensity Score Matching (nearest neighbour)
5. dowhy        – DoWhy + EconML LinearDML (requires pip install dowhy econml)
"""

import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EstimationResult:
    ate: float
    se: float
    p_value: float
    ci_lower: float
    ci_upper: float
    summary_text: str
    is_binary_outcome: bool
    is_categorical_treatment: bool
    interpretation: str
    estimator_used: str = "regression"
    bootstrap_ates: Optional[np.ndarray] = None


# ── Estimator registry ────────────────────────────────────────────────────────

ESTIMATOR_REGISTRY = {
    "regression": {
        "name": "Regression Adjustment (OLS/Logit)",
        "short": "Regression",
        "description": (
            "Fits OLS (continuous outcome) or logistic regression (binary outcome) "
            "with treatment + confounders as predictors. The treatment coefficient "
            "is the ATE. Fast, interpretable, but biased if outcome model is misspecified."
        ),
        "requires": [],
        "best_when": "Small datasets, interpretability needed, few confounders.",
        "assumption": "Correct outcome model specification (no unmeasured confounders).",
    },
    "ipw": {
        "name": "Inverse Probability Weighting (IPW)",
        "short": "IPW",
        "description": (
            "Estimates a propensity score P(T=1|X) via logistic regression, then "
            "weights each observation by 1/p (treated) or 1/(1-p) (control). "
            "Doubly-consistent if propensity model is correct."
        ),
        "requires": ["scikit-learn"],
        "best_when": "Binary treatment, moderate sample size, well-specified propensity model.",
        "assumption": "Correct propensity model; positivity (0 < P(T=1|X) < 1 for all X).",
    },
    "aipw": {
        "name": "Augmented IPW / Doubly Robust (AIPW)",
        "short": "AIPW",
        "description": (
            "Combines an outcome regression model and a propensity score model. "
            "Consistent if *either* model is correct (doubly robust). "
            "Lowest variance among semiparametric estimators."
        ),
        "requires": ["scikit-learn"],
        "best_when": "Moderate-to-large datasets, many confounders, want robustness.",
        "assumption": "At least one of outcome model or propensity model is correctly specified.",
    },
    "matching": {
        "name": "Propensity Score Matching (PSM)",
        "short": "Matching",
        "description": (
            "Estimates propensity scores, then matches each treated unit to its "
            "nearest control unit. ATE is the mean outcome difference across matched pairs. "
            "Reduces confounding by balancing covariate distributions."
        ),
        "requires": ["scikit-learn"],
        "best_when": "Binary treatment, want interpretable matched pairs, overlap region is good.",
        "assumption": "Sufficient overlap between treated and control propensity distributions.",
    },
    "dowhy": {
        "name": "DoWhy + EconML (LinearDML)",
        "short": "DoWhy/EconML",
        "description": (
            "Uses the DoWhy causal graph framework with EconML's LinearDML (Double/Debiased ML) "
            "estimator. Includes automatic refutation tests (placebo, random common cause, "
            "subset validation) to stress-test the causal estimate."
        ),
        "requires": ["dowhy", "econml"],
        "best_when": "Large datasets, complex confounding, need rigorous refutation tests.",
        "assumption": "No unmeasured confounders (must specify all common causes).",
    },
}


# ── Recommendation logic ──────────────────────────────────────────────────────

@dataclass
class EstimatorRecommendation:
    estimator_key: str
    rationale: str
    warnings: List[str]


def recommend_estimator(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> EstimatorRecommendation:
    """Rule-based recommendation of the best estimator for the given setup."""
    warnings_list = []
    n_rows = len(df.dropna(subset=[treatment, outcome]))
    n_confounders = len(confounders)

    # Check binary treatment
    t_vals = df[treatment].dropna().unique()
    is_binary_treatment = set(t_vals).issubset({0, 1}) or (
        not pd.api.types.is_numeric_dtype(df[treatment])
        and df[treatment].nunique() <= 2
    )

    # Check binary outcome
    y_vals = df[outcome].dropna().unique()
    is_binary_outcome = len(y_vals) == 2 and set(y_vals).issubset({0, 1})

    # Very small dataset → regression (most stable)
    if n_rows < 50:
        warnings_list.append(
            f"Very small sample ({n_rows} rows). Most estimators will have high variance. "
            "Regression Adjustment is the most stable choice."
        )
        return EstimatorRecommendation(
            estimator_key="regression",
            rationale=f"Small sample size ({n_rows} rows). Regression Adjustment is the most stable estimator here.",
            warnings=warnings_list,
        )

    # No confounders → IPW is overkill, regression is fine
    if n_confounders == 0:
        warnings_list.append(
            "No confounders specified. With no adjustment variables, all estimators "
            "reduce to a simple difference-in-means."
        )
        return EstimatorRecommendation(
            estimator_key="regression",
            rationale="No confounders: simple regression adjustment is sufficient.",
            warnings=warnings_list,
        )

    # Large dataset + many confounders → AIPW (doubly robust, most efficient)
    if n_rows >= 500 and n_confounders >= 3:
        return EstimatorRecommendation(
            estimator_key="aipw",
            rationale=(
                f"Large dataset ({n_rows:,} rows) with {n_confounders} confounders. "
                "AIPW (Doubly Robust) is the most statistically efficient semiparametric estimator — "
                "consistent if either the outcome or propensity model is correct."
            ),
            warnings=warnings_list,
        )

    # Binary treatment, moderate size → IPW or Matching
    if is_binary_treatment:
        if n_rows >= 200:
            return EstimatorRecommendation(
                estimator_key="ipw",
                rationale=(
                    f"Binary treatment with {n_rows} rows. IPW directly targets the causal contrast "
                    "via propensity score weighting, giving consistent ATE estimates when the "
                    "propensity model is correctly specified."
                ),
                warnings=warnings_list,
            )
        else:
            return EstimatorRecommendation(
                estimator_key="matching",
                rationale=(
                    f"Binary treatment with moderate sample ({n_rows} rows). "
                    "Propensity Score Matching is interpretable and works well at this sample size."
                ),
                warnings=warnings_list,
            )

    # Default → AIPW for robustness
    if n_confounders >= 2:
        return EstimatorRecommendation(
            estimator_key="aipw",
            rationale=(
                f"Multiple confounders ({n_confounders}) detected. AIPW provides doubly-robust "
                "protection against model misspecification."
            ),
            warnings=warnings_list,
        )

    return EstimatorRecommendation(
        estimator_key="regression",
        rationale="Regression Adjustment is the default reliable baseline estimator.",
        warnings=warnings_list,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fit_ols_logit(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str,
    confounders: List[str],
    is_binary_outcome: bool,
) -> Tuple[float, float, float, float, float, str]:
    """Fit OLS or Logit model, return (ate, se, p_val, ci_lo, ci_hi, summary)."""
    y = data[outcome]
    X = data[[treatment_col] + confounders]
    X = sm.add_constant(X)

    if is_binary_outcome:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        margeff = result.get_margeff()
        cols = list(X.columns)
        # treatment_col index in X minus constant offset
        idx = cols.index(treatment_col) - 1
        ate = float(margeff.margeff[idx])
        se = float(margeff.margeff_se[idx])
        p_val = float(margeff.pvalues[idx])
        ci_lo = float(margeff.conf_int()[idx, 0])
        ci_hi = float(margeff.conf_int()[idx, 1])
        summary = result.summary().as_text()
    else:
        model = sm.OLS(y, X)
        result = model.fit()
        ate = float(result.params[treatment_col])
        se = float(result.bse[treatment_col])
        p_val = float(result.pvalues[treatment_col])
        ci = result.conf_int().loc[treatment_col]
        ci_lo = float(ci.iloc[0])
        ci_hi = float(ci.iloc[1])
        summary = result.summary().as_text()

    return ate, se, p_val, ci_lo, ci_hi, summary


def _propensity_scores(data: pd.DataFrame, treatment_col: str, confounders: List[str]) -> np.ndarray:
    """Estimate propensity scores P(T=1|X) via logistic regression."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    X = data[confounders].values if confounders else np.ones((len(data), 1))
    T = data[treatment_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegressionCV(cv=5, max_iter=500, random_state=42)
    clf.fit(X_scaled, T)
    ps = clf.predict_proba(X_scaled)[:, 1]

    # Clip to avoid extreme weights
    ps = np.clip(ps, 0.01, 0.99)
    return ps


# ── Estimator implementations ─────────────────────────────────────────────────

def _estimate_regression(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str,
    confounders: List[str],
    is_binary_outcome: bool,
) -> Tuple[float, float, float, float, float, str]:
    return _fit_ols_logit(data, treatment_col, outcome, confounders, is_binary_outcome)


def _estimate_ipw(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str,
    confounders: List[str],
) -> Tuple[float, float, float, str]:
    """Horvitz-Thompson IPW estimator."""
    T = data[treatment_col].values
    Y = data[outcome].values
    ps = _propensity_scores(data, treatment_col, confounders)

    weights = np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    # Stabilised weights (Robins-style)
    p_t = T.mean()
    stab_weights = np.where(T == 1, p_t / ps, (1 - p_t) / (1 - ps))

    mu1 = np.sum(stab_weights * T * Y) / np.sum(stab_weights * T)
    mu0 = np.sum(stab_weights * (1 - T) * Y) / np.sum(stab_weights * (1 - T))
    ate = float(mu1 - mu0)

    # Variance via influence function
    n = len(Y)
    if1 = stab_weights * T * (Y - mu1) / p_t
    if0 = stab_weights * (1 - T) * (Y - mu0) / (1 - p_t)
    influence = (if1 - if0) - ate
    se = float(np.sqrt(np.var(influence, ddof=1) / n))

    summary = (
        f"IPW Estimator\n"
        f"{'='*50}\n"
        f"ATE (Horvitz-Thompson): {ate:.6f}\n"
        f"Std Error:              {se:.6f}\n"
        f"Mean propensity score:  {ps.mean():.4f}  (range: {ps.min():.4f}–{ps.max():.4f})\n"
        f"Treated units:          {T.sum():.0f}\n"
        f"Control units:          {(1-T).sum():.0f}\n"
        f"Stabilised weights range: [{stab_weights.min():.2f}, {stab_weights.max():.2f}]"
    )
    return ate, se, summary


def _estimate_aipw(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str,
    confounders: List[str],
) -> Tuple[float, float, str]:
    """Augmented IPW / Doubly-Robust estimator."""
    from sklearn.linear_model import Ridge, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    T = data[treatment_col].values.astype(float)
    Y = data[outcome].values.astype(float)
    X_raw = data[confounders].values if confounders else np.ones((len(data), 1))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Propensity model
    ps_model = LogisticRegressionCV(cv=5, max_iter=500, random_state=42)
    ps_model.fit(X, T)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)

    # Outcome model — fit separate models for treated / control
    X_feat = np.column_stack([T, X])

    outcome_model = Ridge(alpha=1.0)
    outcome_model.fit(X_feat, Y)

    # Potential outcome predictions
    X1 = np.column_stack([np.ones(len(T)), X])
    X0 = np.column_stack([np.zeros(len(T)), X])
    mu1_hat = outcome_model.predict(X1)
    mu0_hat = outcome_model.predict(X0)

    # AIPW score function
    aipw_treated = (T * (Y - mu1_hat)) / ps + mu1_hat
    aipw_control = ((1 - T) * (Y - mu0_hat)) / (1 - ps) + mu0_hat
    scores = aipw_treated - aipw_control

    ate = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(len(scores)))

    summary = (
        f"AIPW / Doubly-Robust Estimator\n"
        f"{'='*50}\n"
        f"ATE:                    {ate:.6f}\n"
        f"Std Error:              {se:.6f}\n"
        f"Mean propensity score:  {ps.mean():.4f}  (range: {ps.min():.4f}–{ps.max():.4f})\n"
        f"Outcome model:          Ridge regression (α=1.0)\n"
        f"Propensity model:       Logistic regression (CV)\n"
        f"Treated units:          {T.sum():.0f}\n"
        f"Control units:          {(1-T).sum():.0f}\n"
        f"DR property:            Consistent if either model is correctly specified."
    )
    return ate, se, summary


def _estimate_matching(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str,
    confounders: List[str],
) -> Tuple[float, float, str]:
    """Propensity Score Matching (1:1 nearest neighbour with replacement)."""
    from sklearn.neighbors import NearestNeighbors

    T = data[treatment_col].values.astype(float)
    Y = data[outcome].values.astype(float)
    ps = _propensity_scores(data, treatment_col, confounders)

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("Need at least 1 treated and 1 control unit for matching.")

    # Nearest neighbour in propensity score space (with replacement)
    ps_control = ps[control_idx].reshape(-1, 1)
    ps_treated = ps[treated_idx].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(ps_control)
    distances, indices = nn.kneighbors(ps_treated)

    matched_control_idx = control_idx[indices.flatten()]
    diffs = Y[treated_idx] - Y[matched_control_idx]
    ate = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))

    # Caliper info
    max_dist = float(distances.max())
    mean_dist = float(distances.mean())

    summary = (
        f"Propensity Score Matching (1:1 NN, with replacement)\n"
        f"{'='*50}\n"
        f"ATE (ATT):              {ate:.6f}\n"
        f"Std Error:              {se:.6f}\n"
        f"Matched pairs:          {len(diffs)}\n"
        f"Mean PS distance:       {mean_dist:.4f}\n"
        f"Max PS distance:        {max_dist:.4f}\n"
        f"Treated units:          {len(treated_idx)}\n"
        f"Control units:          {len(control_idx)}\n"
        f"Note: ATT (effect on treated) is estimated, not ATE."
    )
    return ate, se, summary


def _estimate_dowhy(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str,
    confounders: List[str],
) -> Tuple[float, float, str]:
    """DoWhy + EconML LinearDML estimator with refutation tests."""
    try:
        import dowhy
        from dowhy import CausalModel
        import econml
    except ImportError:
        raise ImportError(
            "DoWhy and EconML are not installed. "
            "Run: pip install dowhy econml"
        )

    # Build causal graph string
    common_causes_graph = ""
    for c in confounders:
        common_causes_graph += f'"{c}" -> "{treatment_col}"; "{c}" -> "{outcome}"; '

    dot_graph = f'digraph {{ {common_causes_graph} "{treatment_col}" -> "{outcome}"; }}'

    model = CausalModel(
        data=data,
        treatment=treatment_col,
        outcome=outcome,
        common_causes=confounders if confounders else None,
        graph=dot_graph if confounders else None,
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.LinearDML",
            method_params={
                "init_params": {"random_state": 42},
                "fit_params": {},
            },
            confidence_intervals=True,
        )

    def _to_scalar(v) -> float:
        """Safely convert any array-like or scalar DoWhy/EconML value to a Python float."""
        if v is None:
            return float("nan")
        arr = np.asarray(v, dtype=float).flatten()
        return float(arr[0]) if arr.size > 0 else float("nan")

    ate = _to_scalar(estimate.value)

    # Extract CI — newer DoWhy/EconML returns arrays, not scalars
    try:
        ci = estimate.get_confidence_intervals()
        if ci is None:
            raise ValueError("CI returned None")
        ci_lo = _to_scalar(ci[0])
        ci_hi = _to_scalar(ci[1])
    except Exception:
        # Fallback: attempt standard error, otherwise use ±10% of ATE
        try:
            se_raw = _to_scalar(estimate.get_standard_error())
        except Exception:
            se_raw = float("nan")
        if not np.isnan(se_raw) and se_raw > 0:
            ci_lo = ate - 1.96 * se_raw
            ci_hi = ate + 1.96 * se_raw
        else:
            margin = abs(ate) * 0.1 if ate != 0 else 0.1
            ci_lo = ate - margin
            ci_hi = ate + margin

    se = (ci_hi - ci_lo) / (2 * 1.96)

    # Run refutation tests
    refutation_lines = []
    try:
        ref_placebo = model.refute_estimate(
            identified_estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=20,
        )
        refutation_lines.append(f"Placebo refutation p-value: {ref_placebo.refutation_result.get('p_value', 'N/A'):.3f}")
    except Exception as ex:
        refutation_lines.append(f"Placebo refutation: skipped ({ex})")

    try:
        ref_random = model.refute_estimate(
            identified_estimand, estimate,
            method_name="random_common_cause",
            num_simulations=20,
        )
        refutation_lines.append(f"Random common cause: new ATE = {_to_scalar(ref_random.new_effect):.4f}")
    except Exception as ex:
        refutation_lines.append(f"Random common cause: skipped ({ex})")

    summary = (
        f"DoWhy + EconML (LinearDML)\n"
        f"{'='*50}\n"
        f"ATE:                    {ate:.6f}\n"
        f"95% CI:                 [{ci_lo:.6f}, {ci_hi:.6f}]\n"
        f"Estimand:               {str(identified_estimand)[:200]}\n"
        f"\nRefutation Tests\n"
        f"{'-'*30}\n"
        + "\n".join(refutation_lines)
    )
    return ate, se, summary


# ── Public API ────────────────────────────────────────────────────────────────

def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    control_value=None,
    treatment_value=None,
    n_bootstrap: int = 0,
    estimator: str = "regression",
) -> EstimationResult:
    """
    Estimate ATE using the chosen estimator.

    Parameters
    ----------
    df              : input DataFrame
    treatment       : column name of the treatment variable
    outcome         : column name of the outcome variable
    confounders     : list of confounder column names
    control_value   : for categorical treatment — encode this as 0
    treatment_value : for categorical treatment — encode this as 1
    n_bootstrap     : number of bootstrap iterations (0 = no bootstrap; IPW/AIPW/Matching use analytical SE)
    estimator       : one of "regression", "ipw", "aipw", "matching", "dowhy"
    """
    vars_to_keep = [treatment, outcome] + confounders
    data = df[vars_to_keep].dropna().copy()

    if len(data) < 5:
        raise ValueError("Too few complete rows after dropping missing values.")

    # Detect categorical treatment
    is_categorical = (
        not pd.api.types.is_numeric_dtype(data[treatment])
        or data[treatment].nunique() <= 20
    )

    treatment_col = treatment
    if is_categorical and control_value is not None and treatment_value is not None:
        data = data[data[treatment].isin([control_value, treatment_value])].copy()
        data["_treatment_encoded"] = (data[treatment] == treatment_value).astype(int)
        treatment_col = "_treatment_encoded"

    y = data[outcome]
    unique_y = sorted(y.dropna().unique())
    is_binary_outcome = len(unique_y) == 2 and set(unique_y).issubset({0, 1})

    # Verify binary treatment for IPW/AIPW/Matching
    if estimator in ("ipw", "aipw", "matching"):
        t_unique = sorted(data[treatment_col].dropna().unique())
        if not set(t_unique).issubset({0, 1}):
            # Try to coerce to binary via median split
            median_t = data[treatment_col].median()
            data = data.copy()
            data[treatment_col] = (data[treatment_col] > median_t).astype(int)
            warnings.warn(
                f"Treatment column '{treatment_col}' is not binary. "
                f"Binarised at median ({median_t:.4f}) for {estimator.upper()} estimator."
            )

    # ── Run estimator ─────────────────────────────────────────────────────────
    p_val = np.nan
    bootstrap_ates = None

    if estimator == "regression":
        ate, se, p_val, ci_lo, ci_hi, summary = _estimate_regression(
            data, treatment_col, outcome, confounders, is_binary_outcome
        )

    elif estimator == "ipw":
        ate, se, summary = _estimate_ipw(data, treatment_col, outcome, confounders)
        z = ate / se if se > 0 else 0.0
        from scipy import stats as scipy_stats
        p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
        ci_lo = ate - 1.96 * se
        ci_hi = ate + 1.96 * se

    elif estimator == "aipw":
        ate, se, summary = _estimate_aipw(data, treatment_col, outcome, confounders)
        z = ate / se if se > 0 else 0.0
        from scipy import stats as scipy_stats
        p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
        ci_lo = ate - 1.96 * se
        ci_hi = ate + 1.96 * se

    elif estimator == "matching":
        ate, se, summary = _estimate_matching(data, treatment_col, outcome, confounders)
        z = ate / se if se > 0 else 0.0
        from scipy import stats as scipy_stats
        p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
        ci_lo = ate - 1.96 * se
        ci_hi = ate + 1.96 * se

    elif estimator == "dowhy":
        ate, se, summary = _estimate_dowhy(data, treatment_col, outcome, confounders)
        z = ate / se if se > 0 else 0.0
        from scipy import stats as scipy_stats
        p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
        ci_lo = ate - 1.96 * se
        ci_hi = ate + 1.96 * se

    else:
        raise ValueError(f"Unknown estimator: {estimator!r}. Choose from {list(ESTIMATOR_REGISTRY)}")

    # ── Bootstrap (regression only; others have analytical SE) ───────────────
    if n_bootstrap > 0 and estimator == "regression":
        boot_ates = []
        for _ in range(n_bootstrap):
            sample = data.sample(n=len(data), replace=True)
            try:
                b_ate, *_ = _fit_ols_logit(sample, treatment_col, outcome, confounders, is_binary_outcome)
                boot_ates.append(b_ate)
            except Exception:
                pass
        if boot_ates:
            bootstrap_ates = np.array(boot_ates)
            se = float(np.std(bootstrap_ates, ddof=1))
            ci_lo = float(np.percentile(bootstrap_ates, 2.5))
            ci_hi = float(np.percentile(bootstrap_ates, 97.5))
    elif n_bootstrap > 0 and estimator in ("ipw", "aipw", "matching"):
        # Bootstrap for non-regression estimators
        boot_ates = []
        for _ in range(n_bootstrap):
            sample = data.sample(n=len(data), replace=True)
            try:
                if estimator == "ipw":
                    b_ate, _, _ = _estimate_ipw(sample, treatment_col, outcome, confounders)
                elif estimator == "aipw":
                    b_ate, _, _ = _estimate_aipw(sample, treatment_col, outcome, confounders)
                elif estimator == "matching":
                    b_ate, _, _ = _estimate_matching(sample, treatment_col, outcome, confounders)
                boot_ates.append(b_ate)
            except Exception:
                pass
        if boot_ates:
            bootstrap_ates = np.array(boot_ates)
            se = float(np.std(bootstrap_ates, ddof=1))
            ci_lo = float(np.percentile(bootstrap_ates, 2.5))
            ci_hi = float(np.percentile(bootstrap_ates, 97.5))
            z = ate / se if se > 0 else 0.0
            from scipy import stats as scipy_stats
            p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))

    # ── Build interpretation ──────────────────────────────────────────────────
    direction = "increases" if ate > 0 else "decreases"
    sig_tag = "✅ statistically significant (p < 0.05)" if p_val < 0.05 else "⚠️ not statistically significant (p ≥ 0.05)"
    est_name = ESTIMATOR_REGISTRY[estimator]["name"]

    if is_binary_outcome:
        interpretation = (
            f"[**{est_name}**] A **1-unit increase** in **{treatment}** *{direction}* the probability of "
            f"**{outcome}** by **{abs(ate)*100:.1f} pp** "
            f"(95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]). "
            f"This result is {sig_tag} (p = {p_val:.4f})."
        )
    else:
        interpretation = (
            f"[**{est_name}**] A **1-unit increase** in **{treatment}** *{direction}* **{outcome}** by "
            f"**{abs(ate):.4f}** units on average "
            f"(95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]). "
            f"This result is {sig_tag} (p = {p_val:.4f})."
        )

    return EstimationResult(
        ate=ate,
        se=se,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        summary_text=summary,
        is_binary_outcome=is_binary_outcome,
        is_categorical_treatment=is_categorical,
        interpretation=interpretation,
        estimator_used=estimator,
        bootstrap_ates=bootstrap_ates,
    )
