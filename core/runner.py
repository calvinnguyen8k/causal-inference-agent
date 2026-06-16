"""Algorithm runner — wraps causal-learn and lingam into a uniform interface."""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunResult:
    algorithm_key: str
    adjacency_matrix: np.ndarray   # Shape (n_vars, n_vars); entry [i,j] != 0 means i → j
    column_names: list[str]
    runtime_seconds: float
    extra: dict = field(default_factory=dict)  # Algorithm-specific output (e.g., p-values)


# Registry: display name, description, family
ALGORITHM_REGISTRY = {
    "pc": {
        "name": "PC",
        "family": "Constraint-based",
        "description": "Peter-Clark algorithm. Uses conditional independence tests to find a CPDAG.",
        "requires_gaussian": False,
    },
    "fci": {
        "name": "FCI",
        "family": "Constraint-based",
        "description": "Fast Causal Inference. Handles latent confounders. Returns a PAG.",
        "requires_gaussian": False,
    },
    "ges": {
        "name": "GES",
        "family": "Score-based",
        "description": "Greedy Equivalence Search. BIC-score-based. Assumes Gaussian data.",
        "requires_gaussian": True,
    },
    "fges": {
        "name": "FGES",
        "family": "Score-based",
        "description": "Fast GES. Scales to large datasets. Uses BIC scoring.",
        "requires_gaussian": True,
    },
    "direct_lingam": {
        "name": "DirectLiNGAM",
        "family": "Functional (FCM)",
        "description": "Exploits non-Gaussianity to identify a unique DAG. Assumes linearity.",
        "requires_gaussian": False,
    },
    "ica_lingam": {
        "name": "ICA-LiNGAM",
        "family": "Functional (FCM)",
        "description": "Original LiNGAM via ICA decomposition.",
        "requires_gaussian": False,
    },

    "notears": {
        "name": "NOTEARS",
        "family": "Continuous optimisation",
        "description": "Continuous optimisation over DAG space. L1-regularised.",
        "requires_gaussian": False,
    },
    "granger": {
        "name": "Granger Causality",
        "family": "Time-series",
        "description": "Tests whether past values of X improve prediction of Y. For time-ordered data.",
        "requires_gaussian": False,
    },
    "exact_search": {
        "name": "Exact Search (DP)",
        "family": "Score-based",
        "description": "Dynamic programming exact solution. Only feasible for <10 variables.",
        "requires_gaussian": True,
    },
}


def run_algorithm(
    df: pd.DataFrame,
    algorithm_key: str,
    alpha: float = 0.05,
    max_degree: Optional[int] = None,
) -> RunResult:
    """
    Run the specified causal discovery algorithm on df.
    Only numeric columns are used. Returns RunResult.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    X = numeric_df.values.astype(float)
    col_names = numeric_df.columns.tolist()
    n_vars = X.shape[1]

    start = time.time()

    if algorithm_key == "pc":
        adj = _run_pc(X, alpha, max_degree)
    elif algorithm_key == "fci":
        adj = _run_fci(X, alpha)
    elif algorithm_key == "ges":
        adj = _run_ges(X)
    elif algorithm_key == "fges":
        adj = _run_fges(X)
    elif algorithm_key == "direct_lingam":
        adj = _run_direct_lingam(X)
    elif algorithm_key == "ica_lingam":
        adj = _run_ica_lingam(X)

    elif algorithm_key == "notears":
        adj = _run_notears(X)
    elif algorithm_key == "granger":
        adj = _run_granger(X, alpha)
    elif algorithm_key == "exact_search":
        adj = _run_exact_search(X)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_key}")

    runtime = time.time() - start

    return RunResult(
        algorithm_key=algorithm_key,
        adjacency_matrix=adj,
        column_names=col_names,
        runtime_seconds=runtime,
    )


# ── Internal algorithm wrappers ──────────────────────────────────────────────

def _run_pc(X, alpha, max_degree):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    result = pc(X, alpha=alpha, indep_test=fisherz)
    return _cg_to_adj(result.G, X.shape[1])


def _run_fci(X, alpha):
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
    result, _ = fci(X, fisherz, alpha)
    return _cg_to_adj(result, X.shape[1])


def _run_ges(X):
    from causallearn.search.ScoreBased.GES import ges
    result = ges(X)
    return _cg_to_adj(result['G'], X.shape[1])


def _run_fges(X):
    # causal-learn FGES wrapper
    from causallearn.search.ScoreBased.GES import ges
    result = ges(X, score_func='local_score_BIC')
    return _cg_to_adj(result['G'], X.shape[1])


def _run_direct_lingam(X):
    import lingam
    model = lingam.DirectLiNGAM()
    model.fit(X)
    return model.adjacency_matrix_


def _run_ica_lingam(X):
    import lingam
    model = lingam.ICALiNGAM()
    model.fit(X)
    return model.adjacency_matrix_



def _run_notears(X):
    from core.notears import notears_linear
    return notears_linear(X, lambda1=0.1, loss_type='l2')


def _run_granger(X, alpha):
    from statsmodels.tsa.stattools import grangercausalitytests
    n = X.shape[1]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                data = np.column_stack([X[:, j], X[:, i]])
                result = grangercausalitytests(data, maxlag=2, verbose=False)
                # Use lag-1 F-test p-value
                p_val = result[1][0]['ssr_ftest'][1]
                if p_val < alpha:
                    adj[i, j] = 1.0 - p_val  # Weight by confidence
            except Exception:
                pass
    return adj


def _run_exact_search(X):
    from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
    dag, _ = bic_exact_search(X)
    return dag.astype(float)


def _cg_to_adj(G, n_vars: int) -> np.ndarray:
    """Convert a causal-learn GeneralGraph to an adjacency matrix."""
    adj = np.zeros((n_vars, n_vars))
    try:
        for i in range(n_vars):
            for j in range(n_vars):
                edge = G.get_edge(G.nodes[i], G.nodes[j])
                if edge is not None:
                    adj[i, j] = 1.0
    except Exception:
        # Fallback: try graph_to_adjmat utility
        try:
            from causallearn.utils.GraphUtils import GraphUtils
            adj = GraphUtils.to_adjacency_matrix(G)
        except Exception:
            pass
    return adj
