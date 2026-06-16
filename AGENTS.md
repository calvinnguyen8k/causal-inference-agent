# AGENTS.md — Causal Inference Agent (Streamlit)

> **Purpose:** Step-by-step instructions for an AI coding agent (Claude Code / Codex / Google Antigravity) to implement a production-quality Causal Inference webapp that mirrors `causalinferenceagent.streamlit.app`. Follow every section in order. Do not skip steps or add features not listed.

---

## 0. Guiding Principles

Before writing a single line of code, internalize these rules:

1. **Think before coding** — state assumptions explicitly; surface tradeoffs before acting.
2. **Simplicity first** — minimum code that solves the problem. No speculative abstractions.
3. **Surgical changes** — touch only what you must. Every changed line traces to a requirement.
4. **Define success criteria per step** — each step has a `✅ Verify` section. Do not proceed until it passes.
5. **No invented features** — if a feature is not listed in this file, do not build it.

---

## 1. Project Overview

Build an interactive, single-file Streamlit webapp for **causal structure discovery**. The app allows users to:

- Upload a CSV of tabular data
- Receive an automated recommendation of the best causal discovery algorithm for their data
- Run any of 15+ causal discovery algorithms
- View the resulting DAG with bootstrap edge-confidence overlays
- Export the adjacency matrix and graph image

**Target deployment:** Local machine first. Only move to Streamlit Community Cloud after all local tests pass. No database. No auth.

---

## 2. Repository Structure

```
causal-inference-agent/
├── app.py                  # Single entrypoint — all Streamlit UI lives here
├── core/
│   ├── __init__.py
│   ├── profiler.py         # Dataset profiling (dtypes, normality, sample size)
│   ├── recommender.py      # Algorithm recommendation logic
│   ├── runner.py           # Algorithm execution wrappers
│   ├── bootstrap.py        # Bootstrap confidence graph computation
│   └── visualiser.py       # DAG → graphviz / matplotlib rendering
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Theme configuration
└── README.md
```

Do NOT create sub-packages beyond `core/`. Do not add a `tests/` directory in the first pass.

---

## 3. Local Environment Setup

Complete this section fully before writing any application code. Every step has a verify check — do not proceed until it passes.

### 3.1 Python version

Requires Python 3.10 or 3.11. Check with:

```bash
python --version
```

If you have 3.12+, use `pyenv` or `conda` to install 3.11 — some causal-learn dependencies have not yet published 3.12 wheels.

### 3.2 Create a virtual environment

**Always use a dedicated venv.** Never install into the system Python.

```bash
# From inside the project root
python3.11 -m venv .venv

# Activate (macOS / Linux)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Confirm you are inside the venv
which python   # should print a path ending in .venv/bin/python
```

### 3.3 Install system-level `graphviz` binary

The Python `graphviz` package is just a thin wrapper — it requires the `graphviz` binaries on your PATH.

**macOS:**
```bash
brew install graphviz
dot -V   # should print graphviz version
```

**Ubuntu / Debian:**
```bash
sudo apt-get update && sudo apt-get install -y graphviz
dot -V
```

**Windows:**
Download and install from https://graphviz.org/download/ and add the `bin/` folder to your PATH. Then restart your terminal and confirm `dot -V` works.

✅ **Verify:** `dot -V` prints a version string. If not, fix PATH before continuing.

### 3.4 `requirements.txt`

Create this file in the project root:

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0
networkx>=3.2.0
matplotlib>=3.8.0
graphviz>=0.20.0
causal-learn>=0.1.3.8
lingam>=1.8.3
statsmodels>=0.14.0
```

Install:

```bash
pip install -r requirements.txt
```

This will take 2–5 minutes — `causal-learn` pulls in several heavy dependencies.

### 3.5 Smoke-test all imports

Create a temporary file `check_env.py` in the project root:

```python
# check_env.py — run this after pip install to confirm everything is wired up
import sys

failures = []

deps = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("networkx", "networkx"),
    ("matplotlib", "matplotlib"),
    ("graphviz", "graphviz"),
    ("causallearn", "causal-learn"),
    ("lingam", "lingam"),
    ("statsmodels", "statsmodels"),
]

for module, pip_name in deps:
    try:
        __import__(module)
        print(f"  ✅  {pip_name}")
    except ImportError as e:
        print(f"  ❌  {pip_name} — {e}")
        failures.append(pip_name)

# Check graphviz system binary
import subprocess
try:
    result = subprocess.run(["dot", "-V"], capture_output=True, text=True)
    print(f"  ✅  graphviz binary (dot): {result.stderr.strip()}")
except FileNotFoundError:
    print("  ❌  graphviz binary (dot) not found on PATH — install with brew/apt")
    failures.append("graphviz-binary")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
else:
    print("All dependencies OK. Proceed to Step 4.")
```

Run it:

```bash
python check_env.py
```

✅ **Verify:** Every line prints ✅ and the final message is `All dependencies OK.` Fix any failures before continuing. Delete `check_env.py` after it passes.

---

## 4. Streamlit Theme

### 4.1 `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#4F46E5"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F1F5F9"
font = "sans serif"
```

---

## 5. Core Module: `core/profiler.py`

**Purpose:** Analyse the uploaded DataFrame and return a profile dict consumed by the recommender and the UI.

```python
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
        corr_matrix = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()

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
```

✅ **Verify:** `from core.profiler import profile_dataframe; import pandas as pd; p = profile_dataframe(pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})); assert p.n_rows == 3`

---

## 6. Core Module: `core/recommender.py`

**Purpose:** Map a `DataProfile` to an algorithm recommendation with a plain-English rationale.

```python
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
```

✅ **Verify:** `from core.recommender import recommend; from core.profiler import DataProfile` runs without error.

---

## 7. Core Module: `core/runner.py`

**Purpose:** Thin wrappers around each algorithm. Returns a `RunResult` with an adjacency matrix and metadata.

```python
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
    from causallearn.search.FCMBased.notears.notears_linear import notears_linear
    W = notears_linear(X, lambda1=0.1, loss_type='l2')
    return W


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
```

✅ **Verify:** `from core.runner import ALGORITHM_REGISTRY; assert len(ALGORITHM_REGISTRY) == 9`

---

## 8. Core Module: `core/bootstrap.py`

**Purpose:** Run N bootstrap resamples of an algorithm and aggregate edge frequencies.

```python
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
```

✅ **Verify:** Module imports without error.

---

## 9. Core Module: `core/visualiser.py`

**Purpose:** Render an adjacency matrix + optional edge frequencies into a matplotlib figure.

```python
"""DAG visualiser — adjacency matrix → matplotlib figure via graphviz layout."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from typing import Optional


def adjacency_to_figure(
    adj: np.ndarray,
    col_names: list[str],
    edge_freq: Optional[np.ndarray] = None,
    freq_threshold: float = 0.0,
    title: str = "Causal Graph",
) -> plt.Figure:
    """
    Convert adjacency matrix to a matplotlib figure.

    Parameters
    ----------
    adj          : (n, n) matrix; entry [i,j] != 0 means i → j
    col_names    : variable names matching adj rows/cols
    edge_freq    : optional (n, n) bootstrap frequency matrix; used for edge colour/width
    freq_threshold: edges below this bootstrap frequency are hidden
    title        : figure title
    """
    n = len(col_names)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    edges = []
    edge_weights = []

    for i in range(n):
        for j in range(n):
            if abs(adj[i, j]) > 1e-8:
                freq = edge_freq[i, j] if edge_freq is not None else 1.0
                if freq >= freq_threshold:
                    G.add_edge(i, j)
                    edges.append((i, j))
                    edge_weights.append(freq)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1E293B")
    ax.set_facecolor("#1E293B")

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No edges above threshold", ha="center", va="center",
                color="white", fontsize=14, transform=ax.transAxes)
        ax.set_title(title, color="white", fontsize=14, pad=12)
        return fig

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Node colours
    node_color = "#4F46E5"
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=1200, ax=ax, alpha=0.95)

    # Edge colours from frequency (low=grey, high=indigo)
    cmap = mcolors.LinearSegmentedColormap.from_list("freq", ["#64748B", "#4F46E5"])
    edge_colors = [cmap(w) for w in edge_weights]
    edge_widths = [1.5 + 3.0 * w for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    # Labels
    labels = {i: col_names[i] for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_color="white",
                             font_size=9, font_weight="bold", ax=ax)

    # Colorbar for bootstrap frequency
    if edge_freq is not None and edge_weights:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Bootstrap frequency", color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def edge_frequency_heatmap(
    freq: np.ndarray,
    col_names: list[str],
) -> plt.Figure:
    """Render bootstrap edge frequency as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#1E293B")
    ax.set_facecolor("#1E293B")

    im = ax.imshow(freq, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Bootstrap frequency")

    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", color="white", fontsize=8)
    ax.set_yticklabels(col_names, color="white", fontsize=8)
    ax.set_title("Edge Frequency Heatmap (row → col)", color="white", fontsize=12, pad=10)

    # Annotate cells
    for i in range(len(col_names)):
        for j in range(len(col_names)):
            val = freq[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black" if val > 0.5 else "white", fontsize=7)

    plt.tight_layout()
    return fig
```

✅ **Verify:** Module imports without error. `import matplotlib; matplotlib.use('Agg')` if running headless.

---

## 10. Main App: `app.py`

This is the entire Streamlit UI. Build it section by section.

```python
"""
Causal Inference Agent — Streamlit app entrypoint.
Run: streamlit run app.py
"""

import io
import time
import numpy as np
import pandas as pd
import streamlit as st

from core.profiler import profile_dataframe
from core.recommender import recommend
from core.runner import run_algorithm, ALGORITHM_REGISTRY
from core.bootstrap import run_bootstrap
from core.visualiser import adjacency_to_figure, edge_frequency_heatmap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal Inference Agent",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
}
.warning-box {
    background: #451A03;
    border-left: 4px solid #F97316;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 8px 0;
    color: #FED7AA;
    font-size: 0.9em;
}
.recommendation-box {
    background: #1E1B4B;
    border-left: 4px solid #4F46E5;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 10px 0;
    color: #C7D2FE;
}
.algo-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85em;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔗 Causal Inference Agent")
    st.caption("Automated causal structure discovery")
    st.divider()

    st.subheader("📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV of tabular data. All numeric columns will be used for causal discovery.",
    )

    st.divider()
    st.subheader("⚙️ Algorithm Settings")

    alpha = st.slider(
        "Significance level (α)",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        help="Threshold for conditional independence tests. Lower α → fewer edges.",
    )

    st.divider()
    st.subheader("🔁 Bootstrap Settings")

    run_bootstrap_flag = st.checkbox(
        "Run bootstrap confidence",
        value=False,
        help="Re-runs the algorithm on N resampled datasets to estimate edge confidence. Slower.",
    )

    n_resamples = 100
    freq_threshold = 0.5
    if run_bootstrap_flag:
        n_resamples = st.slider("Bootstrap resamples (N)", 50, 500, 100, step=50)
        freq_threshold = st.slider(
            "Edge frequency threshold",
            0.0, 1.0, 0.5, step=0.05,
            help="Hide edges that appear in fewer than this fraction of bootstrap runs.",
        )

    st.divider()
    st.caption("Built with [causal-learn](https://github.com/py-why/causal-learn) & [LiNGAM](https://github.com/cdt15/lingam)")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Causal Inference Agent")
st.markdown("Discover causal structure in your data — automatically.")

if uploaded_file is None:
    _show_landing()
    st.stop()

# ── Load & profile data ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

with st.spinner("Loading data…"):
    df = load_data(uploaded_file)

profile = profile_dataframe(df)

# ── Data overview ─────────────────────────────────────────────────────────────
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{profile.n_rows:,}")
col2.metric("Columns", f"{profile.n_cols}")
col3.metric("Numeric cols", f"{len(profile.numeric_cols)}")
col4.metric("Missing", f"{profile.missing_pct:.1f}%")

with st.expander("Show data preview"):
    st.dataframe(df.head(50), use_container_width=True)

if not profile.numeric_cols:
    st.error("No numeric columns found. Please upload a dataset with at least 2 numeric variables.")
    st.stop()

if len(profile.numeric_cols) < 2:
    st.error("At least 2 numeric columns are required for causal discovery.")
    st.stop()

# ── Recommendation ────────────────────────────────────────────────────────────
rec = recommend(profile)

st.subheader("🤖 Algorithm Recommendation")

for w in rec.warnings:
    st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

algo_info = ALGORITHM_REGISTRY.get(rec.algorithm_key, {})
st.markdown(
    f'<div class="recommendation-box">'
    f'<strong>Recommended: {algo_info.get("name", rec.algorithm_key)}</strong><br>'
    f'{rec.rationale}'
    f'</div>',
    unsafe_allow_html=True
)

# ── Algorithm selector ────────────────────────────────────────────────────────
st.subheader("🔬 Run Algorithm")

algorithm_options = {
    key: f"{info['name']} — {info['family']}"
    for key, info in ALGORITHM_REGISTRY.items()
}

# Default to recommendation
default_index = list(algorithm_options.keys()).index(rec.algorithm_key)

selected_key = st.selectbox(
    "Choose algorithm (override recommendation if desired)",
    options=list(algorithm_options.keys()),
    format_func=lambda k: algorithm_options[k],
    index=default_index,
)

selected_info = ALGORITHM_REGISTRY[selected_key]
st.markdown(
    f'<div class="algo-card">'
    f'<strong>{selected_info["name"]}</strong> · {selected_info["family"]}<br>'
    f'<em>{selected_info["description"]}</em>'
    f'</div>',
    unsafe_allow_html=True
)

# Warn if Gaussian assumption violated
if selected_info.get("requires_gaussian") and not profile.is_gaussian:
    st.warning(
        f"⚠️ {selected_info['name']} assumes Gaussian data, but only "
        f"{profile.gaussian_ratio * 100:.0f}% of your columns pass the normality test. "
        "Results may be unreliable."
    )

run_button = st.button("▶ Run Causal Discovery", type="primary", use_container_width=True)

if not run_button:
    st.stop()

# ── Execute algorithm ─────────────────────────────────────────────────────────
with st.spinner(f"Running {selected_info['name']}…"):
    try:
        run_result = run_algorithm(df, selected_key, alpha=alpha)
    except Exception as e:
        st.error(f"Algorithm failed: {e}")
        st.stop()

st.success(f"✅ Completed in {run_result.runtime_seconds:.2f}s")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
bootstrap_result = None
if run_bootstrap_flag:
    progress_bar = st.progress(0, text="Running bootstrap resamples…")
    # Run bootstrap in one call; update progress bar after
    with st.spinner(f"Running {n_resamples} bootstrap resamples…"):
        bootstrap_result = run_bootstrap(df, selected_key, n_resamples=n_resamples, alpha=alpha)
    progress_bar.progress(1.0, text=f"Bootstrap complete ({bootstrap_result.n_resamples} successful runs)")

# ── Results tabs ──────────────────────────────────────────────────────────────
st.subheader("📈 Results")

tab_graph, tab_matrix, tab_bootstrap, tab_export = st.tabs([
    "Causal Graph", "Adjacency Matrix", "Bootstrap Confidence", "Export"
])

with tab_graph:
    edge_freq = bootstrap_result.edge_frequency if bootstrap_result else None
    threshold = freq_threshold if bootstrap_result else 0.0

    fig = adjacency_to_figure(
        adj=run_result.adjacency_matrix,
        col_names=run_result.column_names,
        edge_freq=edge_freq,
        freq_threshold=threshold,
        title=f"{selected_info['name']} — Causal Graph",
    )
    st.pyplot(fig, use_container_width=True)

    n_edges = int((np.abs(run_result.adjacency_matrix) > 1e-8).sum())
    st.caption(f"Discovered {n_edges} directed edges across {len(run_result.column_names)} variables.")

with tab_matrix:
    adj_df = pd.DataFrame(
        run_result.adjacency_matrix,
        index=run_result.column_names,
        columns=run_result.column_names,
    )
    st.markdown("**Adjacency matrix** — row causes column if value ≠ 0.")
    st.dataframe(adj_df.style.background_gradient(cmap="Blues"), use_container_width=True)

with tab_bootstrap:
    if bootstrap_result is None:
        st.info("Enable 'Run bootstrap confidence' in the sidebar and re-run to see edge confidence estimates.")
    else:
        st.markdown(
            f"Edge frequencies from **{bootstrap_result.n_resamples}** bootstrap resamples. "
            "Value = fraction of runs in which each directed edge appeared."
        )
        fig_heat = edge_frequency_heatmap(
            bootstrap_result.edge_frequency,
            bootstrap_result.column_names,
        )
        st.pyplot(fig_heat, use_container_width=True)

        # Top edges table
        freq = bootstrap_result.edge_frequency
        cols = bootstrap_result.column_names
        rows = []
        for i in range(len(cols)):
            for j in range(len(cols)):
                if freq[i, j] > 0.05:
                    rows.append({"From": cols[i], "To": cols[j], "Frequency": round(freq[i, j], 3)})
        if rows:
            top_df = pd.DataFrame(rows).sort_values("Frequency", ascending=False)
            st.dataframe(top_df, use_container_width=True, hide_index=True)

with tab_export:
    st.markdown("**Download results**")

    # Adjacency matrix CSV
    csv_buffer = io.StringIO()
    adj_df.to_csv(csv_buffer)
    st.download_button(
        "⬇ Download adjacency matrix (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"causal_adj_{selected_key}.csv",
        mime="text/csv",
    )

    # Graph image PNG
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight",
                facecolor="#1E293B")
    st.download_button(
        "⬇ Download graph image (PNG)",
        data=img_buffer.getvalue(),
        file_name=f"causal_graph_{selected_key}.png",
        mime="image/png",
    )

    if bootstrap_result:
        freq_df = pd.DataFrame(
            bootstrap_result.edge_frequency,
            index=bootstrap_result.column_names,
            columns=bootstrap_result.column_names,
        )
        freq_buffer = io.StringIO()
        freq_df.to_csv(freq_buffer)
        st.download_button(
            "⬇ Download bootstrap frequencies (CSV)",
            data=freq_buffer.getvalue(),
            file_name=f"bootstrap_freq_{selected_key}.csv",
            mime="text/csv",
        )


# ── Landing page helper ───────────────────────────────────────────────────────
def _show_landing():
    st.markdown("""
### How it works

1. **Upload** a CSV of tabular data (all numeric or mixed).
2. The agent **profiles** your data — sample size, distribution, types.
3. It **recommends** the best causal discovery algorithm and explains why.
4. You **run** the algorithm (and optionally bootstrap confidence).
5. **Explore** the resulting causal graph, adjacency matrix, and edge frequencies.
6. **Export** results as CSV or PNG.

---

### Supported algorithms

| Family | Algorithms |
|---|---|
| Constraint-based | PC, FCI |
| Score-based | GES, FGES, Exact Search |
| Functional / FCM | DirectLiNGAM, ICA-LiNGAM, NOTEARS |
| Time-series | Granger Causality |

---
**Upload a CSV in the sidebar to get started.**
""")
```

> **Note:** The `_show_landing()` function must be defined **before** the `if uploaded_file is None:` check that calls it. Move the function definition to the top of the file (after imports) or use Python's forward-reference pattern.

✅ **Verify:** `streamlit run app.py` loads without import errors. The landing page renders.

---

## 11. Sample Data for Testing

Create `data/sample.csv` for local testing:

```bash
python - << 'EOF'
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

# Simple 5-variable DAG: A → B → D, A → C → D, C → E
A = np.random.normal(0, 1, n)
B = 0.8 * A + np.random.normal(0, 0.5, n)
C = 0.6 * A + np.random.normal(0, 0.5, n)
D = 0.7 * B + 0.5 * C + np.random.normal(0, 0.3, n)
E = 0.9 * C + np.random.normal(0, 0.4, n)

df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})
df.to_csv('data/sample.csv', index=False)
print("Written data/sample.csv")
EOF
```

✅ **Verify:** Upload `data/sample.csv`. Profiler recommends PC or DirectLiNGAM. Graph shows edges A→B, A→C, B→D, C→D, C→E.

---

## 12. Running Locally

**Complete this entire section before touching deployment.** The goal is a fully working app on `localhost:8501`.

### 12.1 Start the app

```bash
# Make sure your venv is active
source .venv/bin/activate   # macOS/Linux
# or: .venv\Scripts\Activate.ps1  (Windows)

streamlit run app.py
```

Streamlit will print something like:

```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser.

✅ **Verify:** Landing page renders with the "How it works" instructions and algorithm table. No Python errors in the terminal.

### 12.2 End-to-end test with sample data

Generate the test dataset first (see Section 11), then perform each test manually:

**Test 1 — Data upload and profiling**
1. Upload `data/sample.csv` via the sidebar
2. Confirm the four metric cards show: Rows=300, Columns=5, Numeric cols=5, Missing=0.0%
3. Expand "Show data preview" — confirm 5 columns A–E are visible

✅ **Pass criteria:** All four metrics correct, preview loads.

**Test 2 — Algorithm recommendation**
1. After upload, the recommendation box should appear
2. Expected recommendation: **PC** (300 rows, Gaussian data by construction)
3. Rationale text should mention "Gaussian data" and "Fisher-Z"

✅ **Pass criteria:** Recommendation box appears and names PC.

**Test 3 — Run PC algorithm**
1. Leave algorithm selector on PC, alpha=0.05
2. Click "▶ Run Causal Discovery"
3. A spinner appears, then "✅ Completed in X.XXs"
4. Switch to "Causal Graph" tab — a DAG renders with 5 nodes (A, B, C, D, E)
5. Expected edges: A→B, A→C, B→D, C→D, C→E (exact orientation may vary)

✅ **Pass criteria:** Graph renders without error. At least 3 of the 5 expected edges are present.

**Test 4 — Adjacency matrix tab**
1. Click "Adjacency Matrix" tab
2. A 5×5 DataFrame with gradient colouring should appear

✅ **Pass criteria:** DataFrame shows, non-zero values match edges in the graph.

**Test 5 — Bootstrap confidence**
1. In the sidebar, check "Run bootstrap confidence"
2. Set N=50 (faster for local testing)
3. Set frequency threshold=0.4
4. Click "▶ Run Causal Discovery" again
5. After both point-estimate and bootstrap complete, click "Bootstrap Confidence" tab
6. Heatmap appears, top-edges table below it

✅ **Pass criteria:** Heatmap renders. Table shows at least one edge with frequency > 0.4.

**Test 6 — Export tab**
1. Click "Export" tab
2. Click "⬇ Download adjacency matrix (CSV)" — a file downloads
3. Open the CSV — confirm 5 rows × 5 columns with variable names as header/index
4. Click "⬇ Download graph image (PNG)" — PNG downloads and opens correctly

✅ **Pass criteria:** Both files download and open without corruption.

**Test 7 — DirectLiNGAM algorithm**
1. Change algorithm selector to "DirectLiNGAM — Functional (FCM)"
2. Click "▶ Run Causal Discovery"
3. Graph should render (LiNGAM recovers a full DAG, not a CPDAG)

✅ **Pass criteria:** Graph renders without error in under 10s.

**Test 8 — Error handling: empty numeric data**
1. Create a CSV with only text columns: `name,city\nAlice,Hanoi\nBob,HCMC`
2. Upload it
3. App should show a red error: "No numeric columns found"

✅ **Pass criteria:** Error message displayed, app does not crash.

### 12.3 Common local errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: causallearn` | Wrong venv active | `source .venv/bin/activate` then re-run |
| `ExecutableNotFound: dot` | graphviz binary missing | `brew install graphviz` / `apt install graphviz` |
| `RuntimeError: GLIBCXX not found` | Linux C++ ABI mismatch | `pip install --upgrade numpy scipy` |
| `st.cache_data` warning about unhashable type | pandas version mismatch | `pip install "pandas>=2.0.0"` |
| Graph tab blank (no figure) | All edges below threshold | Lower alpha slider or set freq threshold to 0 |
| `lingam` import error on Windows | Missing Microsoft C++ build tools | Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/ |
| `shapiro` warning: sample too small | <3 rows in a column | Drop or impute short columns before upload |

### 12.4 Confirming all 8 tests pass

Only proceed to deployment after **all 8 tests pass** on your local machine. Document any test failure in a comment at the top of `app.py` before asking for help.

---

## 13. Deployment to Streamlit Community Cloud

**Only start this section after Section 12 is fully complete.**

### 13.1 Add `packages.txt` for the system graphviz binary

Create `packages.txt` in the project root:

```
graphviz
```

Streamlit Community Cloud reads this file and installs the system packages before starting your app.

### 13.2 Push to GitHub

```bash
git init
git add .
git commit -m "feat: initial causal inference agent — all local tests passing"
git remote add origin https://github.com/<your-username>/causal-inference-agent.git
git push -u origin main
```

### 13.3 Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"New app"**
3. Select repo `causal-inference-agent`, branch `main`, main file `app.py`
4. Click **"Deploy"**
5. Wait ~3–5 minutes for the build to complete

✅ **Verify:** Public URL (e.g. `https://yourname-causal-inference-agent-app-xxxx.streamlit.app`) opens and the landing page renders.

### 13.4 Post-deploy smoke test

Repeat Tests 1–4 from Section 12.2 on the live URL. Skip Test 5 (bootstrap N=50) initially — run it once to confirm it doesn't OOM.

### 13.5 Memory guard for large datasets

Add this block in `app.py` immediately after `profile = profile_dataframe(df)`:

```python
_is_large = profile.n_rows * profile.n_cols > 500_000
if _is_large:
    st.warning(
        "Large dataset detected. Bootstrap is disabled to stay within "
        "Streamlit Community Cloud memory limits (~1 GB). "
        "Run locally for full bootstrap support."
    )
```

Then wrap the bootstrap sidebar checkbox:

```python
run_bootstrap_flag = st.checkbox(
    "Run bootstrap confidence",
    value=False,
    disabled=_is_large,
    help="Disabled for large datasets on cloud. Run locally for bootstrap.",
)
```

---

## 13. Known Limitations & Future Work

Document these in `README.md` so users understand scope:

| Limitation | Mitigation |
|---|---|
| Bootstrap is single-threaded | Use `concurrent.futures` with `ProcessPoolExecutor` for local runs |
| No prior knowledge (forbidden edges) | Add an edge table UI in Phase 2 |
| FGES uses GES fallback in causal-learn | Swap for pycausal / Tetrad Java bridge when needed |
| No causal effect estimation | Add EconML / DoWhy integration in Phase 2 |
| Granger only tests lag-1 and lag-2 | Expose max_lag parameter in sidebar |
| No GPU support | NOTEARS is CPU-only in causal-learn |

---

## 14. Step-by-Step Build Checklist for the Agent

Execute in strict order. Each step must pass its ✅ verify before moving to the next.

**Phase 1 — Environment (do this before any code)**

- [ ] **Step 1** — Confirm Python 3.10 or 3.11 is active. Create and activate `.venv`. ✅ `which python` points inside `.venv/`.
- [ ] **Step 2** — Install system `graphviz` binary. ✅ `dot -V` prints a version.
- [ ] **Step 3** — Create `requirements.txt`. Run `pip install -r requirements.txt`. ✅ No pip errors.
- [ ] **Step 4** — Run `python check_env.py`. ✅ All imports print ✅. Delete `check_env.py`.

**Phase 2 — Project scaffold**

- [ ] **Step 5** — Create directory structure: `core/`, `data/`, `.streamlit/`. Create `core/__init__.py` (empty). ✅ `ls core/` shows `__init__.py`.
- [ ] **Step 6** — Create `.streamlit/config.toml` with dark theme. ✅ File exists and is valid TOML.

**Phase 3 — Core modules (implement and unit-verify each)**

- [ ] **Step 7** — Implement `core/profiler.py`. ✅ Unit verify: `python -c "from core.profiler import profile_dataframe; import pandas as pd; p = profile_dataframe(pd.DataFrame({'x':[1,2,3],'y':[4,5,6]})); assert p.n_rows==3; print('OK')"` prints `OK`.
- [ ] **Step 8** — Implement `core/recommender.py`. ✅ `python -c "from core.recommender import recommend; print('OK')"` prints `OK`.
- [ ] **Step 9** — Implement `core/runner.py`. ✅ `python -c "from core.runner import ALGORITHM_REGISTRY; assert len(ALGORITHM_REGISTRY)==9; print('OK')"` prints `OK`.
- [ ] **Step 10** — Implement `core/bootstrap.py`. ✅ `python -c "from core.bootstrap import run_bootstrap; print('OK')"` prints `OK`.
- [ ] **Step 11** — Implement `core/visualiser.py`. ✅ `python -c "from core.visualiser import adjacency_to_figure; print('OK')"` prints `OK`.

**Phase 4 — App (build incrementally, verify in browser each time)**

- [ ] **Step 12** — Implement `app.py` with landing page and file upload only. ✅ `streamlit run app.py` → browser opens → landing page renders, no terminal errors.
- [ ] **Step 13** — Add profiler + recommender sections to `app.py`. ✅ Upload `data/sample.csv` → 4 metric cards appear, recommendation box visible.
- [ ] **Step 14** — Add algorithm selector + runner to `app.py`. ✅ Click "Run" → spinner → "Completed in X.XXs" → Causal Graph tab shows DAG.
- [ ] **Step 15** — Add bootstrap section. ✅ Enable bootstrap, N=50, click Run → Bootstrap Confidence tab shows heatmap and table.
- [ ] **Step 16** — Add export tab. ✅ CSV and PNG download buttons work; files open correctly.

**Phase 5 — Local testing (all 8 tests from Section 12.2)**

- [ ] **Step 17** — Generate `data/sample.csv` (Section 11 script). ✅ File exists with 300 rows and columns A–E.
- [ ] **Step 18** — Run Test 1 (upload + profile). ✅
- [ ] **Step 19** — Run Test 2 (recommendation = PC). ✅
- [ ] **Step 20** — Run Test 3 (PC graph renders, ≥3 correct edges). ✅
- [ ] **Step 21** — Run Test 4 (adjacency matrix tab). ✅
- [ ] **Step 22** — Run Test 5 (bootstrap N=50 heatmap). ✅
- [ ] **Step 23** — Run Test 6 (CSV and PNG export). ✅
- [ ] **Step 24** — Run Test 7 (DirectLiNGAM renders). ✅
- [ ] **Step 25** — Run Test 8 (text-only CSV shows error, no crash). ✅

**Phase 6 — Deployment (only after all Phase 5 steps pass)**

- [ ] **Step 26** — Add `packages.txt` with `graphviz`. Commit everything to git.
- [ ] **Step 27** — Push to GitHub. ✅ `git push` succeeds.
- [ ] **Step 28** — Deploy on Streamlit Community Cloud. ✅ Public URL loads landing page.
- [ ] **Step 29** — Re-run Tests 1–4 on the live URL. ✅ All pass.

---

## 15. File Tree Summary

```
causal-inference-agent/
├── app.py
├── core/
│   ├── __init__.py          # Empty
│   ├── profiler.py
│   ├── recommender.py
│   ├── runner.py
│   ├── bootstrap.py
│   └── visualiser.py
├── data/
│   └── sample.csv           # Generated by Section 11 script
├── requirements.txt
├── packages.txt             # graphviz system dep — only needed for Streamlit Cloud (Section 13)
├── .streamlit/
│   └── config.toml
└── README.md

# Temporary (delete after use):
# check_env.py              # Environment smoke-test from Section 3.5
```

Total Python files: **6**. Total lines (approximate): **~580**.

**Build order summary:** Environment → Scaffold → Core modules → App (incremental) → 8 local tests → Deploy.

---

*End of AGENTS.md*
