# AGENTS.md — Causal Inference Agent (Streamlit)

> **Purpose:** Authoritative step-by-step instructions for an AI coding agent to understand, extend, and maintain this production Streamlit webapp.
> This file reflects the **actual built application** — every section maps directly to real source code.

---

## 0. Guiding Principles

1. **Think before coding** — state assumptions explicitly; surface tradeoffs before acting.
2. **Simplicity first** — minimum code that solves the problem. No speculative abstractions.
3. **Surgical changes** — touch only what you must. Every changed line traces to a requirement.
4. **Define success criteria per step** — each step has a `✅ Verify` section. Do not proceed until it passes.
5. **No invented features** — if a feature is not listed in this file, do not build it.

---

## 1. Project Overview

An interactive, **single-entrypoint** Streamlit webapp for causal structure discovery and causal effect estimation. Users can:

- Upload a CSV of tabular data
- Receive an automated recommendation of the best algorithm for their data
- Run any of **9 causal discovery algorithms**
- View the resulting DAG with bootstrap edge-confidence overlays
- Estimate the **Average Treatment Effect (ATE)** using 5 semiparametric estimators
- Perform exploratory data analysis with an interactive chart builder
- Export adjacency matrices and graph images

**Live demo:** https://causalinferenceagent.streamlit.app

---

## 2. Repository Structure

```
causal-inference-agent/
├── app.py                  # Single Streamlit entrypoint — all UI lives here
├── core/
│   ├── __init__.py
│   ├── profiler.py         # Dataset profiling (dtypes, normality, sample size)
│   ├── recommender.py      # Algorithm recommendation logic (causal discovery)
│   ├── runner.py           # 9 algorithm execution wrappers + ALGORITHM_REGISTRY
│   ├── bootstrap.py        # Bootstrap confidence graph computation
│   ├── visualiser.py       # DAG → matplotlib/graphviz rendering + edge table
│   ├── estimation.py       # ATE estimation: 5 estimators + ESTIMATOR_REGISTRY
│   └── notears.py          # Self-contained NOTEARS implementation
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── requirements.txt
├── .gitignore
└── README.md
```

**Do NOT** create sub-packages beyond `core/`. Do not add a `tests/` directory without explicit instruction.

---

## 3. Local Environment Setup

### 3.1 Python version

Requires **Python 3.10 or 3.11**. Check with:
```bash
python --version
```

If you have 3.12+, use `pyenv` or `conda` to install 3.11 — some causal-learn dependencies have not yet published 3.12 wheels.

### 3.2 Virtual environment

**Always use a dedicated venv.** Never install into system Python.

```bash
cd "causal-inference-agent"
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
which python                     # Must print a path ending in .venv/bin/python
```

### 3.3 System-level graphviz binary

```bash
# macOS
brew install graphviz
dot -V   # ✅ must print version string

# Ubuntu / Debian
sudo apt-get install -y graphviz
```

### 3.4 `requirements.txt`

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
dowhy>=0.11
econml>=0.15
```

Install:
```bash
pip install -r requirements.txt
```

> **Note:** `dowhy` and `econml` are optional — the app works without them for all estimators except DoWhy/EconML. If they are missing, the app shows a clear error message with install instructions.

### 3.5 Running the app

**Always launch using the venv's own streamlit binary** — do not use a globally installed `streamlit`, as it will use a different Python that lacks your installed packages.

```bash
# Correct ✅
.venv/bin/streamlit run app.py

# Incorrect ❌ (may use global Python without your packages)
streamlit run app.py
```

---

## 4. Streamlit Theme

### `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#4F46E5"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F1F5F9"
font = "sans serif"
```

---

## 5. Application Architecture

### 5.1 Navigation

`app.py` uses a sidebar `st.radio` to switch between four workspaces:

| Nav label | Workspace |
|-----------|-----------|
| 🔍 Causal Discovery | Run causal structure algorithms, view DAG, export |
| 📈 Exploratory Analysis | EDA, data preview, correlations, chart builder |
| 📊 Causal Estimation | ATE estimation with 5 estimators |
| 🤖 Causal AI Agent | Chat interface (placeholder for LLM integration) |

### 5.2 Sidebar

- **Navigation radio** (workspace selector)
- **CSV file uploader** — `st.file_uploader`, accepts `.csv` only
- Caption crediting causal-learn and LiNGAM

### 5.3 Custom CSS classes

Defined inline in `app.py` via `st.markdown(..., unsafe_allow_html=True)`:

| Class | Purpose |
|-------|---------|
| `.metric-card` | Dark card for metric display |
| `.warning-box` | Orange left-border warning panel |
| `.recommendation-box` | Indigo left-border recommendation panel |
| `.algo-card` | Small dark card for algorithm info |

---

## 6. Core Module: `core/profiler.py`

**Purpose:** Analyse the uploaded DataFrame and return a `DataProfile` dataclass consumed by the recommender and the UI.

### `DataProfile` fields

| Field | Type | Description |
|-------|------|-------------|
| `n_rows` | `int` | Row count |
| `n_cols` | `int` | Column count |
| `numeric_cols` | `List[str]` | Numeric column names |
| `categorical_cols` | `List[str]` | Non-numeric column names |
| `missing_pct` | `float` | Mean missing % across all columns (0–100) |
| `is_gaussian` | `bool` | True if ≥50% of numeric cols pass Shapiro-Wilk (p>0.05) |
| `gaussian_ratio` | `float` | Fraction of numeric cols that are Gaussian |
| `has_time_index` | `bool` | True if any datetime column or DatetimeIndex detected |
| `max_abs_correlation` | `float` | Max pairwise Pearson |r| among numeric cols |

### Key implementation details

- Shapiro-Wilk is capped at 200 samples per column (W's own limit).
- Diagonal is zeroed before taking `corr_matrix.max()`.

✅ **Verify:** `from core.profiler import profile_dataframe; import pandas as pd; p = profile_dataframe(pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})); assert p.n_rows == 3`

---

## 7. Core Module: `core/recommender.py`

**Purpose:** Map a `DataProfile` to a causal discovery algorithm recommendation with plain-English rationale.

### `Recommendation` fields

| Field | Type | Description |
|-------|------|-------------|
| `algorithm_key` | `str` | Key in `ALGORITHM_REGISTRY` |
| `rationale` | `str` | Plain-English explanation |
| `warnings` | `List[str]` | Assumption violation warnings |

### Decision logic (priority order)

1. **No numeric columns** → `pc` (chi-squared on discrete data)
2. **Time index detected** → `granger` (temporal structure)
3. **n_rows > 5000** → `fges` (scalable score-based)
4. **n_rows < 100** → add small-sample warning, continue
5. **Non-Gaussian** (gaussian_ratio < 0.5) → `direct_lingam`
6. **Default (Gaussian, moderate size)** → `pc`

### Warning conditions

- `missing_pct > 5%` → missing data warning
- `n_rows < 100` → small sample warning
- `has_time_index` → time-series warning (regardless of algorithm chosen)

✅ **Verify:** `from core.recommender import recommend; from core.profiler import DataProfile` runs without error.

---

## 8. Core Module: `core/runner.py`

**Purpose:** Thin wrappers around each algorithm. Returns a `RunResult` with an adjacency matrix.

### `RunResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `algorithm_key` | `str` | Algorithm identifier |
| `adjacency_matrix` | `np.ndarray` | Shape (n_vars, n_vars); `[i,j] != 0` means i → j |
| `column_names` | `List[str]` | Variable names |
| `runtime_seconds` | `float` | Wall-clock time |
| `extra` | `dict` | Algorithm-specific extras (default empty) |

### `ALGORITHM_REGISTRY` (9 algorithms)

| Key | Name | Family | Gaussian required |
|-----|------|--------|------------------|
| `pc` | PC | Constraint-based | No |
| `fci` | FCI | Constraint-based | No |
| `ges` | GES | Score-based | Yes |
| `fges` | FGES | Score-based | Yes |
| `direct_lingam` | DirectLiNGAM | Functional (FCM) | No |
| `ica_lingam` | ICA-LiNGAM | Functional (FCM) | No |
| `notears` | NOTEARS | Continuous optimisation | No |
| `granger` | Granger Causality | Time-series | No |
| `exact_search` | Exact Search (DP) | Score-based | Yes |

### Implementation notes

- **Only numeric columns** are used; NaN rows are dropped before fitting.
- `_cg_to_adj(G, n_vars)` converts causal-learn `GeneralGraph` objects to `np.ndarray`. Has a fallback to `GraphUtils.to_adjacency_matrix` if node iteration fails.
- **NOTEARS** uses `core/notears.py` (self-contained implementation) — not the causal-learn version — to avoid import path issues.
- **Granger**: pairwise F-test (lag=2). Edge weight = `1.0 - p_value` when significant. Diagonal is skipped.
- **FGES**: implemented via `ges(X, score_func='local_score_BIC')` (causal-learn does not expose FGES as a separate function).

✅ **Verify:** `from core.runner import ALGORITHM_REGISTRY; assert len(ALGORITHM_REGISTRY) == 9`

---

## 9. Core Module: `core/bootstrap.py`

**Purpose:** Run N bootstrap resamples and aggregate edge frequencies.

### `BootstrapResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `edge_frequency` | `np.ndarray` | (n_vars, n_vars) fraction of runs where edge appeared |
| `column_names` | `List[str]` | Variable names |
| `n_resamples` | `int` | Number of **successful** resamples (failed ones silently skipped) |
| `algorithm_key` | `str` | Algorithm used |

### Key details

- Samples `n_rows` rows with replacement each iteration.
- Binarises adjacency: any `abs(val) > 1e-8` counts as edge present.
- Failed resamples (algorithm crash) are silently skipped; `n_resamples` reflects only successes.
- Final `edge_frequency` is normalised by `success_count`, not total `n_resamples`.

✅ **Verify:** Module imports without error.

---

## 10. Core Module: `core/visualiser.py`

**Purpose:** Render an adjacency matrix + optional edge frequencies into a matplotlib figure.

### Public functions

#### `adjacency_to_figure(adj, col_names, edge_freq, freq_threshold, title)`

**Dual visual encoding:**
- **Edge colour** → connection strength (absolute adjacency weight): grey → amber → red
- **Edge width** → bootstrap confidence: thin = uncertain, thick = high confidence
- When all edges have binary weight (e.g., PC/GES outputs 0/1), colour falls back to bootstrap frequency so there is still visual variation.

**Layout priority:**
1. `nx.nx_agraph.graphviz_layout(G, prog="dot")` — best for DAGs
2. `nx.kamada_kawai_layout` — fallback if graphviz unavailable
3. `nx.spring_layout` — last resort

**Node colour:** Hue shifts from indigo (#4F46E5) to blue (#6366F1) based on out-degree — nodes with more outgoing edges appear more saturated.

**Bidirectional edges** get increased arc radius (`rad=0.25`) to avoid overlap.

#### `edge_frequency_heatmap(freq, col_names)`

- `YlOrRd` colormap, vmin=0, vmax=1
- Annotates cells with frequency value when > 0.05
- Dark text on high-frequency cells, white on low

#### `get_edge_table(adj, col_names, edge_freq, freq_threshold)`

Returns a `pd.DataFrame` with columns: `From`, `→`, `To`, `Edge weight`, `Bootstrap conf.`

- Sorted by bootstrap confidence descending (if available), else by abs(weight)
- Bootstrap conf. formatted as percentage string or `"—"` if no bootstrap

---

## 11. Core Module: `core/estimation.py`

**Purpose:** Estimate the Average Treatment Effect (ATE) of a treatment variable on an outcome, adjusting for confounders.

### `EstimationResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `ate` | `float` | Point estimate of ATE |
| `se` | `float` | Standard error |
| `p_value` | `float` | Two-tailed z-test p-value |
| `ci_lower` | `float` | 95% CI lower bound |
| `ci_upper` | `float` | 95% CI upper bound |
| `summary_text` | `str` | Full text summary for expander |
| `is_binary_outcome` | `bool` | True if outcome has exactly 2 unique values in {0,1} |
| `is_categorical_treatment` | `bool` | True if treatment is non-numeric or ≤20 unique values |
| `interpretation` | `str` | Human-readable result sentence |
| `estimator_used` | `str` | Estimator key used |
| `bootstrap_ates` | `Optional[np.ndarray]` | Bootstrap ATE samples if bootstrap was run |

### `ESTIMATOR_REGISTRY` (5 estimators)

| Key | Name | Extra deps | Best when |
|-----|------|-----------|-----------|
| `regression` | Regression Adjustment (OLS/Logit) | None | Small data, interpretability |
| `ipw` | Inverse Probability Weighting | scikit-learn | Binary treatment, good PS model |
| `aipw` | Augmented IPW / Doubly Robust | scikit-learn | Many confounders, robustness |
| `matching` | Propensity Score Matching | scikit-learn | Binary treatment, matched pairs |
| `dowhy` | DoWhy + EconML (LinearDML) | dowhy, econml | Large data, refutation tests |

### Estimator recommendation logic (`recommend_estimator`)

Priority:
1. **n_rows < 50** → `regression` (most stable for tiny data)
2. **No confounders** → `regression` (IPW is overkill)
3. **n_rows ≥ 500 AND n_confounders ≥ 3** → `aipw`
4. **Binary treatment AND n_rows ≥ 200** → `ipw`
5. **Binary treatment AND n_rows < 200** → `matching`
6. **n_confounders ≥ 2** → `aipw`
7. **Default** → `regression`

### Categorical treatment handling

When `is_categorical=True` and `control_value`/`treatment_value` are specified:
- Rows are filtered to only these two values
- A binary `_treatment_encoded` column (0/1) is created

### Continuous treatment binarisation

For IPW/AIPW/Matching with continuous treatment: automatically binarised at the **median** with a user-facing warning.

### Interpretation text format

```
[EstimatorName] A 1-unit increase in treatment *decreases/increases* outcome
by X units on average (95% CI: [lo, hi]).
This result is ✅/⚠️ statistically significant/not significant (p = X).
```

- Binary outcome: effect stated in **percentage points (pp)**, multiplied by 100.
- Continuous outcome: effect stated in **raw units**.
- "A **1-unit increase**" is always used (not "a unit change") to be unambiguous.
- Effect direction (`decreases`/`increases`) is *italicised* in the Streamlit markdown render.

### DoWhy scalar extraction

Newer DoWhy/EconML (≥0.11/≥0.15) returns **numpy arrays** for `estimate.value` and CI bounds, not plain Python floats. The `_to_scalar(v)` helper uses `np.asarray(v).flatten()[0]` to safely extract a scalar.

CI fallback chain:
1. `estimate.get_confidence_intervals()` → flatten to scalar
2. If fails → `estimate.get_standard_error()` → compute `ATE ± 1.96 × SE`
3. If fails → `ATE ± 10% of ATE` (last resort)

### Bootstrap for non-regression estimators

When `n_bootstrap > 0` and estimator is `ipw`/`aipw`/`matching`:
- Re-runs full estimation pipeline on each bootstrap sample
- Final SE = `std(boot_ates, ddof=1)`; CI = `[2.5th, 97.5th percentile]`
- p-value recomputed from bootstrap SE

---

## 12. Core Module: `core/notears.py`

**Purpose:** Self-contained NOTEARS (No Tears) continuous DAG optimisation implementation.

- Avoids dependency on `causallearn.search.FCMBased.notears` which has import path issues in some versions.
- L1-regularised (`lambda1=0.1`), L2 loss by default.
- Called by `runner._run_notears(X)`.

---

## 13. App Workspace: Causal Discovery (`🔍`)

**File:** `app.py` lines ~135–409

### Flow

1. Load CSV via `load_data()` (`@st.cache_data`)
2. Profile with `profile_dataframe(df)`
3. Display 4 metric columns: Rows, Columns, Numeric cols, Missing %
4. Show data preview (`st.expander`)
5. Validate: require ≥2 numeric columns
6. Recommend algorithm with `recommend(profile)`
7. Show warnings + recommendation box
8. Algorithm selector dropdown (defaults to recommended)
9. Gaussian warning if selected algorithm requires Gaussian but data isn't
10. Alpha slider (0.01–0.20, default 0.05)
11. Bootstrap checkbox + settings (N resamples, frequency threshold)
12. "▶ Run Causal Discovery" button

### Results (4 tabs)

| Tab | Contents |
|-----|---------|
| Causal Graph | `adjacency_to_figure()` rendered via `st.pyplot` |
| Adjacency Matrix | `pd.DataFrame` with `background_gradient(cmap="Blues")` |
| Bootstrap Confidence | `edge_frequency_heatmap()` + top edges table |
| Export | Download adjacency CSV + graph PNG + bootstrap frequency CSV |

### Directed Connections table

Colour-coded by bootstrap confidence:
- ≥80%: green background (`#14532D`)
- ≥50%: blue background (`#1E3A5F`)
- >0%: dark grey (`#1E293B`)

---

## 14. App Workspace: Exploratory Analysis (`📈`)

**File:** `app.py` lines ~411–656

### Sections

1. **Dataset Overview** — 5 metrics: Rows, Columns, Numeric, Categorical, Missing %
2. **Data Preview** — slider for row count, download full CSV
3. **Data Summary** — `df.describe(include="all").T` in expander + correlation matrix
4. **Data Preprocessing** — missing value strategy + normalisation (Min-Max or Z-score), preview + download
5. **Visualization (Chart Builder)** — 5 chart types

### Chart Builder

| Chart type | Controls |
|-----------|---------|
| Scatter Plot | X var, Y var, optional colour/group variable |
| Histogram | Variable, bins slider (5–100) |
| Box Plot | Multi-select variables, optional group by |
| Line Chart | X axis, multi-select Y variables |
| Bar Chart | Category X, numeric Y (aggregated by mean) |

All charts use dark theme (`#1E293B` background).

### Known matplotlib compatibility fixes

| Bug | Fix applied |
|-----|------------|
| `ax.boxplot(..., labels=...)` removed in matplotlib ≥3.9 | Try `tick_labels=` first, fallback to `labels=` via `TypeError` catch |
| `plt.cm.get_cmap("tab10", N)` deprecated in matplotlib ≥3.7 | Replaced with `plt.colormaps["tab10"].resampled(N)` |

---

## 15. App Workspace: Causal Estimation (`📊`)

**File:** `app.py` lines ~659–885

### Configuration flow

1. **Treatment** — dropdown from all columns; detect if categorical (≤20 unique values)
2. **Categorical treatment** — two extra dropdowns: Control Value (0) and Treatment Value (1)
3. **Outcome** — dropdown excluding treatment
4. **Confounders** — multi-select of remaining columns
5. **Estimator recommendation** — `recommend_estimator(df, treatment, outcome, confounders)`
6. **Estimator selector** — dropdown defaulting to recommendation
7. **Estimator detail card** — shows description, best-when, assumption, extra deps
8. **DoWhy warning** — shown if `dowhy` selected (advises pip install)
9. **Non-binary treatment warning** — shown for IPW/AIPW/Matching with continuous treatment
10. **Estimator comparison table** — expandable; all 5 estimators side-by-side
11. **Bootstrap** — checkbox + N iterations input
12. "▶ Run Causal Analysis" button

### Results display

- **4 metric columns:** ATE, SE, 95% CI, Significance (✅/⚠️ with p-value delta)
- **`st.info(result.interpretation)`** — plain-English result with bold formatting
- **Bootstrap ATE distribution** — histogram with ATE line and CI bounds (shown if bootstrap ran)
- **Full Regression Summary expander** — raw statsmodels/IPW/AIPW text summary

---

## 16. App Workspace: Causal AI Agent (`🤖`)

**File:** `app.py` lines ~887–915

**Status:** Placeholder only. Implements a basic `st.chat_input` + `st.chat_message` loop. The agent response is a fixed string noting that full LLM integration is planned.

**To extend:** Replace the mock response with an actual LLM call (e.g., Gemini, GPT-4) using the causal inference context.

---

## 17. Known Issues & Fixes Applied

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `boxplot() got unexpected kwarg 'labels'` | matplotlib ≥3.9 renamed param | `try tick_labels= except TypeError: labels=` |
| `plt.cm.get_cmap()` deprecated | matplotlib ≥3.7 | `plt.colormaps["tab10"].resampled(N)` |
| DoWhy: "only 0-dimensional arrays can be converted to Python scalars" | DoWhy ≥0.11 returns arrays not floats | `_to_scalar()` helper using `np.asarray().flatten()[0]` |
| Streamlit running wrong Python (no dowhy/econml) | Global `streamlit` binary used instead of venv's | Always run `.venv/bin/streamlit run app.py` |
| `ATE interpretation ambiguous ("a unit change")` | Phrase doesn't specify direction of treatment change | Changed to "A **1-unit increase** in... *decreases/increases*..." |

---

## 18. Deployment

### Local (development)

```bash
cd "causal-inference-agent"
source .venv/bin/activate
.venv/bin/streamlit run app.py
# Open http://localhost:8501
```

### Streamlit Community Cloud

1. Push to GitHub (all files except `.venv/`, data files, secrets)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set: repo = `calvinnguyen8k/causal-inference-agent`, branch = `main`, main file = `app.py`
4. Requirements are read automatically from `requirements.txt`
5. **Note:** `graphviz` system binary must be available — Streamlit Cloud has it pre-installed.
6. `dowhy` and `econml` in `requirements.txt` will be installed automatically.

### `.gitignore` — what is excluded

```
.venv/           # Virtual environment (never commit)
__pycache__/     # Python bytecode
*.csv            # Data files (user uploads, test data)
data/ files/     # Local data directories
*.zip            # Large archives
.DS_Store        # macOS metadata
.streamlit/secrets.toml  # Secrets (if added later)
```

---

## 19. Git Workflow

```bash
# First push (after creating repo on GitHub)
git init
git add .
git commit -m "Initial commit: Causal Inference Agent"
git remote add origin https://github.com/calvinnguyen8k/causal-inference-agent.git
git push -u origin main

# Subsequent pushes
git add -A
git commit -m "feat: describe your change"
git push

# If macOS Keychain overrides credentials (403 error):
git credential-osxkeychain erase  # then press Enter twice
git remote set-url origin https://calvinnguyen8k:YOUR_TOKEN@github.com/calvinnguyen8k/causal-inference-agent.git
```

**PAT security:** Never paste GitHub Personal Access Tokens into chat or commit them to the repo. Generate fresh tokens at [github.com/settings/tokens](https://github.com/settings/tokens) with `repo` scope only.

---

## 20. Adding New Features (Agent Instructions)

### Adding a new causal discovery algorithm

1. Add entry to `ALGORITHM_REGISTRY` in `core/runner.py`
2. Add `elif algorithm_key == "new_algo":` branch in `run_algorithm()`
3. Implement `_run_new_algo(X)` returning `np.ndarray` of shape `(n_vars, n_vars)`
4. Update recommendation logic in `core/recommender.py` if it should be auto-recommended
5. **Do not** modify `app.py` — the UI picks up new algorithms automatically from `ALGORITHM_REGISTRY`

### Adding a new ATE estimator

1. Add entry to `ESTIMATOR_REGISTRY` in `core/estimation.py`
2. Implement `_estimate_new(data, treatment_col, outcome, confounders)` returning `(ate, se, summary)`
3. Add `elif estimator == "new":` branch in `estimate_effect()`
4. Add z-test p-value and CI calculation consistent with other branches
5. Update `recommend_estimator()` if applicable
6. **Do not** modify `app.py` — the UI picks up new estimators automatically from `ESTIMATOR_REGISTRY`

### Adding a new EDA chart type

1. Add the chart name to the `chart_type` selectbox options in the Exploratory Analysis section of `app.py`
2. Add an `elif chart_type == "New Chart":` block following the existing pattern
3. Always: set `fig.patch.set_facecolor("#1E293B")`, `ax.set_facecolor("#1E293B")`, `plt.tight_layout()`, `plt.close(fig)`

---

## 21. Dependency Compatibility Matrix

| Package | Min version | Notes |
|---------|------------|-------|
| streamlit | 1.35.0 | `st.cache_data`, `st.file_uploader` API |
| matplotlib | 3.8.0 | Use `tick_labels=` for boxplot; use `plt.colormaps[]` for cmap |
| causal-learn | 0.1.3.8 | `PC`, `FCI`, `GES`, `ExactSearch` APIs |
| lingam | 1.8.3 | `DirectLiNGAM`, `ICALiNGAM` |
| dowhy | 0.11+ | `estimate.value` returns array — use `_to_scalar()` |
| econml | 0.15+ | LinearDML estimator |
| networkx | 3.2.0 | `min_source_margin`, `min_target_margin` in `draw_networkx_edges` |
| Python | 3.10–3.11 | 3.12 not yet supported by all causal-learn wheels |
