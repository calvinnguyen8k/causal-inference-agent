# Causal Inference Agent

An interactive Streamlit web app for **causal structure discovery** and **causal effect estimation**.


---

## Features

- 📂 Upload any CSV dataset
- 🤖 **Auto-recommend** the best causal discovery algorithm for your data
- 🔬 Run **15+ causal discovery algorithms** (PC, FCI, GES, LiNGAM, NOTEARS, Granger, and more)
- 📈 Visualise the resulting **DAG** with bootstrap edge-confidence overlays
- 📊 **Causal Effect Estimation** — estimate ATE with 5 estimators:
  - Regression Adjustment (OLS/Logit)
  - Inverse Probability Weighting (IPW)
  - Augmented IPW / Doubly Robust (AIPW)
  - Propensity Score Matching (PSM)
  - DoWhy + EconML (LinearDML)
- 📉 Exploratory Data Analysis — distributions, correlations, chart builder
- ⬇️ Export adjacency matrix (CSV) and graph image (PNG)

---

## Quickstart

### 1. Requirements
- Python 3.10 or 3.11
- [Graphviz](https://graphviz.org/download/) installed on your system (`brew install graphviz` on macOS)

### 2. Install

```bash
git clone https://github.com/calvinnguyen8k/causal-inference-agent.git
cd causal-inference-agent

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Optional: DoWhy + EconML estimator
pip install dowhy econml
```

### 3. Run

```bash
.venv/bin/streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
causal-inference-agent/
├── app.py                  # Streamlit UI (single entrypoint)
├── core/
│   ├── profiler.py         # Dataset profiling (dtypes, normality, sample size)
│   ├── recommender.py      # Algorithm recommendation logic
│   ├── runner.py           # Causal discovery algorithm wrappers
│   ├── bootstrap.py        # Bootstrap edge confidence
│   ├── estimation.py       # Causal effect estimation (ATE)
│   └── visualiser.py       # DAG → matplotlib rendering
├── .streamlit/
│   └── config.toml         # Dark theme configuration
└── requirements.txt
```

---

## Algorithms Supported

| Algorithm | Family | Notes |
|-----------|--------|-------|
| PC | Constraint-based | Fisher-Z CI test |
| FCI | Constraint-based | Handles latent confounders |
| GES | Score-based | BIC scoring, Gaussian |
| FGES | Score-based | Fast GES, large datasets |
| DirectLiNGAM | Functional (FCM) | Non-Gaussian data |
| ICA-LiNGAM | Functional (FCM) | ICA-based |
| NOTEARS | Continuous optimisation | L1-regularised |
| Granger Causality | Time-series | Temporal data |
| Exact Search (DP) | Score-based | Feasible for <10 vars |

---

## Tech Stack

- [Streamlit](https://streamlit.io) — UI framework
- [causal-learn](https://github.com/py-why/causal-learn) — Causal discovery algorithms
- [LiNGAM](https://github.com/cdt15/lingam) — LiNGAM family
- [DoWhy](https://github.com/py-why/dowhy) + [EconML](https://github.com/microsoft/EconML) — Causal estimation
- [NetworkX](https://networkx.org) + [Graphviz](https://graphviz.org) — Graph rendering
- [scikit-learn](https://scikit-learn.org) / [statsmodels](https://www.statsmodels.org) — ML & stats

---

## License

MIT
