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
from core.visualiser import adjacency_to_figure, edge_frequency_heatmap, get_edge_table
from core.estimation import estimate_effect, ESTIMATOR_REGISTRY, recommend_estimator

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


# ── Landing page helper ───────────────────────────────────────────────────────
# IMPORTANT: Must be defined before the `if uploaded_file is None:` check below.
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔗 Causal Inference Agent")
    st.caption("Automated causal structure & effect analysis")
    st.divider()

    st.subheader("🧭 Navigation")
    nav_selection = st.radio(
        "Choose a workspace:",
        ["🔍 Causal Discovery", "📈 Exploratory Analysis", "📊 Causal Estimation", "🤖 Causal AI Agent"],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("📂 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset (CSV)",
        type=["csv"],
        help="CSV file containing tabular data (rows = samples, cols = variables)."
    )
    
    if uploaded_file is None:
        import os
        if os.path.exists("test_data.csv"):
            uploaded_file = "test_data.csv"
        else:
            _show_landing()
            st.stop()

    st.divider()
    st.caption("Built with [causal-learn](https://github.com/py-why/causal-learn) & [LiNGAM](https://github.com/cdt15/lingam)")


# ── Main area ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

if nav_selection == "🔍 Causal Discovery":
    st.title("Causal Discovery")
    st.markdown("Discover causal structure in your data — automatically.")

    if uploaded_file is None:
        _show_landing()
        st.stop()

    # ── Load & profile data ───────────────────────────────────────────────────────
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

        # ── Text connections table ────────────────────────────────────────────────
        st.subheader("🔗 Directed Connections")
        edge_tbl = get_edge_table(
            adj=run_result.adjacency_matrix,
            col_names=run_result.column_names,
            edge_freq=edge_freq,
            freq_threshold=threshold,
        )
        if edge_tbl.empty:
            st.info("No edges above the current threshold.")
        else:
            # Render a styled table
            st.markdown(
                "Each row is a **directed causal link** discovered by the algorithm. "
                "*Edge weight* is the raw coefficient/strength; "
                "*Bootstrap conf.* shows how often this edge appeared across resamples."
            )
            # Colour-code by bootstrap confidence when available
            if edge_freq is not None:
                def _colour_conf(val):
                    try:
                        pct = float(val.strip("%")) / 100
                        if pct >= 0.8:
                            return "background-color:#14532D; color:#86EFAC"
                        elif pct >= 0.5:
                            return "background-color:#1E3A5F; color:#93C5FD"
                        elif pct > 0:
                            return "background-color:#1E293B; color:#CBD5E1"
                        return ""
                    except Exception:
                        return ""
                st.dataframe(
                    edge_tbl.style.map(_colour_conf, subset=["Bootstrap conf."]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(edge_tbl, use_container_width=True, hide_index=True)

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


elif nav_selection == "📈 Exploratory Analysis":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Understand your dataset before running causal analysis — inspect distributions, correlations, and visualise relationships.")

    if uploaded_file is None:
        st.info("⬆ Upload a CSV file in the sidebar to get started.")
    else:
        with st.spinner("Loading data…"):
            df = load_data(uploaded_file)

        profile = profile_dataframe(df)
        cols = df.columns.tolist()
        num_cols = profile.numeric_cols

        # ── 1. Dataset Metrics ────────────────────────────────────────────────
        st.subheader("1. Dataset Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{profile.n_rows:,}")
        c2.metric("Columns", f"{profile.n_cols}")
        c3.metric("Numeric", f"{len(num_cols)}")
        c4.metric("Categorical", f"{len(profile.categorical_cols)}")
        c5.metric("Missing %", f"{profile.missing_pct:.1f}%")

        # ── 2. Data Preview ───────────────────────────────────────────────────
        st.divider()
        st.subheader("2. Data Preview")
        preview_n = st.slider("Rows to preview", min_value=5, max_value=min(500, profile.n_rows), value=min(20, profile.n_rows), step=5)
        st.dataframe(df.head(preview_n), use_container_width=True)

        # Download full dataset
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("⬇ Download full dataset (CSV)", data=csv_buf.getvalue(), file_name="dataset.csv", mime="text/csv")

        # ── 3. Data Summary ───────────────────────────────────────────────────
        st.divider()
        st.subheader("3. Data Summary")

        with st.expander("📋 Show Summary Statistics", expanded=False):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        with st.expander("🔗 Show Correlation Matrix", expanded=False):
            if len(num_cols) >= 2:
                import matplotlib.pyplot as plt
                corr = df[num_cols].corr()
                fig_corr, ax_corr = plt.subplots(figsize=(max(6, len(num_cols)), max(5, len(num_cols) - 1)))
                fig_corr.patch.set_facecolor("#1E293B")
                ax_corr.set_facecolor("#1E293B")
                import matplotlib.colors as mcolors
                cmap = plt.cm.RdYlGn
                im = ax_corr.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
                ax_corr.set_xticks(range(len(num_cols)))
                ax_corr.set_yticks(range(len(num_cols)))
                ax_corr.set_xticklabels(num_cols, rotation=45, ha="right", color="white", fontsize=8)
                ax_corr.set_yticklabels(num_cols, color="white", fontsize=8)
                for i in range(len(num_cols)):
                    for j in range(len(num_cols)):
                        ax_corr.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                                     color="black" if abs(corr.values[i, j]) > 0.3 else "white", fontsize=7)
                cbar = plt.colorbar(im, ax=ax_corr, shrink=0.8)
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
                cbar.set_label("Pearson r", color="white")
                ax_corr.set_title("Correlation Matrix", color="white", pad=12)
                plt.tight_layout()
                st.pyplot(fig_corr, use_container_width=True)
                plt.close(fig_corr)

                # Highlight top correlations
                pairs = []
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        pairs.append({"Variable A": num_cols[i], "Variable B": num_cols[j],
                                      "Correlation": round(corr.values[i, j], 4)})
                if pairs:
                    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
                    st.markdown("**Top correlated pairs:**")
                    st.dataframe(pairs_df.head(10), use_container_width=True, hide_index=True)
            else:
                st.info("Need at least 2 numeric columns for a correlation matrix.")

        # ── 4. Data Preprocessing ─────────────────────────────────────────────
        st.divider()
        st.subheader("4. Data Preprocessing")
        with st.expander("⚙️ Preprocessing Options", expanded=False):
            col_prep1, col_prep2 = st.columns(2)
            with col_prep1:
                missing_strategy = st.selectbox(
                    "Handle missing values",
                    ["None (keep as-is)", "Drop rows with missing", "Fill with column mean", "Fill with column median"],
                    index=0,
                )
            with col_prep2:
                normalize = st.selectbox(
                    "Normalise numeric columns",
                    ["None", "Min-Max (0–1)", "Z-score (standardise)"],
                    index=0,
                )

            if st.button("Apply Preprocessing & Preview", key="apply_preproc"):
                df_proc = df.copy()
                if missing_strategy == "Drop rows with missing":
                    df_proc = df_proc.dropna()
                elif missing_strategy == "Fill with column mean":
                    df_proc[num_cols] = df_proc[num_cols].fillna(df_proc[num_cols].mean())
                elif missing_strategy == "Fill with column median":
                    df_proc[num_cols] = df_proc[num_cols].fillna(df_proc[num_cols].median())

                if normalize == "Min-Max (0–1)" and num_cols:
                    mn = df_proc[num_cols].min()
                    rng = df_proc[num_cols].max() - mn
                    rng = rng.replace(0, 1)  # avoid division by zero
                    df_proc[num_cols] = (df_proc[num_cols] - mn) / rng
                elif normalize == "Z-score (standardise)" and num_cols:
                    df_proc[num_cols] = (df_proc[num_cols] - df_proc[num_cols].mean()) / df_proc[num_cols].std()

                st.success(f"After preprocessing: {len(df_proc)} rows × {len(df_proc.columns)} columns")
                st.dataframe(df_proc.head(20), use_container_width=True)

                proc_buf = io.StringIO()
                df_proc.to_csv(proc_buf, index=False)
                st.download_button("⬇ Download preprocessed dataset (CSV)", data=proc_buf.getvalue(),
                                   file_name="preprocessed_dataset.csv", mime="text/csv")

        # ── 5. Chart Builder ──────────────────────────────────────────────────
        st.divider()
        st.subheader("5. Visualization (Chart Builder)")

        chart_type = st.selectbox(
            "Chart Type",
            ["Scatter Plot", "Histogram", "Box Plot", "Line Chart", "Bar Chart"],
            index=0,
        )

        import matplotlib.pyplot as plt

        if chart_type == "Scatter Plot":
            c1, c2, c3 = st.columns(3)
            with c1:
                x_var = st.selectbox("X Variable", options=num_cols, index=0, key="sc_x")
            with c2:
                y_opts = [c for c in num_cols if c != x_var]
                y_var = st.selectbox("Y Variable(s)", options=y_opts, index=0, key="sc_y")
            with c3:
                color_opts = ["None"] + cols
                color_var = st.selectbox("Color/Group (Optional)", options=color_opts, index=0, key="sc_col")

            if st.button("📊 Generate Chart", key="gen_scatter"):
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("#1E293B"); ax.set_facecolor("#1E293B")
                if color_var != "None" and color_var in df.columns:
                    groups = df[color_var].unique()
                    cmap_g = plt.colormaps["tab10"].resampled(len(groups))
                    for idx, grp in enumerate(groups):
                        mask = df[color_var] == grp
                        ax.scatter(df.loc[mask, x_var], df.loc[mask, y_var],
                                   label=str(grp), color=cmap_g(idx), alpha=0.7, s=30)
                    ax.legend(facecolor="#1E293B", labelcolor="white")
                else:
                    ax.scatter(df[x_var], df[y_var], color="#4F46E5", alpha=0.7, s=30)
                ax.set_xlabel(x_var, color="white"); ax.set_ylabel(y_var, color="white")
                ax.tick_params(colors="white"); ax.spines[:].set_color("#334155")
                ax.set_title(f"{x_var} vs {y_var}", color="white")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        elif chart_type == "Histogram":
            c1, c2 = st.columns(2)
            with c1:
                hist_var = st.selectbox("Variable", options=num_cols, index=0, key="hist_v")
            with c2:
                hist_bins = st.slider("Bins", 5, 100, 30, key="hist_bins")

            if st.button("📊 Generate Chart", key="gen_hist"):
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor("#1E293B"); ax.set_facecolor("#1E293B")
                ax.hist(df[hist_var].dropna(), bins=hist_bins, color="#4F46E5", alpha=0.85, edgecolor="#1E293B")
                ax.set_xlabel(hist_var, color="white"); ax.set_ylabel("Frequency", color="white")
                ax.tick_params(colors="white"); ax.spines[:].set_color("#334155")
                ax.set_title(f"Distribution of {hist_var}", color="white")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        elif chart_type == "Box Plot":
            c1, c2 = st.columns(2)
            with c1:
                box_vars = st.multiselect("Variables", options=num_cols, default=num_cols[:min(4, len(num_cols))], key="box_v")
            with c2:
                group_var = st.selectbox("Group By (Optional)", options=["None"] + cols, index=0, key="box_grp")

            if st.button("📊 Generate Chart", key="gen_box") and box_vars:
                fig, ax = plt.subplots(figsize=(max(6, len(box_vars) * 1.5), 5))
                fig.patch.set_facecolor("#1E293B"); ax.set_facecolor("#1E293B")
                data_to_plot = [df[v].dropna().values for v in box_vars]
                try:
                    bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=box_vars)
                except TypeError:
                    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=box_vars)
                colors = ["#4F46E5", "#7C3AED", "#2563EB", "#0EA5E9", "#06B6D4"]
                for patch, color in zip(bp["boxes"], colors * 10):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                for element in ["whiskers", "caps", "medians"]:
                    for line in bp[element]:
                        line.set_color("white")
                ax.tick_params(colors="white", axis="x", rotation=30)
                ax.tick_params(colors="white", axis="y")
                ax.spines[:].set_color("#334155")
                ax.set_title("Box Plots", color="white")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        elif chart_type == "Line Chart":
            c1, c2 = st.columns(2)
            with c1:
                line_x = st.selectbox("X Axis", options=cols, index=0, key="line_x")
            with c2:
                line_y_opts = [c for c in num_cols if c != line_x]
                line_ys = st.multiselect("Y Variable(s)", options=line_y_opts, default=line_y_opts[:min(2, len(line_y_opts))], key="line_ys")

            if st.button("📊 Generate Chart", key="gen_line") and line_ys:
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor("#1E293B"); ax.set_facecolor("#1E293B")
                line_colors = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6"]
                for idx, yv in enumerate(line_ys):
                    ax.plot(df[line_x], df[yv], label=yv, color=line_colors[idx % len(line_colors)], linewidth=1.5)
                ax.legend(facecolor="#1E293B", labelcolor="white")
                ax.set_xlabel(line_x, color="white"); ax.set_ylabel("Value", color="white")
                ax.tick_params(colors="white", axis="x", rotation=30)
                ax.tick_params(colors="white", axis="y")
                ax.spines[:].set_color("#334155")
                ax.set_title("Line Chart", color="white")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        elif chart_type == "Bar Chart":
            c1, c2 = st.columns(2)
            with c1:
                bar_x = st.selectbox("Category (X)", options=cols, index=0, key="bar_x")
            with c2:
                bar_y = st.selectbox("Value (Y)", options=num_cols, index=0, key="bar_y")

            if st.button("📊 Generate Chart", key="gen_bar"):
                agg = df.groupby(bar_x)[bar_y].mean().reset_index().sort_values(bar_y, ascending=False)
                fig, ax = plt.subplots(figsize=(max(6, len(agg) * 0.5), 5))
                fig.patch.set_facecolor("#1E293B"); ax.set_facecolor("#1E293B")
                ax.bar(agg[bar_x].astype(str), agg[bar_y], color="#4F46E5", alpha=0.85)
                ax.set_xlabel(bar_x, color="white"); ax.set_ylabel(f"Mean {bar_y}", color="white")
                ax.tick_params(colors="white", axis="x", rotation=45)
                ax.tick_params(colors="white", axis="y")
                ax.spines[:].set_color("#334155")
                ax.set_title(f"Average {bar_y} by {bar_x}", color="white")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)


elif nav_selection == "📊 Causal Estimation":
    st.title("📊 Causal Analysis")
    st.markdown(
        "Estimate the **Average Treatment Effect (ATE)** of an action on an outcome, "
        "adjusting for confounders. Choose from 5 estimators below."
    )

    if uploaded_file is None:
        st.info("⬆ Upload a CSV file in the sidebar to get started.")
    else:
        with st.spinner("Loading data…"):
            df = load_data(uploaded_file)

        cols = df.columns.tolist()

        st.divider()
        st.subheader("⚙️ Configuration Guide")

        # ── Treatment selection ───────────────────────────────────────────────
        treatment = st.selectbox("Treatment (Action)", options=cols, index=0, key="est_treatment")

        # Detect categorical / binary treatment
        n_unique = df[treatment].nunique()
        is_cat = (
            not pd.api.types.is_numeric_dtype(df[treatment])
            or n_unique <= 20
        )

        control_value = None
        treatment_value = None

        if is_cat:
            unique_vals = sorted(df[treatment].dropna().unique().tolist(), key=str)
            st.info(f"ℹ️ Detected categorical treatment: **{treatment}** ({n_unique} unique values). Encoding as binary.")
            c1, c2 = st.columns(2)
            with c1:
                ctrl_key = f"ctrl_val_{treatment}"
                control_value = st.selectbox(
                    "Control Value (0)", options=unique_vals, index=0, key=ctrl_key,
                )
            with c2:
                remaining = [v for v in unique_vals if v != control_value]
                if not remaining:
                    remaining = unique_vals
                treat_key = f"treat_val_{treatment}"
                treatment_value = st.selectbox(
                    "Treatment Value (1)", options=remaining, index=0, key=treat_key,
                )

        # ── Outcome ───────────────────────────────────────────────────────────
        outcome = st.selectbox(
            "Outcome (Result)",
            options=[c for c in cols if c != treatment],
            index=0,
        )

        # ── Confounders ───────────────────────────────────────────────────────
        confounders = st.multiselect(
            "Confounders (Common Causes)",
            options=[c for c in cols if c not in [treatment, outcome]],
            default=[],
            help="Variables that causally affect both the treatment and the outcome.",
        )

        # ── Estimator selection ───────────────────────────────────────────────
        st.divider()
        st.subheader("🔬 Estimator Selection")

        # Auto-recommendation
        rec = recommend_estimator(df, treatment, outcome, list(confounders))
        rec_info = ESTIMATOR_REGISTRY[rec.estimator_key]

        # Show recommendation box
        st.markdown(
            f'<div class="recommendation-box">'
            f'<strong>🤖 Recommended: {rec_info["name"]}</strong><br>'
            f'{rec.rationale}'
            f'</div>',
            unsafe_allow_html=True,
        )
        for w in rec.warnings:
            st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

        # Estimator dropdown
        estimator_options = {k: v["name"] for k, v in ESTIMATOR_REGISTRY.items()}
        default_est_idx = list(estimator_options.keys()).index(rec.estimator_key)

        selected_estimator = st.selectbox(
            "Choose estimator (override recommendation if desired)",
            options=list(estimator_options.keys()),
            format_func=lambda k: estimator_options[k],
            index=default_est_idx,
            key="est_estimator",
        )

        # Show selected estimator details
        sel_info = ESTIMATOR_REGISTRY[selected_estimator]
        req_str = ", ".join(sel_info["requires"]) if sel_info["requires"] else "None (built-in)"

        st.markdown(
            f'<div class="algo-card">'
            f'<strong>{sel_info["name"]}</strong><br>'
            f'<em>{sel_info["description"]}</em><br><br>'
            f'📌 <strong>Best when:</strong> {sel_info["best_when"]}<br>'
            f'⚠️ <strong>Key assumption:</strong> {sel_info["assumption"]}<br>'
            f'📦 <strong>Extra dependencies:</strong> {req_str}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # DoWhy warning
        if selected_estimator == "dowhy":
            st.warning(
                "⚠️ DoWhy + EconML require extra packages: `pip install dowhy econml`. "
                "If not installed, this will raise an ImportError. All other estimators work out of the box."
            )

        # Non-binary treatment warning for IPW/AIPW/Matching
        if selected_estimator in ("ipw", "aipw", "matching"):
            t_vals = df[treatment].dropna().unique()
            if not (set(t_vals).issubset({0, 1}) or (is_cat and control_value is not None)):
                st.warning(
                    f"⚠️ **{sel_info['short']}** works best with a binary (0/1) treatment. "
                    "For continuous treatments, the app will binarise at the median automatically."
                )

        # ── Estimator comparison table ────────────────────────────────────────
        with st.expander("📊 Compare All Estimators", expanded=False):
            comp_data = []
            for k, v in ESTIMATOR_REGISTRY.items():
                comp_data.append({
                    "Estimator": v["short"],
                    "Full Name": v["name"],
                    "Best When": v["best_when"],
                    "Key Assumption": v["assumption"],
                    "Extra deps": ", ".join(v["requires"]) or "None",
                    "Recommended ✓": "✅" if k == rec.estimator_key else "",
                })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # ── Bootstrap options ─────────────────────────────────────────────────
        st.divider()
        enable_bootstrap = st.checkbox(
            "Enable Bootstrapping (Calculate Standard Errors)",
            value=(selected_estimator == "regression"),
            help=(
                "For Regression: re-fits the model on N bootstrap samples for robust CIs. "
                "For IPW/AIPW/Matching: bootstraps the entire estimation pipeline. "
                "DoWhy always uses analytical CIs."
            ),
        )
        n_bootstrap = 0
        if enable_bootstrap and selected_estimator != "dowhy":
            n_bootstrap = st.number_input(
                "Bootstrap Iterations", min_value=10, max_value=500, value=50, step=10
            )

        st.divider()

        if st.button("▶ Run Causal Analysis", type="primary", use_container_width=True):
            with st.spinner(f"Running {sel_info['short']} estimator…"):
                try:
                    result = estimate_effect(
                        df,
                        treatment=treatment,
                        outcome=outcome,
                        confounders=list(confounders),
                        control_value=control_value,
                        treatment_value=treatment_value,
                        n_bootstrap=int(n_bootstrap),
                        estimator=selected_estimator,
                    )
                except ImportError as e:
                    st.error(
                        f"❌ Missing dependency: {e}\n\n"
                        "Run `pip install dowhy econml` in your virtual environment, then restart the app."
                    )
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.stop()

            st.success(f"✅ {sel_info['name']} complete!")
            st.divider()

            # ── Key metrics ───────────────────────────────────────────────────
            st.subheader("📈 Results")
            sig = result.p_value < 0.05
            sig_label = "✅ Significant" if sig else "⚠️ Not Significant"
            sig_delta = f"p = {result.p_value:.4f}"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ATE (Point Estimate)", f"{result.ate:.4f}")
            m2.metric("Standard Error", f"{result.se:.4f}")
            m3.metric("95% CI", f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
            m4.metric("Significance", sig_label, delta=sig_delta, delta_color="normal")

            # ── Interpretation ─────────────────────────────────────────────────
            st.info(result.interpretation)

            # ── Bootstrap ATE distribution ─────────────────────────────────────
            if result.bootstrap_ates is not None and len(result.bootstrap_ates) > 0:
                st.subheader("🔁 Bootstrap ATE Distribution")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 3))
                fig.patch.set_facecolor("#1E293B")
                ax.set_facecolor("#1E293B")
                ax.hist(
                    result.bootstrap_ates, bins=30,
                    color="#4F46E5", alpha=0.85, edgecolor="#1E293B"
                )
                ax.axvline(result.ate, color="#F59E0B", linewidth=2, label=f"ATE = {result.ate:.4f}")
                ax.axvline(result.ci_lower, color="#EF4444", linewidth=1.5, linestyle="--", label="95% CI")
                ax.axvline(result.ci_upper, color="#EF4444", linewidth=1.5, linestyle="--")
                ax.tick_params(colors="white")
                ax.spines[:].set_color("#334155")
                ax.set_xlabel("Estimated ATE", color="white")
                ax.set_ylabel("Frequency", color="white")
                ax.set_title("Bootstrap Distribution of ATE", color="white")
                ax.legend(facecolor="#1E293B", labelcolor="white")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # ── Regression summary ─────────────────────────────────────────────
            with st.expander("📋 Full Regression Summary", expanded=False):
                st.text(result.summary_text)

elif nav_selection == "🤖 Causal AI Agent":
    st.title("Causal AI Agent")
    st.markdown("Chat with the Causal AI Agent for guidance on causal inference.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about causal inference..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Mock agent response
        response = f"I am a Causal AI Agent. You asked: '{prompt}'. In a future update, I will be integrated with a full LLM backend to answer your causal inference queries!"
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

