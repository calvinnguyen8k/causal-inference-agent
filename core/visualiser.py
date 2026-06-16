"""DAG visualiser — adjacency matrix → matplotlib figure via graphviz layout."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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

    Edge COLOUR encodes connection strength (abs weight):
        grey  →  amber  →  red   (weak → strong)

    Edge WIDTH encodes bootstrap confidence (when available):
        thin = uncertain,  thick = high confidence
    """
    n = len(col_names)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    edges            = []
    edge_raw_weights = []   # absolute adj values — drives COLOUR
    edge_boot_freqs  = []   # bootstrap frequency  — drives WIDTH

    for i in range(n):
        for j in range(n):
            if abs(adj[i, j]) > 1e-8:
                freq = edge_freq[i, j] if edge_freq is not None else 1.0
                if freq >= freq_threshold:
                    G.add_edge(i, j)
                    edges.append((i, j))
                    edge_raw_weights.append(abs(float(adj[i, j])))
                    edge_boot_freqs.append(float(freq))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_w = max(14, n * 1.4)
    fig_h = max(10, n * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    if not edges:
        ax.text(0.5, 0.5, "No edges above threshold", ha="center", va="center",
                color="#94A3B8", fontsize=14, transform=ax.transAxes)
        ax.set_title(title, color="#F1F5F9", fontsize=15, pad=16, fontweight="bold")
        ax.axis("off")
        return fig

    # ── Normalise strength to [0, 1] ──────────────────────────────────────────
    raw_max = max(edge_raw_weights) if edge_raw_weights else 1.0
    raw_min = min(edge_raw_weights) if edge_raw_weights else 0.0
    raw_range = raw_max - raw_min or 1.0
    edge_strength_norm = [(w - raw_min) / raw_range for w in edge_raw_weights]

    # If all edges have the same weight (binary 0/1 from PC / GES), show gradient
    # based on bootstrap confidence instead so there's still visual variation.
    all_same_strength = (raw_range < 1e-8)

    # ── Layout ────────────────────────────────────────────────────────────────
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.kamada_kawai_layout(G, scale=3.0)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=3.0 / max(1, n ** 0.5))

    _pos_arr = np.array(list(pos.values()))
    x_range = _pos_arr[:, 0].max() - _pos_arr[:, 0].min() or 1
    y_range = _pos_arr[:, 1].max() - _pos_arr[:, 1].min() or 1
    pos = {k: ((v[0] - _pos_arr[:, 0].min()) / x_range,
               (v[1] - _pos_arr[:, 1].min()) / y_range)
           for k, v in pos.items()}

    # ── Node style ─────────────────────────────────────────────────────────────
    out_degrees = dict(G.out_degree())
    max_out = max(out_degrees.values()) if out_degrees else 1
    node_colors = []
    for node in G.nodes():
        ratio = out_degrees[node] / max(max_out, 1)
        r = int(79 + (99 - 79) * ratio)
        g = int(70 + (102 - 70) * (1 - ratio))
        b = int(229 + (241 - 229) * (1 - ratio))
        node_colors.append(f"#{r:02x}{g:02x}{b:02x}")

    node_size = max(2500, 4000 - n * 80)

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        ax=ax,
        alpha=0.92,
        linewidths=2,
        edgecolors="#818CF8",
    )

    # ── Edge colour: strength gradient (grey → amber → red) ───────────────────
    # weak  = #475569 (slate)
    # mid   = #F59E0B (amber)
    # strong= #EF4444 (red)
    strength_cmap = mcolors.LinearSegmentedColormap.from_list(
        "strength",
        ["#475569", "#F59E0B", "#EF4444"],
    )

    # If all edges binary (same weight), fall back to bootstrap freq for colour
    colour_vals = edge_boot_freqs if all_same_strength else edge_strength_norm
    edge_colors = [strength_cmap(v) for v in colour_vals]

    # Edge width: bootstrap confidence if available, else uniform
    if edge_freq is not None:
        edge_widths = [1.0 + 4.0 * f for f in edge_boot_freqs]
    else:
        # When no bootstrap, width also reflects strength
        edge_widths = [1.5 + 3.5 * s for s in edge_strength_norm]

    # ── Draw edges (one at a time to allow per-edge arc radius) ───────────────
    edge_set = set(edges)
    for idx, (src, dst) in enumerate(edges):
        rad = 0.25 if (dst, src) in edge_set else 0.08
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(src, dst)],
            edge_color=[edge_colors[idx]],
            width=edge_widths[idx],
            arrows=True,
            arrowsize=22,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            ax=ax,
            min_source_margin=28,
            min_target_margin=28,
        )

    # ── Labels ─────────────────────────────────────────────────────────────────
    labels = {i: col_names[i] for i in range(n)}
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_color="white",
        font_size=max(7, 11 - n // 4),
        font_weight="bold",
        ax=ax,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#1E293B",
            edgecolor="#475569",
            alpha=0.85,
            linewidth=1,
        ),
    )

    # ── Colorbars ──────────────────────────────────────────────────────────────
    # Strength colorbar (always shown)
    label_str = ("Bootstrap confidence" if all_same_strength
                 else f"Connection strength  [{raw_min:.3g} – {raw_max:.3g}]")
    sm_strength = plt.cm.ScalarMappable(
        cmap=strength_cmap, norm=plt.Normalize(0, 1)
    )
    sm_strength.set_array([])
    cbar1 = plt.colorbar(sm_strength, ax=ax, shrink=0.45, pad=0.01,
                         aspect=22, location="right")
    cbar1.set_label(label_str, color="#CBD5E1", fontsize=9)
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color="#CBD5E1")
    cbar1.outline.set_edgecolor("#475569")
    # Custom tick labels: weak / medium / strong
    cbar1.set_ticks([0, 0.5, 1.0])
    cbar1.set_ticklabels(["Weak", "Medium", "Strong"], color="#CBD5E1")

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#475569", label="Weak connection"),
        mpatches.Patch(color="#F59E0B", label="Medium connection"),
        mpatches.Patch(color="#EF4444", label="Strong connection"),
    ]
    if edge_freq is not None:
        # Add width legend entries
        from matplotlib.lines import Line2D
        legend_items += [
            Line2D([0], [0], color="white", linewidth=1.5, label="Low confidence (narrow)"),
            Line2D([0], [0], color="white", linewidth=5.0, label="High confidence (wide)"),
        ]
        legend_items.append(mpatches.Patch(color="#1E293B", label="Colour = strength  │  Width = confidence"))

    ax.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=8,
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="white",
        framealpha=0.9,
    )

    ax.set_title(title, color="#F1F5F9", fontsize=15, pad=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout(pad=1.5)
    return fig


def edge_frequency_heatmap(
    freq: np.ndarray,
    col_names: list[str],
) -> plt.Figure:
    """Render bootstrap edge frequency as a heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, len(col_names) * 0.8),
                                    max(7, len(col_names) * 0.7)))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    im = ax.imshow(freq, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Bootstrap frequency", color="#CBD5E1", fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#CBD5E1")

    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", color="#CBD5E1", fontsize=8)
    ax.set_yticklabels(col_names, color="#CBD5E1", fontsize=8)
    ax.set_title("Edge Frequency Heatmap (row → col)", color="#F1F5F9", fontsize=12, pad=10)

    for i in range(len(col_names)):
        for j in range(len(col_names)):
            val = freq[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="#0F172A" if val > 0.55 else "white", fontsize=7,
                        fontweight="bold")

    plt.tight_layout()
    return fig


def get_edge_table(
    adj: np.ndarray,
    col_names: list[str],
    edge_freq: Optional[np.ndarray] = None,
    freq_threshold: float = 0.0,
) -> pd.DataFrame:
    """Return a DataFrame of directed edges for text display."""
    rows = []
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            w = adj[i, j]
            if abs(w) > 1e-8:
                freq = edge_freq[i, j] if edge_freq is not None else None
                if freq is None or freq >= freq_threshold:
                    rows.append({
                        "From": col_names[i],
                        "→": "→",
                        "To": col_names[j],
                        "Edge weight": round(float(w), 4),
                        "Bootstrap conf.": f"{freq:.0%}" if freq is not None else "—",
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if edge_freq is not None:
        df["_sort"] = [edge_freq[col_names.index(r["From"]), col_names.index(r["To"])]
                       for _, r in df.iterrows()]
        df = df.sort_values("_sort", ascending=False).drop(columns="_sort")
    else:
        df = df.sort_values("Edge weight", ascending=False, key=abs)
    return df.reset_index(drop=True)
