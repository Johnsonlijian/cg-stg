"""Fig 1 — CG-STG concept diagram.

A 5-panel sketch:
    (a) Climate driver (CMIP6 SSP scenarios at 2020 / 2050 / 2100)
    (b) Soil-state mediator (GW depth trajectory)
    (c) Time-varying liquefaction susceptibility (CRR × site amp)
    (d) Lifeline dependency graph (water/power/transport)
    (e) Cascading failure → per-node damage map

Output: figures/fig1_concept.png and figures/fig1_concept.pdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_fig1():
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # (a) climate driver — synthetic dGW under SSP
    ax = fig.add_subplot(gs[0, 0])
    epochs = np.array([2020, 2050, 2100])
    rng = np.random.default_rng(0)
    for label, mus, sds, color in [
        ("SSP5-8.5", [0.0, -0.5, -1.0], [0.0, 0.2, 0.2], "#d62728"),
        ("SSP2-4.5", [0.0, -0.25, -0.5], [0.0, 0.15, 0.15], "#1f77b4"),
        ("Control-NoCC", [0.0, 0.0, 0.0], [0.0, 0.1, 0.1], "#2ca02c"),
    ]:
        ys = []
        es = []
        for mu, sd in zip(mus, sds):
            samples = rng.normal(mu, max(sd, 1e-3), size=200)
            ys.append(samples.mean())
            es.append(samples.std())
        ax.errorbar(epochs, ys, yerr=es, marker="o", capsize=4, color=color, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ groundwater depth (m)\n(negative = water rose)")
    ax.set_title("(a) Climate driver (CMIP6 SSP)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) soil-state mediator (illustrative water table profile)
    ax = fig.add_subplot(gs[0, 1])
    depths = np.linspace(0, 30, 200)
    for archetype, mu, color in [
        ("Deltaic (Tianjin-like)", 2.0, "#d62728"),
        ("Coastal (Bangkok-like)", 4.0, "#ff7f0e"),
        ("Mixed (Beijing-like)", 11.0, "#1f77b4"),
        ("Inland (Tangshan-like)", 16.0, "#2ca02c"),
    ]:
        # Smoothed water table profile
        wt = 0.95 / (1 + np.exp(-3 * (depths - mu)))
        ax.plot(depths, wt, label=archetype, color=color, linewidth=2)
    ax.set_xlabel("Depth below surface (m)")
    ax.set_ylabel("Saturation fraction")
    ax.set_title("(b) Soil-state mediator")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) time-varying susceptibility CRR (state-dependent)
    ax = fig.add_subplot(gs[0, 2])
    qc = np.linspace(20, 200, 200)
    base_crr = np.exp(qc / 113.0 + (qc / 1000.0) ** 2 - (qc / 140.0) ** 3 + (qc / 137.0) ** 4 - 2.8)
    for fact, label, color in [(1.0, "2020 baseline", "#1f77b4"),
                                (0.85, "2050 SSP5-8.5", "#ff7f0e"),
                                (0.70, "2100 SSP5-8.5", "#d62728")]:
        ax.plot(qc, np.minimum(base_crr * fact, 0.6), label=label, color=color, linewidth=2)
    ax.set_xlabel("$q_{c1Ncs}$ (CPT-equivalent)")
    ax.set_ylabel("$CRR_{7.5}$ (clean sand)")
    ax.set_title("(c) Time-varying CRR\n(Boulanger-Idriss 2014 + GW mediator)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) lifeline dependency graph (synthetic, illustrative)
    ax = fig.add_subplot(gs[1, 0])
    rng_g = np.random.default_rng(42)
    G = nx.Graph()
    n = 25
    coords = rng_g.uniform(0, 10, size=(n, 2))
    types = rng_g.choice([0, 1, 2, 3], size=n, p=[0.6, 0.15, 0.15, 0.10])
    type_color = {0: "#dddddd", 1: "#3498db", 2: "#e74c3c", 3: "#2ecc71"}
    type_size = {0: 60, 1: 200, 2: 200, 3: 200}
    type_label = {0: "Building", 1: "Water", 2: "Power", 3: "Transport"}
    for i in range(n):
        G.add_node(i, pos=tuple(coords[i]), type=int(types[i]))
    for c in (1, 2, 3):
        idx = np.where(types == c)[0]
        if len(idx) < 2:
            continue
        for ii, i in enumerate(idx):
            d = np.linalg.norm(coords[idx] - coords[i], axis=1)
            d[ii] = np.inf
            for j in idx[np.argsort(d)[:2]]:
                if j != i:
                    G.add_edge(int(i), int(j))
    # building → nearest utility
    for i in np.where(types == 0)[0]:
        for c in (1, 2, 3):
            idx_c = np.where(types == c)[0]
            if len(idx_c) == 0:
                continue
            j = idx_c[np.argmin(np.linalg.norm(coords[idx_c] - coords[i], axis=1))]
            G.add_edge(int(i), int(j))
    pos = nx.get_node_attributes(G, "pos")
    for c in (0, 1, 2, 3):
        idx = [i for i in G.nodes() if G.nodes[i]["type"] == c]
        nx.draw_networkx_nodes(G, pos, nodelist=idx, node_color=type_color[c],
                                node_size=type_size[c], ax=ax,
                                edgecolors="k", linewidths=0.5,
                                label=type_label[c])
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    ax.set_title("(d) Lifeline dependency graph\n(buildings + utilities)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xticks([]); ax.set_yticks([])

    # (e) cascading propagation: show t=0 vs t=8 damage spreading on the same graph
    for ax_idx, (subtitle, step) in enumerate([("(e1) Initial damage (t=0)", 0),
                                                ("(e2) Post-cascade (t=8)", 8)]):
        ax = fig.add_subplot(gs[1, 1 + ax_idx])
        # synthetic damage: high near "epicentre" (right side), low far
        epi = np.array([8.0, 5.0])
        dist = np.linalg.norm(coords - epi, axis=1)
        d0 = np.clip(0.5 * np.exp(-dist / 4.0), 0, 1)
        if step == 0:
            d = d0
        else:
            d = d0.copy()
            A = nx.to_numpy_array(G)
            for _ in range(step):
                src = np.maximum(d - 0.10, 0)
                p = A.T @ src / np.clip(A.sum(axis=0), 1, None)
                d = d + 0.15 * p * (1 - d)
                d = np.clip(d, 0, 1)
        cmap = plt.cm.Reds
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
        sc = nx.draw_networkx_nodes(G, pos, node_color=d, cmap=cmap, vmin=0, vmax=1,
                                     node_size=[type_size[G.nodes[i]['type']] for i in G.nodes()],
                                     ax=ax, edgecolors="k", linewidths=0.5)
        # epicentre marker
        ax.plot(*epi, marker="*", color="black", markersize=14, markeredgecolor="white", markeredgewidth=0.7)
        ax.set_title(subtitle)
        ax.set_xticks([]); ax.set_yticks([])
    # Add colorbar to last
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.Reds),
                        ax=fig.axes[-1], shrink=0.7, pad=0.02, label="Damage rate")

    fig.suptitle("CG-STG: Climate-Compounded Seismic Risk to Urban Lifelines",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(FIG_DIR / "fig1_concept.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_concept.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Fig 1 written: {FIG_DIR / 'fig1_concept.png'} and .pdf")


if __name__ == "__main__":
    make_fig1()
