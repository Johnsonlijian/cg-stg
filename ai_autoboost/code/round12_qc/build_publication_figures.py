"""Build publication/QC figure copies for the Markdown/Word submission package.

The script keeps the original round outputs intact and writes cleaned copies to
`ai_autoboost/outputs/final_qc_figures/`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT = PROJECT_ROOT / "ai_autoboost" / "outputs" / "final_qc_figures"
OUT.mkdir(parents=True, exist_ok=True)

ROUND3 = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round3"
ROUND4 = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round4" / "final_figures"
ROUND9 = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round9"
ROUND10 = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round10"


plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig01_concept() -> None:
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    epochs = np.array([2020, 2050, 2100])
    for label, y, color in [
        ("SSP5-8.5", [0.0, -0.5, -1.0], "#d62728"),
        ("SSP2-4.5", [0.0, -0.25, -0.5], "#1f77b4"),
        ("SSP1-2.6", [0.0, -0.05, -0.08], "#2ca02c"),
    ]:
        ax.plot(epochs, y, "o-", color=color, lw=1.6, ms=4, label=label)
    ax.invert_yaxis()
    ax.set_title("(a) Climate driver")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Groundwater change (m)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    depths = np.linspace(0, 30, 200)
    for label, mu, color in [
        ("deltaic", 2.0, "#d62728"),
        ("coastal", 4.0, "#ff7f0e"),
        ("mixed", 11.0, "#1f77b4"),
        ("inland", 16.0, "#2ca02c"),
    ]:
        wt = 0.95 / (1 + np.exp(-3 * (depths - mu)))
        ax.plot(depths, wt, label=label, color=color, lw=1.6)
    ax.set_title("(b) Soil-state mediator")
    ax.set_xlabel("Depth below surface (m)")
    ax.set_ylabel("Saturation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    qc = np.linspace(20, 200, 200)
    base = np.exp(qc / 113.0 + (qc / 1000.0) ** 2 - (qc / 140.0) ** 3 + (qc / 137.0) ** 4 - 2.8)
    for fact, label, color in [(1.0, "2020", "#1f77b4"), (0.85, "2050", "#ff7f0e"), (0.70, "2100", "#d62728")]:
        ax.plot(qc, np.minimum(base * fact, 0.6), color=color, lw=1.6, label=label)
    ax.set_title("(c) Liquefaction resistance")
    ax.set_xlabel("CPT-equivalent qc")
    ax.set_ylabel("CRR7.5")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    G = nx.Graph()
    n = 22
    coords = rng.uniform(0, 10, size=(n, 2))
    types = rng.choice(["Building", "Water", "Power", "Transport"], size=n, p=[0.58, 0.16, 0.16, 0.10])
    colors = {"Building": "#dddddd", "Water": "#3498db", "Power": "#e74c3c", "Transport": "#2ecc71"}
    sizes = {"Building": 35, "Water": 115, "Power": 115, "Transport": 115}
    for i in range(n):
        G.add_node(i, pos=coords[i], kind=types[i])
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        for j in np.argsort(d)[1:4]:
            G.add_edge(i, int(j))
    pos = {i: tuple(coords[i]) for i in range(n)}

    ax = fig.add_subplot(gs[1, 0])
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.8)
    for kind in colors:
        nodes = [i for i in G if G.nodes[i]["kind"] == kind]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors[kind],
                               node_size=sizes[kind], edgecolors="k", linewidths=0.4,
                               label=kind, ax=ax)
    ax.set_title("(d) Lifeline graph")
    ax.legend(frameon=False, loc="lower left", fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])

    epic = np.array([7.8, 4.8])
    d0 = np.clip(0.55 * np.exp(-np.linalg.norm(coords - epic, axis=1) / 4), 0, 1)
    A = nx.to_numpy_array(G)
    for col, (title, steps) in enumerate([("(e1) Initial damage", 0), ("(e2) Post-cascade", 8)], start=1):
        ax = fig.add_subplot(gs[1, col])
        d = d0.copy()
        for _ in range(steps):
            src = np.maximum(d - 0.1, 0)
            p = A.T @ src / np.clip(A.sum(axis=0), 1, None)
            d = np.clip(d + 0.15 * p * (1 - d), 0, 1)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.18, width=0.8)
        nodes = nx.draw_networkx_nodes(G, pos, node_color=d, cmap="Reds", vmin=0, vmax=1,
                                       node_size=[sizes[G.nodes[i]["kind"]] for i in G],
                                       edgecolors="k", linewidths=0.35, ax=ax)
        ax.plot(*epic, "*", color="black", ms=8, markeredgecolor="white", markeredgewidth=0.5)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(nodes, ax=fig.axes[-1], fraction=0.046, pad=0.02, label="Damage")

    fig.suptitle("CG-STG: Climate-Coupled Seismic Risk to Urban Lifelines",
                 fontsize=11, fontweight="bold", y=0.99)
    save(fig, "Fig01_concept_qc")


def fig02_regression() -> None:
    df = pd.read_csv(ROUND9 / "cohort100_summary.csv")
    ssps = ["Control-NoCC", "SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for ax, ssp in zip(axes.ravel(), ssps):
        sub = df[df["ssp"] == ssp].copy()
        x = 1 / sub["gw_base_m"].to_numpy()
        y = sub["mean_gap"].to_numpy()
        colors = sub["sign"].map({"positive": "#d62728", "negative": "#1f77b4", "zero-crossing": "#999999"}).fillna("#999999")
        ax.scatter(x, y, s=18, c=colors, edgecolors="black", linewidths=0.35, alpha=0.86)
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        xs = np.linspace(x.min(), x.max(), 80)
        ax.plot(xs, model.predict(xs.reshape(-1, 1)), "k--", lw=1)
        r, p = pearsonr(x, y)
        pos = int((sub["sign"] == "positive").sum())
        ax.set_title(
            f"{ssp}\nR2={model.score(x.reshape(-1,1), y):.3f}; "
            f"r={r:.3f}; pos-CI={pos}/100",
            pad=5,
        )
        ax.axhline(0, color="0.55", lw=0.6)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("1 / baseline groundwater depth (m-1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean damage gap (2100 - 2020)")
    fig.suptitle("100-city cohort regression across climate scenarios", fontsize=10, y=0.995)
    fig.subplots_adjust(hspace=0.34, wspace=0.16, top=0.88)
    save(fig, "Fig02_cohort100_regression_qc")


def try_plot_world(ax) -> None:
    try:
        import geopandas as gpd
        import urllib.request
        import zipfile
        import tempfile
        url = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"
        cache = OUT / "ne_110m_land.zip"
        if not cache.exists():
            urllib.request.urlretrieve(url, cache)
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(cache, "r") as zf:
                zf.extractall(td)
            shp = next(Path(td).glob("*.shp"))
            world = gpd.read_file(shp)
            world.plot(ax=ax, color="#f3f3f3", edgecolor="#c9c9c9", linewidth=0.35)
    except Exception:
        ax.set_facecolor("#fafafa")
        for lat in range(-60, 81, 20):
            ax.axhline(lat, color="#dddddd", lw=0.45, zorder=0)
        for lon in range(-180, 181, 30):
            ax.axvline(lon, color="#dddddd", lw=0.45, zorder=0)


def fig03_map() -> None:
    df = pd.read_csv(ROUND9 / "cohort100_summary.csv")
    sub = df[df["ssp"] == "SSP5-8.5"].copy()
    size = 24 + 1200 * sub["mean_gap"].abs()
    colors = sub["sign"].map({"positive": "#d62728", "negative": "#1f77b4", "zero-crossing": "#9c9c9c"}).fillna("#9c9c9c")
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    try_plot_world(ax)
    ax.scatter(sub["lon"], sub["lat"], s=size, c=colors, edgecolors="black",
               linewidths=0.45, alpha=0.84, zorder=5)
    offsets = {
        "NewOrleans": (6, 10), "Miami": (8, -10), "Honolulu": (8, 7),
        "Rotterdam": (8, 12), "Amsterdam": (-46, -12), "Tianjin": (10, 16),
        "Shanghai": (14, -14), "Guangzhou": (-42, 6), "Hanoi": (-42, -12),
        "Yangon": (-42, -22), "HoChiMinh": (8, -2), "Singapore": (8, -16),
        "Dhaka": (-38, 12), "Christchurch": (-78, 12),
    }
    top = sub[sub["sign"] == "positive"].sort_values("mean_gap", ascending=False)
    for _, row in top.iterrows():
        dx, dy = offsets.get(row["city"], (6, 6))
        ha = "right" if dx < 0 else "left"
        ax.annotate(f"{row['city']}\n{row['mean_gap']:+.3f}",
                    (row["lon"], row["lat"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=6.2, color="#a01418",
                    bbox=dict(boxstyle="round,pad=0.18", fc="#fff9df", ec="#d62728", lw=0.45, alpha=0.88),
                    arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.35, alpha=0.65))
    ax.set_xlim(-180, 185)
    ax.set_ylim(-60, 82)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.4)
    ax.set_title("Climate-isolated lifeline damage gap, SSP5-8.5 (2100 - 2020)\n"
                 "n=100 OSM-anchored cities; size proportional to |gap|")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#d62728", markeredgecolor="black", label="Positive-CI (14)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#9c9c9c", markeredgecolor="black", label="Zero-crossing (86)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#1f77b4", markeredgecolor="black", label="Negative-CI (0)"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92)
    save(fig, "Fig03_global_map_ssp585_2100_qc")


def si_b_pcmci() -> None:
    edges = pd.read_csv(ROUND3 / "pcmci_per_archetype.csv")
    d = edges[edges["archetype"] == "deltaic"].copy()
    def val(src, dst):
        row = d[(d["src"] == src) & (d["dst"] == dst) & (d["tau"] == 0)]
        return float(row.iloc[0]["val"]) if not row.empty else np.nan
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    pos = {
        "dGW": (0.0, 0.0),
        "soil moisture": (1.55, 0.62),
        "liquefaction": (3.1, 0.0),
        "damage": (4.65, 0.0),
    }
    for node, (x, y) in pos.items():
        ax.scatter(x, y, s=850, facecolor="#d8ecff", edgecolor="black", linewidth=0.9, zorder=4)
        ax.text(x, y, node, ha="center", va="center", fontsize=8, zorder=5)
    arrows = [
        ("dGW", "soil moisture", val("dGW", "soil_moist"), "tau=0"),
        ("soil moisture", "liquefaction", val("soil_moist", "p_liq"), "tau=0"),
        ("liquefaction", "damage", val("p_liq", "damage"), "tau=0"),
        ("dGW", "damage", val("dGW", "damage"), "weak direct"),
    ]
    label_positions = [
        (0.72, 1.02),
        (2.32, 0.96),
        (3.88, 0.48),
        (2.28, -0.42),
    ]
    for idx, (src, dst, r, tau) in enumerate(arrows):
        sx, sy = pos[src]; dx, dy = pos[dst]
        color = "#d62728" if r > 0 else "#1f77b4"
        style = "->"
        is_weak_direct = src == "dGW" and dst == "damage"
        rad = 0.24 if is_weak_direct else 0.0
        lw = 1.1 if is_weak_direct else 1.0 + min(abs(r), 1) * 3.0
        linestyle = (0, (3, 2)) if is_weak_direct else "solid"
        alpha = 0.46 if is_weak_direct else 0.78
        ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle=style, lw=lw, color=color, alpha=alpha,
                                    linestyle=linestyle,
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=22, shrinkB=22),
                    zorder=1 if is_weak_direct else 2)
        mx, my = label_positions[idx]
        ax.text(mx, my, f"{tau}, r={r:.2f}", ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    ax.text(2.3, -0.75, "PCMCI deltaic archetype: mediator chain dominates; direct dGW -> damage is weak.",
            ha="center", va="center", fontsize=8)
    ax.set_xlim(-0.55, 5.2)
    ax.set_ylim(-1.0, 1.15)
    ax.axis("off")
    ax.set_title("PCMCI causal graph -- deltaic archetype")
    save(fig, "SI_FigB_PCMCI_deltaic_qc")


def si_d_mexico_sensitivity() -> None:
    df = pd.read_csv(ROUND3 / "mexico_sensitivity_grid.csv")
    arch_amps = sorted(df["arch_amp"].unique(), reverse=True)
    dgws = sorted(df["dGW_2100_mean"].unique())

    mean = (
        df.pivot(index="arch_amp", columns="dGW_2100_mean", values="mean_gap_2100_minus_2020")
        .loc[arch_amps, dgws]
    )
    sign = (
        df.pivot(index="arch_amp", columns="dGW_2100_mean", values="sign")
        .loc[arch_amps, dgws]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(mean.to_numpy())))

    im = axes[0].imshow(mean.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title("(a) Mean damage gap, 2100-2020")
    axes[0].set_xlabel("dGW_2100_mean (m)")
    axes[0].set_ylabel("arch_amp")
    axes[0].set_xticks(range(len(dgws)), [f"{x:+.1f}" for x in dgws])
    axes[0].set_yticks(range(len(arch_amps)), [f"{x:.2f}" for x in arch_amps])
    for i in range(len(arch_amps)):
        for j in range(len(dgws)):
            value = mean.iat[i, j]
            color = "white" if abs(value) > 0.0045 else "black"
            axes[0].text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=axes[0], shrink=0.82, label="Mean damage gap")

    sign_code = sign.replace({"negative": 0, "zero-crossing": 1, "positive": 2}).astype(int).to_numpy()
    cmap = ListedColormap(["#1f77b4", "#f7f7f7", "#d62728"])
    axes[1].imshow(sign_code, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    axes[1].set_title("(b) Sign of 95% BCa CI")
    axes[1].set_xlabel("dGW_2100_mean (m)")
    axes[1].set_ylabel("arch_amp")
    axes[1].set_xticks(range(len(dgws)), [f"{x:+.1f}" for x in dgws])
    axes[1].set_yticks(range(len(arch_amps)), [f"{x:.2f}" for x in arch_amps])
    short = {"negative": "N", "zero-crossing": "Z", "positive": "P"}
    for i in range(len(arch_amps)):
        for j in range(len(dgws)):
            label = short[sign.iat[i, j]]
            color = "white" if label in {"N", "P"} else "black"
            axes[1].text(j, i, label, ha="center", va="center", fontsize=8, fontweight="bold", color=color)
    axes[1].legend(
        handles=[
            Patch(facecolor="#1f77b4", edgecolor="black", label="N: negative CI"),
            Patch(facecolor="#f7f7f7", edgecolor="black", label="Z: zero-crossing"),
            Patch(facecolor="#d62728", edgecolor="black", label="P: positive CI"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    for ax in axes:
        ax.set_xticks(np.arange(-0.5, len(dgws), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(arch_amps), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle("Mexico City high-altitude archetype: climate-driver sensitivity", fontsize=10)
    save(fig, "SI_FigD_mexico_sensitivity_qc")


def copy_or_upscale(src: Path, stem: str, scale: int = 2) -> None:
    im = Image.open(src).convert("RGB")
    if min(im.size) < 900:
        im = im.resize((im.width * scale, im.height * scale), Image.Resampling.LANCZOS)
    im.save(OUT / f"{stem}.png", dpi=(300, 300))


def main() -> int:
    fig01_concept()
    fig02_regression()
    fig03_map()
    si_b_pcmci()
    si_d_mexico_sensitivity()
    copies = {
        "Fig04_LOAO_R2_stability_qc": ROUND4 / "Fig04_LOAO_R2_stability.png",
        "Fig05_robustness_qc": ROUND4 / "Fig05_robustness.png",
        "Fig06_per_class_per_archetype_qc": ROUND4 / "Fig06_per_class_per_archetype.png",
        "SI_FigA_GNN_vs_MLP_LOCO_qc": ROUND4 / "SI_FigA_GNN_vs_MLP_LOCO.png",
        "SI_FigC_OSM_vs_archetype_qc": ROUND4 / "SI_FigC_OSM_vs_archetype.png",
        "SI_FigE_baseline_comparison_qc": ROUND4 / "SI_FigE_baseline_comparison.png",
        "SI_FigF_Mw_R_heatmap_qc": ROUND4 / "SI_FigF_Mw_R_heatmap.png",
    }
    for stem, src in copies.items():
        copy_or_upscale(src, stem)
    print(f"Wrote publication/QC figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
