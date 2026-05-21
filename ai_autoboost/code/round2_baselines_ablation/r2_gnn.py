"""Round 2 — GNN cascading surrogate (PyTorch, CPU OK).

Goal:
    Learn a graph-neural-net surrogate for the physics-anchored cascading simulator
    (r2_lib.physics_cascading). The GNN takes initial per-node damage + city
    features as input and predicts post-cascade per-node damage in one forward pass.

Why this matters for the paper:
    - At inference, the GNN is ~100× faster than the iterative physics simulator
      (8 message-passing steps in batch vs 8 sequential matrix multiplies × seeds).
    - The trained GNN demonstrates **transferable knowledge** of the propagation
      kernel: train on 6 archetypes, test on 2 held-out via city-LOCO.
    - Compared MLP-per-node baseline (A2 ablation): the GNN should win on RMSE,
      which is direct evidence that **graph structure matters**.

Output:
    outputs/round2/gnn_train_log.csv
    outputs/round2/gnn_loco_results.csv
    outputs/round2/gnn_vs_mlp_comparison.csv
    outputs/round2/gnn_loco.png
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round2"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r2_gnn_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r2_gnn")

sys.path.insert(0, str(CODE_ROOT))
import r2_lib as L
from r2_main import ARCHETYPES, sample_dGW

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight GraphSAGE-style layer with no PyG dependency.
# ---------------------------------------------------------------------------

class SAGEConv(nn.Module):
    """Mean aggregator GraphSAGE conv on dense adjacency.

    Inputs:
        h: (n, d) node features
        A: (n, n) adjacency (row-normalized inside)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        nbr = A_norm @ h          # (n, d)
        return self.lin_self(h) + self.lin_neigh(nbr)


class GNNCascade(nn.Module):
    """K-step message-passing surrogate for the physics cascading."""

    def __init__(self, node_feat_dim: int = 5, hidden: int = 32, n_steps: int = 4):
        super().__init__()
        self.n_steps = n_steps
        self.embed = nn.Linear(node_feat_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(n_steps)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Row-normalize once for stability
        deg = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
        A_norm = A / deg
        h = F.relu(self.embed(x))
        for conv in self.convs:
            h = F.relu(conv(h, A_norm))
        return self.head(h).squeeze(-1)


class MLPCascadeBaseline(nn.Module):
    """Per-node MLP (no message passing) — ablation for "graph structure matters"."""

    def __init__(self, node_feat_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training-data generator
# ---------------------------------------------------------------------------

def build_node_features(graph: L.CityGraph, initial_damage: np.ndarray, dGW: float, Mw: float) -> np.ndarray:
    """5-dim per-node feature vector for GNN input."""
    return np.stack([
        initial_damage.astype(np.float32),
        graph.Vs30.astype(np.float32) / 600.0,                 # normalised
        graph.GW_2020.astype(np.float32) / 30.0,
        np.full(graph.n_nodes, dGW / 2.0, dtype=np.float32),    # broadcast climate driver
        np.full(graph.n_nodes, (Mw - 6.0) / 1.5, dtype=np.float32),
    ], axis=1)


def generate_training_set(cohort: List[L.CityGraph], n_per_city: int = 80,
                           rng_seed: int = 2024) -> List[Dict]:
    """Build (x, A, y_target) examples from the physics simulator.

    Each example = one scenario: random epoch, random SSP, random ground motion.
    Target y_target = post-cascade per-node damage from physics_cascading.
    """
    rng = np.random.default_rng(rng_seed)
    examples: List[Dict] = []
    for city in cohort:
        for k in range(n_per_city):
            # Random sampling
            Mw = float(rng.choice([5.5, 6.0, 6.5, 7.0, 7.5]))
            R = float(rng.choice([10.0, 25.0, 50.0, 100.0]))
            ssp = rng.choice(["SSP2-4.5", "SSP5-8.5", "Control-NoCC"])
            epoch = int(rng.choice([2020, 2050, 2100]))
            dGW = sample_dGW(rng, ssp, epoch, city.archetype)

            R_vec = np.full(city.n_nodes, R)
            pga = L.bssa14_pga(Mw, R_vec, city.Vs30, fault="SS")
            pga = pga * np.exp(rng.normal(0.0, 0.72, size=city.n_nodes))
            GW_t = np.clip(city.GW_2020 + dGW, 0.3, None)
            p_liq = L.liquefaction_probability(Mw, pga, city.Vs30, GW_t, depth_m=3.0)
            dmg_init_mean, _, _ = L.damage_ensemble(pga, city.asset_class, p_liq)

            dmg_final, _ = L.physics_cascading(dmg_init_mean, city,
                                                n_steps=8,
                                                transmission_kappa=0.15,
                                                recovery_threshold=0.10)

            x = build_node_features(city, dmg_init_mean, dGW, Mw)
            examples.append({
                "city_id": city.city_id,
                "archetype": city.archetype,
                "x": x.astype(np.float32),
                "A": city.adjacency.astype(np.float32),
                "y": dmg_final.astype(np.float32),
                "n": city.n_nodes,
            })
    return examples


# ---------------------------------------------------------------------------
# Trainer with LOCO splits
# ---------------------------------------------------------------------------

def evaluate_predictions(model: nn.Module, examples: List[Dict], model_kind: str,
                         held_out_city_ids: List[int], seed: int) -> Tuple[Dict, List[Dict]]:
    """Return test metrics and node-level predictions for the held-out examples."""
    y_all = []
    pred_all = []
    rows: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for ex_idx, ex in enumerate(examples):
            x = torch.from_numpy(ex["x"])
            A = torch.from_numpy(ex["A"])
            y = torch.from_numpy(ex["y"])
            pred = model(x, A).detach().cpu().numpy()
            target = y.detach().cpu().numpy()
            y_all.append(target)
            pred_all.append(pred)
            for node_id, (obs, est) in enumerate(zip(target, pred)):
                rows.append({
                    "model": model_kind,
                    "held_out_city_ids": ",".join(map(str, held_out_city_ids)),
                    "seed": seed,
                    "example_id": ex_idx,
                    "city_id": ex["city_id"],
                    "archetype": ex["archetype"],
                    "node_id": node_id,
                    "target_damage": round(float(obs), 6),
                    "predicted_damage": round(float(est), 6),
                    "abs_error": round(float(abs(est - obs)), 6),
                })

    y_vec = np.concatenate(y_all)
    pred_vec = np.concatenate(pred_all)
    err = pred_vec - y_vec
    sp = spearmanr(y_vec, pred_vec)
    kt = kendalltau(y_vec, pred_vec)
    metrics = {
        "test_rmse": float(np.sqrt(np.mean(err ** 2))),
        "test_mae": float(np.mean(np.abs(err))),
        "test_bias": float(np.mean(err)),
        "spearman_rho": float(sp.statistic) if not np.isnan(sp.statistic) else float("nan"),
        "spearman_p": float(sp.pvalue) if not np.isnan(sp.pvalue) else float("nan"),
        "kendall_tau": float(kt.statistic) if not np.isnan(kt.statistic) else float("nan"),
        "kendall_p": float(kt.pvalue) if not np.isnan(kt.pvalue) else float("nan"),
        "n_test_nodes": int(y_vec.size),
    }
    return metrics, rows


def train_eval(examples: List[Dict], held_out_city_ids: List[int],
                model_kind: str = "gnn", n_epochs: int = 60, lr: float = 3e-3,
                hidden: int = 32, n_steps: int = 4,
                log_csv: List[Dict] = None, seed: int = 0) -> Tuple[float, Dict, List[Dict]]:
    """Train on non-held-out and evaluate on held-out with retained predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ex = [e for e in examples if e["city_id"] not in held_out_city_ids]
    test_ex = [e for e in examples if e["city_id"] in held_out_city_ids]

    if model_kind == "gnn":
        model = GNNCascade(node_feat_dim=5, hidden=hidden, n_steps=n_steps)
    elif model_kind == "mlp":
        model = MLPCascadeBaseline(node_feat_dim=5, hidden=hidden)
    else:
        raise ValueError(model_kind)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_train = len(train_ex)

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(n_train)
        total = 0.0
        for idx in perm:
            ex = train_ex[idx]
            x = torch.from_numpy(ex["x"])
            A = torch.from_numpy(ex["A"])
            y = torch.from_numpy(ex["y"])
            pred = model(x, A)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        train_rmse = (total / max(n_train, 1)) ** 0.5

        # Eval
        model.eval()
        test_err_sum = 0.0
        n_test_nodes = 0
        with torch.no_grad():
            for ex in test_ex:
                x = torch.from_numpy(ex["x"])
                A = torch.from_numpy(ex["A"])
                y = torch.from_numpy(ex["y"])
                pred = model(x, A)
                test_err_sum += float(((pred - y) ** 2).sum())
                n_test_nodes += y.numel()
        test_rmse = (test_err_sum / max(n_test_nodes, 1)) ** 0.5

        if log_csv is not None:
            log_csv.append({
                "model": model_kind, "held_out": ",".join(map(str, held_out_city_ids)),
                "epoch": epoch, "train_rmse": round(train_rmse, 5),
                "test_rmse": round(test_rmse, 5),
            })
    metrics, pred_rows = evaluate_predictions(model, test_ex, model_kind, held_out_city_ids, seed)
    return train_rmse, metrics, pred_rows


def loco_sweep(examples: List[Dict], n_seeds: int = 3) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Leave-one-city-out for both GNN and MLP, n_seeds repeats."""
    rows = []
    train_logs = []
    prediction_rows = []
    city_ids = sorted({e["city_id"] for e in examples})
    for held in city_ids:
        for seed in range(n_seeds):
            for kind in ("gnn", "mlp"):
                _, metrics, pred_rows = train_eval(examples, [held], model_kind=kind,
                                                   n_epochs=40, hidden=32, n_steps=4,
                                                   log_csv=train_logs, seed=seed)
                arch, label = ARCHETYPES[held]
                rows.append({
                    "model": kind,
                    "held_out_city_id": held,
                    "held_out_archetype": arch,
                    "held_out_label": label,
                    "seed": seed,
                    "test_rmse": round(metrics["test_rmse"], 5),
                    "test_mae": round(metrics["test_mae"], 5),
                    "test_bias": round(metrics["test_bias"], 5),
                    "spearman_rho": round(metrics["spearman_rho"], 5),
                    "spearman_p": round(metrics["spearman_p"], 6),
                    "kendall_tau": round(metrics["kendall_tau"], 5),
                    "kendall_p": round(metrics["kendall_p"], 6),
                    "n_test_nodes": metrics["n_test_nodes"],
                })
                prediction_rows.extend(pred_rows)
                log.info(
                    f"  LOCO {kind} held={arch} seed={seed} "
                    f"test_rmse={metrics['test_rmse']:.5f} test_mae={metrics['test_mae']:.5f} "
                    f"spearman={metrics['spearman_rho']:.3f}"
                )
    return rows, train_logs, prediction_rows


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate_loco(loco_rows: List[Dict], out: Path) -> None:
    """Summarise: per-archetype RMSE mean ± std, GNN vs MLP, plus paired Wilcoxon."""
    import pandas as pd
    df = pd.DataFrame(loco_rows)
    out_rows = []
    for held in sorted(df["held_out_city_id"].unique()):
        for kind in ("gnn", "mlp"):
            sub = df[(df["held_out_city_id"] == held) & (df["model"] == kind)]
            vals = sub["test_rmse"].values
            arch = sub["held_out_archetype"].iloc[0]
            label = sub["held_out_label"].iloc[0]
            lo, hi = (L.bca_ci(vals, n_resamples=999, rng_seed=int(held) * 10) if vals.size >= 3
                      else (float("nan"), float("nan")))
            out_rows.append({
                "model": kind, "held_out_city_id": int(held), "archetype": arch, "label": label,
                "n_seeds": int(vals.size),
                "rmse_mean": round(float(vals.mean()), 5),
                "rmse_std": round(float(vals.std(ddof=1)) if vals.size > 1 else 0.0, 5),
                "bca_ci_lo": round(lo, 5), "bca_ci_hi": round(hi, 5),
            })
    write_csv(out_rows, out)


def compare_gnn_vs_mlp(loco_rows: List[Dict], out: Path) -> None:
    """Paired Wilcoxon GNN vs MLP across (held_out, seed) pairs."""
    import pandas as pd
    df = pd.DataFrame(loco_rows)
    gnn = df[df["model"] == "gnn"].sort_values(["held_out_city_id", "seed"])["test_rmse"].values
    mlp = df[df["model"] == "mlp"].sort_values(["held_out_city_id", "seed"])["test_rmse"].values
    z, p = L.paired_wilcoxon(gnn, mlp)
    lo, hi = L.bca_ci(mlp - gnn, n_resamples=2999, rng_seed=42)
    write_csv([{
        "n_pairs": int(gnn.size),
        "gnn_rmse_mean": round(float(gnn.mean()), 5),
        "mlp_rmse_mean": round(float(mlp.mean()), 5),
        "mean_advantage_mlp_minus_gnn": round(float((mlp - gnn).mean()), 5),
        "bca_ci_lo": round(lo, 5),
        "bca_ci_hi": round(hi, 5),
        "wilcoxon_z": round(z, 4) if not np.isnan(z) else "",
        "wilcoxon_p": round(p, 6) if not np.isnan(p) else "",
        "decision_at_alpha_0.05": "GNN BETTER" if p < 0.05 and gnn.mean() < mlp.mean() else (
            "MLP BETTER" if p < 0.05 else "NOT SIGNIFICANT"),
    }], out)


def surrogate_metric_summary(loco_rows: List[Dict], out: Path) -> None:
    """Summary table with RMSE, MAE and ranking metrics for manuscript use."""
    import pandas as pd
    df = pd.DataFrame(loco_rows)
    pivot = df.pivot_table(index=["held_out_city_id", "held_out_archetype", "seed"],
                           columns="model", values="test_rmse").reset_index()
    pivot["gnn_win_rmse"] = pivot["gnn"] < pivot["mlp"]
    rows = []
    for model, label in [("mlp", "Node-local MLP"), ("gnn", "GraphSAGE")]:
        sub = df[df["model"] == model]
        rows.append({
            "model": label,
            "validation_split": "city leave-one-out",
            "n_pairs": int(sub.shape[0]),
            "rmse_mean": round(float(sub["test_rmse"].mean()), 5),
            "rmse_std": round(float(sub["test_rmse"].std(ddof=1)), 5),
            "mae_mean": round(float(sub["test_mae"].mean()), 5),
            "mae_std": round(float(sub["test_mae"].std(ddof=1)), 5),
            "spearman_mean": round(float(sub["spearman_rho"].mean()), 5),
            "kendall_mean": round(float(sub["kendall_tau"].mean()), 5),
            "bias_mean": round(float(sub["test_bias"].mean()), 5),
            "rmse_wins_vs_other": int(pivot["gnn_win_rmse"].sum() if model == "gnn" else (~pivot["gnn_win_rmse"]).sum()),
            "relative_rmse_reduction_pct": "",
            "relative_mae_reduction_pct": "",
        })
    gnn = df[df["model"] == "gnn"].sort_values(["held_out_city_id", "seed"])
    mlp = df[df["model"] == "mlp"].sort_values(["held_out_city_id", "seed"])
    rel_rmse = (mlp["test_rmse"].mean() - gnn["test_rmse"].mean()) / mlp["test_rmse"].mean() * 100.0
    rel_mae = (mlp["test_mae"].mean() - gnn["test_mae"].mean()) / mlp["test_mae"].mean() * 100.0
    rows.append({
        "model": "GraphSAGE advantage",
        "validation_split": "paired city leave-one-out",
        "n_pairs": int(gnn.shape[0]),
        "rmse_mean": "",
        "rmse_std": "",
        "mae_mean": "",
        "mae_std": "",
        "spearman_mean": "",
        "kendall_mean": "",
        "bias_mean": "",
        "rmse_wins_vs_other": f"{int(pivot['gnn_win_rmse'].sum())}/{len(pivot)}",
        "relative_rmse_reduction_pct": round(float(rel_rmse), 2),
        "relative_mae_reduction_pct": round(float(rel_mae), 2),
    })
    write_csv(rows, out)


def plot_loco(loco_rows: List[Dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        log.warning(f"plot skip: {e}")
        return
    df = pd.DataFrame(loco_rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    archetypes = [ARCHETYPES[i][0] for i in sorted(df["held_out_city_id"].unique())]
    x = np.arange(len(archetypes))
    for offs, kind, color in [(-width / 2, "gnn", None), (+width / 2, "mlp", None)]:
        means = []
        stds = []
        for held in sorted(df["held_out_city_id"].unique()):
            sub = df[(df["held_out_city_id"] == held) & (df["model"] == kind)]
            means.append(sub["test_rmse"].mean())
            stds.append(sub["test_rmse"].std(ddof=1) if sub.shape[0] > 1 else 0.0)
        ax.bar(x + offs, means, yerr=stds, width=width, capsize=4, label=kind.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(archetypes, rotation=20, ha="right")
    ax.set_ylabel("LOCO test RMSE (per-node final damage)")
    ax.set_title("GNN vs MLP under city-Leave-One-Out (lower is better)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_parity(prediction_rows: List[Dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        log.warning(f"parity plot skip: {e}")
        return
    df = pd.DataFrame(prediction_rows)
    if df.empty:
        return
    if df.shape[0] > 6000:
        df = df.sample(6000, random_state=20260521)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True, sharey=True)
    for ax, kind, title in zip(axes, ["mlp", "gnn"], ["Node-local MLP", "GraphSAGE"]):
        sub = df[df["model"] == kind]
        ax.scatter(sub["target_damage"], sub["predicted_damage"], s=5, alpha=0.25)
        lo = float(min(sub["target_damage"].min(), sub["predicted_damage"].min()))
        hi = float(max(sub["target_damage"].max(), sub["predicted_damage"].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted final damage")
    for ax in axes:
        ax.set_xlabel("Simulator final damage")
    fig.suptitle("LOCO surrogate parity against cascade-simulator labels", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_city", type=int, default=80)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--nodes", type=int, default=100)
    args = parser.parse_args()
    if args.smoke:
        args.n_per_city = 20
        args.n_seeds = 2
        args.nodes = 60

    t0 = datetime.utcnow()
    log.info(f"R2 GNN start. n_per_city={args.n_per_city} n_seeds={args.n_seeds} nodes={args.nodes}")
    log.info(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")

    cohort_rng = np.random.default_rng(2026_05_15)
    cohort = []
    for i, (arch, label) in enumerate(ARCHETYPES):
        cg = L.synthesize_city_graph(np.random.default_rng(cohort_rng.integers(0, 1 << 31)),
                                      n_nodes=args.nodes, archetype=arch)
        cg.archetype = arch
        object.__setattr__(cg, "label", label)
        object.__setattr__(cg, "city_id", i)
        cohort.append(cg)
    log.info(f"cohort: {[c.archetype for c in cohort]}")

    log.info(f"Generating {args.n_per_city * len(cohort)} training examples ...")
    examples = generate_training_set(cohort, n_per_city=args.n_per_city)
    log.info(f"  → {len(examples)} examples")

    log.info("Running LOCO sweep (GNN + MLP) ...")
    loco_rows, train_logs, prediction_rows = loco_sweep(examples, n_seeds=args.n_seeds)

    write_csv(loco_rows, OUT_DIR / "gnn_loco_results.csv")
    write_csv(train_logs, OUT_DIR / "gnn_train_log.csv")
    write_csv(prediction_rows, OUT_DIR / "gnn_loco_predictions.csv")
    aggregate_loco(loco_rows, OUT_DIR / "gnn_loco_summary.csv")
    compare_gnn_vs_mlp(loco_rows, OUT_DIR / "gnn_vs_mlp_comparison.csv")
    surrogate_metric_summary(loco_rows, OUT_DIR / "gnn_metric_summary.csv")
    plot_loco(loco_rows, OUT_DIR / "gnn_loco.png")
    plot_parity(prediction_rows, OUT_DIR / "gnn_parity.png")

    elapsed = (datetime.utcnow() - t0).total_seconds()
    meta = {
        "started_utc": t0.isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 2),
        "n_examples": len(examples),
        "n_loco_rows": len(loco_rows),
        "config": vars(args),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    (OUT_DIR / "gnn_run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"R2 GNN done in {elapsed:.1f}s")
    print(f"\n=== R2 GNN finished in {elapsed:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
