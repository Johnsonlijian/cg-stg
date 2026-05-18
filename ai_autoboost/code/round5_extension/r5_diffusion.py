"""Round 5.2 — Conditional DDPM for climate-seismic scenario synthesis (C15 D → B).

We train a small conditional diffusion model to generate (dGW, soil_moisture, p_liq,
damage) 4-tuples conditioned on (archetype, ssp, epoch). The model is trained on
~13,000 samples from r3_pcmci's 8-archetype × 17-epoch trajectory set + a fresh
re-generation.

Evaluation:
    1. Generated samples should reproduce per-archetype marginal distributions of
       (dGW, soil, p_liq, damage).
    2. Generated samples should respect the PCMCI mediator chain: partial r
       (dGW → damage controlling for soil + p_liq) should be **small** in generated
       data, matching the training distribution.
    3. Compute KS test between training and generated marginals.

If the generator passes both tests, C15 ("conditional diffusion synthesis preserves
causal structure") upgrades D → B.

Architecture: 3-layer MLP score model, ~10K params, CPU torch.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
OUT_DIR = PROJECT_ROOT / "ai_autoboost" / "outputs" / "round5"
LOG_DIR = PROJECT_ROOT / "ai_autoboost" / "logs"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"r5_diff_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger("r5_diff")

sys.path.insert(0, str(CODE_ROOT.parent / "round2_baselines_ablation"))
sys.path.insert(0, str(CODE_ROOT.parent / "round3_mechanism_error"))
import r2_lib as L
from r2_main import ARCHETYPES, sample_dGW
from r3_pcmci import simulate_dense_trajectory, chain_strength_climate_to_damage

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Diffusion: discrete-time linear-beta schedule, 200 steps
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    def __init__(self, T: int = 200, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self.alpha_bars[t].unsqueeze(1)
        eps = torch.randn_like(x0)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps, eps


class CondMLPDenoiser(nn.Module):
    """Conditional MLP that predicts noise ε given (x_t, t, cond)."""

    def __init__(self, x_dim: int = 4, cond_dim: int = 11, hidden: int = 64):
        super().__init__()
        self.t_embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, 32), nn.SiLU(), nn.Linear(32, 32))
        self.body = nn.Sequential(
            nn.Linear(x_dim + 32 + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_e = self.t_embed(t.unsqueeze(1).float() / 200.0)
        c_e = self.cond_embed(cond.float())
        return self.body(torch.cat([x, t_e, c_e], dim=1))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

ARCHETYPE_LIST = [a for a, _ in ARCHETYPES]
SSP_LIST = ["SSP2-4.5", "SSP5-8.5", "Control-NoCC"]


def build_training_set(n_seeds_per_arch: int = 12) -> tuple[np.ndarray, np.ndarray, Dict]:
    """Build (x, cond) training set across 8 archetypes × 17 epochs × n_seeds × 3 SSPs.

    cond is a 11-dim one-hot/normalized condition vector:
        [arch_onehot (8 dims)] + [ssp_onehot (3 dims)]
        - epoch is encoded via the diffusion timestep t conditioning indirectly through training.
        Actually we'll separate: use (arch_onehot + ssp_onehot + epoch_norm) = 12 dims; but
        for clean conditioning we encode epoch_norm into cond too — making it 12 dims.

    Re-implementation: cond_dim = 12 = 8 (arch) + 3 (ssp) + 1 (epoch normalized).
    """
    xs: List[np.ndarray] = []
    conds: List[np.ndarray] = []
    for ai, arch in enumerate(ARCHETYPE_LIST):
        for ssp in SSP_LIST:
            # Use r3_pcmci to generate trajectory data across 17 epochs × n_seeds
            data = simulate_dense_trajectory(arch, n_nodes=100, n_seeds=n_seeds_per_arch,
                                              ssp=ssp, rng_seed=hash((arch, ssp)) & 0xFFFFFF)
            v1 = data["v1"]; v2 = data["v2"]; v3 = data["v3"]; v4 = data["v4"]
            # build conds: epoch index encoded as scalar from 0..1 across the 17 epochs
            # we stacked n_seeds * 17 epochs in v1, so 0..n_seeds-1 of epoch 0, then 1...
            EPOCHS_DENSE = tuple(range(2020, 2101, 5))
            for k in range(v1.size):
                epoch_idx = k % len(EPOCHS_DENSE)
                epoch_norm = epoch_idx / (len(EPOCHS_DENSE) - 1)
                arch_oh = np.zeros(len(ARCHETYPE_LIST), dtype=np.float32)
                arch_oh[ai] = 1.0
                ssp_oh = np.zeros(len(SSP_LIST), dtype=np.float32)
                ssp_oh[SSP_LIST.index(ssp)] = 1.0
                cond = np.concatenate([arch_oh, ssp_oh, [epoch_norm]])
                xs.append([v1[k], v2[k], v3[k], v4[k]])
                conds.append(cond)
    X = np.array(xs, dtype=np.float32)
    C = np.array(conds, dtype=np.float32)
    # Normalize x by per-feature mean/std for training stability
    stats = {
        "mu": X.mean(axis=0).tolist(),
        "sd": X.std(axis=0).tolist(),
    }
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return X, C, stats


def denormalize(x_norm: np.ndarray, stats: Dict) -> np.ndarray:
    mu = np.array(stats["mu"])
    sd = np.array(stats["sd"])
    return x_norm * sd + mu


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: nn.Module, X: np.ndarray, C: np.ndarray, sched: DiffusionSchedule,
          n_epochs: int = 200, batch: int = 512, lr: float = 1e-3) -> List[float]:
    Xt = torch.from_numpy(X)
    Ct = torch.from_numpy(C)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n = X.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            x0 = Xt[idx]
            c = Ct[idx]
            t = torch.randint(0, sched.T, (x0.shape[0],))
            x_noisy, eps = sched.add_noise(x0, t)
            pred = model(x_noisy, t, c)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 20 == 0:
            log.info(f"  epoch {epoch + 1}/{n_epochs}  loss={avg:.4f}")
    return losses


@torch.no_grad()
def sample(model: nn.Module, cond: np.ndarray, sched: DiffusionSchedule, x_dim: int = 4) -> np.ndarray:
    model.eval()
    n = cond.shape[0]
    x = torch.randn(n, x_dim)
    c = torch.from_numpy(cond.astype(np.float32))
    for t in reversed(range(sched.T)):
        tt = torch.full((n,), t, dtype=torch.long)
        eps_pred = model(x, tt, c)
        alpha = sched.alphas[t]
        alpha_bar = sched.alpha_bars[t]
        beta = sched.betas[t]
        coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
        mean = (x - coef * eps_pred) / torch.sqrt(alpha)
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    return x.numpy()


# ---------------------------------------------------------------------------
# Causal-fidelity evaluation
# ---------------------------------------------------------------------------

def causal_fidelity(real: np.ndarray, gen: np.ndarray) -> Dict:
    """Compare:
        partial r (v1 → v4 | v2, v3) on real vs generated
        marginal KS test on each variable
    """
    from sklearn.linear_model import LinearRegression
    from scipy.stats import ks_2samp

    def partial_r(arr: np.ndarray) -> float:
        v1 = arr[:, 0]; v2 = arr[:, 1]; v3 = arr[:, 2]; v4 = arr[:, 3]
        X_med = np.stack([v2, v3], axis=1)
        r1 = v1 - LinearRegression().fit(X_med, v1).predict(X_med)
        r4 = v4 - LinearRegression().fit(X_med, v4).predict(X_med)
        if r1.std() < 1e-9 or r4.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(r1, r4)[0, 1])

    real_partial = partial_r(real)
    gen_partial = partial_r(gen)
    ks = []
    for j, name in enumerate(["dGW", "soil_moist", "p_liq", "damage"]):
        stat, p = ks_2samp(real[:, j], gen[:, j])
        ks.append({"var": name, "ks_stat": float(stat), "ks_p": float(p)})
    return {
        "real_partial_r": real_partial,
        "gen_partial_r": gen_partial,
        "partial_r_difference": gen_partial - real_partial,
        "ks": ks,
    }


def main() -> int:
    t0 = datetime.utcnow()
    log.info("R5 diffusion start")

    # 1. Build dataset
    log.info("Building training set ...")
    X, C, stats = build_training_set(n_seeds_per_arch=12)
    log.info(f"  N samples: {X.shape[0]}, cond_dim={C.shape[1]}")
    cond_dim = C.shape[1]

    # 2. Train
    sched = DiffusionSchedule(T=200, beta_start=1e-4, beta_end=2e-2)
    model = CondMLPDenoiser(x_dim=4, cond_dim=cond_dim, hidden=64)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  model params: {n_params}")
    log.info("Training ...")
    torch.manual_seed(42)
    losses = train(model, X, C, sched, n_epochs=200, batch=512, lr=1e-3)
    log.info(f"  final loss: {losses[-1]:.4f}")

    # 3. Sample: 1000 samples per (archetype, ssp, epoch=2100)
    log.info("Sampling for evaluation ...")
    all_real_per_cell = {}
    all_gen_per_cell = {}
    for ai, arch in enumerate(ARCHETYPE_LIST):
        for si, ssp in enumerate(SSP_LIST):
            # Real samples at epoch=2100
            real_data = simulate_dense_trajectory(arch, n_nodes=100, n_seeds=12,
                                                    ssp=ssp, rng_seed=hash((arch, ssp, "eval")) & 0xFFFFFF)
            # Filter to epoch 2100 only
            EPOCHS_DENSE = tuple(range(2020, 2101, 5))
            ep_idx_2100 = len(EPOCHS_DENSE) - 1
            real_slice = []
            v1 = real_data["v1"]; v2 = real_data["v2"]; v3 = real_data["v3"]; v4 = real_data["v4"]
            for k in range(v1.size):
                if k % len(EPOCHS_DENSE) == ep_idx_2100:
                    real_slice.append([v1[k], v2[k], v3[k], v4[k]])
            real_slice = np.array(real_slice, dtype=np.float32)
            # Generated samples: condition with arch + ssp + epoch_norm=1.0
            n_gen = 200
            arch_oh = np.zeros((n_gen, len(ARCHETYPE_LIST)), dtype=np.float32); arch_oh[:, ai] = 1.0
            ssp_oh = np.zeros((n_gen, len(SSP_LIST)), dtype=np.float32); ssp_oh[:, si] = 1.0
            cond = np.concatenate([arch_oh, ssp_oh, np.ones((n_gen, 1), dtype=np.float32)], axis=1)
            gen_norm = sample(model, cond, sched)
            gen = denormalize(gen_norm, stats)
            all_real_per_cell[(arch, ssp)] = real_slice
            all_gen_per_cell[(arch, ssp)] = gen

    # 4. Causal fidelity per (archetype, ssp)
    fid_rows = []
    for (arch, ssp), real in all_real_per_cell.items():
        gen = all_gen_per_cell[(arch, ssp)]
        f = causal_fidelity(real, gen)
        fid_rows.append({
            "archetype": arch, "ssp": ssp,
            "real_partial_r": round(f["real_partial_r"], 4),
            "gen_partial_r": round(f["gen_partial_r"], 4),
            "diff": round(f["partial_r_difference"], 4),
            **{f"ks_{k['var']}_stat": round(k["ks_stat"], 4) for k in f["ks"]},
            **{f"ks_{k['var']}_p": round(k["ks_p"], 6) for k in f["ks"]},
        })

    # Pooled aggregation
    real_all = np.concatenate(list(all_real_per_cell.values()), axis=0)
    gen_all = np.concatenate(list(all_gen_per_cell.values()), axis=0)
    pooled = causal_fidelity(real_all, gen_all)
    fid_rows.append({
        "archetype": "POOLED", "ssp": "ALL",
        "real_partial_r": round(pooled["real_partial_r"], 4),
        "gen_partial_r": round(pooled["gen_partial_r"], 4),
        "diff": round(pooled["partial_r_difference"], 4),
        **{f"ks_{k['var']}_stat": round(k["ks_stat"], 4) for k in pooled["ks"]},
        **{f"ks_{k['var']}_p": round(k["ks_p"], 6) for k in pooled["ks"]},
    })

    fields = sorted(fid_rows[0].keys(), key=lambda k: (k != "archetype", k != "ssp", k))
    with (OUT_DIR / "diffusion_causal_fidelity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(fid_rows)

    log.info(f"POOLED real partial r = {pooled['real_partial_r']:.4f}, gen partial r = {pooled['gen_partial_r']:.4f}, diff = {pooled['partial_r_difference']:.4f}")
    log.info(f"POOLED KS test on damage: stat={pooled['ks'][3]['ks_stat']:.4f}, p={pooled['ks'][3]['ks_p']:.4g}")

    # 5. Plot loss + marginal comparison
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
        # Loss
        axes[0].plot(losses)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE loss")
        axes[0].set_title("(a) Training loss")
        axes[0].grid(alpha=0.3)
        # Marginals
        names = ["dGW", "soil_moist", "p_liq", "damage"]
        for j, name in enumerate(names):
            ax = axes[j + 1]
            ax.hist(real_all[:, j], bins=40, alpha=0.5, label="real", density=True, color="#1f77b4")
            ax.hist(gen_all[:, j], bins=40, alpha=0.5, label="generated", density=True, color="#d62728")
            ax.set_xlabel(name); ax.set_title(f"({chr(98 + j)}) {name}\nKS p = {pooled['ks'][j]['ks_p']:.3f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle("R5 conditional DDPM — training + marginal fidelity")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "diffusion_marginals.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"plot fail: {e}")

    # 6. Meta
    meta = {
        "model_params": n_params,
        "n_train_samples": X.shape[0],
        "n_diffusion_steps": sched.T,
        "n_epochs": 200,
        "final_loss": round(losses[-1], 4),
        "pooled_real_partial_r": pooled["real_partial_r"],
        "pooled_gen_partial_r": pooled["gen_partial_r"],
        "ks_damage_p": pooled["ks"][3]["ks_p"],
        "verdict_C15_grade": (
            "B" if abs(pooled["partial_r_difference"]) < 0.15 and pooled["ks"][3]["ks_p"] > 0.005 else
            "C" if abs(pooled["partial_r_difference"]) < 0.30 else "D"
        ),
        "elapsed_seconds": round((datetime.utcnow() - t0).total_seconds(), 2),
    }
    (OUT_DIR / "diffusion_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    log.info(f"DIFFUSION verdict: C15 → {meta['verdict_C15_grade']}")
    log.info(f"R5 diffusion done in {meta['elapsed_seconds']}s")
    print(f"\n=== R5 diffusion done; C15 → grade {meta['verdict_C15_grade']} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
