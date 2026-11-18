"""
Train the plain ODE (stage 1) and a 5-seed UDE ensemble (stage 2) on the NOMINAL train split.

    python -m ml.train [--float64] [--window-days 14] [--quick]

Stage 1: correction frozen (g = 1); fit log K_dep, log K_rem with Adam on a Huber loss of the
standardised U and dP residuals. Saved as ml/artifacts/ode.pt.
Stage 2 (per seed): init from stage 1, unfreeze the net; lower LR for the net than for the K's;
weight decay on the net; L2 penalty on log g; gradient clipping; early stopping on the nominal
validation windows. Saved as ml/artifacts/ude_seed{s}.pt.

The loss never sees Rf or delta: only measured U and dP. Hyperparameters were chosen on the
nominal validation windows only. Checkpoints hold CPU tensors.
"""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass

import ml  # noqa: F401
import torch
from torch import nn

from core import config
from ml import data as D
from ml.device import ARTIFACTS_DIR, default_dtype, get_device
from ml.ude import FoulingUDE

ODE_PATH = ARTIFACTS_DIR / "ode.pt"
LOG_PATH = ARTIFACTS_DIR / "training_log.json"
ENSEMBLE_SEEDS = (11, 12, 13, 14, 15)


def ude_path(seed: int):
    return ARTIFACTS_DIR / f"ude_seed{seed}.pt"


@dataclass
class TrainConfig:
    window_days: int = 14
    stride_h: int = 24
    huber_delta: float = 1.0
    stage1_iters: int = 400
    stage1_lr: float = 0.03
    stage2_max_iters: int = 1500
    stage2_lr_k: float = 3e-3
    stage2_lr_net: float = 3e-3   # chosen on nominal validation (1e-3..1e-2 gave equal val loss)
    weight_decay: float = 1e-4
    log_g_penalty: float = 1e-2
    grad_clip: float = 1.0
    eval_every: int = 25
    patience_evals: int = 8


def huber_loss(batch: D.WindowBatch, model: FoulingUDE, cfg: TrainConfig, penalty: float = 0.0):
    r_u, r_dp, log_g = D.observation_residuals(model, batch)
    hub = nn.functional.huber_loss
    zero = torch.zeros_like(r_u)
    data_loss = hub(r_u, zero, delta=cfg.huber_delta) + hub(r_dp, zero, delta=cfg.huber_delta)
    return data_loss + penalty * (log_g ** 2).mean(), data_loss


def build_batches(cfg: TrainConfig, offset_h: int = 0, device="cpu", dtype=torch.float64):
    df = D.load("nominal")
    ranges = D.split_ranges(df)
    L = cfg.window_days * 24
    train = D.make_windows(df, D.window_starts(df, ranges["train"], L, cfg.stride_h, offset_h), L)
    val = D.make_windows(df, D.window_starts(df, ranges["val"], L, cfg.stride_h, 0), L)
    return train.to(device, dtype), val.to(device, dtype)


def _to_cpu_state(model):
    return {k: v.detach().to("cpu", torch.float64) for k, v in model.state_dict().items()}


def save_model(model: FoulingUDE, path, extra=None):
    torch.save({"state_dict": _to_cpu_state(model), "use_correction": model.use_correction, "extra": extra or {}}, path)


def load_model(path, dtype=torch.float64) -> FoulingUDE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FoulingUDE(use_correction=ckpt["use_correction"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(dtype=dtype).eval()


def fit_stage1(train, val, cfg: TrainConfig, device, dtype, log):
    torch.manual_seed(0)
    model = FoulingUDE(use_correction=False).to(device=device, dtype=dtype)
    opt = torch.optim.Adam([model.log_k_dep, model.log_k_rem], lr=cfg.stage1_lr)
    for it in range(cfg.stage1_iters):
        opt.zero_grad()
        loss, _ = huber_loss(train, model, cfg)
        loss.backward()
        opt.step()
        if it % cfg.eval_every == 0 or it == cfg.stage1_iters - 1:
            with torch.no_grad():
                _, v = huber_loss(val, model, cfg)
            log.append({"stage": "ode", "iter": it, "train": loss.item(), "val": float(v),
                        "k_dep": model.k_dep.item(), "k_rem": model.k_rem.item()})
    return model


def fit_stage2(ode: FoulingUDE, train, val, cfg: TrainConfig, seed: int, device, dtype, log):
    torch.manual_seed(seed)
    model = FoulingUDE(use_correction=True)
    model.log_k_dep.data.copy_(ode.log_k_dep.detach().cpu())
    model.log_k_rem.data.copy_(ode.log_k_rem.detach().cpu())
    model = model.to(device=device, dtype=dtype)
    opt = torch.optim.AdamW([
        {"params": [model.log_k_dep, model.log_k_rem], "lr": cfg.stage2_lr_k, "weight_decay": 0.0},
        {"params": model.net.parameters(), "lr": cfg.stage2_lr_net, "weight_decay": cfg.weight_decay},
    ])
    with torch.no_grad():
        _, best_val = huber_loss(val, model, cfg)
    best_val, best_state, bad, it = float(best_val), copy.deepcopy(model.state_dict()), 0, 0
    log.append({"stage": f"ude_seed{seed}", "iter": 0, "train": None, "val": best_val})
    for it in range(1, cfg.stage2_max_iters + 1):
        opt.zero_grad()
        loss, _ = huber_loss(train, model, cfg, cfg.log_g_penalty)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if it % cfg.eval_every == 0:
            with torch.no_grad():
                _, v = huber_loss(val, model, cfg)
            v = float(v)
            log.append({"stage": f"ude_seed{seed}", "iter": it, "train": loss.item(), "val": v,
                        "k_dep": model.k_dep.item(), "k_rem": model.k_rem.item()})
            if v < best_val - 1e-6:
                best_val, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience_evals:
                    break
    model.load_state_dict(best_state)
    return model, best_val, it


def main():
    ap = argparse.ArgumentParser(description="Train ODE + UDE ensemble on nominal data")
    ap.add_argument("--float64", action="store_true", help="float64 on CPU (ignored on MPS)")
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--quick", action="store_true", help="few iterations (smoke test)")
    args = ap.parse_args()

    cfg = TrainConfig(window_days=args.window_days)
    if args.quick:
        cfg.stage1_iters, cfg.stage2_max_iters = 20, 50
    device = get_device()
    dtype = default_dtype(device, args.float64)
    print(f"device={device} dtype={dtype} window={cfg.window_days} d")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()

    train, val = build_batches(cfg, 0, device, dtype)
    print(f"stage 1: {len(train.starts)} train / {len(val.starts)} val windows")
    ode = fit_stage1(train, val, cfg, device, dtype, log)
    save_model(ode, ODE_PATH, {"stage": "ode"})
    print(f"  ODE  K_dep {ode.k_dep.item():.4e} (config {config.K_DEPOSITION:.4e})  "
          f"K_rem {ode.k_rem.item():.4e} (config {config.K_REMOVAL:.4e})  [{time.time() - t0:.0f}s]")

    members = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        train_s, _ = build_batches(cfg, offset_h=(5 * i) % 24, device=device, dtype=dtype)
        model, best_val, iters = fit_stage2(ode, train_s, val, cfg, seed, device, dtype, log)
        save_model(model, ude_path(seed), {"stage": "ude", "seed": seed, "best_val": best_val, "iters": iters})
        members.append({"seed": seed, "best_val": best_val, "iters": iters,
                        "k_dep": model.k_dep.item(), "k_rem": model.k_rem.item()})
        print(f"  UDE seed {seed}: K_dep {model.k_dep.item():.4e}  K_rem {model.k_rem.item():.4e}  "
              f"best val {best_val:.4f} @ {iters} it  [{time.time() - t0:.0f}s]")

    with torch.no_grad():
        _, ode_val = huber_loss(val, ode, cfg)
    LOG_PATH.write_text(json.dumps({
        "config": asdict(cfg), "device": str(device), "dtype": str(dtype),
        "train_windows": len(train.starts), "val_windows": len(val.starts),
        "splits": D.split_ranges(D.load("nominal")),
        "ode": {"k_dep": ode.k_dep.item(), "k_rem": ode.k_rem.item(), "val": float(ode_val)},
        "ude_members": members, "config_values": {"k_dep": config.K_DEPOSITION, "k_rem": config.K_REMOVAL},
        "history": log, "seconds": time.time() - t0,
    }, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
