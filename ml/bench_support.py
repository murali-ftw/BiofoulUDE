"""Real-model benchmark case for ml.bench_device: one UDE training step on nominal train windows."""

import ml  # noqa: F401
import torch

from ml import data as D
from ml.train import TrainConfig, huber_loss
from ml.ude import FoulingUDE


def make_benchmark_case(device, dtype, batch, hours):
    cfg = TrainConfig(window_days=max(1, hours // 24))
    df = D.load("nominal")
    L = cfg.window_days * 24
    starts = D.window_starts(df, D.split_ranges(df)["train"], L)[:batch]
    windows = D.make_windows(df, starts, L).to(device, dtype)
    model = FoulingUDE().to(device=device, dtype=dtype)
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 0.01)
    return model, (lambda: huber_loss(windows, model, cfg, cfg.log_g_penalty)[0]), f"FoulingUDE ({len(starts)} windows)"
