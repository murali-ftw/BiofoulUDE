"""
Benchmark one representative training step on CPU and on MPS, and record the faster device.

    python -m ml.bench_device [--batch 64] [--hours 336] [--repeats 5]

A step = batched hourly rollout (vectorised across windows, sequential in time) + loss +
backward + optimiser update. If ml.ude.FoulingUDE exists it is used; otherwise a stand-in
with the same operation mix (Ratkowsky factor, Haaland friction, tiny tanh MLP) is timed.
The model is tiny, so CPU may well beat MPS; the result is measured, not assumed.
"""

import argparse
import json
import platform
import time

import ml  # noqa: F401
import torch
from torch import nn

from ml.device import ARTIFACTS_DIR, DEVICE_FILE, _mps_available, synchronize


class _StandIn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
        self.log_k = nn.Parameter(torch.tensor([-15.6, -12.4]))

    def forward(self, x):  # x: [B, L, 6]
        B, L, _ = x.shape
        delta = torch.zeros(B, device=x.device, dtype=x.dtype)
        out = []
        for t in range(L):
            xt = x[:, t, :]
            g = torch.exp(self.net(xt).squeeze(-1))
            T = 6.0 + 39.0 * xt[:, 0]
            r = ((T - 45.0) * (T - 6.0) ** 2) / (22.0 * (22.0 * (T - 28.0) + 17.0 * (34.0 - 2.0 * T)))
            d_eff = 0.021 - 2.0 * delta
            rel = (1.5e-6 + delta) / d_eff
            inv = -1.8 * torch.log10((rel / 3.7) ** 1.11 + 6.9 / 3e4)
            tau = (1.0 / inv) ** 2 / 8.0 * 998.0 * 2.0
            delta = torch.clamp(delta + torch.exp(self.log_k[0]) * r.clamp(0, 1) * g - torch.exp(self.log_k[1]) * tau * delta, 0.0, 350e-6)
            out.append(delta)
        return torch.stack(out, dim=1)


def _make_model_and_input(device, dtype, batch, hours):
    torch.manual_seed(0)
    try:
        from ml.bench_support import make_benchmark_case  # provided once the UDE exists
        return make_benchmark_case(device, dtype, batch, hours)
    except ImportError:
        model = _StandIn().to(device=device, dtype=dtype)
        x = torch.rand(batch, hours, 6, device=device, dtype=dtype) * 0.5
        target = torch.rand(batch, hours, device=device, dtype=dtype) * 1e-4

        def loss_fn():
            return ((model(x) - target) ** 2).mean() * 1e8

        return model, loss_fn, "stand-in"


def time_step(device, dtype=torch.float32, batch=64, hours=336, repeats=5, warmup=2):
    device = torch.device(device)
    model, loss_fn, label = _make_model_and_input(device, dtype, batch, hours)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def one():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one()
    synchronize(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        one()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    return {"median_s": times[len(times) // 2], "min_s": times[0], "model": label}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hours", type=int, default=336)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    results = {"cpu_float32": time_step("cpu", torch.float32, args.batch, args.hours, args.repeats),
               "cpu_float64": time_step("cpu", torch.float64, args.batch, args.hours, args.repeats)}
    if _mps_available():
        results["mps_float32"] = time_step("mps", torch.float32, args.batch, args.hours, args.repeats)

    cpu_best = min(results["cpu_float32"]["median_s"], results["cpu_float64"]["median_s"])
    winner = "mps" if "mps_float32" in results and results["mps_float32"]["median_s"] < cpu_best else "cpu"
    record = {
        "device": winner,
        "timings": results,
        "batch_windows": args.batch,
        "rollout_hours": args.hours,
        "torch": torch.__version__,
        "machine": platform.platform(),
        "processor": platform.processor(),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE_FILE.write_text(json.dumps(record, indent=2))
    for k, v in results.items():
        print(f"{k:12s} median {v['median_s'] * 1000:8.1f} ms  (model: {v['model']})")
    print(f"winner: {winner}  -> {DEVICE_FILE}")


if __name__ == "__main__":
    main()
