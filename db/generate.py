#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
db/generate.py

Regenerate every synthetic dataset in db/. Runs offline; same seed gives byte-identical files.

Usage (from the repository root):
    python db/generate.py                  # all datasets, base seed 42
    python db/generate.py --seed 7         # different base seed (file seeds = base + offset)
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import fouling_indicators
import synthetic_generator
from synthetic_generator import GeneratorScenario

DB_DIR = Path(__file__).resolve().parent
DATASET_PATH = DB_DIR / "biofouling_synthetic_dataset.csv"

# Mismatch mechanisms the nominal ODE does not contain: +50% deposition drift over the year,
# a slow temperature-independent scale deposit, and a 21-day x3 summer bloom from late June.
MISMATCH_SCENARIO = GeneratorScenario(
    k_dep_drift=0.5,
    scale_rate_m_per_h=5e-9,
    bloom_events=((4300, 504, 3.0),),
)

# name -> (seed offset, generator kwargs, limits the file must cross)
DATASETS = {
    "biofouling_synthetic_dataset.csv": (0, dict(inject_cleanings=True), ()),
    "stress_no_clean.csv": (1, dict(inject_cleanings=False), ("Rf", "dP", "BSI")),
    "stress_long_interval.csv": (2, dict(cleaning_interval_hours=90 * 24), ("Rf", "BSI")),
    "mismatch_stress.csv": (3, dict(cleaning_interval_hours=60 * 24, scenario=MISMATCH_SCENARIO), ("Rf", "dP", "BSI")),
}


def generate(name: str, base_seed: int = config.DEFAULT_RANDOM_SEED, hours: int = config.DATASET_DEFAULT_HOURS):
    """Generate one named dataset as a DataFrame."""
    offset, kwargs, _ = DATASETS[name]
    return synthetic_generator.generate_biofouling_dataset(hours=hours, seed=base_seed + offset, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Generate all synthetic datasets into db/")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_RANDOM_SEED, help="Base random seed")
    parser.add_argument("--hours", type=int, default=config.DATASET_DEFAULT_HOURS, help="Simulation duration in hours")
    parser.add_argument("--out-dir", type=Path, default=DB_DIR, help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        df = generate(name, args.seed, args.hours)
        df.to_csv(args.out_dir / name, index=False)
        crossings = fouling_indicators.first_limit_crossings(df)
        summary = ", ".join(
            f"{k} peak {v['peak']:.4g} first {'never' if v['first_hour'] is None else v['first_hour']}"
            for k, v in crossings.items()
        )
        print(f"{name:34s} seed {args.seed + DATASETS[name][0]:3d}  {summary}")


if __name__ == "__main__":
    main()
