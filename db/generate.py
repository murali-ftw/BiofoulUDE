#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
db/generate.py

Regenerate the synthetic biofouling dataset into db/. Runs offline.

Usage (from the repository root):
    python db/generate.py
    python db/generate.py --seed 42 --hours 8760
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import synthetic_generator

DB_DIR = Path(__file__).resolve().parent
DATASET_PATH = DB_DIR / "biofouling_synthetic_dataset.csv"


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic biofouling dataset into db/")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument("--hours", type=int, default=config.DATASET_DEFAULT_HOURS, help="Simulation duration in hours")
    parser.add_argument("--out", type=Path, default=DATASET_PATH, help="Output CSV path")
    args = parser.parse_args()

    df = synthetic_generator.generate_biofouling_dataset(hours=args.hours, seed=args.seed, inject_cleanings=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
