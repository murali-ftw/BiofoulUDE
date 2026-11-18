#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_stress_datasets.py

Stress and mismatch datasets in db/: present, same schema as the nominal file, and crossing
the limits they are generated to cross.
"""

from pathlib import Path
import sys
import unittest

import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
for p in (APP_DIR, APP_DIR / "db"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import fouling_indicators
from core import synthetic_generator
import generate as db_generate

DB_DIR = APP_DIR / "db"
NOMINAL = "biofouling_synthetic_dataset.csv"


class TestStressDatasets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.expected_columns = list(pd.read_csv(DB_DIR / NOMINAL, nrows=1).columns)

    def test_files_exist_with_expected_columns(self):
        for name in db_generate.DATASETS:
            with self.subTest(name=name):
                path = DB_DIR / name
                self.assertTrue(path.exists(), f"{name} missing; run python db/generate.py")
                df = pd.read_csv(path)
                self.assertEqual(list(df.columns), self.expected_columns)
                self.assertEqual(len(df), 8760)

    def test_stress_files_cross_claimed_limits(self):
        for name, (_, _, claimed) in db_generate.DATASETS.items():
            df = pd.read_csv(DB_DIR / name)
            crossings = fouling_indicators.first_limit_crossings(df)
            for limit_name in claimed:
                with self.subTest(name=name, limit=limit_name):
                    self.assertIsNotNone(crossings[limit_name]["first_hour"])

    def test_nominal_file_crosses_no_limit(self):
        crossings = fouling_indicators.first_limit_crossings(pd.read_csv(DB_DIR / NOMINAL))
        for limit_name, info in crossings.items():
            with self.subTest(limit=limit_name):
                self.assertIsNone(info["first_hour"])

    def test_files_match_generator(self):
        for name in db_generate.DATASETS:
            with self.subTest(name=name):
                regenerated = db_generate.generate(name).to_csv(index=False)
                self.assertEqual(regenerated, (DB_DIR / name).read_text(), f"{name} is stale; run python db/generate.py")

    def test_nominal_scenario_is_identity(self):
        explicit = synthetic_generator.generate_biofouling_dataset(
            hours=500, seed=3, scenario=synthetic_generator.GeneratorScenario()
        )
        default = synthetic_generator.generate_biofouling_dataset(hours=500, seed=3)
        self.assertTrue(explicit.equals(default))


if __name__ == "__main__":
    unittest.main()
