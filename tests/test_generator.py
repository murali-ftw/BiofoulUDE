#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_generator.py

Synthetic generator: step function, cleaning policies and cleaning-type alternation.
"""

from pathlib import Path
import sys
import unittest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import synthetic_generator


class TestGeneratorCleaningPolicy(unittest.TestCase):
    """Default cleaning policy and the pluggable policy hook."""

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_generator.generate_biofouling_dataset(hours=8760, seed=42, inject_cleanings=True)

    def test_both_cleaning_types_occur_in_default_year(self):
        counts = self.df.loc[self.df["cleaning_event"] == 1, "cleaning_type"].value_counts()
        self.assertGreater(counts.get("Mechanical_Balls", 0), 0)
        self.assertGreater(counts.get("Biocide_Shock", 0), 0)

    def test_default_policy_alternates_starting_with_mechanical(self):
        types = self.df.loc[self.df["cleaning_event"] == 1, "cleaning_type"].tolist()
        expected = ["Mechanical_Balls" if i % 2 == 0 else "Biocide_Shock" for i in range(len(types))]
        self.assertEqual(types, expected)

    def test_custom_policy_is_used(self):
        def every_100h_chemical(hour, state):
            return synthetic_generator.CHEMICAL if state.time_since_last_clean >= 100.0 else None

        df = synthetic_generator.generate_biofouling_dataset(hours=1000, seed=1, cleaning_policy=every_100h_chemical)
        events = df[df["cleaning_event"] == 1]
        self.assertEqual(len(events), 10)
        self.assertTrue((events["cleaning_type"] == "Biocide_Shock").all())

    def test_no_cleaning_when_disabled(self):
        df = synthetic_generator.generate_biofouling_dataset(hours=2000, seed=42, inject_cleanings=False)
        self.assertEqual(int(df["cleaning_event"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
