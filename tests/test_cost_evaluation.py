#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_cost_evaluation.py

Re-simulation based cost evaluation: determinism, cost additivity, feasibility on stress
data, sweep optimality, and that every strategy is priced on its own simulated trajectory.
"""

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
for p in (APP_DIR, APP_DIR / "db"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import config
from optimizer import cleaning_optimizer as co
from core import synthetic_generator
import generate as db_generate

COST_COMPONENTS = ("cleaning_cost", "downtime_cost", "pumping_energy_cost", "thermal_penalty_cost")


class TestCostEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.opt = co.CleaningScheduleOptimizer()
        cls.fixed30 = cls.opt.evaluate_fixed_calendar_schedule(30)
        cls.sweep = cls.opt.best_fixed_interval()

    # (a) determinism
    def test_same_policy_same_cost(self):
        for make in (lambda: co.fixed_interval_policy(21), lambda: co.condition_based_policy(0.7, 0.7)):
            with self.subTest(policy=make):
                r1 = co.evaluate_policy(make(), hours=3000, seed=5)
                r2 = co.evaluate_policy(make(), hours=3000, seed=5)
                self.assertEqual(r1["total_cost"], r2["total_cost"])
                self.assertEqual(r1["cleanings_total"], r2["cleanings_total"])

    # (b) additivity
    def test_components_sum_to_total(self):
        rows = [self.fixed30, self.sweep["best"], self.opt.evaluate_never_clean(hours=3000)]
        for r in rows:
            with self.subTest(strategy=r["strategy"]):
                self.assertAlmostEqual(sum(r[k] for k in COST_COMPONENTS), r["total_cost"], places=6)
                self.assertEqual(r["cleanings_mechanical"] + r["cleanings_chemical"], r["cleanings_total"])
                self.assertAlmostEqual(r["biocide_kg"], r["cleanings_chemical"] * config.BIOCIDE_KG_PER_EVENT)
                self.assertAlmostEqual(
                    r["downtime_cost"],
                    r["cleanings_total"] * config.DOWNTIME_HOURS_PER_CLEANING * config.DOWNTIME_COST_PER_HOUR,
                )

    # (c) never-clean on the stress_no_clean configuration is infeasible
    def test_never_clean_on_stress_no_clean_is_infeasible(self):
        offset, kwargs, _ = db_generate.DATASETS["stress_no_clean.csv"]
        r = self.opt.evaluate_never_clean(seed=config.DEFAULT_RANDOM_SEED + offset)
        self.assertFalse(r["is_feasible"])
        self.assertGreater(r["hours_above_limit"], 0)
        # and it is the same trajectory as the stored stress file
        run = co.simulate_policy(co.never_clean_policy(), seed=config.DEFAULT_RANDOM_SEED + offset)
        self.assertEqual(run.df.to_csv(index=False), (APP_DIR / "db" / "stress_no_clean.csv").read_text())

    # (d) sweep optimum is at least as cheap as fixed 30 d when both are feasible
    def test_sweep_not_worse_than_fixed30(self):
        best = self.sweep["best"]
        self.assertIsNotNone(best)
        self.assertTrue(best["is_feasible"])
        if self.fixed30["is_feasible"]:
            self.assertLessEqual(best["total_cost"], self.fixed30["total_cost"])
        self.assertEqual(len(self.sweep["results"]), 90)

    # (e) each strategy is priced on the trajectory produced by its own cleanings
    def test_evaluators_do_not_accept_external_trajectories(self):
        for fn in (
            self.opt.evaluate_fixed_calendar_schedule,
            self.opt.evaluate_condition_based_schedule,
            self.opt.evaluate_never_clean,
        ):
            with self.subTest(fn=fn.__name__):
                self.assertNotIn("df", inspect.signature(fn).parameters)

    def test_fixed_run_is_generator_run_under_same_schedule(self):
        run = co.simulate_policy(co.fixed_interval_policy(16), hours=4000, seed=9)
        reference = synthetic_generator.generate_biofouling_dataset(hours=4000, seed=9, cleaning_interval_hours=16 * 24)
        pd.testing.assert_frame_equal(run.df, reference)
        cleaning_hours = run.df.loc[run.df["cleaning_event"] == 1, "time_hours"].tolist()
        self.assertEqual(cleaning_hours, [16 * 24 * k - 1 for k in range(1, len(cleaning_hours) + 1)])
        # it differs from the default dataset, which uses a different schedule
        default = synthetic_generator.generate_biofouling_dataset(hours=4000, seed=9)
        self.assertFalse(np.allclose(run.df["biofilm_thickness_um"], default["biofilm_thickness_um"]))

    def test_cleanings_reset_the_priced_state(self):
        run = co.simulate_policy(co.condition_based_policy(0.7, 0.7), hours=8760, seed=42)
        idx = np.flatnonzero(run.df["cleaning_event"].to_numpy() == 1)
        self.assertGreater(len(idx), 0)
        self.assertEqual(len(idx), len(run.cleaning_types))
        for i in idx[idx > 0]:
            self.assertLess(run.delta_m[i], 0.2 * run.delta_m[i - 1])
        # the priced state is the run's own state
        np.testing.assert_allclose(run.delta_m * 1e6, run.df["biofilm_thickness_um"], atol=1e-3)

    def test_default_policy_reproduces_nominal_dataset(self):
        run = co.simulate_policy(synthetic_generator.default_cleaning_policy)
        self.assertEqual(run.df.to_csv(index=False), (APP_DIR / "db" / "biofouling_synthetic_dataset.csv").read_text())


if __name__ == "__main__":
    unittest.main()
