"""Device layer: selection order and dtype rules."""

import os
import unittest
from unittest import mock

import ml  # noqa: F401
import torch

from ml import device as ml_device


class TestDevice(unittest.TestCase):
    def test_mps_fallback_env_set(self):
        self.assertEqual(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1")

    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"ML_DEVICE": "cpu"}):
            self.assertEqual(ml_device.get_device().type, "cpu")

    def test_dtypes(self):
        self.assertEqual(ml_device.default_dtype("mps"), torch.float32)
        self.assertEqual(ml_device.default_dtype("mps", prefer_float64=True), torch.float32)
        self.assertEqual(ml_device.default_dtype("cpu"), torch.float32)
        self.assertEqual(ml_device.default_dtype("cpu", prefer_float64=True), torch.float64)


if __name__ == "__main__":
    unittest.main()
