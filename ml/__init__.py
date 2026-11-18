"""
ml: Universal Differential Equation (UDE) model of biofilm fouling. Synthetic data only;
academic demonstration, not for plant use.
"""

import os

# Must be set before torch is imported anywhere: lets unsupported MPS ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
