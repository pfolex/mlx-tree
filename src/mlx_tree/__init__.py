"""
mlx_tree: Apple Silicon Prefix-Shared Tree Inference & Local Parameter Delta Trainer.

Provides:
- `RadixKVCache`: Prefix-shared Copy-on-Write KV cache across `branchbox` tree rollouts.
- `VerifiablePolicyModel` & `LoRALayer`: Modular low-rank policy & value network with
  activation hook points (`register_hook`) for `steerlens`.
- `GRPOLoRATrainer` & `ParameterDelta`: Group Relative Policy Optimization (RLVR) trainer
  that updates compact LoRA deltas and exports portable "Weight Pull Request" artifacts
  for `everbench`.
"""

from .model import LoRALayer, VerifiablePolicyModel, detect_compute_backend
from .radix_kv import KVBlock, RadixKVCache
from .trainer import GRPOLoRATrainer, GRPOUpdateMetrics, ParameterDelta

__all__ = [
    "RadixKVCache",
    "KVBlock",
    "LoRALayer",
    "VerifiablePolicyModel",
    "detect_compute_backend",
    "GRPOLoRATrainer",
    "GRPOUpdateMetrics",
    "ParameterDelta",
]

__version__ = "0.1.0"
