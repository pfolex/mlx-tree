"""
`VerifiablePolicyModel` and `LoRALayer`.

A self-contained neural policy & value network that works out-of-the-box with NumPy 2.0
on Apple Silicon and automatically detects MLX (`mlx.core`) or PyTorch MPS if installed.
Includes:
- Frozen base weights `W_0` + trainable Low-Rank Adaptation matrices (`lora_A`, `lora_B`)
- Residual activation hook points (`register_hook`) compatible with `steerlens`
  for reading hidden states or injecting Contrastive Activation Addition steering vectors.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def detect_compute_backend() -> str:
    """Detect available Apple Silicon / local array acceleration backend."""
    if importlib.util.find_spec("mlx") is not None:
        return "mlx-apple-silicon"
    if importlib.util.find_spec("torch") is not None:
        return "torch-mps-cpu"
    return "numpy-accelerated"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


class LoRALayer:
    """
    Low-Rank Adaptation linear layer:
        y = x @ (W_0 + scale * (lora_A @ lora_B)) + b
    where `W_0` is frozen (`d_in x d_out`), `lora_A` is (`d_in x rank`),
    and `lora_B` is (`rank x d_out`), initialized so `lora_B = 0` at start.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int = 4,
        alpha: float = 8.0,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / float(rank)

        # Frozen base weights
        self.W_0 = rng.normal(0.0, 1.0 / np.sqrt(d_in), size=(d_in, d_out)).astype(np.float64)
        self.b_0 = np.zeros((d_out,), dtype=np.float64)

        # Trainable LoRA low-rank matrices
        self.lora_A = rng.normal(0.0, 0.08, size=(d_in, rank)).astype(np.float64)
        self.lora_B = np.zeros((rank, d_out), dtype=np.float64)

    def delta_weight(self) -> np.ndarray:
        """Return the effective weight delta: `scale * (lora_A @ lora_B)`."""
        return self.scale * (self.lora_A @ self.lora_B)

    def effective_weight(self) -> np.ndarray:
        """Return `W_0 + delta_weight()`."""
        return self.W_0 + self.delta_weight()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.effective_weight() + self.b_0


# Hook signature: fn(layer_name, activation_array) -> Optional[modified_activation_array]
ActivationHook = Callable[[str, np.ndarray], Optional[np.ndarray]]


class VerifiablePolicyModel:
    """
    Two-layer residual policy & value network with LoRA adapters and `steerlens`
    activation hook registration points.
    """

    def __init__(
        self,
        d_model: int = 16,
        n_actions: int = 6,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        seed: int = 7,
    ) -> None:
        self.d_model = d_model
        self.n_actions = n_actions
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.backend = detect_compute_backend()

        self.hidden_lora = LoRALayer(
            d_in=d_model, d_out=d_model, rank=lora_rank, alpha=lora_alpha, seed=seed
        )
        self.policy_lora = LoRALayer(
            d_in=d_model, d_out=n_actions, rank=lora_rank, alpha=lora_alpha, seed=seed + 1
        )
        rng = np.random.default_rng(seed + 2)
        self.value_head = rng.normal(0.0, 0.1, size=(d_model, 1)).astype(np.float64)

        self._hooks: Dict[str, List[ActivationHook]] = {}
        self.last_activations: Dict[str, np.ndarray] = {}

    def register_hook(self, layer_name: str, hook_fn: ActivationHook) -> None:
        """
        Register an activation read/steering hook on `layer_name` (e.g. `'residual_mid'`).
        If `hook_fn` returns a modified array, it replaces the activation in the forward pass.
        """
        self._hooks.setdefault(layer_name, []).append(hook_fn)

    def clear_hooks(self) -> None:
        """Remove all registered activation hooks."""
        self._hooks.clear()

    def _apply_hooks(self, layer_name: str, act: np.ndarray) -> np.ndarray:
        current = act
        for fn in self._hooks.get(layer_name, []):
            out = fn(layer_name, current)
            if out is not None:
                current = out
        self.last_activations[layer_name] = current.copy()
        return current

    def encode_state(self, state: Any) -> np.ndarray:
        """
        Deterministically encode an arbitrary state object or feature vector into a
        normalized `(d_model,)` float64 embedding vector.
        """
        if isinstance(state, np.ndarray):
            vec = state.astype(np.float64).ravel()
            if vec.size < self.d_model:
                vec = np.pad(vec, (0, self.d_model - vec.size))
            else:
                vec = vec[: self.d_model]
        else:
            text = repr(state)
            vec = np.zeros((self.d_model,), dtype=np.float64)
            for idx, ch in enumerate(text.encode("utf-8")):
                slot = idx % self.d_model
                vec[slot] += ((ch % 31) - 15.0) * 0.1 + (idx + 1) * 0.01
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def forward(
        self, state_or_vec: Any
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Run a forward pass returning `(logits, action_probs, value_estimate, hidden_state)`.
        """
        x = self.encode_state(state_or_vec)
        x = self._apply_hooks("embed", x)

        # Residual block with LoRA adapter
        h_pre = np.tanh(self.hidden_lora.forward(x))
        h = x + h_pre
        h = self._apply_hooks("residual_mid", h)

        logits = self.policy_lora.forward(h)
        logits = self._apply_hooks("logits", logits)
        probs = softmax(logits)
        value = float((h @ self.value_head).item())
        return logits, probs, value, h

    def get_lora_parameters(self) -> Dict[str, np.ndarray]:
        """Return a copy of all trainable LoRA matrices."""
        return {
            "hidden_lora.A": self.hidden_lora.lora_A.copy(),
            "hidden_lora.B": self.hidden_lora.lora_B.copy(),
            "policy_lora.A": self.policy_lora.lora_A.copy(),
            "policy_lora.B": self.policy_lora.lora_B.copy(),
        }

    def set_lora_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Load LoRA adapter matrices into the model."""
        if "hidden_lora.A" in params:
            self.hidden_lora.lora_A = params["hidden_lora.A"].astype(np.float64).copy()
        if "hidden_lora.B" in params:
            self.hidden_lora.lora_B = params["hidden_lora.B"].astype(np.float64).copy()
        if "policy_lora.A" in params:
            self.policy_lora.lora_A = params["policy_lora.A"].astype(np.float64).copy()
        if "policy_lora.B" in params:
            self.policy_lora.lora_B = params["policy_lora.B"].astype(np.float64).copy()
