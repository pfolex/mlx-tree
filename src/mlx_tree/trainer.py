"""
`GRPOLoRATrainer` and `ParameterDelta`.

Implements Reinforcement Learning from Verifiable Rewards (RLVR) via Group Relative
Policy Optimization (GRPO) on the compact LoRA adapter matrices (`lora_A`, `lora_B`).
Exports portable `ParameterDelta` artifacts ready to be verified by `everbench` as a
"Weight Pull Request".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .model import VerifiablePolicyModel, softmax


@dataclass
class ParameterDelta:
    """
    Portable, verifiable LoRA parameter delta artifact ("Weight Pull Request").
    """

    model_id: str
    lora_rank: int
    lora_alpha: float
    tensors: Dict[str, np.ndarray]
    layer_norms: Dict[str, float]
    total_frobenius_norm: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Deterministic SHA-256 fingerprint of the parameter delta tensors."""
        hasher = hashlib.sha256()
        for key in sorted(self.tensors.keys()):
            hasher.update(key.encode("utf-8"))
            hasher.update(np.ascontiguousarray(self.tensors[key]).tobytes())
        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize delta to a JSON-compatible dictionary."""
        return {
            "model_id": self.model_id,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "digest": self.digest,
            "layer_norms": self.layer_norms,
            "total_frobenius_norm": self.total_frobenius_norm,
            "metadata": self.metadata,
            "tensors": {k: v.tolist() for k, v in self.tensors.items()},
        }

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save parameter delta to disk as either `.npz` (with companion metadata) or `.json`.
        """
        out_path = Path(path)
        if out_path.suffix == ".npz":
            np.savez(out_path, **self.tensors)
            meta_path = out_path.with_suffix(".meta.json")
            meta_payload = {
                "model_id": self.model_id,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "digest": self.digest,
                "layer_norms": self.layer_norms,
                "total_frobenius_norm": self.total_frobenius_norm,
                "metadata": self.metadata,
            }
            meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        else:
            out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParameterDelta":
        """Load a `ParameterDelta` from a `.npz` or `.json` file."""
        in_path = Path(path)
        if in_path.suffix == ".npz":
            archive = np.load(in_path)
            tensors = {k: archive[k].astype(np.float64) for k in archive.files}
            meta_path = in_path.with_suffix(".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return cls(
                model_id=meta["model_id"],
                lora_rank=int(meta["lora_rank"]),
                lora_alpha=float(meta["lora_alpha"]),
                tensors=tensors,
                layer_norms={k: float(v) for k, v in meta["layer_norms"].items()},
                total_frobenius_norm=float(meta["total_frobenius_norm"]),
                metadata=meta.get("metadata", {}),
            )
        raw = json.loads(in_path.read_text(encoding="utf-8"))
        tensors = {k: np.array(v, dtype=np.float64) for k, v in raw["tensors"].items()}
        return cls(
            model_id=raw["model_id"],
            lora_rank=int(raw["lora_rank"]),
            lora_alpha=float(raw["lora_alpha"]),
            tensors=tensors,
            layer_norms={k: float(v) for k, v in raw["layer_norms"].items()},
            total_frobenius_norm=float(raw["total_frobenius_norm"]),
            metadata=raw.get("metadata", {}),
        )


@dataclass
class GRPOUpdateMetrics:
    """Telemetry returned after a GRPO LoRA optimization step."""

    group_size: int
    mean_reward: float
    std_reward: float
    advantages: List[float]
    winning_prob_before: float
    winning_prob_after: float
    kl_divergence: float
    delta_frobenius_norm: float


class GRPOLoRATrainer:
    """
    Group Relative Policy Optimization (GRPO) trainer for `VerifiablePolicyModel` LoRA adapters.

    Given a group of sibling rollouts `(state, action_idx, reward, verified)` branched from
    a common ancestor:
    1. Computes group-normalized relative advantages:
           A_i = (r_i - mean(r)) / (std(r) + eps)
    2. Computes analytical policy-gradient updates for the LoRA adapters (`policy_lora`
       and `hidden_lora`) with KL-divergence regularization against the reference policy.
    3. Strictly leaves frozen base weights `W_0` untouched.
    """

    def __init__(
        self,
        model: VerifiablePolicyModel,
        lr: float = 0.15,
        kl_coeff: float = 0.01,
        VerifiedBonus: float = 0.5,
    ) -> None:
        self.model = model
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.verified_bonus = VerifiedBonus
        self.reference_params = model.get_lora_parameters()
        self.steps_completed: int = 0

    def compute_group_advantages(
        self, rewards: Sequence[float], verified_flags: Optional[Sequence[bool]] = None
    ) -> np.ndarray:
        """
        Compute group-normalized GRPO advantages:
            A_i = (R_i - mean(R)) / (std(R) + 1e-6)
        """
        r = np.array(rewards, dtype=np.float64)
        if verified_flags is not None:
            bonuses = np.array(
                [self.verified_bonus if v else 0.0 for v in verified_flags],
                dtype=np.float64,
            )
            r = r + bonuses
        mean_r = float(np.mean(r))
        std_r = float(np.std(r))
        if std_r < 1e-8:
            return np.zeros_like(r)
        return (r - mean_r) / (std_r + 1e-6)

    def step_group(
        self,
        state: Any,
        actions: Sequence[int],
        rewards: Sequence[float],
        verified_flags: Optional[Sequence[bool]] = None,
        num_inner_epochs: int = 4,
    ) -> GRPOUpdateMetrics:
        """
        Perform a GRPO LoRA parameter update on a group of sibling rollouts sharing `state`.
        """
        if len(actions) != len(rewards) or len(actions) < 2:
            raise ValueError("GRPO requires at least 2 sibling rollouts with matching actions/rewards")

        advantages = self.compute_group_advantages(rewards, verified_flags)
        best_idx = int(np.argmax(advantages))
        winning_action = int(actions[best_idx])

        _, probs_before, _, _ = self.model.forward(state)
        winning_prob_before = float(probs_before[winning_action])
        ref_probs = probs_before.copy()

        x = self.model.encode_state(state)

        for _ in range(num_inner_epochs):
            # Forward through hidden LoRA
            z_hidden = self.model.hidden_lora.forward(x)
            h_pre = np.tanh(z_hidden)
            h = x + h_pre
            logits = self.model.policy_lora.forward(h)
            probs = softmax(logits)

            # Accumulate gradient of GRPO objective w.r.t. logits:
            # dL/dlogits = sum_i A_i * (e_{a_i} - probs) - kl_coeff * (probs - ref_probs)
            grad_logits = np.zeros_like(logits)
            for act_idx, adv in zip(actions, advantages):
                one_hot = np.zeros_like(logits)
                one_hot[int(act_idx)] = 1.0
                grad_logits += float(adv) * (one_hot - probs)

            grad_logits -= self.kl_coeff * (probs - ref_probs)
            grad_logits /= float(len(actions))

            # Policy LoRA gradients: logits = h @ (W_0 + scale * A_p @ B_p)
            scale_p = self.model.policy_lora.scale
            A_p = self.model.policy_lora.lora_A
            B_p = self.model.policy_lora.lora_B

            # dL/dB_p = scale * A_p^T @ (h^T @ grad_logits)
            outer_h_g = np.outer(h, grad_logits)
            grad_B_p = scale_p * (A_p.T @ outer_h_g)
            grad_A_p = scale_p * (outer_h_g @ B_p.T)

            # Backpropagate into hidden state h and hidden LoRA
            W_eff_p = self.model.policy_lora.effective_weight()
            grad_h = W_eff_p @ grad_logits
            grad_z_hidden = grad_h * (1.0 - h_pre**2)

            scale_h = self.model.hidden_lora.scale
            A_h = self.model.hidden_lora.lora_A
            B_h = self.model.hidden_lora.lora_B
            outer_x_gh = np.outer(x, grad_z_hidden)
            grad_B_h = scale_h * (A_h.T @ outer_x_gh)
            grad_A_h = scale_h * (outer_x_gh @ B_h.T)

            # Ascent step on LoRA parameters only (W_0 stays strictly frozen)
            self.model.policy_lora.lora_B += self.lr * grad_B_p
            self.model.policy_lora.lora_A += self.lr * grad_A_p
            self.model.hidden_lora.lora_B += self.lr * grad_B_h
            self.model.hidden_lora.lora_A += self.lr * grad_A_h

        self.steps_completed += 1
        _, probs_after, _, _ = self.model.forward(state)
        winning_prob_after = float(probs_after[winning_action])

        # Compute KL(probs_after || probs_before)
        kl_div = float(
            np.sum(probs_after * np.log((probs_after + 1e-12) / (probs_before + 1e-12)))
        )
        delta = self.export_parameter_delta()

        return GRPOUpdateMetrics(
            group_size=len(actions),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            advantages=[round(float(a), 4) for a in advantages],
            winning_prob_before=winning_prob_before,
            winning_prob_after=winning_prob_after,
            kl_divergence=kl_div,
            delta_frobenius_norm=delta.total_frobenius_norm,
        )

    def export_parameter_delta(
        self,
        model_id: str = "verifiable-policy-lora-v1",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParameterDelta:
        """
        Export the trained LoRA parameter delta (`ParameterDelta`) ready for `everbench`
        Weight Pull Request verification.
        """
        tensors = {
            "hidden_lora.A": self.model.hidden_lora.lora_A.copy(),
            "hidden_lora.B": self.model.hidden_lora.lora_B.copy(),
            "hidden_lora.delta_W": self.model.hidden_lora.delta_weight(),
            "policy_lora.A": self.model.policy_lora.lora_A.copy(),
            "policy_lora.B": self.model.policy_lora.lora_B.copy(),
            "policy_lora.delta_W": self.model.policy_lora.delta_weight(),
        }
        layer_norms = {
            name: float(np.linalg.norm(arr)) for name, arr in tensors.items()
        }
        total_norm = float(
            np.sqrt(
                layer_norms["hidden_lora.delta_W"] ** 2
                + layer_norms["policy_lora.delta_W"] ** 2
            )
        )
        meta = {"steps_completed": self.steps_completed, "backend": self.model.backend}
        if metadata:
            meta.update(metadata)
        return ParameterDelta(
            model_id=model_id,
            lora_rank=self.model.lora_rank,
            lora_alpha=self.model.lora_alpha,
            tensors=tensors,
            layer_norms=layer_norms,
            total_frobenius_norm=total_norm,
            metadata=meta,
        )

    def apply_parameter_delta(self, delta: ParameterDelta) -> None:
        """Load a `ParameterDelta` into the model's LoRA adapters."""
        self.model.set_lora_parameters(delta.tensors)
