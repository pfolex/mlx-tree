"""
Unit tests for `mlx-tree`:
1. `RadixKVCache` prefix sharing, O(1) branch forking, reference counting, and bytes_saved telemetry.
2. `VerifiablePolicyModel` forward pass and `register_hook` steering compatibility.
3. `GRPOLoRATrainer` RLVR update: verifies that frozen base weights `W_0` never mutate while
   policy probability on winning verifiable actions strictly increases after a GRPO step.
4. `ParameterDelta` export, digest verification, and `.npz` / `.json` round-trip serialization.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_tree import (
    GRPOLoRATrainer,
    ParameterDelta,
    RadixKVCache,
    VerifiablePolicyModel,
)


class TestMLXTree(unittest.TestCase):
    def test_radix_kv_cache_prefix_sharing_and_ref_counting(self) -> None:
        cache = RadixKVCache(d_head=8, n_heads=2)
        prompt_tokens = [101, 2054, 2003, 1037, 7953]

        # Root branch allocates prompt prefix (cache miss)
        root_block = cache.allocate_or_share_prefix("branch_root", prompt_tokens)
        self.assertEqual(cache.cache_misses, 1)
        self.assertEqual(cache.cache_hits, 0)
        self.assertEqual(root_block.ref_count, 1)

        # Fork 3 child branches sharing the prompt prefix in O(1)
        for i in range(3):
            cache.fork_branch("branch_root", f"branch_child_{i}")
            cache.append_tokens(f"branch_child_{i}", [500 + i, 600 + i])

        self.assertEqual(root_block.ref_count, 4)
        self.assertGreater(cache.bytes_saved, 0)
        self.assertGreater(cache.cache_hit_ratio, 0.35)

        # Verify concatenated KV tensor dimensions for child 0 (5 prompt + 2 continuation = 7 tokens)
        k_tensor, v_tensor = cache.get_branch_kv("branch_child_0")
        self.assertEqual(k_tensor.shape, (7, 2, 8))
        self.assertEqual(v_tensor.shape, (7, 2, 8))

        # Free branches and verify reference count cleanup
        cache.free_branch("branch_child_0")
        self.assertEqual(root_block.ref_count, 3)

    def test_model_forward_and_activation_steering_hooks(self) -> None:
        model = VerifiablePolicyModel(d_model=16, n_actions=5, lora_rank=4)
        state = {"task": "synthesize_verified_sort", "step": 1}

        logits_base, probs_base, val_base, hidden_base = model.forward(state)
        self.assertEqual(logits_base.shape, (5,))
        self.assertAlmostEqual(float(np.sum(probs_base)), 1.0, places=6)
        self.assertIsInstance(val_base, float)
        self.assertEqual(hidden_base.shape, (16,))

        # Register a steering hook on 'residual_mid' (simulating steerlens CAA injection)
        steering_vec = np.ones((16,), dtype=np.float64) * 0.5
        captured = {}

        def steer_hook(layer_name: str, act: np.ndarray):
            captured[layer_name] = act.copy()
            return act + steering_vec

        model.register_hook("residual_mid", steer_hook)
        logits_steered, probs_steered, _, _ = model.forward(state)
        self.assertIn("residual_mid", captured)
        self.assertFalse(np.allclose(probs_base, probs_steered))

    def test_grpo_lora_update_strictly_increases_winning_action_probability(self) -> None:
        model = VerifiablePolicyModel(d_model=16, n_actions=6, lora_rank=4, seed=19)
        trainer = GRPOLoRATrainer(model, lr=0.2, kl_coeff=0.01)

        # Snapshot frozen base weights W_0 before training
        w0_hidden_before = model.hidden_lora.W_0.copy()
        w0_policy_before = model.policy_lora.W_0.copy()

        state = {"goal": "prove_invariant_x_gt_0", "context": "loop_body"}
        # Group of 4 sibling rollouts from branchbox: action 3 is verified winner!
        actions = [0, 1, 3, 5]
        rewards = [-0.6, -0.2, 1.0, -0.4]
        verified = [False, False, True, False]

        metrics = trainer.step_group(
            state=state,
            actions=actions,
            rewards=rewards,
            verified_flags=verified,
            num_inner_epochs=5,
        )

        # 1. Policy probability on winning verifiable action (action 3) must strictly increase
        self.assertGreater(metrics.winning_prob_after, metrics.winning_prob_before)
        self.assertGreater(metrics.delta_frobenius_norm, 0.0)

        # 2. Frozen base weights W_0 must remain 100% bit-for-bit identical
        np.testing.assert_array_equal(model.hidden_lora.W_0, w0_hidden_before)
        np.testing.assert_array_equal(model.policy_lora.W_0, w0_policy_before)

    def test_parameter_delta_serialization_roundtrip(self) -> None:
        model = VerifiablePolicyModel(d_model=12, n_actions=4, lora_rank=3, seed=99)
        trainer = GRPOLoRATrainer(model, lr=0.15)
        trainer.step_group(
            state={"prompt": "test"},
            actions=[0, 2, 1],
            rewards=[-0.5, 1.0, 0.0],
            verified_flags=[False, True, False],
        )

        delta = trainer.export_parameter_delta(
            model_id="mlx-tree-test-lora",
            metadata={"benchmark": "vericode_arith"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "weight_pr.npz"
            json_path = Path(tmpdir) / "weight_pr.json"

            delta.save(npz_path)
            delta.save(json_path)

            loaded_npz = ParameterDelta.load(npz_path)
            loaded_json = ParameterDelta.load(json_path)

            self.assertEqual(delta.digest, loaded_npz.digest)
            self.assertEqual(delta.digest, loaded_json.digest)
            self.assertAlmostEqual(
                delta.total_frobenius_norm, loaded_npz.total_frobenius_norm, places=7
            )

            # Apply loaded delta to a fresh model and verify identical action probabilities
            fresh_model = VerifiablePolicyModel(d_model=12, n_actions=4, lora_rank=3, seed=99)
            fresh_trainer = GRPOLoRATrainer(fresh_model)
            fresh_trainer.apply_parameter_delta(loaded_npz)

            _, probs_orig, _, _ = model.forward({"prompt": "test"})
            _, probs_loaded, _, _ = fresh_model.forward({"prompt": "test"})
            np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
