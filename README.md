# `mlx-tree` — Apple Silicon Prefix-Shared Tree Inference & Local Parameter Delta Trainer

**Pillar 4 of the Closed-Loop Verifiable AI Stack**

`mlx-tree` is a prefix-shared tree inference and local **Reinforcement Learning from Verifiable Rewards (RLVR)** engine designed for Apple Silicon unified memory (with out-of-the-box NumPy 2.0 acceleration and automatic MLX / PyTorch MPS backend detection). It shares attention KV caches across `branchbox` forks without redundant computation, exposes residual activation hooks for `steerlens`, and trains compact **Low-Rank Adaptation (LoRA)** parameter deltas ($\Delta W = \frac{\alpha}{r} A B$) via **Group Relative Policy Optimization (GRPO)** that export directly as portable **"Weight Pull Requests"** for `everbench`.

---

## Key Capabilities

1. **Prefix-Shared Tree KV Cache (`RadixKVCache`)**
   - Deduplicates shared prompt and reasoning prefixes across `branchbox` sibling rollouts in $O(1)$ time using reference-counted `KVBlock` tables and Copy-on-Write continuation blocks.
   - Tracks `cache_hit_ratio`, `bytes_allocated`, and `bytes_saved`.
2. **Modular Low-Rank Policy & Value Network (`VerifiablePolicyModel` & `LoRALayer`)**
   - Keeps base weights $W_0$ strictly frozen while attaching low-rank adapters (`lora_A`, `lora_B`).
   - Exposes `register_hook(layer_name, fn)` at `embed`, `residual_mid`, and `logits` layers so `steerlens` can capture internal activations or inject **Contrastive Activation Addition (CAA)** steering vectors at inference time.
3. **RLVR Group Relative Policy Optimization (`GRPOLoRATrainer`)**
   - Computes group-normalized relative advantages $A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$ across sibling rollouts sharing a common fork ancestor.
   - Analytically updates only the LoRA adapter weights with KL regularization so verified winning actions strictly gain probability mass without catastrophic forgetting.
4. **Portable Weight Pull Requests (`ParameterDelta`)**
   - Exports deterministic, SHA-256 fingerprinted `.npz` and `.json` LoRA parameter deltas with per-layer Frobenius norms ready to be audited by `everbench`.

---

## Quickstart

```python
from mlx_tree import RadixKVCache, VerifiablePolicyModel, GRPOLoRATrainer

# 1. Share prefix KV cache across 3 sibling tree branches
kv_cache = RadixKVCache(d_head=16, n_heads=2)
kv_cache.allocate_or_share_prefix("root", [101, 202, 303])
for b in ("child_0", "child_1", "child_2"):
    kv_cache.fork_branch("root", b)
    kv_cache.append_tokens(b, [404, 505])

print("KV Cache Telemetry:", kv_cache.telemetry())

# 2. Train a LoRA Parameter Delta from Verifiable Rewards via GRPO
model = VerifiablePolicyModel(d_model=16, n_actions=6, lora_rank=4)
trainer = GRPOLoRATrainer(model, lr=0.2)

metrics = trainer.step_group(
    state={"goal": "synthesize_verified_patch"},
    actions=[0, 2, 4],
    rewards=[-0.5, 1.0, -0.2],
    verified_flags=[False, True, False],
)
print(f"Winning action prob: {metrics.winning_prob_before:.3f} -> {metrics.winning_prob_after:.3f}")

# 3. Export a Weight Pull Request for EverBench
delta = trainer.export_parameter_delta(model_id="verifiable-coder-lora-v1")
print(f"Exported ParameterDelta digest={delta.digest[:16]}... norm={delta.total_frobenius_norm:.4f}")
```

---

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
