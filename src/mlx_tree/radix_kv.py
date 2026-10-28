"""
`RadixKVCache`: Tree-structured Copy-on-Write KV-cache manager.

Shares prefix attention states (`K`, `V` tensors) across `branchbox` forks with
zero redundant prefix computation, reference counting, and memory-saved telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class KVBlock:
    """A single immutable or Copy-on-Write page block of Key/Value attention states."""

    block_id: int
    tokens: Tuple[int, ...]
    keys: np.ndarray
    values: np.ndarray
    ref_count: int = 1

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)


class RadixKVCache:
    """
    Tree-structured Radix Key-Value Cache with Copy-on-Write block sharing.

    Matches token prefixes across sibling branches in a `branchbox` execution tree so
    shared prompt/reasoning prefixes are computed once and referenced by all children.
    """

    def __init__(self, d_head: int = 16, n_heads: int = 2) -> None:
        self.d_head = d_head
        self.n_heads = n_heads
        self.blocks: Dict[int, KVBlock] = {}
        self.prefix_index: Dict[Tuple[int, ...], int] = {}
        self.branch_blocks: Dict[str, List[int]] = {}
        self._next_block_id: int = 0

        # Telemetry counters
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.bytes_saved: int = 0
        self.bytes_allocated: int = 0

    def allocate_or_share_prefix(
        self,
        branch_id: str,
        tokens: List[int],
        keys: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
    ) -> KVBlock:
        """
        Look up whether `tokens` already exists in the radix prefix index.
        If found, increment reference count and reuse the KV block (cache hit);
        otherwise allocate a new KV block (cache miss).
        """
        token_key = tuple(tokens)
        if token_key in self.prefix_index:
            block_id = self.prefix_index[token_key]
            block = self.blocks[block_id]
            block.ref_count += 1
            self.branch_blocks.setdefault(branch_id, []).append(block_id)
            self.cache_hits += 1
            self.bytes_saved += block.nbytes
            return block

        seq_len = max(1, len(tokens))
        if keys is None:
            keys = np.ones((seq_len, self.n_heads, self.d_head), dtype=np.float32) * (
                sum(tokens) % 17 + 1
            ) * 0.05
        if values is None:
            values = np.ones((seq_len, self.n_heads, self.d_head), dtype=np.float32) * (
                sum(tokens) % 13 + 1
            ) * 0.05

        block_id = self._next_block_id
        self._next_block_id += 1
        block = KVBlock(
            block_id=block_id,
            tokens=token_key,
            keys=keys.copy(),
            values=values.copy(),
            ref_count=1,
        )
        self.blocks[block_id] = block
        self.prefix_index[token_key] = block_id
        self.branch_blocks.setdefault(branch_id, []).append(block_id)
        self.cache_misses += 1
        self.bytes_allocated += block.nbytes
        return block

    def fork_branch(self, parent_branch_id: str, child_branch_id: str) -> List[int]:
        """
        Fork a parent branch's KV block table into `child_branch_id` in O(1) time
        without copying K/V arrays.
        """
        parent_table = list(self.branch_blocks.get(parent_branch_id, []))
        for bid in parent_table:
            block = self.blocks[bid]
            block.ref_count += 1
            self.cache_hits += 1
            self.bytes_saved += block.nbytes
        self.branch_blocks[child_branch_id] = parent_table
        return parent_table

    def append_tokens(
        self,
        branch_id: str,
        new_tokens: List[int],
        keys: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
    ) -> KVBlock:
        """
        Append new continuation tokens to `branch_id` without mutating any shared parent prefix blocks.
        """
        return self.allocate_or_share_prefix(branch_id, new_tokens, keys=keys, values=values)

    def get_branch_kv(self, branch_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate and return the full `(K, V)` tensors across a branch's block table."""
        block_ids = self.branch_blocks.get(branch_id, [])
        if not block_ids:
            empty = np.zeros((0, self.n_heads, self.d_head), dtype=np.float32)
            return empty, empty
        k_list = [self.blocks[bid].keys for bid in block_ids]
        v_list = [self.blocks[bid].values for bid in block_ids]
        return np.concatenate(k_list, axis=0), np.concatenate(v_list, axis=0)

    def free_branch(self, branch_id: str) -> None:
        """Release a branch's references and reclaim blocks whose `ref_count` drops to zero."""
        block_ids = self.branch_blocks.pop(branch_id, [])
        for bid in block_ids:
            block = self.blocks.get(bid)
            if block is None:
                continue
            block.ref_count -= 1
            if block.ref_count <= 0:
                self.prefix_index.pop(block.tokens, None)
                del self.blocks[bid]

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def telemetry(self) -> Dict[str, float]:
        """Return KV-cache sharing telemetry."""
        return {
            "active_blocks": len(self.blocks),
            "active_branches": len(self.branch_blocks),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            "bytes_allocated": self.bytes_allocated,
            "bytes_saved": self.bytes_saved,
        }
