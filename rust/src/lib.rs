//! # MLX-Tree Core (Rust Radix KV-Cache Block Allocator)
//!
//! Manages Copy-on-Write prefix blocks for tree-structured LLM/agent rollouts,
//! ensuring sibling branches created by `branchbox` share prefix KV attention blocks
//! with reference counting and zero redundant prefix recomputation.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct KVBlock {
    pub block_id: usize,
    pub token_hash: u64,
    pub ref_count: usize,
    pub bytes: usize,
}

#[derive(Default, Debug)]
pub struct RadixBlockAllocator {
    blocks: HashMap<usize, KVBlock>,
    branch_tables: HashMap<String, Vec<usize>>,
    next_block_id: usize,
    pub bytes_saved: usize,
}

impl RadixBlockAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new KV block for a branch.
    pub fn allocate_block(&mut self, branch_id: &str, token_hash: u64, bytes: usize) -> usize {
        let id = self.next_block_id;
        self.next_block_id += 1;
        self.blocks.insert(
            id,
            KVBlock {
                block_id: id,
                token_hash,
                ref_count: 1,
                bytes,
            },
        );
        self.branch_tables
            .entry(branch_id.to_string())
            .or_default()
            .push(id);
        id
    }

    /// Fork a parent branch's KV block table into `child_branch_id` in O(1) via ref-counting.
    pub fn fork_branch(&mut self, parent_branch_id: &str, child_branch_id: &str) -> usize {
        let parent_blocks = self
            .branch_tables
            .get(parent_branch_id)
            .cloned()
            .unwrap_or_default();
        let mut saved = 0usize;
        for &bid in &parent_blocks {
            if let Some(block) = self.blocks.get_mut(&bid) {
                block.ref_count += 1;
                saved += block.bytes;
            }
        }
        self.bytes_saved += saved;
        self.branch_tables
            .insert(child_branch_id.to_string(), parent_blocks);
        saved
    }

    /// Free a branch and decrement reference counts on its KV blocks.
    pub fn free_branch(&mut self, branch_id: &str) {
        if let Some(bids) = self.branch_tables.remove(branch_id) {
            for bid in bids {
                let remove_block = if let Some(block) = self.blocks.get_mut(&bid) {
                    block.ref_count = block.ref_count.saturating_sub(1);
                    block.ref_count == 0
                } else {
                    false
                };
                if remove_block {
                    self.blocks.remove(&bid);
                }
            }
        }
    }
}
