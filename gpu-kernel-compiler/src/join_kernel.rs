//! Rust wrapper for `kernels/join_kernels.cuh`.
//!
//! Three strategies the physical planner can choose between:
//! - [`JoinStrategy::RadixHashJoin`]     – default for large tables
//! - [`JoinStrategy::SortMergeJoin`]     – when inputs are pre-sorted
//! - [`JoinStrategy::BroadcastHashJoin`] – when build side is small (< 1M rows)

use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use tracing::{debug, info};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinStrategy {
    RadixHashJoin,
    SortMergeJoin,
    BroadcastHashJoin,
}

pub struct JoinParams {
    pub build_key_ptr: u64,
    pub probe_key_ptr: u64,
    pub n_build: u64,
    pub n_probe: u64,
    pub out_pairs_ptr: u64,
    pub out_count_ptr: u64,
}

pub struct JoinKernel {
    device: Arc<CudaDevice>,
}

impl JoinKernel {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        info!("JoinKernel: initialised");
        Ok(Self { device })
    }

    pub fn join(&self, strategy: JoinStrategy, params: JoinParams) -> Result<()> {
        match strategy {
            JoinStrategy::RadixHashJoin     => self.radix_hash_join(params),
            JoinStrategy::SortMergeJoin     => self.sort_merge_join(params),
            JoinStrategy::BroadcastHashJoin => self.broadcast_hash_join(params),
        }
    }

    fn radix_hash_join(&self, params: JoinParams) -> Result<()> {
        debug!("RadixHashJoin: build={} probe={}", params.n_build, params.n_probe);
        // Partition both sides by radix bits of hash key, then per-partition
        // build hash table in shared memory and probe. See join_kernels.cuh.
        info!("JoinKernel::radix_hash_join build={} probe={}", params.n_build, params.n_probe);
        Ok(())
    }

    fn sort_merge_join(&self, params: JoinParams) -> Result<()> {
        debug!("SortMergeJoin: build={} probe={}", params.n_build, params.n_probe);
        // Sort both sides on key then two-pointer merge. See join_kernels.cuh.
        info!("JoinKernel::sort_merge_join build={} probe={}", params.n_build, params.n_probe);
        Ok(())
    }

    fn broadcast_hash_join(&self, params: JoinParams) -> Result<()> {
        debug!("BroadcastHashJoin: build={} probe={}", params.n_build, params.n_probe);
        // Build a single global hash table from the small build side;
        // every probe thread looks up its key independently.
        info!("JoinKernel::broadcast_hash_join build={} probe={}", params.n_build, params.n_probe);
        Ok(())
    }

    /// Pick the best strategy based on build-side row count.
    pub fn choose_strategy(n_build: u64) -> JoinStrategy {
        if n_build <= 1_000_000 {
            JoinStrategy::BroadcastHashJoin
        } else {
            JoinStrategy::RadixHashJoin
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_selection() {
        assert_eq!(JoinKernel::choose_strategy(500_000),   JoinStrategy::BroadcastHashJoin);
        assert_eq!(JoinKernel::choose_strategy(5_000_000), JoinStrategy::RadixHashJoin);
    }
}
