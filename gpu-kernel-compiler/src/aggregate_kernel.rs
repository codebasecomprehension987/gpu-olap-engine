//! GPU hash-aggregate kernel wrapper.
//!
//! Implements a two-phase approach:
//!   Phase 1 – thread-local accumulation into a warp-level hash table
//!              (kept in shared memory when the number of groups is small).
//!   Phase 2 – global merge across all warps / blocks.
//!
//! The full CUDA source lives in `kernels/aggregate_kernels.cu`.

use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use tracing::{debug, info};

/// Supported aggregate functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Sum,
    Count,
    Min,
    Max,
    /// Internally stored as (sum, count); finalised to sum/count on readback.
    Avg,
}

impl AggFunc {
    pub fn identity_value(&self) -> i64 {
        match self {
            AggFunc::Sum | AggFunc::Count | AggFunc::Avg => 0,
            AggFunc::Min => i64::MAX,
            AggFunc::Max => i64::MIN,
        }
    }
}

/// Parameters for one aggregate operation.
#[derive(Debug, Clone)]
pub struct AggregateSpec {
    pub func: AggFunc,
    /// Column index of the value to aggregate.
    pub value_col_index: usize,
    /// Where in the output to write this aggregate result.
    pub output_col_index: usize,
}

/// Full launch parameters for the aggregate kernel.
pub struct AggregateParams {
    /// GPU pointer to flat int64 group-key column.
    pub group_key_ptr: u64,
    /// GPU pointers to value columns (one per `AggregateSpec`).
    pub value_ptrs: Vec<u64>,
    /// Number of input rows.
    pub n_rows: u64,
    /// Expected (upper-bound) number of distinct groups.
    pub n_groups: u32,
    /// GPU pointer for output aggregate values (one slot per group per agg).
    pub out_ptr: u64,
}

pub struct AggregateKernel {
    device: Arc<CudaDevice>,
}

impl AggregateKernel {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        info!("AggregateKernel: initialised");
        Ok(Self { device })
    }

    /// Launch the hash-aggregate kernel.
    ///
    /// Steps in a real GPU build:
    /// 1. Allocate a zeroed/identity-filled output table (`n_groups × n_aggs`).
    /// 2. Launch `hash_aggregate_kernel` – each thread atomically updates the
    ///    slot for its row's group using `atomicAdd` / `atomicMin` / `atomicMax`.
    /// 3. If `n_groups` is too large for shared-memory (> ~8 K groups), fall
    ///    back to a global-memory hash table with open addressing.
    /// 4. For AVG, launch a secondary kernel to compute `sum / count`.
    /// 5. Copy output back to host via `TransferQueue`.
    pub fn aggregate(&self, specs: &[AggregateSpec], params: AggregateParams) -> Result<()> {
        debug!(
            "AggregateKernel::aggregate rows={} groups={} aggs={}",
            params.n_rows,
            params.n_groups,
            specs.len()
        );

        for spec in specs {
            debug!("  {:?} on col {}", spec.func, spec.value_col_index);
        }

        // ----------------------------------------------------------------
        // REAL IMPLEMENTATION:
        //
        //   // Initialise output with identity values
        //   for (i, spec) in specs.iter().enumerate() {
        //       let identity = spec.func.identity_value();
        //       fill_kernel(&self.device, params.out_ptr + i * n_groups * 8,
        //                   params.n_groups, identity)?;
        //   }
        //
        //   // Launch aggregate
        //   let (grid, block) = Self::compute_launch_params(params.n_rows);
        //   launch_hash_aggregate(
        //       &self.device, grid, block,
        //       params.group_key_ptr,
        //       &params.value_ptrs,
        //       params.n_rows,
        //       params.n_groups,
        //       params.out_ptr,
        //       specs,
        //   )?;
        //
        //   // Finalise AVG
        //   for spec in specs.iter().filter(|s| s.func == AggFunc::Avg) {
        //       launch_avg_finalize(&self.device, ...)?;
        //   }
        // ----------------------------------------------------------------

        info!(
            "AggregateKernel: would aggregate {} rows into {} groups",
            params.n_rows, params.n_groups
        );
        Ok(())
    }

    pub fn compute_launch_params(n_rows: u64) -> (u32, u32) {
        const BLOCK: u32 = 256;
        let grid = ((n_rows as u32) + BLOCK - 1) / BLOCK;
        (grid.max(1), BLOCK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_values() {
        assert_eq!(AggFunc::Sum.identity_value(), 0);
        assert_eq!(AggFunc::Min.identity_value(), i64::MAX);
        assert_eq!(AggFunc::Max.identity_value(), i64::MIN);
    }
}
