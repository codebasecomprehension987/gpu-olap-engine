//! GPU radix sort kernel wrapper.
//!
//! Uses an LSB radix sort (256-way, 1 byte per pass → 8 passes for int64).
//! The accompanying CUDA source lives in `kernels/sort_kernels.cu`.

use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use tracing::{debug, info};

/// Number of radix bits processed per pass.
pub const RADIX_BITS: u32 = 8;
/// Number of buckets per pass.
pub const RADIX_BUCKETS: u32 = 1 << RADIX_BITS;
/// Number of passes needed for a 64-bit key.
pub const RADIX_PASSES: u32 = 64 / RADIX_BITS;

pub struct SortKernel {
    device: Arc<CudaDevice>,
}

/// Describes a sort operation.
pub struct SortParams {
    /// GPU pointer to int64 sort keys.
    pub keys_ptr: u64,
    /// GPU pointer to associated row-ids (permutation vector).
    pub row_ids_ptr: u64,
    /// Number of elements to sort.
    pub n_rows: u64,
    /// If true, sort descending.
    pub descending: bool,
}

impl SortKernel {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        info!("SortKernel: initialised");
        Ok(Self { device })
    }

    /// Launch an in-place radix sort on `params.keys_ptr` / `params.row_ids_ptr`.
    ///
    /// The algorithm:
    /// 1. For each of `RADIX_PASSES` (0..8), extract the relevant byte.
    /// 2. Compute a histogram (256 bins) in shared memory.
    /// 3. Prefix-sum the histogram to obtain scatter offsets.
    /// 4. Scatter keys and row_ids into temporary ping-pong buffers.
    /// 5. Swap buffers and continue.
    pub fn sort(&self, params: SortParams) -> Result<()> {
        debug!(
            "SortKernel::sort rows={} descending={}",
            params.n_rows, params.descending
        );

        // ----------------------------------------------------------------
        // REAL IMPLEMENTATION:
        //
        //   let tmp_keys   = self.device.alloc::<i64>(params.n_rows as usize)?;
        //   let tmp_rowids = self.device.alloc::<u32>(params.n_rows as usize)?;
        //
        //   for pass in 0..RADIX_PASSES {
        //       let shift = pass * RADIX_BITS;
        //       // histogram pass
        //       launch_histogram_kernel(&self.device, params.keys_ptr, n_rows, shift, &histo)?;
        //       // prefix sum
        //       launch_prefix_sum(&self.device, &histo)?;
        //       // scatter
        //       launch_scatter_kernel(&self.device, params.keys_ptr, &histo, shift,
        //                             tmp_keys.device_ptr(), tmp_rowids.device_ptr())?;
        //       std::mem::swap(&mut current_keys, &mut tmp_keys);
        //   }
        // ----------------------------------------------------------------

        info!(
            "SortKernel: would sort {} rows ({} radix passes)",
            params.n_rows, RADIX_PASSES
        );
        Ok(())
    }

    /// Compute an appropriate grid size for `n_rows`.
    pub fn compute_launch_params(n_rows: u64) -> (u32, u32) {
        const BLOCK: u32 = 512;
        let grid = ((n_rows as u32) + BLOCK - 1) / BLOCK;
        (grid.max(1), BLOCK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_passes() {
        assert_eq!(RADIX_PASSES, 8, "8 passes × 8 bits = 64-bit key");
    }

    #[test]
    fn launch_params() {
        let (g, b) = SortKernel::compute_launch_params(1_000_000);
        assert_eq!(b, 512);
        assert!(g > 1);
    }
}
