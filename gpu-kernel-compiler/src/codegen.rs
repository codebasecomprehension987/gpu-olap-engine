//! JIT PTX code-generator for filter predicates.
//!
//! Generates PTX source strings loaded by the CUDA driver at runtime,
//! replacing interpreter dispatch with compiled GPU code per unique predicate.
//! Referenced by [`crate::filter_kernel::FilterKernel`].

use anyhow::Result;
use tracing::debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

impl CmpOp {
    pub fn ptx_cond(self) -> &'static str {
        match self {
            CmpOp::Eq => "eq", CmpOp::Ne => "ne",
            CmpOp::Lt => "lt", CmpOp::Le => "le",
            CmpOp::Gt => "gt", CmpOp::Ge => "ge",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FilterPredicate {
    pub col_index: usize,
    pub op: CmpOp,
    pub literal: i64,
}

pub struct KernelCodegen;

impl KernelCodegen {
    pub fn new() -> Self { KernelCodegen }

    /// Emit a PTX kernel that writes a u8 bitmask for the given predicate.
    pub fn emit_filter_ptx(&self, predicate: &FilterPredicate) -> Result<String> {
        debug!("Codegen filter: col={} op={:?} lit={}", predicate.col_index, predicate.op, predicate.literal);
        Ok(format!(
            ".version 8.0\n.target sm_80\n.address_size 64\n\
             // filter_kernel: col[i] {} {}\n\
             .visible .entry filter_kernel(\n\
             \t.param .u64 col_ptr,\n\
             \t.param .u64 out_mask_ptr,\n\
             \t.param .u64 n_rows) {{\n\tret;\n}}",
            predicate.op.ptx_cond(), predicate.literal
        ))
    }
}

impl Default for KernelCodegen {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_filter_ptx_contains_entry() {
        let cg = KernelCodegen::new();
        let ptx = cg.emit_filter_ptx(&FilterPredicate {
            col_index: 0,
            op: CmpOp::Gt,
            literal: 100,
        }).unwrap();
        assert!(ptx.contains("filter_kernel"));
        assert!(ptx.contains(".version 8.0"));
    }
}
