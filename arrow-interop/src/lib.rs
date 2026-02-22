//! Arrow ↔ GPU columnar conversion utilities.
//!
//! This crate provides zero-copy (where possible) conversion between Apache
//! Arrow `RecordBatch`es and the flat, column-major `u8` buffers that GPU
//! kernels consume.
//!
//! # Layout contract
//!
//! Each column is stored as a **flat, row-contiguous** byte buffer:
//!
//! ```text
//! [  value[0]  |  value[1]  | ... |  value[n-1]  ]
//! ```
//!
//! Null bitmaps from Arrow are *not* currently forwarded to the GPU; columns
//! with nulls are materialised with a sentinel (0 / `i64::MIN`) before
//! transfer.

pub mod column_buffer;
pub mod record_batch_convert;
pub mod schema_utils;

pub use column_buffer::ColumnBuffer;
pub use record_batch_convert::{record_batch_to_gpu_buffers, gpu_buffers_to_record_batch};
pub use schema_utils::SchemaExt;
