//! Conversion between Arrow `RecordBatch` and `Vec<ColumnBuffer>`.

use anyhow::{bail, Context, Result};
use arrow_array::{
    Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::column_buffer::{ColumnBuffer, GpuDataType};

// ---------------------------------------------------------------------------
// Arrow → GPU
// ---------------------------------------------------------------------------

/// Convert an Arrow `RecordBatch` into a `Vec<ColumnBuffer>` (one per column).
///
/// All numeric types are widened to `i64` / `f64` (both stored as 8 bytes).
/// String columns are hashed to `i64` (dict encoding).
/// Null values are replaced with a sentinel (`0` / `i64::MIN`).
pub fn record_batch_to_gpu_buffers(batch: &RecordBatch) -> Result<Vec<ColumnBuffer>> {
    let n_rows = batch.num_rows();
    let mut buffers = Vec::with_capacity(batch.num_columns());

    for (field, col) in batch.schema().fields().iter().zip(batch.columns()) {
        let buf = column_to_buffer(field.name(), col.as_ref(), n_rows)
            .with_context(|| format!("Converting column '{}'", field.name()))?;
        buffers.push(buf);
    }

    Ok(buffers)
}

fn column_to_buffer(name: &str, array: &dyn Array, n_rows: usize) -> Result<ColumnBuffer> {
    let validity: Option<Vec<u8>> = if array.null_count() > 0 {
        Some((0..n_rows).map(|i| if array.is_valid(i) { 1u8 } else { 0u8 }).collect())
    } else {
        None
    };

    match array.data_type() {
        DataType::Int8 => {
            let a = array.as_any().downcast_ref::<Int8Array>().unwrap();
            let data = int_to_i64_bytes(a.iter().map(|v| v.map(|x| x as i64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::Int16 => {
            let a = array.as_any().downcast_ref::<Int16Array>().unwrap();
            let data = int_to_i64_bytes(a.iter().map(|v| v.map(|x| x as i64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>().unwrap();
            let data = int_to_i64_bytes(a.iter().map(|v| v.map(|x| x as i64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let data = int_to_i64_bytes(a.iter());
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::UInt8 => {
            let a = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            let data = int_to_i64_bytes(a.iter().map(|v| v.map(|x| x as i64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::UInt16 => {
            let a = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            let data = int_to_i64_bytes(a.iter().map(|v| v.map(|x| x as i64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::UInt32 => {
            let a = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            let data = int_to_i64_bytes(a.iter().map(|v| v.map(|x| x as i64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::UInt64 => {
            let a = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            let data = int_to_i64_bytes(a.iter().map(|v| v.map(|x| x as i64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Int64, data, n_rows, validity)
        }
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let data = float_to_f64_bytes(a.iter().map(|v| v.map(|x| x as f64)));
            ColumnBuffer::from_bytes(name, GpuDataType::Float64, data, n_rows, validity)
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let data = float_to_f64_bytes(a.iter());
            ColumnBuffer::from_bytes(name, GpuDataType::Float64, data, n_rows, validity)
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().unwrap();
            let data = string_to_hash_bytes(a.iter());
            ColumnBuffer::from_bytes(name, GpuDataType::DictEncodedString, data, n_rows, validity)
        }
        other => bail!("Unsupported column type for GPU conversion: {:?}", other),
    }
}

fn int_to_i64_bytes(iter: impl Iterator<Item = Option<i64>>) -> Vec<u8> {
    iter.flat_map(|v| v.unwrap_or(0i64).to_le_bytes())
        .collect()
}

fn float_to_f64_bytes(iter: impl Iterator<Item = Option<f64>>) -> Vec<u8> {
    iter.flat_map(|v| v.unwrap_or(0.0f64).to_le_bytes())
        .collect()
}

fn string_to_hash_bytes(iter: impl Iterator<Item = Option<&'static str>>) -> Vec<u8> {
    iter.flat_map(|v| {
        let hash: i64 = match v {
            Some(s) => fnv1a_hash(s) as i64,
            None => 0i64,
        };
        hash.to_le_bytes()
    })
    .collect()
}

fn fnv1a_hash(s: &str) -> u64 {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

// ---------------------------------------------------------------------------
// GPU → Arrow
// ---------------------------------------------------------------------------

/// Reconstruct an Arrow `RecordBatch` from GPU column buffers.
///
/// The caller must supply the target `Schema` so we know the intended Arrow
/// types.
pub fn gpu_buffers_to_record_batch(
    buffers: &[ColumnBuffer],
    schema: Arc<Schema>,
) -> Result<RecordBatch> {
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(buffers.len());

    for (buf, field) in buffers.iter().zip(schema.fields()) {
        let col = buffer_to_array(buf, field)?;
        columns.push(col);
    }

    RecordBatch::try_new(schema, columns).context("Building RecordBatch from GPU buffers")
}

fn buffer_to_array(buf: &ColumnBuffer, field: &Field) -> Result<Arc<dyn Array>> {
    let n = buf.n_rows;

    match field.data_type() {
        DataType::Int64 => {
            let mut b = arrow_array::builder::Int64Builder::with_capacity(n);
            for i in 0..n {
                let val = i64::from_le_bytes(buf.data[i * 8..(i + 1) * 8].try_into().unwrap());
                let is_valid = buf.validity.as_ref().map(|v| v[i] == 1).unwrap_or(true);
                if is_valid { b.append_value(val); } else { b.append_null(); }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Float64 => {
            let mut b = arrow_array::builder::Float64Builder::with_capacity(n);
            for i in 0..n {
                let val = f64::from_le_bytes(buf.data[i * 8..(i + 1) * 8].try_into().unwrap());
                let is_valid = buf.validity.as_ref().map(|v| v[i] == 1).unwrap_or(true);
                if is_valid { b.append_value(val); } else { b.append_null(); }
            }
            Ok(Arc::new(b.finish()))
        }
        other => bail!("gpu_buffers_to_record_batch: unsupported target type {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let b = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]));
        RecordBatch::try_new(schema, vec![a, b]).unwrap()
    }

    #[test]
    fn round_trip_metadata() {
        let batch = make_batch();
        let buffers = record_batch_to_gpu_buffers(&batch).unwrap();
        assert_eq!(buffers.len(), 2);
        assert_eq!(buffers[0].n_rows, 5);
        assert_eq!(buffers[0].byte_len(), 5 * 8);
        assert_eq!(buffers[1].dtype, GpuDataType::Float64);
    }

    #[test]
    fn int32_widens_to_i64() {
        let batch = make_batch();
        let buffers = record_batch_to_gpu_buffers(&batch).unwrap();
        let val = i64::from_le_bytes(buffers[0].data[0..8].try_into().unwrap());
        assert_eq!(val, 1);
    }
}
