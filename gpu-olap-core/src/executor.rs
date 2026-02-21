use anyhow::{Context, Result};
use arrow::datatypes::DataType;
use arrow_array::{
    builder::{Float64Builder, Int64Builder},
    Array, Float64Array, Int64Array, RecordBatch,
};
use arrow_schema::{Field, Schema};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::physical_plan::{AggregateExpr, PhysicalExpr, PhysicalPlan};
use crate::EngineConfig;

/// Execute physical plan on GPU (or CPU fallback when no device is present).
pub async fn execute(
    plan: PhysicalPlan,
    config: &EngineConfig,
) -> Result<Vec<RecordBatch>> {
    info!("Executing physical plan");
    debug!("Plan: {:?}", plan);

    let executor = GpuExecutor::new(config)?;
    executor.execute(plan).await
}

struct GpuExecutor {
    config: EngineConfig,
}

impl GpuExecutor {
    fn new(config: &EngineConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    async fn execute(&self, plan: PhysicalPlan) -> Result<Vec<RecordBatch>> {
        match plan {
            PhysicalPlan::GpuTableScan {
                table_name,
                schema,
                projection,
            } => {
                self.execute_table_scan(&table_name, &schema, projection.as_ref())
                    .await
            }

            PhysicalPlan::GpuFilter { input, predicate } => {
                let input_data = self.execute(*input).await?;
                self.execute_filter(input_data, predicate).await
            }

            PhysicalPlan::GpuProjection { input, exprs } => {
                let input_data = self.execute(*input).await?;
                self.execute_projection(input_data, exprs).await
            }

            PhysicalPlan::GpuHashJoin {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
                schema,
            } => {
                let left_data = self.execute(*left).await?;
                let right_data = self.execute(*right).await?;
                self.execute_hash_join(left_data, right_data, left_keys, right_keys, join_type, &schema)
                    .await
            }

            PhysicalPlan::GpuSortMergeJoin {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
                schema,
            } => {
                let left_data = self.execute(*left).await?;
                let right_data = self.execute(*right).await?;
                self.execute_sort_merge_join(
                    left_data, right_data, left_keys, right_keys, join_type, &schema,
                )
                .await
            }

            PhysicalPlan::GpuAggregate {
                input,
                group_by,
                aggr_exprs,
                schema,
            } => {
                let input_data = self.execute(*input).await?;
                self.execute_aggregate(input_data, group_by, aggr_exprs, &schema)
                    .await
            }

            PhysicalPlan::GpuSort { input, exprs } => {
                let input_data = self.execute(*input).await?;
                self.execute_sort(input_data, exprs).await
            }
        }
    }

    // -----------------------------------------------------------------------
    // Table scan
    // -----------------------------------------------------------------------

    async fn execute_table_scan(
        &self,
        table_name: &str,
        schema: &arrow_schema::Schema,
        projection: Option<&Vec<usize>>,
    ) -> Result<Vec<RecordBatch>> {
        info!("GPU table scan: '{}'", table_name);

        // Build an empty RecordBatch with the correct schema so that the rest
        // of the pipeline has something to work with.  A real implementation
        // would page through the Parquet file and stream chunks to the GPU.
        let projected_schema = match projection {
            Some(indices) => {
                let fields: Vec<Field> = indices
                    .iter()
                    .map(|&i| schema.field(i).clone())
                    .collect();
                Arc::new(Schema::new(fields))
            }
            None => Arc::new(schema.clone()),
        };

        warn!(
            "Table scan for '{}' returning empty batch (no Parquet path in physical plan). \
             Wire up Catalog in a real deployment.",
            table_name
        );

        let columns: Vec<Arc<dyn Array>> = projected_schema
            .fields()
            .iter()
            .map(|f| -> Arc<dyn Array> {
                match f.data_type() {
                    DataType::Float32 | DataType::Float64 => {
                        Arc::new(Float64Array::from(Vec::<f64>::new()))
                    }
                    _ => Arc::new(Int64Array::from(Vec::<i64>::new())),
                }
            })
            .collect();

        let batch = RecordBatch::try_new(projected_schema, columns)
            .context("Building empty scan RecordBatch")?;

        Ok(vec![batch])
    }

    // -----------------------------------------------------------------------
    // Filter (CPU reference implementation – GPU version uses codegen PTX)
    // -----------------------------------------------------------------------

    async fn execute_filter(
        &self,
        input: Vec<RecordBatch>,
        predicate: PhysicalExpr,
    ) -> Result<Vec<RecordBatch>> {
        info!("Executing filter");

        // For each batch, compute a boolean mask, then compact.
        let mut out = Vec::with_capacity(input.len());
        for batch in input {
            let mask = eval_filter_mask(&batch, &predicate)?;
            let filtered = filter_record_batch(&batch, &mask)?;
            if filtered.num_rows() > 0 {
                out.push(filtered);
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Projection
    // -----------------------------------------------------------------------

    async fn execute_projection(
        &self,
        input: Vec<RecordBatch>,
        exprs: Vec<PhysicalExpr>,
    ) -> Result<Vec<RecordBatch>> {
        info!("Executing projection ({} exprs)", exprs.len());

        let mut out = Vec::with_capacity(input.len());
        for batch in &input {
            let columns: Result<Vec<Arc<dyn Array>>> = exprs
                .iter()
                .map(|e| eval_expr(e, batch))
                .collect();
            let columns = columns?;

            // Derive schema from the expression results
            let fields: Vec<Field> = exprs
                .iter()
                .zip(columns.iter())
                .map(|(e, col)| {
                    let name = expr_name(e);
                    Field::new(name, col.data_type().clone(), col.null_count() > 0)
                })
                .collect();

            let schema = Arc::new(Schema::new(fields));
            let rb = RecordBatch::try_new(schema, columns)?;
            out.push(rb);
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Hash join (CPU reference using nested-loop; GPU uses join_kernels.cuh)
    // -----------------------------------------------------------------------

    async fn execute_hash_join(
        &self,
        left: Vec<RecordBatch>,
        right: Vec<RecordBatch>,
        left_keys: Vec<PhysicalExpr>,
        right_keys: Vec<PhysicalExpr>,
        join_type: crate::logical_plan::JoinType,
        schema: &arrow_schema::Schema,
    ) -> Result<Vec<RecordBatch>> {
        info!("Executing hash join");

        if left.is_empty() || right.is_empty() {
            return Ok(vec![]);
        }

        // Build phase: extract left keys as i64 values
        // Probe phase: for each right row, look up in hash map
        use std::collections::HashMap;
        let mut build_map: HashMap<i64, Vec<usize>> = HashMap::new();
        let mut left_row_offset = 0usize;

        for batch in &left {
            if left_keys.is_empty() {
                break;
            }
            let key_col = eval_expr(&left_keys[0], batch)?;
            if let Some(arr) = key_col.as_any().downcast_ref::<Int64Array>() {
                for i in 0..arr.len() {
                    let k = arr.value(i);
                    build_map.entry(k).or_default().push(left_row_offset + i);
                }
            }
            left_row_offset += batch.num_rows();
        }

        // For this reference impl, just return an empty schema-correct batch
        let empty_cols: Vec<Arc<dyn Array>> = schema
            .fields()
            .iter()
            .map(|f| -> Arc<dyn Array> { Arc::new(Int64Array::from(Vec::<i64>::new())) })
            .collect();

        let rb = RecordBatch::try_new(Arc::new(schema.clone()), empty_cols)
            .context("Hash join output batch")?;
        Ok(vec![rb])
    }

    // -----------------------------------------------------------------------
    // Sort-merge join
    // -----------------------------------------------------------------------

    async fn execute_sort_merge_join(
        &self,
        left: Vec<RecordBatch>,
        right: Vec<RecordBatch>,
        _left_keys: Vec<PhysicalExpr>,
        _right_keys: Vec<PhysicalExpr>,
        _join_type: crate::logical_plan::JoinType,
        schema: &arrow_schema::Schema,
    ) -> Result<Vec<RecordBatch>> {
        info!("Executing sort-merge join");

        // Placeholder: return empty, schema-correct batch
        let empty_cols: Vec<Arc<dyn Array>> = schema
            .fields()
            .iter()
            .map(|_| -> Arc<dyn Array> { Arc::new(Int64Array::from(Vec::<i64>::new())) })
            .collect();

        let rb = RecordBatch::try_new(Arc::new(schema.clone()), empty_cols)?;
        Ok(vec![rb])
    }

    // -----------------------------------------------------------------------
    // Aggregation (CPU reference)
    // -----------------------------------------------------------------------

    async fn execute_aggregate(
        &self,
        input: Vec<RecordBatch>,
        group_by: Vec<PhysicalExpr>,
        aggr_exprs: Vec<AggregateExpr>,
        schema: &arrow_schema::Schema,
    ) -> Result<Vec<RecordBatch>> {
        info!("Executing aggregate");
        use std::collections::BTreeMap;

        // Simple single-group (no GROUP BY keys) aggregation for reference impl
        let mut sum_accum: Vec<f64> = vec![0.0; aggr_exprs.len()];
        let mut count_accum: Vec<u64> = vec![0; aggr_exprs.len()];
        let mut min_accum: Vec<f64> = vec![f64::MAX; aggr_exprs.len()];
        let mut max_accum: Vec<f64> = vec![f64::MIN; aggr_exprs.len()];

        for batch in &input {
            for (i, agg) in aggr_exprs.iter().enumerate() {
                let col = eval_agg_input(agg, batch)?;
                for v in col {
                    count_accum[i] += 1;
                    sum_accum[i] += v;
                    if v < min_accum[i] {
                        min_accum[i] = v;
                    }
                    if v > max_accum[i] {
                        max_accum[i] = v;
                    }
                }
            }
        }

        // Build output batch
        let columns: Vec<Arc<dyn Array>> = aggr_exprs
            .iter()
            .enumerate()
            .map(|(i, agg)| -> Arc<dyn Array> {
                let val = match agg {
                    AggregateExpr::Sum { .. } => sum_accum[i],
                    AggregateExpr::Count { .. } => count_accum[i] as f64,
                    AggregateExpr::Min { .. } => min_accum[i],
                    AggregateExpr::Max { .. } => max_accum[i],
                    AggregateExpr::Avg { .. } => {
                        if count_accum[i] > 0 {
                            sum_accum[i] / count_accum[i] as f64
                        } else {
                            0.0
                        }
                    }
                };
                Arc::new(Float64Array::from(vec![val]))
            })
            .collect();

        let rb = RecordBatch::try_new(Arc::new(schema.clone()), columns)
            .unwrap_or_else(|_| RecordBatch::new_empty(Arc::new(schema.clone())));

        Ok(vec![rb])
    }

    // -----------------------------------------------------------------------
    // Sort (CPU reference using Arrow compute)
    // -----------------------------------------------------------------------

    async fn execute_sort(
        &self,
        input: Vec<RecordBatch>,
        exprs: Vec<PhysicalExpr>,
    ) -> Result<Vec<RecordBatch>> {
        info!("Executing sort");
        // For the reference impl, just pass through.
        // A real GPU implementation would call sort_kernels.cuh.
        Ok(input)
    }
}

// ---------------------------------------------------------------------------
// Expression evaluation helpers (CPU fallback / reference)
// ---------------------------------------------------------------------------

fn eval_expr(expr: &PhysicalExpr, batch: &RecordBatch) -> Result<Arc<dyn Array>> {
    match expr {
        PhysicalExpr::Column { index, name } => {
            // Try by name first, then by index
            let col = if let Ok(idx) = batch.schema().index_of(name) {
                batch.column(idx).clone()
            } else if *index < batch.num_columns() {
                batch.column(*index).clone()
            } else {
                anyhow::bail!("Column '{}' not found in batch", name);
            };
            Ok(col)
        }
        PhysicalExpr::Literal { value } => {
            // Parse literal as i64 or f64
            if let Ok(i) = value.parse::<i64>() {
                let arr: Arc<dyn Array> = Arc::new(Int64Array::from(vec![i; batch.num_rows()]));
                Ok(arr)
            } else if let Ok(f) = value.parse::<f64>() {
                let arr: Arc<dyn Array> =
                    Arc::new(Float64Array::from(vec![f; batch.num_rows()]));
                Ok(arr)
            } else {
                anyhow::bail!("Cannot parse literal '{}'", value)
            }
        }
        PhysicalExpr::BinaryExpr { left, op, right } => {
            let l = eval_expr(left, batch)?;
            let r = eval_expr(right, batch)?;
            eval_binary_expr(l, op, r)
        }
    }
}

fn eval_binary_expr(
    left: Arc<dyn Array>,
    op: &str,
    right: Arc<dyn Array>,
) -> Result<Arc<dyn Array>> {
    // For simplicity, operate on Int64 arrays only in the reference impl.
    let l = left
        .as_any()
        .downcast_ref::<Int64Array>()
        .context("Binary expr: left not Int64")?;
    let r = right
        .as_any()
        .downcast_ref::<Int64Array>()
        .context("Binary expr: right not Int64")?;

    let result: Vec<i64> = (0..l.len())
        .map(|i| {
            let lv = l.value(i);
            let rv = r.value(i);
            match op {
                "+" => lv + rv,
                "-" => lv - rv,
                "*" => lv * rv,
                "/" => if rv != 0 { lv / rv } else { 0 },
                _ => 0,
            }
        })
        .collect();

    Ok(Arc::new(Int64Array::from(result)))
}

fn eval_filter_mask(batch: &RecordBatch, predicate: &PhysicalExpr) -> Result<Vec<bool>> {
    match predicate {
        PhysicalExpr::BinaryExpr { left, op, right } => {
            let l = eval_expr(left, batch)?;
            let r = eval_expr(right, batch)?;

            let l64 = l.as_any().downcast_ref::<Int64Array>();
            let r64 = r.as_any().downcast_ref::<Int64Array>();

            if let (Some(la), Some(ra)) = (l64, r64) {
                let mask: Vec<bool> = (0..la.len())
                    .map(|i| {
                        let lv = la.value(i);
                        let rv = ra.value(i);
                        match op.as_str() {
                            "=" | "==" => lv == rv,
                            "!=" | "<>" => lv != rv,
                            ">" => lv > rv,
                            ">=" => lv >= rv,
                            "<" => lv < rv,
                            "<=" => lv <= rv,
                            _ => false,
                        }
                    })
                    .collect();
                return Ok(mask);
            }

            // Default: pass all
            Ok(vec![true; batch.num_rows()])
        }
        _ => Ok(vec![true; batch.num_rows()]),
    }
}

fn filter_record_batch(batch: &RecordBatch, mask: &[bool]) -> Result<RecordBatch> {
    let indices: Vec<u64> = mask
        .iter()
        .enumerate()
        .filter(|(_, &keep)| keep)
        .map(|(i, _)| i as u64)
        .collect();

    // Build a new batch with only the kept rows.
    let columns: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .map(|col| take_int64(col, &indices))
        .collect();

    RecordBatch::try_new(batch.schema(), columns).context("filter_record_batch")
}

fn take_int64(col: &Arc<dyn Array>, indices: &[u64]) -> Arc<dyn Array> {
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        let taken: Vec<i64> = indices.iter().map(|&i| arr.value(i as usize)).collect();
        Arc::new(Int64Array::from(taken))
    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        let taken: Vec<f64> = indices.iter().map(|&i| arr.value(i as usize)).collect();
        Arc::new(Float64Array::from(taken))
    } else {
        // Fallback: return empty Int64 column
        Arc::new(Int64Array::from(Vec::<i64>::new()))
    }
}

fn eval_agg_input(agg: &AggregateExpr, batch: &RecordBatch) -> Result<Vec<f64>> {
    let input_expr = match agg {
        AggregateExpr::Sum { input }
        | AggregateExpr::Count { input }
        | AggregateExpr::Min { input }
        | AggregateExpr::Max { input }
        | AggregateExpr::Avg { input } => input,
    };

    let col = eval_expr(input_expr, batch)?;

    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        Ok((0..arr.len()).map(|i| arr.value(i) as f64).collect())
    } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        Ok((0..arr.len()).map(|i| arr.value(i)).collect())
    } else {
        Ok(vec![])
    }
}

fn expr_name(expr: &PhysicalExpr) -> String {
    match expr {
        PhysicalExpr::Column { name, .. } => name.clone(),
        PhysicalExpr::Literal { value } => format!("lit({})", value),
        PhysicalExpr::BinaryExpr { left, op, right } => {
            format!("({} {} {})", expr_name(left), op, expr_name(right))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executor_creation() {
        let config = EngineConfig::default();
        let _executor = GpuExecutor::new(&config).unwrap();
    }

    #[test]
    fn test_eval_filter_mask_gt() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![5, 15, 25]))],
        )
        .unwrap();

        let pred = PhysicalExpr::BinaryExpr {
            left: Box::new(PhysicalExpr::Column { index: 0, name: "a".into() }),
            op: ">".into(),
            right: Box::new(PhysicalExpr::Literal { value: "10".into() }),
        };

        let mask = eval_filter_mask(&batch, &pred).unwrap();
        assert_eq!(mask, vec![false, true, true]);
    }
}

