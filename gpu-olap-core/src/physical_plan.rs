use anyhow::Result;
use arrow_schema::Schema;
use std::sync::Arc;

use crate::catalog::Catalog;
use crate::logical_plan::{LogicalPlan, LogicalExpr, JoinType};
use crate::EngineConfig;

/// Physical execution plan - operators that execute on GPU
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// GPU table scan
    GpuTableScan {
        table_name: String,
        schema: Arc<Schema>,
        projection: Option<Vec<usize>>,
    },
    
    /// GPU filter
    GpuFilter {
        input: Box<PhysicalPlan>,
        predicate: PhysicalExpr,
    },
    
    /// GPU projection
    GpuProjection {
        input: Box<PhysicalPlan>,
        exprs: Vec<PhysicalExpr>,
    },
    
    /// GPU hash join (radix partitioned)
    GpuHashJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        left_keys: Vec<PhysicalExpr>,
        right_keys: Vec<PhysicalExpr>,
        join_type: JoinType,
        schema: Arc<Schema>,
    },
    
    /// GPU sort-merge join
    GpuSortMergeJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        left_keys: Vec<PhysicalExpr>,
        right_keys: Vec<PhysicalExpr>,
        join_type: JoinType,
        schema: Arc<Schema>,
    },
    
    /// GPU aggregation
    GpuAggregate {
        input: Box<PhysicalPlan>,
        group_by: Vec<PhysicalExpr>,
        aggr_exprs: Vec<AggregateExpr>,
        schema: Arc<Schema>,
    },
    
    /// GPU sort
    GpuSort {
        input: Box<PhysicalPlan>,
        exprs: Vec<PhysicalExpr>,
    },
}

#[derive(Debug, Clone)]
pub enum PhysicalExpr {
    Column { index: usize, name: String },
    Literal { value: String },
    BinaryExpr {
        left: Box<PhysicalExpr>,
        op: String,
        right: Box<PhysicalExpr>,
    },
}

#[derive(Debug, Clone)]
pub enum AggregateExpr {
    Sum { input: Box<PhysicalExpr> },
    Count { input: Box<PhysicalExpr> },
    Min { input: Box<PhysicalExpr> },
    Max { input: Box<PhysicalExpr> },
    Avg { input: Box<PhysicalExpr> },
}

/// Create physical plan from logical plan
pub fn create_physical_plan(
    logical_plan: LogicalPlan,
    catalog: &Catalog,
    config: &EngineConfig,
) -> Result<PhysicalPlan> {
    match logical_plan {
        LogicalPlan::TableScan { table_name, projection } => {
            let schema = catalog.get_schema(&table_name)?;
            
            let projection_indices = projection.map(|cols| {
                cols.iter()
                    .filter_map(|col| {
                        schema.fields().iter().position(|f| f.name() == col)
                    })
                    .collect()
            });
            
            Ok(PhysicalPlan::GpuTableScan {
                table_name,
                schema,
                projection: projection_indices,
            })
        },
        
        LogicalPlan::Projection { input, exprs } => {
            let input_plan = create_physical_plan(*input, catalog, config)?;
            let physical_exprs = exprs.into_iter()
                .map(|e| logical_to_physical_expr(e))
                .collect::<Result<Vec<_>>>()?;
            
            Ok(PhysicalPlan::GpuProjection {
                input: Box::new(input_plan),
                exprs: physical_exprs,
            })
        },
        
        LogicalPlan::Filter { input, predicate } => {
            let input_plan = create_physical_plan(*input, catalog, config)?;
            let physical_pred = logical_to_physical_expr(predicate)?;
            
            Ok(PhysicalPlan::GpuFilter {
                input: Box::new(input_plan),
                predicate: physical_pred,
            })
        },
        
        LogicalPlan::Join { left, right, on, join_type } => {
            let left_plan = create_physical_plan(*left, catalog, config)?;
            let right_plan = create_physical_plan(*right, catalog, config)?;
            
            // Extract join keys from ON condition
            let (left_keys, right_keys) = extract_join_keys(on)?;
            
            // Decide between hash join and sort-merge join
            // Hash join is preferred for smaller tables
            // Sort-merge join for larger tables or when data is already sorted
            
            // For now, always use hash join
            let schema = derive_join_schema(&left_plan, &right_plan)?;
            
            Ok(PhysicalPlan::GpuHashJoin {
                left: Box::new(left_plan),
                right: Box::new(right_plan),
                left_keys,
                right_keys,
                join_type,
                schema,
            })
        },
        
        LogicalPlan::Aggregate { input, group_by, aggr_exprs } => {
            let input_plan = create_physical_plan(*input, catalog, config)?;
            
            let group_exprs = group_by.into_iter()
                .map(logical_to_physical_expr)
                .collect::<Result<Vec<_>>>()?;
            
            let aggr_physical = aggr_exprs.into_iter()
                .map(logical_to_aggregate_expr)
                .collect::<Result<Vec<_>>>()?;
            
            let schema = derive_aggregate_schema(&input_plan, &group_exprs, &aggr_physical)?;
            
            Ok(PhysicalPlan::GpuAggregate {
                input: Box::new(input_plan),
                group_by: group_exprs,
                aggr_exprs: aggr_physical,
                schema,
            })
        },
        
        LogicalPlan::Sort { input, exprs } => {
            let input_plan = create_physical_plan(*input, catalog, config)?;
            let sort_exprs = exprs.into_iter()
                .map(logical_to_physical_expr)
                .collect::<Result<Vec<_>>>()?;
            
            Ok(PhysicalPlan::GpuSort {
                input: Box::new(input_plan),
                exprs: sort_exprs,
            })
        },
        
        LogicalPlan::Limit { input, limit } => {
            // Limit is handled in executor, just pass through
            create_physical_plan(*input, catalog, config)
        },
    }
}

fn logical_to_physical_expr(expr: LogicalExpr) -> Result<PhysicalExpr> {
    match expr {
        LogicalExpr::Column(name) => Ok(PhysicalExpr::Column { index: 0, name }),
        LogicalExpr::Literal(value) => Ok(PhysicalExpr::Literal { value }),
        LogicalExpr::BinaryExpr { left, op, right } => {
            Ok(PhysicalExpr::BinaryExpr {
                left: Box::new(logical_to_physical_expr(*left)?),
                op,
                right: Box::new(logical_to_physical_expr(*right)?),
            })
        },
        LogicalExpr::Alias { expr, .. } => logical_to_physical_expr(*expr),
        _ => anyhow::bail!("Unsupported logical expression in physical plan"),
    }
}

fn logical_to_aggregate_expr(expr: LogicalExpr) -> Result<AggregateExpr> {
    match expr {
        LogicalExpr::AggregateFunction { name, args } => {
            if args.len() != 1 {
                anyhow::bail!("Aggregate functions expect exactly 1 argument");
            }
            
            let input = Box::new(logical_to_physical_expr(args[0].clone())?);
            
            match name.as_str() {
                "sum" => Ok(AggregateExpr::Sum { input }),
                "count" => Ok(AggregateExpr::Count { input }),
                "min" => Ok(AggregateExpr::Min { input }),
                "max" => Ok(AggregateExpr::Max { input }),
                "avg" => Ok(AggregateExpr::Avg { input }),
                _ => anyhow::bail!("Unsupported aggregate function: {}", name),
            }
        },
        _ => anyhow::bail!("Expected aggregate function"),
    }
}

fn extract_join_keys(on: LogicalExpr) -> Result<(Vec<PhysicalExpr>, Vec<PhysicalExpr>)> {
    // Simplified - in real implementation, we'd parse the ON condition properly
    // For now, assume simple equality: left.col = right.col
    
    match on {
        LogicalExpr::BinaryExpr { left, op, right } if op == "=" => {
            Ok((
                vec![logical_to_physical_expr(*left)?],
                vec![logical_to_physical_expr(*right)?],
            ))
        },
        _ => anyhow::bail!("Only simple equality joins are supported"),
    }
}

fn derive_join_schema(
    _left: &PhysicalPlan,
    _right: &PhysicalPlan,
) -> Result<Arc<Schema>> {
    // Placeholder - combine schemas from left and right
    Ok(Arc::new(Schema::empty()))
}

fn derive_aggregate_schema(
    _input: &PhysicalPlan,
    _group_by: &[PhysicalExpr],
    _aggr: &[AggregateExpr],
) -> Result<Arc<Schema>> {
    // Placeholder - derive schema from group by + aggregates
    Ok(Arc::new(Schema::empty()))
}
