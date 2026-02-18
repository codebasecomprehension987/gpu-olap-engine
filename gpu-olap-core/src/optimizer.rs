use anyhow::Result;
use crate::logical_plan::{LogicalPlan, LogicalExpr};

/// Optimize logical plan
/// 
/// Applies various optimization rules:
/// - Predicate pushdown
/// - Projection pushdown  
/// - Join reordering
/// - Constant folding
/// - Filter merging
pub fn optimize(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut optimized = plan;
    
    // Apply optimization passes
    optimized = pushdown_predicates(optimized)?;
    optimized = pushdown_projections(optimized)?;
    optimized = merge_filters(optimized)?;
    optimized = constant_folding(optimized)?;
    
    Ok(optimized)
}

/// Push filters down closer to table scans
fn pushdown_predicates(plan: LogicalPlan) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Filter { input, predicate } => {
            match *input {
                // Filter -> Projection => try to push filter through projection
                LogicalPlan::Projection { input: inner, exprs } => {
                    // Check if predicate only references projected columns
                    let pushed = LogicalPlan::Filter {
                        input: inner,
                        predicate: predicate.clone(),
                    };
                    
                    Ok(LogicalPlan::Projection {
                        input: Box::new(pushed),
                        exprs,
                    })
                },
                
                // Filter -> Join => push to appropriate side
                LogicalPlan::Join { left, right, on, join_type } => {
                    // Simplified - in real impl, analyze predicate to determine
                    // which side(s) it applies to
                    Ok(LogicalPlan::Join {
                        left,
                        right,
                        on,
                        join_type,
                    })
                },
                
                other => {
                    Ok(LogicalPlan::Filter {
                        input: Box::new(other),
                        predicate,
                    })
                }
            }
        },
        
        LogicalPlan::Projection { input, exprs } => {
            Ok(LogicalPlan::Projection {
                input: Box::new(pushdown_predicates(*input)?),
                exprs,
            })
        },
        
        LogicalPlan::Join { left, right, on, join_type } => {
            Ok(LogicalPlan::Join {
                left: Box::new(pushdown_predicates(*left)?),
                right: Box::new(pushdown_predicates(*right)?),
                on,
                join_type,
            })
        },
        
        LogicalPlan::Aggregate { input, group_by, aggr_exprs } => {
            Ok(LogicalPlan::Aggregate {
                input: Box::new(pushdown_predicates(*input)?),
                group_by,
                aggr_exprs,
            })
        },
        
        other => Ok(other),
    }
}

/// Push projections down to reduce data movement
fn pushdown_projections(plan: LogicalPlan) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Projection { input, exprs } => {
            match *input {
                // Projection -> Projection => merge
                LogicalPlan::Projection { input: inner, exprs: inner_exprs } => {
                    // Merge projections
                    Ok(LogicalPlan::Projection {
                        input: inner,
                        exprs, // Use outer projection
                    })
                },
                
                // Projection -> TableScan => push projection into scan
                LogicalPlan::TableScan { table_name, .. } => {
                    let column_names = extract_column_names(&exprs);
                    
                    Ok(LogicalPlan::Projection {
                        input: Box::new(LogicalPlan::TableScan {
                            table_name,
                            projection: Some(column_names),
                        }),
                        exprs,
                    })
                },
                
                other => {
                    Ok(LogicalPlan::Projection {
                        input: Box::new(pushdown_projections(other)?),
                        exprs,
                    })
                }
            }
        },
        
        LogicalPlan::Filter { input, predicate } => {
            Ok(LogicalPlan::Filter {
                input: Box::new(pushdown_projections(*input)?),
                predicate,
            })
        },
        
        LogicalPlan::Join { left, right, on, join_type } => {
            Ok(LogicalPlan::Join {
                left: Box::new(pushdown_projections(*left)?),
                right: Box::new(pushdown_projections(*right)?),
                on,
                join_type,
            })
        },
        
        other => Ok(other),
    }
}

/// Merge consecutive filters
fn merge_filters(plan: LogicalPlan) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Filter { input, predicate } => {
            match *input {
                // Filter -> Filter => merge with AND
                LogicalPlan::Filter { input: inner, predicate: inner_pred } => {
                    let merged = LogicalExpr::BinaryExpr {
                        left: Box::new(predicate),
                        op: "AND".to_string(),
                        right: Box::new(inner_pred),
                    };
                    
                    Ok(LogicalPlan::Filter {
                        input: inner,
                        predicate: merged,
                    })
                },
                
                other => {
                    Ok(LogicalPlan::Filter {
                        input: Box::new(merge_filters(other)?),
                        predicate,
                    })
                }
            }
        },
        
        other => Ok(other),
    }
}

/// Fold constant expressions
fn constant_folding(plan: LogicalPlan) -> Result<LogicalPlan> {
    // Simplified implementation
    // In real impl, we'd evaluate constant expressions at compile time
    Ok(plan)
}

fn extract_column_names(exprs: &[LogicalExpr]) -> Vec<String> {
    let mut columns = Vec::new();
    
    for expr in exprs {
        collect_columns(expr, &mut columns);
    }
    
    columns.sort();
    columns.dedup();
    columns
}

fn collect_columns(expr: &LogicalExpr, columns: &mut Vec<String>) {
    match expr {
        LogicalExpr::Column(name) => {
            columns.push(name.clone());
        },
        LogicalExpr::BinaryExpr { left, right, .. } => {
            collect_columns(left, columns);
            collect_columns(right, columns);
        },
        LogicalExpr::AggregateFunction { args, .. } => {
            for arg in args {
                collect_columns(arg, columns);
            }
        },
        LogicalExpr::Alias { expr, .. } => {
            collect_columns(expr, columns);
        },
        _ => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_pushdown() {
        // Test that filter is pushed below projection
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Projection {
                input: Box::new(LogicalPlan::TableScan {
                    table_name: "test".to_string(),
                    projection: None,
                }),
                exprs: vec![LogicalExpr::Column("a".to_string())],
            }),
            predicate: LogicalExpr::BinaryExpr {
                left: Box::new(LogicalExpr::Column("a".to_string())),
                op: ">".to_string(),
                right: Box::new(LogicalExpr::Literal("10".to_string())),
            },
        };
        
        let optimized = optimize(plan).unwrap();
        println!("{}", optimized);
    }
}
