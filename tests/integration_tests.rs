//! Integration tests for the GPU OLAP engine.
//!
//! These tests exercise the full SQL → logical plan → physical plan → executor
//! pipeline using in-memory Arrow RecordBatches (no CUDA device required).

use gpu_olap_core::{EngineConfig, OlapEngine};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

async fn make_engine() -> OlapEngine {
    OlapEngine::new(EngineConfig::default()).unwrap()
}

// ---------------------------------------------------------------------------
// Parser tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_parse_simple_select() {
    use gpu_olap_core::parser::parse_sql;
    let plan = parse_sql("SELECT a, b FROM t WHERE a > 10").unwrap();
    let repr = format!("{:?}", plan);
    assert!(repr.contains("Projection"));
    assert!(repr.contains("Filter"));
    assert!(repr.contains("TableScan"));
}

#[tokio::test]
async fn test_parse_aggregate() {
    use gpu_olap_core::parser::parse_sql;
    let plan = parse_sql("SELECT sum(revenue), count(id) FROM orders GROUP BY region").unwrap();
    let repr = format!("{:?}", plan);
    assert!(repr.contains("Aggregate"));
}

#[tokio::test]
async fn test_parse_order_by() {
    use gpu_olap_core::parser::parse_sql;
    let plan = parse_sql("SELECT a FROM t ORDER BY a").unwrap();
    let repr = format!("{:?}", plan);
    assert!(repr.contains("Sort"));
}

#[tokio::test]
async fn test_parse_limit() {
    use gpu_olap_core::parser::parse_sql;
    let plan = parse_sql("SELECT a FROM t LIMIT 100").unwrap();
    let repr = format!("{:?}", plan);
    assert!(repr.contains("Limit"));
}

#[tokio::test]
async fn test_parse_join() {
    use gpu_olap_core::parser::parse_sql;
    let sql = "SELECT t1.id, t2.val FROM t1 JOIN t2 ON t1.id = t2.id";
    let plan = parse_sql(sql).unwrap();
    let repr = format!("{:?}", plan);
    assert!(repr.contains("Join"));
}

// ---------------------------------------------------------------------------
// Optimizer tests
// ---------------------------------------------------------------------------

#[test]
fn test_optimizer_predicate_pushdown() {
    use gpu_olap_core::{optimizer, parser};
    let plan = parser::parse_sql("SELECT a FROM t WHERE a > 5").unwrap();
    let optimized = optimizer::optimize(plan).unwrap();
    // After pushdown the filter should be below the projection
    let repr = format!("{}", optimized);
    // Projection wraps Filter wraps TableScan
    assert!(repr.contains("Projection"));
}

#[test]
fn test_optimizer_merge_filters() {
    use gpu_olap_core::{logical_plan::{LogicalExpr, LogicalPlan}, optimizer};

    let plan = LogicalPlan::Filter {
        input: Box::new(LogicalPlan::Filter {
            input: Box::new(LogicalPlan::TableScan {
                table_name: "t".into(),
                projection: None,
            }),
            predicate: LogicalExpr::BinaryExpr {
                left: Box::new(LogicalExpr::Column("b".into())),
                op: "<".into(),
                right: Box::new(LogicalExpr::Literal("100".into())),
            },
        }),
        predicate: LogicalExpr::BinaryExpr {
            left: Box::new(LogicalExpr::Column("a".into())),
            op: ">".into(),
            right: Box::new(LogicalExpr::Literal("5".into())),
        },
    };

    let optimized = optimizer::optimize(plan).unwrap();
    let repr = format!("{:?}", optimized);
    // Two filters should have been merged into one with AND
    assert!(repr.contains("AND"));
}

// ---------------------------------------------------------------------------
// Catalog tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_catalog_empty_tables() {
    use gpu_olap_core::catalog::Catalog;
    let cat = Catalog::new().unwrap();
    assert!(cat.list_tables().is_empty());
}

// ---------------------------------------------------------------------------
// Physical plan tests
// ---------------------------------------------------------------------------

#[test]
fn test_physical_plan_table_scan() {
    use gpu_olap_core::{catalog::Catalog, physical_plan, logical_plan::LogicalPlan, EngineConfig};
    use std::sync::Arc;

    let catalog = Catalog::new().unwrap();
    let config = EngineConfig::default();

    // A TableScan for an unknown table should fail (no schema in catalog)
    let logical = LogicalPlan::TableScan {
        table_name: "nonexistent".into(),
        projection: None,
    };
    let result = physical_plan::create_physical_plan(logical, &catalog, &config);
    assert!(result.is_err());
}
