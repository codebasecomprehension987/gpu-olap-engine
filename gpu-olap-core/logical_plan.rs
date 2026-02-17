use std::fmt;

/// Logical query plan representation
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan a table
    TableScan {
        table_name: String,
        projection: Option<Vec<String>>,
    },
    
    /// Project columns
    Projection {
        input: Box<LogicalPlan>,
        exprs: Vec<LogicalExpr>,
    },
    
    /// Filter rows
    Filter {
        input: Box<LogicalPlan>,
        predicate: LogicalExpr,
    },
    
    /// Join two tables
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        on: LogicalExpr,
        join_type: JoinType,
    },
    
    /// Aggregate
    Aggregate {
        input: Box<LogicalPlan>,
        group_by: Vec<LogicalExpr>,
        aggr_exprs: Vec<LogicalExpr>,
    },
    
    /// Sort
    Sort {
        input: Box<LogicalPlan>,
        exprs: Vec<LogicalExpr>,
    },
    
    /// Limit
    Limit {
        input: Box<LogicalPlan>,
        limit: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Logical expression
#[derive(Debug, Clone)]
pub enum LogicalExpr {
    /// Column reference
    Column(String),
    
    /// Literal value
    Literal(String),
    
    /// Binary operation
    BinaryExpr {
        left: Box<LogicalExpr>,
        op: String,
        right: Box<LogicalExpr>,
    },
    
    /// Aggregate function
    AggregateFunction {
        name: String,
        args: Vec<LogicalExpr>,
    },
    
    /// Alias
    Alias {
        expr: Box<LogicalExpr>,
        alias: String,
    },
    
    /// Wildcard (*)
    Wildcard,
}

impl fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LogicalPlan::TableScan { table_name, .. } => {
                write!(f, "TableScan: {}", table_name)
            },
            LogicalPlan::Projection { input, exprs } => {
                write!(f, "Projection: {:?}\n  {}", exprs, input)
            },
            LogicalPlan::Filter { input, predicate } => {
                write!(f, "Filter: {:?}\n  {}", predicate, input)
            },
            LogicalPlan::Join { left, right, join_type, .. } => {
                write!(f, "Join: {:?}\n  Left: {}\n  Right: {}", join_type, left, right)
            },
            LogicalPlan::Aggregate { input, group_by, aggr_exprs } => {
                write!(f, "Aggregate: group_by={:?}, aggr={:?}\n  {}", 
                       group_by, aggr_exprs, input)
            },
            LogicalPlan::Sort { input, exprs } => {
                write!(f, "Sort: {:?}\n  {}", exprs, input)
            },
            LogicalPlan::Limit { input, limit } => {
                write!(f, "Limit: {}\n  {}", limit, input)
            },
        }
    }
}
