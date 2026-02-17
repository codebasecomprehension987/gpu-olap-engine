use anyhow::{Context, Result};
use sqlparser::ast::{Statement, Select, SelectItem, TableFactor, JoinOperator};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::logical_plan::{LogicalPlan, LogicalExpr, JoinType};

/// Parse SQL string into a logical plan
pub fn parse_sql(sql: &str) -> Result<LogicalPlan> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .context("Failed to parse SQL")?;
    
    if statements.len() != 1 {
        anyhow::bail!("Expected exactly one statement, got {}", statements.len());
    }
    
    match &statements[0] {
        Statement::Query(query) => parse_query(query),
        _ => anyhow::bail!("Only SELECT queries are supported"),
    }
}

fn parse_query(query: &sqlparser::ast::Query) -> Result<LogicalPlan> {
    let select = match query.body.as_ref() {
        sqlparser::ast::SetExpr::Select(select) => select,
        _ => anyhow::bail!("Only simple SELECT queries are supported"),
    };
    
    parse_select(select)
}

fn parse_select(select: &Select) -> Result<LogicalPlan> {
    // Parse FROM clause
    if select.from.is_empty() {
        anyhow::bail!("FROM clause is required");
    }
    
    let mut plan = parse_table_factor(&select.from[0].relation)?;
    
    // Parse JOINs
    for join in &select.from[0].joins {
        let right = parse_table_factor(&join.relation)?;
        let join_type = parse_join_type(&join.join_operator)?;
        
        let on_condition = match &join.join_operator {
            JoinOperator::Inner(constraint) | 
            JoinOperator::LeftOuter(constraint) |
            JoinOperator::RightOuter(constraint) |
            JoinOperator::FullOuter(constraint) => {
                parse_join_constraint(constraint)?
            },
            _ => anyhow::bail!("Unsupported join type"),
        };
        
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(right),
            on: on_condition,
            join_type,
        };
    }
    
    // Parse WHERE clause
    if let Some(expr) = &select.selection {
        let filter = parse_expr(expr)?;
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: filter,
        };
    }
    
    // Parse SELECT clause (projections)
    let projections = parse_projections(&select.projection)?;
    plan = LogicalPlan::Projection {
        input: Box::new(plan),
        exprs: projections,
    };
    
    // Parse GROUP BY
    if !select.group_by.is_empty() {
        let group_exprs = select.group_by.iter()
            .map(|e| parse_expr(e))
            .collect::<Result<Vec<_>>>()?;
        
        plan = LogicalPlan::Aggregate {
            input: Box::new(plan),
            group_by: group_exprs,
            aggr_exprs: vec![], // TODO: extract from projections
        };
    }
    
    // Parse ORDER BY
    if !query.order_by.is_empty() {
        // TODO: implement ORDER BY
    }
    
    // Parse LIMIT
    if let Some(limit) = &query.limit {
        // TODO: implement LIMIT
    }
    
    Ok(plan)
}

fn parse_table_factor(table: &TableFactor) -> Result<LogicalPlan> {
    match table {
        TableFactor::Table { name, alias, .. } => {
            let table_name = name.to_string();
            Ok(LogicalPlan::TableScan {
                table_name,
                projection: None,
            })
        },
        _ => anyhow::bail!("Unsupported table factor"),
    }
}

fn parse_join_type(op: &JoinOperator) -> Result<JoinType> {
    match op {
        JoinOperator::Inner(_) => Ok(JoinType::Inner),
        JoinOperator::LeftOuter(_) => Ok(JoinType::Left),
        JoinOperator::RightOuter(_) => Ok(JoinType::Right),
        JoinOperator::FullOuter(_) => Ok(JoinType::Full),
        _ => anyhow::bail!("Unsupported join type"),
    }
}

fn parse_join_constraint(constraint: &sqlparser::ast::JoinConstraint) -> Result<LogicalExpr> {
    match constraint {
        sqlparser::ast::JoinConstraint::On(expr) => parse_expr(expr),
        _ => anyhow::bail!("Only ON join constraints are supported"),
    }
}

fn parse_projections(items: &[SelectItem]) -> Result<Vec<LogicalExpr>> {
    items.iter().map(|item| {
        match item {
            SelectItem::UnnamedExpr(expr) => parse_expr(expr),
            SelectItem::ExprWithAlias { expr, alias } => {
                let expr = parse_expr(expr)?;
                Ok(LogicalExpr::Alias {
                    expr: Box::new(expr),
                    alias: alias.value.clone(),
                })
            },
            SelectItem::Wildcard(_) => Ok(LogicalExpr::Wildcard),
            _ => anyhow::bail!("Unsupported select item"),
        }
    }).collect()
}

fn parse_expr(expr: &sqlparser::ast::Expr) -> Result<LogicalExpr> {
    use sqlparser::ast::Expr;
    
    match expr {
        Expr::Identifier(ident) => Ok(LogicalExpr::Column(ident.value.clone())),
        
        Expr::BinaryOp { left, op, right } => {
            let left = parse_expr(left)?;
            let right = parse_expr(right)?;
            let op_str = op.to_string();
            
            Ok(LogicalExpr::BinaryExpr {
                left: Box::new(left),
                op: op_str,
                right: Box::new(right),
            })
        },
        
        Expr::Function(func) => {
            let name = func.name.to_string().to_lowercase();
            let args = func.args.iter()
                .map(|arg| {
                    match arg {
                        sqlparser::ast::FunctionArg::Unnamed(arg_expr) => {
                            match arg_expr {
                                sqlparser::ast::FunctionArgExpr::Expr(e) => parse_expr(e),
                                _ => anyhow::bail!("Unsupported function argument"),
                            }
                        },
                        _ => anyhow::bail!("Named arguments not supported"),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            
            Ok(LogicalExpr::AggregateFunction { name, args })
        },
        
        Expr::Value(value) => {
            use sqlparser::ast::Value;
            match value {
                Value::Number(n, _) => {
                    if let Ok(i) = n.parse::<i64>() {
                        Ok(LogicalExpr::Literal(format!("{}", i)))
                    } else {
                        Ok(LogicalExpr::Literal(n.clone()))
                    }
                },
                Value::SingleQuotedString(s) => Ok(LogicalExpr::Literal(s.clone())),
                _ => anyhow::bail!("Unsupported literal value"),
            }
        },
        
        _ => anyhow::bail!("Unsupported expression: {:?}", expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let sql = "SELECT a, b FROM table1 WHERE a > 10";
        let plan = parse_sql(sql).unwrap();
        println!("{:?}", plan);
    }

    #[test]
    fn test_join_query() {
        let sql = "SELECT t1.a, t2.b FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id";
        let plan = parse_sql(sql).unwrap();
        println!("{:?}", plan);
    }
}
