//! Benchmarks for the GPU OLAP engine core path.
//!
//! Run with:  `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gpu_olap_core::{optimizer, parser};

fn bench_parse_simple(c: &mut Criterion) {
    let sql = "SELECT a, b, c FROM sales WHERE revenue > 1000";
    c.bench_function("parse_simple_select", |b| {
        b.iter(|| {
            let plan = parser::parse_sql(black_box(sql)).unwrap();
            black_box(plan);
        })
    });
}

fn bench_parse_join(c: &mut Criterion) {
    let sql = "SELECT o.id, c.name, sum(o.amount) \
               FROM orders o JOIN customers c ON o.cust_id = c.id \
               WHERE o.amount > 100 \
               GROUP BY o.id, c.name \
               ORDER BY o.id \
               LIMIT 50";
    c.bench_function("parse_complex_join", |b| {
        b.iter(|| {
            let plan = parser::parse_sql(black_box(sql)).unwrap();
            black_box(plan);
        })
    });
}

fn bench_optimize(c: &mut Criterion) {
    let sql = "SELECT a FROM t WHERE a > 10";
    let plan = parser::parse_sql(sql).unwrap();

    c.bench_function("optimize_simple", |b| {
        b.iter(|| {
            let optimized = optimizer::optimize(black_box(plan.clone())).unwrap();
            black_box(optimized);
        })
    });
}

criterion_group!(benches, bench_parse_simple, bench_parse_join, bench_optimize);
criterion_main!(benches);
