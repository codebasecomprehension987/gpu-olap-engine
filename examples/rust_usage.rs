use anyhow::Result;
use gpu_olap_core::{OlapEngine, EngineConfig};
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    info!("GPU OLAP Engine - Rust Examples");
    
    // Example 1: Basic query execution
    basic_query_example().await?;
    
    // Example 2: Complex join query
    join_query_example().await?;
    
    // Example 3: Aggregation query
    aggregation_example().await?;
    
    // Example 4: Memory management configuration
    memory_config_example().await?;
    
    Ok(())
}

async fn basic_query_example() -> Result<()> {
    info!("=== Example 1: Basic Query ===");
    
    // Create engine with default configuration
    let config = EngineConfig::default();
    let engine = OlapEngine::new(config)?;
    
    // Load table from Parquet
    engine.load_table("products", "data/products.parquet").await?;
    
    // Execute simple SELECT query
    let results = engine.execute_query(
        "SELECT product_id, name, price FROM products WHERE price > 100"
    ).await?;
    
    info!("Query returned {} batches", results.len());
    for (i, batch) in results.iter().enumerate() {
        info!("Batch {}: {} rows", i, batch.num_rows());
    }
    
    Ok(())
}

async fn join_query_example() -> Result<()> {
    info!("=== Example 2: Join Query ===");
    
    let config = EngineConfig {
        max_gpu_memory: 8 * 1024 * 1024 * 1024, // 8GB
        num_streams: 8,
        use_unified_memory: true,
        batch_size: 1024 * 1024,
        enable_cache: true,
    };
    
    let engine = OlapEngine::new(config)?;
    
    // Load multiple tables
    engine.load_table("orders", "data/orders.parquet").await?;
    engine.load_table("customers", "data/customers.parquet").await?;
    
    // Execute join query
    let query = "
        SELECT 
            c.customer_name,
            COUNT(*) as num_orders,
            SUM(o.total_amount) as total_spent
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_date >= '2024-01-01'
        GROUP BY c.customer_name
        ORDER BY total_spent DESC
        LIMIT 10
    ";
    
    let results = engine.execute_query(query).await?;
    
    info!("Top customers by spending:");
    for batch in results {
        info!("  {} rows", batch.num_rows());
        // In real implementation, print actual data
    }
    
    Ok(())
}

async fn aggregation_example() -> Result<()> {
    info!("=== Example 3: Aggregation ===");
    
    let config = EngineConfig::default();
    let engine = OlapEngine::new(config)?;
    
    // Load sales data
    engine.load_table("sales", "data/sales.parquet").await?;
    
    // Complex aggregation query
    let query = "
        SELECT 
            region,
            product_category,
            COUNT(*) as num_sales,
            SUM(amount) as total_revenue,
            AVG(amount) as avg_sale,
            MIN(amount) as min_sale,
            MAX(amount) as max_sale
        FROM sales
        WHERE year = 2024
        GROUP BY region, product_category
        HAVING total_revenue > 100000
        ORDER BY total_revenue DESC
    ";
    
    let results = engine.execute_query(query).await?;
    
    info!("Aggregation results: {} batches", results.len());
    
    Ok(())
}

async fn memory_config_example() -> Result<()> {
    info!("=== Example 4: Memory Configuration ===");
    
    // Configure for specific hardware
    let config = EngineConfig {
        // Limit GPU memory usage
        max_gpu_memory: 4 * 1024 * 1024 * 1024, // 4GB
        
        // More streams for better overlap
        num_streams: 16,
        
        // Enable unified memory for large datasets
        use_unified_memory: true,
        
        // Smaller batch size for memory-constrained environments
        batch_size: 512 * 1024,
        
        // Disable cache to save memory
        enable_cache: false,
    };
    
    let engine = OlapEngine::new(config)?;
    
    info!("Engine configured with:");
    info!("  Max GPU Memory: {} GB", config.max_gpu_memory / (1024 * 1024 * 1024));
    info!("  Num Streams: {}", config.num_streams);
    info!("  Unified Memory: {}", config.use_unified_memory);
    info!("  Batch Size: {} rows", config.batch_size);
    
    // Load and query large dataset
    engine.load_table("large_table", "data/large_dataset.parquet").await?;
    
    // This will use streaming and unified memory automatically
    let results = engine.execute_query(
        "SELECT COUNT(*), SUM(value) FROM large_table WHERE condition = true"
    ).await?;
    
    info!("Large query completed successfully");
    
    Ok(())
}

// Example: Custom error handling
async fn error_handling_example() -> Result<()> {
    let config = EngineConfig::default();
    let engine = OlapEngine::new(config)?;
    
    // Handle table not found
    match engine.get_table_schema("nonexistent") {
        Ok(schema) => info!("Schema: {:?}", schema),
        Err(e) => info!("Expected error: {}", e),
    }
    
    // Handle invalid SQL
    match engine.execute_query("INVALID SQL QUERY").await {
        Ok(_) => {},
        Err(e) => info!("Expected SQL error: {}", e),
    }
    
    Ok(())
}

// Example: Benchmark utilities
async fn benchmark_example() -> Result<()> {
    use std::time::Instant;
    
    let config = EngineConfig {
        max_gpu_memory: 8 * 1024 * 1024 * 1024,
        num_streams: 8,
        ..Default::default()
    };
    
    let engine = OlapEngine::new(config)?;
    
    // Load test data
    engine.load_table("test_data", "data/benchmark.parquet").await?;
    
    // Warm up
    let _ = engine.execute_query("SELECT COUNT(*) FROM test_data").await?;
    
    // Benchmark different queries
    let queries = vec![
        ("Simple scan", "SELECT * FROM test_data WHERE id < 1000"),
        ("Aggregation", "SELECT region, COUNT(*) FROM test_data GROUP BY region"),
        ("Join", "SELECT * FROM test_data t1 JOIN test_data t2 ON t1.id = t2.id"),
    ];
    
    for (name, query) in queries {
        let start = Instant::now();
        let _ = engine.execute_query(query).await?;
        let elapsed = start.elapsed();
        
        info!("{}: {:?}", name, elapsed);
    }
    
    Ok(())
}
