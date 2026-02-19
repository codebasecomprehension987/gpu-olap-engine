"""
GPU OLAP Engine - Python Examples

Demonstrates usage of the GPU-accelerated OLAP database engine
from Python with Pandas and Polars integration.
"""

import gpu_olap_py
import pandas as pd
import polars as pl
import numpy as np
import time


def example_basic_queries():
    """Basic SQL query examples"""
    print("=== Basic Queries ===\n")
    
    # Create engine
    engine = gpu_olap_py.GpuOlapEngine(
        max_gpu_memory=8 * 1024**3,  # 8GB
        num_streams=8,
        use_unified_memory=True
    )
    
    # Load table from Parquet
    print("Loading sales table...")
    engine.load_table('sales', 'data/sales.parquet')
    
    # Simple SELECT
    print("\n1. Simple SELECT with WHERE:")
    result = engine.query("""
        SELECT product_id, amount, customer_id
        FROM sales
        WHERE amount > 1000
        LIMIT 10
    """)
    print(result.to_pandas())
    
    # Aggregation
    print("\n2. GROUP BY aggregation:")
    result = engine.query("""
        SELECT 
            region,
            COUNT(*) as num_sales,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM sales
        GROUP BY region
        ORDER BY total_amount DESC
    """)
    print(result.to_pandas())
    
    # Join query
    print("\n3. JOIN with customers:")
    engine.load_table('customers', 'data/customers.parquet')
    
    result = engine.query("""
        SELECT 
            c.customer_name,
            c.region,
            SUM(s.amount) as total_purchases,
            COUNT(*) as num_purchases
        FROM sales s
        JOIN customers c ON s.customer_id = c.customer_id
        WHERE s.year = 2024
        GROUP BY c.customer_name, c.region
        ORDER BY total_purchases DESC
        LIMIT 20
    """)
    print(result.to_pandas())


def example_pandas_integration():
    """Pandas DataFrame integration with zero-copy"""
    print("\n=== Pandas Integration ===\n")
    
    engine = gpu_olap_py.GpuOlapEngine()
    
    # Create sample DataFrame
    print("Creating sample DataFrame...")
    df = pd.DataFrame({
        'id': range(1000000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000000),
        'value': np.random.randn(1000000) * 100,
        'quantity': np.random.randint(1, 100, 1000000)
    })
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Query the DataFrame (zero-copy through Arrow)
    print("\nQuerying DataFrame on GPU:")
    start = time.time()
    
    result = engine.query_pandas(df, """
        SELECT 
            category,
            COUNT(*) as count,
            SUM(value * quantity) as total_value,
            AVG(value) as avg_value,
            MAX(quantity) as max_quantity
        FROM df
        WHERE value > 0
        GROUP BY category
        ORDER BY total_value DESC
    """)
    
    elapsed = time.time() - start
    
    result_df = result.to_pandas()
    print(result_df)
    print(f"\nQuery time: {elapsed*1000:.2f}ms")
    
    # Compare with native Pandas
    print("\nCompare with native Pandas:")
    start = time.time()
    
    pandas_result = (
        df[df['value'] > 0]
        .groupby('category')
        .agg({
            'id': 'count',
            'value': lambda x: (x * df.loc[x.index, 'quantity']).sum(),
            'value': 'mean',
            'quantity': 'max'
        })
        .reset_index()
        .sort_values('value', ascending=False)
    )
    
    elapsed_pandas = time.time() - start
    print(f"Pandas time: {elapsed_pandas*1000:.2f}ms")
    print(f"Speedup: {elapsed_pandas/elapsed:.2f}x")


def example_polars_integration():
    """Polars DataFrame integration"""
    print("\n=== Polars Integration ===\n")
    
    engine = gpu_olap_py.GpuOlapEngine()
    
    # Create sample DataFrame
    print("Creating Polars DataFrame...")
    df = pl.DataFrame({
        'timestamp': pl.date_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 12, 31),
            interval='1h',
            eager=True
        ),
        'sensor_id': np.random.randint(1, 100, 8760),
        'temperature': np.random.randn(8760) * 10 + 20,
        'humidity': np.random.randn(8760) * 15 + 60,
        'pressure': np.random.randn(8760) * 5 + 1013
    })
    
    print(f"DataFrame shape: {df.shape}")
    
    # Query with GPU (zero-copy through Arrow)
    print("\nQuerying with GPU:")
    start = time.time()
    
    result = engine.query_polars(df, """
        SELECT 
            sensor_id,
            COUNT(*) as readings,
            AVG(temperature) as avg_temp,
            AVG(humidity) as avg_humidity,
            AVG(pressure) as avg_pressure,
            MAX(temperature) - MIN(temperature) as temp_range
        FROM df
        GROUP BY sensor_id
        HAVING avg_temp > 20
        ORDER BY temp_range DESC
        LIMIT 10
    """)
    
    elapsed_gpu = time.time() - start
    
    result_df = pl.from_arrow(result)
    print(result_df)
    print(f"\nGPU time: {elapsed_gpu*1000:.2f}ms")
    
    # Compare with native Polars
    print("\nCompare with native Polars:")
    start = time.time()
    
    polars_result = (
        df.groupby('sensor_id')
        .agg([
            pl.count().alias('readings'),
            pl.col('temperature').mean().alias('avg_temp'),
            pl.col('humidity').mean().alias('avg_humidity'),
            pl.col('pressure').mean().alias('avg_pressure'),
            (pl.col('temperature').max() - pl.col('temperature').min()).alias('temp_range')
        ])
        .filter(pl.col('avg_temp') > 20)
        .sort('temp_range', descending=True)
        .head(10)
    )
    
    elapsed_polars = time.time() - start
    print(f"Polars time: {elapsed_polars*1000:.2f}ms")
    print(f"Speedup: {elapsed_polars/elapsed_gpu:.2f}x")


def example_complex_analytics():
    """Complex analytical queries"""
    print("\n=== Complex Analytics ===\n")
    
    engine = gpu_olap_py.GpuOlapEngine(
        max_gpu_memory=16 * 1024**3,  # 16GB for large datasets
        num_streams=16
    )
    
    # Load multiple tables
    print("Loading tables...")
    engine.load_table('orders', 'data/orders.parquet')
    engine.load_table('order_items', 'data/order_items.parquet')
    engine.load_table('products', 'data/products.parquet')
    engine.load_table('customers', 'data/customers.parquet')
    
    # Complex multi-table join with aggregations
    print("\nComplex analytics query:")
    query = """
    SELECT 
        c.region,
        p.category,
        COUNT(DISTINCT o.order_id) as num_orders,
        COUNT(DISTINCT c.customer_id) as num_customers,
        SUM(oi.quantity * p.price) as total_revenue,
        AVG(oi.quantity * p.price) as avg_order_value
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE 
        o.order_date >= '2024-01-01'
        AND o.order_date < '2024-07-01'
        AND o.status = 'completed'
    GROUP BY c.region, p.category
    HAVING total_revenue > 100000
    ORDER BY total_revenue DESC
    """
    
    start = time.time()
    result = engine.query(query)
    elapsed = time.time() - start
    
    df = result.to_pandas()
    print(df)
    
    print(f"\nQuery processed in {elapsed:.2f}s")
    print(f"Rows returned: {len(df)}")


def example_memory_management():
    """Demonstrate memory management features"""
    print("\n=== Memory Management ===\n")
    
    # Configure for large datasets
    engine = gpu_olap_py.GpuOlapEngine(
        max_gpu_memory=4 * 1024**3,     # 4GB GPU memory
        num_streams=8,
        use_unified_memory=True  # Enable for out-of-core processing
    )
    
    # Load large table (>GPU memory)
    print("Loading large table (may exceed GPU memory)...")
    engine.load_table('large_table', 'data/large_dataset.parquet')
    
    # Query will automatically use unified memory and streaming
    print("\nQuerying large dataset:")
    result = engine.query("""
        SELECT 
            date_part('year', timestamp) as year,
            date_part('month', timestamp) as month,
            COUNT(*) as num_events,
            SUM(value) as total_value
        FROM large_table
        GROUP BY year, month
        ORDER BY year, month
    """)
    
    print(result.to_pandas())


def benchmark_join_performance():
    """Benchmark different join sizes"""
    print("\n=== Join Performance Benchmark ===\n")
    
    engine = gpu_olap_py.GpuOlapEngine()
    
    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    
    for size in sizes:
        print(f"\nBenchmarking join with {size:,} rows each side:")
        
        # Create test DataFrames
        left = pd.DataFrame({
            'key': np.random.randint(0, size // 2, size),
            'left_value': np.random.randn(size)
        })
        
        right = pd.DataFrame({
            'key': np.random.randint(0, size // 2, size),
            'right_value': np.random.randn(size)
        })
        
        # GPU join
        start = time.time()
        result_gpu = engine.query_pandas(left, """
            SELECT l.*, r.right_value
            FROM df l
            JOIN (SELECT * FROM df) r ON l.key = r.key
        """)
        gpu_time = time.time() - start
        
        # Pandas join
        start = time.time()
        result_pandas = left.merge(right, on='key')
        pandas_time = time.time() - start
        
        print(f"  GPU:    {gpu_time*1000:>8.2f}ms")
        print(f"  Pandas: {pandas_time*1000:>8.2f}ms")
        print(f"  Speedup: {pandas_time/gpu_time:.2f}x")


if __name__ == '__main__':
    print("GPU OLAP Engine - Python Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_basic_queries()
    except Exception as e:
        print(f"Error in basic queries: {e}")
    
    try:
        example_pandas_integration()
    except Exception as e:
        print(f"Error in Pandas integration: {e}")
    
    try:
        example_polars_integration()
    except Exception as e:
        print(f"Error in Polars integration: {e}")
    
    try:
        example_complex_analytics()
    except Exception as e:
        print(f"Error in complex analytics: {e}")
    
    try:
        benchmark_join_performance()
    except Exception as e:
        print(f"Error in benchmarks: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
