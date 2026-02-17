use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use std::sync::Arc;

pub mod catalog;
pub mod executor;
pub mod optimizer;
pub mod parser;
pub mod physical_plan;
pub mod logical_plan;

/// Main database engine
pub struct OlapEngine {
    catalog: Arc<catalog::Catalog>,
    config: EngineConfig,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum GPU memory to use (bytes)
    pub max_gpu_memory: usize,
    /// Number of CUDA streams for async transfers
    pub num_streams: usize,
    /// Enable unified memory
    pub use_unified_memory: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable query result caching
    pub enable_cache: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory: 8 * 1024 * 1024 * 1024, // 8GB
            num_streams: 8,
            use_unified_memory: true,
            batch_size: 1024 * 1024, // 1M rows
            enable_cache: true,
        }
    }
}

impl OlapEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        let catalog = Arc::new(catalog::Catalog::new()?);
        Ok(Self { catalog, config })
    }

    /// Execute a SQL query and return results as Arrow RecordBatches
    pub async fn execute_query(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        // Parse SQL
        let logical_plan = parser::parse_sql(sql)?;
        
        // Optimize logical plan
        let optimized = optimizer::optimize(logical_plan)?;
        
        // Generate physical plan
        let physical_plan = physical_plan::create_physical_plan(
            optimized,
            &self.catalog,
            &self.config,
        )?;
        
        // Execute on GPU
        executor::execute(physical_plan, &self.config).await
    }

    /// Load table from Parquet file
    pub async fn load_table(&self, table_name: &str, path: &str) -> Result<()> {
        self.catalog.load_table(table_name, path).await
    }

    /// Get table schema
    pub fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>> {
        self.catalog.get_schema(table_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = OlapEngine::new(EngineConfig::default()).unwrap();
        assert!(true);
    }
}
