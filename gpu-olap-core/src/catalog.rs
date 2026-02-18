use anyhow::{Context, Result};
use arrow::datatypes::Schema;
use arrow_array::RecordBatch;
use dashmap::DashMap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::sync::Arc;
use tracing::info;

/// Catalog manages table metadata and data storage
pub struct Catalog {
    tables: DashMap<String, TableMetadata>,
}

#[derive(Clone)]
struct TableMetadata {
    schema: Arc<Schema>,
    location: String,
    row_count: usize,
    /// In-memory cache of data (for small tables)
    data_cache: Option<Vec<RecordBatch>>,
}

impl Catalog {
    pub fn new() -> Result<Self> {
        Ok(Self {
            tables: DashMap::new(),
        })
    }
    
    /// Load a table from Parquet file
    pub async fn load_table(&self, table_name: &str, path: &str) -> Result<()> {
        info!("Loading table '{}' from {}", table_name, path);
        
        let file = File::open(path)
            .context(format!("Failed to open file: {}", path))?;
        
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create Parquet reader")?;
        
        let schema = Arc::new(builder.schema().as_ref().clone());
        let metadata = builder.metadata();
        
        let row_count = metadata.file_metadata().num_rows() as usize;
        
        info!("  Schema: {:?}", schema);
        info!("  Rows: {}", row_count);
        
        // For small tables (<100MB), load into memory
        let data_cache = if row_count < 10_000_000 {
            info!("  Loading into memory cache");
            let reader = builder.build()?;
            let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()
                .context("Failed to read record batches")?;
            Some(batches)
        } else {
            info!("  Table too large for memory cache");
            None
        };
        
        let metadata = TableMetadata {
            schema,
            location: path.to_string(),
            row_count,
            data_cache,
        };
        
        self.tables.insert(table_name.to_string(), metadata);
        
        info!("Table '{}' loaded successfully", table_name);
        
        Ok(())
    }
    
    /// Get table schema
    pub fn get_schema(&self, table_name: &str) -> Result<Arc<Schema>> {
        let entry = self.tables.get(table_name)
            .context(format!("Table not found: {}", table_name))?;
        Ok(entry.schema.clone())
    }
    
    /// Get table data (from cache if available)
    pub fn get_table_data(&self, table_name: &str) -> Result<Option<Vec<RecordBatch>>> {
        let entry = self.tables.get(table_name)
            .context(format!("Table not found: {}", table_name))?;
        Ok(entry.data_cache.clone())
    }
    
    /// Get table location
    pub fn get_table_location(&self, table_name: &str) -> Result<String> {
        let entry = self.tables.get(table_name)
            .context(format!("Table not found: {}", table_name))?;
        Ok(entry.location.clone())
    }
    
    /// Get table row count
    pub fn get_row_count(&self, table_name: &str) -> Result<usize> {
        let entry = self.tables.get(table_name)
            .context(format!("Table not found: {}", table_name))?;
        Ok(entry.row_count)
    }
    
    /// List all tables
    pub fn list_tables(&self) -> Vec<String> {
        self.tables.iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
    
    /// Drop a table
    pub fn drop_table(&self, table_name: &str) -> Result<()> {
        self.tables.remove(table_name)
            .context(format!("Table not found: {}", table_name))?;
        info!("Dropped table '{}'", table_name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_catalog_creation() {
        let catalog = Catalog::new().unwrap();
        assert_eq!(catalog.list_tables().len(), 0);
    }
}
