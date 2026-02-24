use anyhow::Result;
use arrow_schema::{DataType, Schema};

pub trait SchemaExt {
    fn project_by_name(&self, names: &[&str]) -> Result<Schema>;
    fn row_byte_width(&self) -> usize;
    fn is_gpu_compatible(&self) -> bool;
}

impl SchemaExt for Schema {
    fn project_by_name(&self, names: &[&str]) -> Result<Schema> {
        let fields = names.iter().map(|n| {
            self.field_with_name(n)
                .map(|f| f.clone())
                .map_err(|_| anyhow::anyhow!("Column '{}' not found", n))
        }).collect::<Result<Vec<_>>>()?;
        Ok(Schema::new(fields))
    }

    fn row_byte_width(&self) -> usize {
        self.fields().iter().map(|f| match f.data_type() {
            DataType::Int8 | DataType::UInt8 | DataType::Boolean => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
            _ => 8,
        }).sum()
    }

    fn is_gpu_compatible(&self) -> bool {
        self.fields().iter().all(|f| !matches!(f.data_type(),
            DataType::List(_) | DataType::LargeList(_) | DataType::Struct(_) | DataType::Union(_, _)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{Field, Schema};

    #[test]
    fn test_project_by_name() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]);
        let proj = schema.project_by_name(&["a"]).unwrap();
        assert_eq!(proj.fields().len(), 1);
    }

    #[test]
    fn test_row_byte_width() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),   // 8
            Field::new("b", DataType::Float32, false),  // 4
        ]);
        assert_eq!(schema.row_byte_width(), 12);
    }
}
