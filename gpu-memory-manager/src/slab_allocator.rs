use anyhow::{Context, Result};
use cudarc::driver::CudaDevice;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Slab allocator for GPU memory
/// 
/// Manages memory in fixed-size slabs to reduce fragmentation
/// and improve allocation/deallocation performance.
pub struct SlabAllocator {
    device: Arc<CudaDevice>,
    slab_classes: Vec<SlabClass>,
}

struct SlabClass {
    size: usize,
    free_slabs: Mutex<VecDeque<usize>>, // GPU pointers
    max_slabs: usize,
    allocated_count: Mutex<usize>,
}

impl SlabAllocator {
    pub fn new(
        device: Arc<CudaDevice>,
        slab_sizes: &[usize],
        max_slabs_per_class: usize,
    ) -> Result<Self> {
        info!("Initializing Slab Allocator");
        info!("  Slab sizes: {:?}", slab_sizes);
        info!("  Max slabs per class: {}", max_slabs_per_class);
        
        let slab_classes = slab_sizes.iter().map(|&size| {
            SlabClass {
                size,
                free_slabs: Mutex::new(VecDeque::new()),
                max_slabs: max_slabs_per_class,
                allocated_count: Mutex::new(0),
            }
        }).collect();
        
        Ok(Self {
            device,
            slab_classes,
        })
    }
    
    /// Allocate memory from appropriate slab class
    pub fn allocate(&self, size: usize) -> Result<(usize, usize)> {
        // Find appropriate slab class (smallest slab >= requested size)
        let slab_class_idx = self.find_slab_class(size)?;
        let slab_class = &self.slab_classes[slab_class_idx];
        
        // Try to get a free slab
        let ptr = {
            let mut free_slabs = slab_class.free_slabs.lock();
            if let Some(ptr) = free_slabs.pop_front() {
                debug!("Reusing slab from class {} (size {})", slab_class_idx, slab_class.size);
                ptr
            } else {
                drop(free_slabs);
                self.allocate_new_slab(slab_class, slab_class_idx)?
            }
        };
        
        Ok((ptr, slab_class_idx))
    }
    
    /// Free memory back to slab pool
    pub fn free(&self, ptr: usize, slab_class_idx: usize) -> Result<()> {
        if slab_class_idx >= self.slab_classes.len() {
            anyhow::bail!("Invalid slab class index: {}", slab_class_idx);
        }
        
        let slab_class = &self.slab_classes[slab_class_idx];
        let mut free_slabs = slab_class.free_slabs.lock();
        
        // Check if we're at max capacity
        if free_slabs.len() >= slab_class.max_slabs {
            // Actually free the memory
            debug!("Freeing slab (not pooling) from class {}", slab_class_idx);
            // Note: In real implementation, we'd call cudaFree here
            // For now, we just don't add it back to the pool
            let mut count = slab_class.allocated_count.lock();
            *count -= 1;
        } else {
            debug!("Returning slab to pool (class {})", slab_class_idx);
            free_slabs.push_back(ptr);
        }
        
        Ok(())
    }
    
    fn find_slab_class(&self, size: usize) -> Result<usize> {
        for (idx, slab_class) in self.slab_classes.iter().enumerate() {
            if slab_class.size >= size {
                return Ok(idx);
            }
        }
        
        anyhow::bail!(
            "Requested size {} exceeds largest slab size {}",
            size,
            self.slab_classes.last().unwrap().size
        );
    }
    
    fn allocate_new_slab(&self, slab_class: &SlabClass, class_idx: usize) -> Result<usize> {
        let mut count = slab_class.allocated_count.lock();
        
        if *count >= slab_class.max_slabs {
            anyhow::bail!(
                "Maximum number of slabs ({}) reached for class {}",
                slab_class.max_slabs,
                class_idx
            );
        }
        
        debug!("Allocating new slab for class {} (size {})", class_idx, slab_class.size);
        
        // Allocate GPU memory
        // In real implementation, use cudaMalloc
        // For now, use a placeholder
        let ptr = self.allocate_gpu_memory(slab_class.size)?;
        
        *count += 1;
        
        Ok(ptr)
    }
    
    fn allocate_gpu_memory(&self, size: usize) -> Result<usize> {
        // Placeholder implementation
        // In real code, this would use:
        // let ptr = self.device.alloc::<u8>(size)?;
        // For now, return a mock pointer
        
        // Use the device to allocate
        let buffer = self.device.alloc::<u8>(size)
            .context("Failed to allocate GPU memory")?;
        
        // Get raw pointer
        let ptr = buffer.device_ptr() as usize;
        
        // Leak the buffer so it's not freed
        std::mem::forget(buffer);
        
        Ok(ptr)
    }
}

impl Drop for SlabAllocator {
    fn drop(&mut self) {
        info!("Shutting down Slab Allocator");
        
        // Free all slabs
        for (idx, slab_class) in self.slab_classes.iter().enumerate() {
            let free_slabs = slab_class.free_slabs.lock();
            info!("  Class {}: {} free slabs", idx, free_slabs.len());
            
            // In real implementation, we'd free each slab here
            // for ptr in free_slabs.iter() {
            //     cudaFree(*ptr);
            // }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_class_selection() {
        // Test that we select the correct slab class
        let slab_sizes = vec![1024, 4096, 16384];
        
        // For size 500, should select 1024
        // For size 2000, should select 4096
        // For size 10000, should select 16384
    }
}
