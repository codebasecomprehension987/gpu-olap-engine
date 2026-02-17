#ifndef GPU_JOIN_KERNELS_H
#define GPU_JOIN_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

// Hash join data structures
struct HashEntry {
    uint32_t hash;
    uint32_t row_id;
    uint32_t next;  // Next entry in chain (for collisions)
};

struct HashTable {
    HashEntry* entries;
    uint32_t* buckets;  // Index into entries array
    uint32_t capacity;
    uint32_t size;
};

// Radix bits for radix hash join
#define RADIX_BITS 8
#define NUM_RADIX_PARTITIONS (1 << RADIX_BITS)

// Device functions
__device__ __forceinline__ uint32_t hash_int32(int32_t key) {
    // MurmurHash3 32-bit finalizer
    uint32_t h = (uint32_t)key;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ uint32_t hash_int64(int64_t key) {
    // Simple hash for 64-bit keys
    uint32_t h = (uint32_t)(key ^ (key >> 32));
    return hash_int32((int32_t)h);
}

// Radix partitioning kernel
// Partitions input data based on radix bits of hash value
template<typename KeyType>
__global__ void radix_partition_kernel(
    const KeyType* __restrict__ keys,
    const uint32_t* __restrict__ row_ids,
    uint32_t num_rows,
    uint32_t* __restrict__ partition_offsets,  // [NUM_RADIX_PARTITIONS]
    KeyType* __restrict__ partitioned_keys,
    uint32_t* __restrict__ partitioned_row_ids,
    int radix_shift
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_rows) return;
    
    KeyType key = keys[tid];
    uint32_t hash;
    
    if constexpr (sizeof(KeyType) == 4) {
        hash = hash_int32((int32_t)key);
    } else {
        hash = hash_int64((int64_t)key);
    }
    
    // Extract radix bits
    uint32_t partition = (hash >> radix_shift) & (NUM_RADIX_PARTITIONS - 1);
    
    // Atomic increment to get position in partition
    uint32_t pos = atomicAdd(&partition_offsets[partition], 1);
    
    partitioned_keys[pos] = key;
    partitioned_row_ids[pos] = row_ids ? row_ids[tid] : tid;
}

// Hash table build kernel
// Builds hash table from partitioned data
template<typename KeyType>
__global__ void build_hash_table_kernel(
    const KeyType* __restrict__ keys,
    const uint32_t* __restrict__ row_ids,
    uint32_t num_rows,
    HashEntry* __restrict__ entries,
    uint32_t* __restrict__ buckets,
    uint32_t capacity
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_rows) return;
    
    KeyType key = keys[tid];
    uint32_t hash;
    
    if constexpr (sizeof(KeyType) == 4) {
        hash = hash_int32((int32_t)key);
    } else {
        hash = hash_int64((int64_t)key);
    }
    
    uint32_t bucket = hash % capacity;
    
    // Create entry
    entries[tid].hash = hash;
    entries[tid].row_id = row_ids ? row_ids[tid] : tid;
    
    // Insert into hash table using chaining
    uint32_t old = atomicExch(&buckets[bucket], tid);
    entries[tid].next = old;
}

// Probe kernel
// Probes hash table with right side keys
template<typename KeyType>
__global__ void probe_hash_table_kernel(
    const KeyType* __restrict__ probe_keys,
    const uint32_t* __restrict__ probe_row_ids,
    uint32_t num_probe_rows,
    const HashEntry* __restrict__ entries,
    const uint32_t* __restrict__ buckets,
    uint32_t capacity,
    const KeyType* __restrict__ build_keys,  // For key comparison
    uint32_t* __restrict__ match_count,
    uint32_t* __restrict__ left_matches,   // Output: left row IDs
    uint32_t* __restrict__ right_matches,  // Output: right row IDs
    uint32_t max_matches
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_probe_rows) return;
    
    KeyType probe_key = probe_keys[tid];
    uint32_t hash;
    
    if constexpr (sizeof(KeyType) == 4) {
        hash = hash_int32((int32_t)probe_key);
    } else {
        hash = hash_int64((int64_t)probe_key);
    }
    
    uint32_t bucket = hash % capacity;
    uint32_t entry_idx = buckets[bucket];
    
    // Walk the chain
    while (entry_idx != 0xFFFFFFFF) {
        const HashEntry& entry = entries[entry_idx];
        
        // Check hash first (cheap)
        if (entry.hash == hash) {
            // Compare actual keys
            KeyType build_key = build_keys[entry.row_id];
            if (build_key == probe_key) {
                // Found a match
                uint32_t pos = atomicAdd(match_count, 1);
                
                if (pos < max_matches) {
                    left_matches[pos] = entry.row_id;
                    right_matches[pos] = probe_row_ids ? probe_row_ids[tid] : tid;
                }
            }
        }
        
        entry_idx = entry.next;
    }
}

// Sort-merge join kernels
template<typename KeyType>
__global__ void merge_join_kernel(
    const KeyType* __restrict__ left_keys,
    const uint32_t* __restrict__ left_row_ids,
    uint32_t num_left,
    const KeyType* __restrict__ right_keys,
    const uint32_t* __restrict__ right_row_ids,
    uint32_t num_right,
    uint32_t* __restrict__ match_count,
    uint32_t* __restrict__ left_matches,
    uint32_t* __restrict__ right_matches,
    uint32_t max_matches
) {
    // Each thread processes a segment of left keys
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = tid; i < num_left; i += stride) {
        KeyType left_key = left_keys[i];
        
        // Binary search for matching range in right
        uint32_t right_start = 0;
        uint32_t right_end = num_right;
        
        // Find first match
        while (right_start < right_end) {
            uint32_t mid = (right_start + right_end) / 2;
            if (right_keys[mid] < left_key) {
                right_start = mid + 1;
            } else {
                right_end = mid;
            }
        }
        
        // Generate matches for all equal keys
        for (uint32_t j = right_start; j < num_right && right_keys[j] == left_key; j++) {
            uint32_t pos = atomicAdd(match_count, 1);
            
            if (pos < max_matches) {
                left_matches[pos] = left_row_ids ? left_row_ids[i] : i;
                right_matches[pos] = right_row_ids ? right_row_ids[j] : j;
            }
        }
    }
}

// Aggregation kernels
template<typename T>
__global__ void hash_aggregate_kernel(
    const uint32_t* __restrict__ group_keys,
    const T* __restrict__ values,
    uint32_t num_rows,
    uint32_t* __restrict__ group_ids,
    T* __restrict__ aggregates,
    uint32_t num_groups
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_rows) return;
    
    uint32_t group_key = group_keys[tid];
    uint32_t group_id = group_key % num_groups;
    T value = values[tid];
    
    // Atomic aggregate (SUM example)
    if constexpr (sizeof(T) == 4) {
        atomicAdd(&aggregates[group_id], value);
    } else if constexpr (sizeof(T) == 8) {
        atomicAdd((unsigned long long*)&aggregates[group_id], 
                  (unsigned long long)value);
    }
}

#endif // GPU_JOIN_KERNELS_H
