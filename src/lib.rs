//! Inference Lab's simulation library.
//!
//! Native builds use jemalloc by default because simulation workloads create
//! and retire many small KV-tree and event structures. The allocator remains a
//! feature so embedders can opt out, and it is excluded from WASM builds where
//! the host controls allocation.

#[cfg(all(feature = "jemalloc", not(target_arch = "wasm32")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(test, feature = "jemalloc", not(target_arch = "wasm32")))]
mod allocator_tests {
    use super::GLOBAL_ALLOCATOR;
    use std::alloc::{GlobalAlloc, Layout};

    #[test]
    fn jemalloc_handles_native_allocations() {
        let layout = Layout::from_size_align(4096, 64).unwrap();

        // SAFETY: the allocation is checked for null, written only within its
        // layout, and returned to the same allocator with that exact layout.
        unsafe {
            let ptr = GLOBAL_ALLOCATOR.alloc(layout);
            assert!(!ptr.is_null());
            ptr.write_bytes(0xa5, layout.size());
            GLOBAL_ALLOCATOR.dealloc(ptr, layout);
        }
    }
}

pub mod catalog;
pub mod compute;
pub mod config;
pub mod dataset;
pub mod kv_cache;
pub mod metrics;
pub mod request;
pub mod router;
pub mod scheduler;
pub mod simulation;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(feature = "serve")]
pub mod serve;
