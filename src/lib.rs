//! Inference Lab's simulation library.
//!
//! Native builds use jemalloc by default because simulation workloads create
//! and retire many small KV-tree and event structures. The allocator remains a
//! feature so embedders can opt out, and it is excluded from WASM builds where
//! the host controls allocation.

#[cfg(all(feature = "jemalloc", not(target_arch = "wasm32")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

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
