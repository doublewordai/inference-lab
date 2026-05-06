//! Small helpers shared between the engine's bookkeeping and tests.
//! Per-stream roofline assembly itself lives in `compute::engine`.

/// Number of leading prompt blocks shared by every request in the batch.
/// Uses the incremental prompt block hashes as the equality check: hash N
/// covers tokens 0..N*block_size, so two requests share a prefix of K blocks
/// iff their first K block hashes are pairwise equal.
pub fn shared_prefix_blocks(batch_requests: &[&crate::request::Request]) -> u32 {
    if batch_requests.len() < 2 {
        return 0;
    }
    let first = batch_requests[0].get_prompt_block_hashes();
    let mut shared = first.len();
    for req in &batch_requests[1..] {
        let other = req.get_prompt_block_hashes();
        let mut i = 0;
        while i < shared && i < other.len() && first[i] == other[i] {
            i += 1;
        }
        shared = i;
        if shared == 0 {
            return 0;
        }
    }
    shared as u32
}
