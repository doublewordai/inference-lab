//! Request routing across the replicas of one deployment.
//!
//! ```toml
//! [router]
//! policy = "round_robin"                    # the default
//!
//! [router]
//! policy = "least_loaded"
//!
//! [router]
//! policy = "session_affinity"                 # session ordinal modulo workers
//!
//! [router]
//! policy = "prefix_affinity"
//! max_load_ratio = 1.5                      # optional bounded-load cap
//!
//! [router]
//! policy = "kv_aware"
//! load_weight = 1.0
//!
//! [decode_router]                           # disaggregated: the decode pool
//! policy = "kv_aware_decode"                # (defaults to [router])
//! load_weight = 64.0
//! ```

use serde::Deserialize;

/// Which router fronts the replicas. See [`crate::router`] for the policies.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum RouterConfig {
    /// Cycle through the replicas in order.
    RoundRobin {},
    /// Fewest requests in the system (running + waiting); ties by queued
    /// prefill tokens, then replica index.
    LeastLoaded {},
    /// Keep every step of a session on `session ordinal % workers`.
    /// Non-session requests fall back to round-robin. This is a deliberately
    /// load-oblivious control for experiments that must hold routing fixed.
    SessionAffinity {},
    /// The replica holding the longest cached prefix of the prompt. With no
    /// cached prefix anywhere, or when the holder's request count exceeds
    /// `max_load_ratio × mean`, falls back to `least_loaded`.
    PrefixAffinity {
        #[serde(default)]
        max_load_ratio: Option<f64>,
    },
    /// Minimise `(prompt − cached prefix) + load_weight × queued prefill
    /// tokens`: the prefill work the request adds plus the prefill work
    /// queued ahead of it, in tokens. A prefill-side policy.
    KvAware {
        #[serde(default = "default_load_weight")]
        load_weight: f64,
    },
    /// Decode-side counterpart for hand-offs: minimise `(context − prompt
    /// prefix resident in the decoder's HBM) + load_weight × running
    /// sequences`, in tokens — the KV the transfer must move plus the
    /// decode batch it joins. Decoders whose free KV cannot hold the
    /// context are passed over while any can.
    KvAwareDecode {
        #[serde(default = "default_decode_load_weight")]
        load_weight: f64,
    },
}

fn default_load_weight() -> f64 {
    1.0
}

/// One running sequence weighs one block of transfer by default.
fn default_decode_load_weight() -> f64 {
    64.0
}

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig::RoundRobin {}
    }
}

impl RouterConfig {
    /// Short name for reports.
    pub fn name(&self) -> &'static str {
        match self {
            RouterConfig::RoundRobin {} => "round_robin",
            RouterConfig::LeastLoaded {} => "least_loaded",
            RouterConfig::SessionAffinity {} => "session_affinity",
            RouterConfig::PrefixAffinity { .. } => "prefix_affinity",
            RouterConfig::KvAware { .. } => "kv_aware",
            RouterConfig::KvAwareDecode { .. } => "kv_aware_decode",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_policy() {
        let rr: RouterConfig = toml::from_str("policy = \"round_robin\"").unwrap();
        assert_eq!(rr, RouterConfig::RoundRobin {});
        let ll: RouterConfig = toml::from_str("policy = \"least_loaded\"").unwrap();
        assert_eq!(ll, RouterConfig::LeastLoaded {});
        let sa: RouterConfig = toml::from_str("policy = \"session_affinity\"").unwrap();
        assert_eq!(sa, RouterConfig::SessionAffinity {});
        let pa: RouterConfig =
            toml::from_str("policy = \"prefix_affinity\"\nmax_load_ratio = 1.5").unwrap();
        assert_eq!(
            pa,
            RouterConfig::PrefixAffinity {
                max_load_ratio: Some(1.5)
            }
        );
        let kv: RouterConfig = toml::from_str("policy = \"kv_aware\"").unwrap();
        assert_eq!(kv, RouterConfig::KvAware { load_weight: 1.0 });
        let kd: RouterConfig = toml::from_str("policy = \"kv_aware_decode\"").unwrap();
        assert_eq!(kd, RouterConfig::KvAwareDecode { load_weight: 64.0 });
    }

    #[test]
    fn rejects_unknown_fields() {
        assert!(toml::from_str::<RouterConfig>("policy = \"round_robin\"\nfoo = 1").is_err());
        assert!(toml::from_str::<RouterConfig>("policy = \"nope\"").is_err());
    }
}
