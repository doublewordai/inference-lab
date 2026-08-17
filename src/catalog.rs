//! Hardware and model presets that ship with the crate.
//!
//! The source of truth is `catalog/hardware/*.toml` and
//! `catalog/models/*.toml` in the repository; the build script embeds them,
//! so `catalog::hardware("b200")` and `catalog::model("deepseek-v4-flash")`
//! work in any build, wasm included, and a serving config can name an entry
//! (`hardware = "b200"`, `model = "deepseek-v4-flash"`) instead of copying
//! its numbers.

use crate::config::{HardwareConfig, ModelSpec};

include!(concat!(env!("OUT_DIR"), "/catalog_entries.rs"));

fn source(kind: &str, name: &str) -> Option<&'static str> {
    ENTRIES
        .iter()
        .find(|(k, n, _)| *k == kind && *n == name)
        .map(|(_, _, src)| *src)
}

fn names(kind: &str) -> Vec<&'static str> {
    ENTRIES
        .iter()
        .filter(|(k, _, _)| *k == kind)
        .map(|(_, n, _)| *n)
        .collect()
}

/// Names of the shipped hardware presets, sorted.
pub fn hardware_names() -> Vec<&'static str> {
    names("hardware")
}

/// Names of the shipped model presets, sorted.
pub fn model_names() -> Vec<&'static str> {
    names("models")
}

/// A hardware preset by name.
pub fn hardware(name: &str) -> Result<HardwareConfig, String> {
    let src = source("hardware", name).ok_or_else(|| {
        format!(
            "unknown hardware preset {name:?}; available: {}",
            hardware_names().join(", ")
        )
    })?;
    let hw: HardwareConfig =
        toml::from_str(src).map_err(|e| format!("catalog hardware {name:?}: {e}"))?;
    if let Some(m) = &hw.memory {
        m.validate()
            .map_err(|e| format!("catalog hardware {name:?}: {e}"))?;
    }
    Ok(hw)
}

/// A model preset by name, validated.
pub fn model(name: &str) -> Result<ModelSpec, String> {
    let src = source("models", name).ok_or_else(|| {
        format!(
            "unknown model preset {name:?}; available: {}",
            model_names().join(", ")
        )
    })?;
    let spec: ModelSpec =
        toml::from_str(src).map_err(|e| format!("catalog model {name:?}: {e}"))?;
    spec.validate()?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shipped_entry_parses_and_validates() {
        assert!(!hardware_names().is_empty());
        assert!(!model_names().is_empty());
        for n in hardware_names() {
            let hw = hardware(n).unwrap_or_else(|e| panic!("{e}"));
            // Every shipped preset offers at least one store reachable
            // from a GPU by a direct link.
            let m = hw
                .memory
                .as_ref()
                .unwrap_or_else(|| panic!("{n}: no [memory]"));
            assert!(
                m.stores.iter().any(|s| m.gpu_link_to(&s.name).is_some()),
                "{n}: no store with a gpu link"
            );
        }
        for n in model_names() {
            model(n).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    #[test]
    fn unknown_names_list_the_alternatives() {
        let e = hardware("nope").unwrap_err();
        assert!(e.contains("available"));
        assert!(model("nope").is_err());
    }
}
