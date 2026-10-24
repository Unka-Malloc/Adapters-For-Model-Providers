//! Profiles reconciled from the frozen reference observations
//! (Magpie presets, Pi factories, Hermes plugins, Magpie community
//! extensions). Entries are added only with documented endpoints.

use super::super::ProviderProfile;

/// Reference-observed providers not already in `major`.
pub fn profiles() -> Vec<ProviderProfile> {
    vec![]
}
