//! Profiles reconciled from the broader models.dev provider
//! directory. Entries are added only with documented endpoints;
//! defunct or non-provider directory entries are skipped.

use super::super::ProviderProfile;

/// Broader-directory providers not already in `major`/`observed`.
pub fn profiles() -> Vec<ProviderProfile> {
    vec![]
}
