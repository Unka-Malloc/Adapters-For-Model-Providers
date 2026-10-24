//! Built-in provider profiles, partitioned by observation source.
//!
//! `major` holds protocol-family owners and documented majors;
//! `observed` holds profiles reconciled from the frozen reference
//! observations; `directory` holds broader-directory reconciliation.
//! Every partition accrues independently — a profile's presence is a
//! supported access surface, never a live-availability claim.

mod major;
mod observed;
mod directory;

/// The complete reconciled profile set.
pub fn builtin_profiles() -> Vec<super::ProviderProfile> {
    let mut all = major::profiles();
    all.extend(observed::profiles());
    all.extend(directory::profiles());
    all
}
