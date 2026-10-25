//! A4MP — Adapters For Model Providers.
//!
//! Independently implemented, provider-neutral adapters for model-provider
//! protocols. One typed SDK for ordinary Rust callers, plus optional wire
//! codecs for gateway-style consumers. Provider identity is open-ended;
//! all model operation families are first-class; credential custody,
//! local effects and service lifetime stay with the host.

pub mod api;
pub mod auth;
pub mod broker;
pub mod catalog;
pub mod effects;
pub mod oauth;
pub mod protocols;
pub mod providers;
pub mod sigv4;
pub mod transport;
pub mod wire;

pub use api::*;
pub use auth::*;
pub use catalog::{AuthSpec, Catalog, Client, ProviderProfile, ServiceDef};
pub use effects::*;
pub use providers::*;
pub use transport::*;

/// Compile the README quickstart as a doctest so the published example
/// cannot drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
#[doc(hidden)]
pub struct ReadmeDoctests;
