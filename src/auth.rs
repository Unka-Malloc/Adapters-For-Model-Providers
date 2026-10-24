//! Credential types and the host-implemented credential port.
//!
//! The library owns authentication mechanics (headers, signing, OAuth
//! flows); the host owns custody: storage, authorization, vaults and
//! leases. Credential values never appear in logs or debug output.

use std::fmt;
use std::time::SystemTime;

use crate::api::{Error, ServiceTarget};

/// Authentication mechanism a service target requires.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum AuthKind {
    /// No credential.
    None,
    /// Static API key; placement (header/query) is the adapter's own.
    ApiKey,
    /// Bearer token (OAuth access token or equivalent).
    Bearer,
    /// Cloud identity: signed requests or identity-brokered tokens
    /// (AWS SigV4, GCP/Azure identity). The adapter owns the mechanics;
    /// the host supplies identity material through the resolver.
    CloudIdentity { service: String },
    /// External client or plugin-backed access: an explicit adapter with
    /// declared runtime requirements. Never silently uses another
    /// application's login.
    External { client: String },
}

/// Credential material with a redacted `Debug`/`Display`.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    /// The only way to read the value; call sites are greppable.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}
impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// Resolved credential bound to an authentication mechanism.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Credential {
    None,
    /// Static API key material.
    ApiKey(Secret),
    /// Bearer/OAuth access token with optional expiry for refresh.
    Bearer { token: Secret, expires_at: Option<SystemTime> },
    /// Cloud-identity material; shape is mechanism-specific and opaque
    /// to callers (signing key, session token, workload-identity token…).
    Cloud {
        token: Secret,
        expires_at: Option<SystemTime>,
        /// Mechanism-scoped fields (region, session name, …).
        context: serde_json::Map<String, serde_json::Value>,
    },
}

/// What the resolver must produce a credential for.
#[derive(Clone, Debug)]
pub struct CredentialScope<'a> {
    pub target: &'a ServiceTarget,
    pub auth: &'a AuthKind,
}

/// Host-implemented credential custody port.
///
/// Implementations live outside the library (application vault, OS
/// keychain, managed identity, user interaction). The library asks for
/// material per request; it never stores secrets itself.
#[async_trait::async_trait]
pub trait CredentialResolver: Send + Sync {
    /// Resolve material for the given scope.
    async fn resolve(&self, scope: CredentialScope<'_>) -> Result<Credential, Error>;

    /// Called after a credential was rejected by the provider (e.g. 401).
    /// Default re-resolves; hosts with refreshable material override this
    /// to rotate or re-authorize.
    async fn refresh(
        &self,
        scope: CredentialScope<'_>,
        _rejected: &Credential,
    ) -> Result<Credential, Error> {
        self.resolve(scope).await
    }
}

/// A resolver for services that need no credential.
pub struct NoCredentials;

#[async_trait::async_trait]
impl CredentialResolver for NoCredentials {
    async fn resolve(&self, _scope: CredentialScope<'_>) -> Result<Credential, Error> {
        Ok(Credential::None)
    }
}

/// A resolver for a single static credential (tests, direct API-key use).
/// Cloning shares the material; the secret stays redacted in output.
pub struct StaticCredential(pub Credential);

#[async_trait::async_trait]
impl CredentialResolver for StaticCredential {
    async fn resolve(&self, _scope: CredentialScope<'_>) -> Result<Credential, Error> {
        Ok(self.0.clone())
    }
}
