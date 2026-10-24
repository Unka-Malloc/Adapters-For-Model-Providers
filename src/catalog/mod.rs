//! Canonical provider registry: declarative profiles, alias resolution,
//! protocol-dispatched adapter binding, and the `Client` entry point.
//!
//! Profiles are data, not code: one entry per provider service surface.
//! The registry reconciles them; it never manufactures capability or
//! availability claims.

mod profiles;
pub use profiles::builtin_profiles;

use std::collections::HashMap;
use std::sync::Arc;

use crate::api::*;
use crate::auth::{AuthKind, CredentialResolver};
use crate::effects::EffectHost;
use crate::transport::Transport;

/// Authentication mechanism a profile accepts. The spec is declarative;
/// adapter binding constructs the concrete `AuthKind`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AuthSpec {
    /// No credential required (local servers, public endpoints).
    None,
    /// Static API key.
    ApiKey,
    /// Bearer token (static or OAuth-issued).
    Bearer,
    /// OAuth authorization flow (PKCE/device) through host interaction.
    OAuth,
    /// Cloud identity (signed requests, workload identity).
    CloudIdentity,
    /// External client/plugin-backed access with declared requirements.
    External { client: &'static str },
}

/// One service endpoint a profile exposes: protocol + base URL +
/// optional region label + required non-secret headers.
#[derive(Clone, Debug)]
pub struct ServiceDef {
    pub protocol: Protocol,
    pub base_url: &'static str,
    pub region: Option<&'static str>,
    pub headers: &'static [(&'static str, &'static str)],
}

impl ServiceDef {
    /// Materialize into a `ServiceTarget` for adapter binding.
    pub fn target(&self, provider: &ProviderId) -> Result<ServiceTarget, Error> {
        let url = url::Url::parse(self.base_url).map_err(|e| {
            Error::Protocol(format!("profile {} endpoint {e}", provider.as_str()))
        })?;
        let mut target = ServiceTarget::new(provider.clone(), self.protocol.clone(), url);
        target.region = self.region.map(str::to_owned);
        for (k, v) in self.headers {
            if let (Ok(name), Ok(value)) = (
                http::HeaderName::try_from(*k),
                http::HeaderValue::from_str(v),
            ) {
                target.headers.insert(name, value);
            }
        }
        Ok(target)
    }
}

/// Declarative profile for one provider's access surface.
///
/// `targets` enumerates service endpoints × protocols (regional plans,
/// native vs compatible dialects). `auth` enumerates accepted credential
/// mechanisms. Neither asserts live availability.
#[derive(Clone, Debug)]
pub struct ProviderProfile {
    pub id: ProviderId,
    pub name: &'static str,
    /// Alternative spellings/legacy ids resolving to this provider.
    pub aliases: &'static [&'static str],
    /// Accepted authentication mechanisms, preferred first.
    pub auth: Vec<AuthSpec>,
    /// Service endpoints this profile exposes.
    pub targets: Vec<ServiceDef>,
    /// Model catalogue provenance hint; never a capability promise.
    pub catalogue: Option<&'static str>,
}

impl ProviderProfile {
    /// First service target speaking `protocol`, else the profile's
    /// primary target, else `None`.
    pub fn target_for(&self, protocol: &Protocol, region: Option<&str>) -> Option<ServiceTarget> {
        let candidates: Vec<&ServiceDef> = self
            .targets
            .iter()
            .filter(|t| region.is_none() || t.region == region)
            .collect();
        let chosen = candidates
            .iter()
            .find(|t| t.protocol == *protocol)
            .or(candidates.first())
            .copied()?;
        chosen.target(&self.id).ok()
    }
}

/// Adapter construction bound to a protocol family.
type AdapterCtor = Arc<
    dyn Fn(
            ServiceTarget,
            AuthKind,
            Arc<dyn Transport>,
            Arc<dyn CredentialResolver>,
            Option<Arc<dyn EffectHost>>,
        ) -> Result<Arc<dyn Adapter>, Error>
        + Send
        + Sync,
>;

/// The canonical registry: profiles by id, aliases, and protocol-dispatched
/// adapter factories.
pub struct Catalog {
    profiles: HashMap<ProviderId, ProviderProfile>,
    aliases: HashMap<String, ProviderId>,
    adapters: HashMap<Protocol, AdapterCtor>,
}

impl Default for Catalog {
    fn default() -> Self {
        Self::with_builtin_adapters()
    }
}

impl Catalog {
    /// Registry with the crate's built-in adapter factories registered.
    pub fn with_builtin_adapters() -> Self {
        let mut c = Self::new(builtin_profiles());
        c.register_openai();
        c.register_anthropic();
        c.register_gemini();
        c.register_bedrock();
        c.register_cohere();
        c.register_ollama();
        c
    }

    /// Registry over an explicit profile set, no adapter factories.
    pub fn new(profiles: Vec<ProviderProfile>) -> Self {
        let mut aliases = HashMap::new();
        let mut by_id = HashMap::new();
        for p in profiles {
            for a in p.aliases {
                aliases.insert(a.to_ascii_lowercase(), p.id.clone());
            }
            by_id.insert(p.id.clone(), p);
        }
        Self {
            profiles: by_id,
            aliases,
            adapters: HashMap::new(),
        }
    }

    /// Register an adapter factory for one protocol family.
    pub fn register_adapter(&mut self, protocol: Protocol, ctor: AdapterCtor) {
        self.adapters.insert(protocol, ctor);
    }

    /// A profile by canonical id or alias (case-insensitive).
    pub fn resolve(&self, id_or_alias: &str) -> Option<&ProviderProfile> {
        let key = id_or_alias.to_ascii_lowercase();
        self.profiles
            .get(&ProviderId::new(id_or_alias))
            .or_else(|| self.aliases.get(&key).and_then(|id| self.profiles.get(id)))
            .or_else(|| {
                self.profiles
                    .values()
                    .find(|p| p.id.as_str().eq_ignore_ascii_case(id_or_alias))
            })
    }

    /// Iterate all registered profiles.
    pub fn profiles(&self) -> impl Iterator<Item = &ProviderProfile> {
        self.profiles.values()
    }

    /// Bind a service target to a live adapter via its protocol family.
    pub fn bind(
        &self,
        target: ServiceTarget,
        auth: AuthKind,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
        effects: Option<Arc<dyn EffectHost>>,
    ) -> Result<Arc<dyn Adapter>, Error> {
        let ctor = self.adapters.get(&target.protocol).ok_or_else(|| {
            Error::Unsupported(format!("no adapter for protocol {:?}", target.protocol))
        })?;
        ctor(target, auth, transport, resolver, effects)
    }

    fn register_openai(&mut self) {
        let ctor: AdapterCtor = Arc::new(|target, auth, transport, resolver, effects| {
            let adapter = crate::providers::OpenAiAdapter::new(target, auth, transport, resolver)?;
            Ok(Arc::new(match effects {
                Some(e) => adapter.with_effects(e),
                None => adapter,
            }))
        });
        self.adapters
            .insert(Protocol::OpenAiChatCompletions, ctor.clone());
        self.adapters.insert(Protocol::OpenAiResponses, ctor);
    }

    fn register_anthropic(&mut self) {
        self.adapters.insert(
            Protocol::AnthropicMessages,
            Arc::new(|target, auth, transport, resolver, effects| {
                let adapter =
                    crate::providers::AnthropicAdapter::new(target, auth, transport, resolver)?;
                Ok(Arc::new(match effects {
                    Some(e) => adapter.with_effects(e),
                    None => adapter,
                }))
            }),
        );
    }

    fn register_cohere(&mut self) {
        self.adapters.insert(
            Protocol::Other("cohere".into()),
            Arc::new(|target, auth, transport, resolver, effects| {
                let adapter =
                    crate::providers::CohereAdapter::new(target, auth, transport, resolver)?;
                Ok(Arc::new(match effects {
                    Some(e) => adapter.with_effects(e),
                    None => adapter,
                }))
            }),
        );
    }

    fn register_ollama(&mut self) {
        self.adapters.insert(
            Protocol::Other("ollama".into()),
            Arc::new(|target, auth, transport, resolver, effects| {
                let adapter =
                    crate::providers::OllamaAdapter::new(target, auth, transport, resolver)?;
                Ok(Arc::new(match effects {
                    Some(e) => adapter.with_effects(e),
                    None => adapter,
                }))
            }),
        );
    }

    fn register_bedrock(&mut self) {
        self.adapters.insert(
            Protocol::BedrockConverse,
            Arc::new(|target, auth, transport, resolver, effects| {
                let adapter =
                    crate::providers::BedrockAdapter::new(target, auth, transport, resolver)?;
                Ok(Arc::new(match effects {
                    Some(e) => adapter.with_effects(e),
                    None => adapter,
                }))
            }),
        );
    }

    fn register_gemini(&mut self) {
        self.adapters.insert(
            Protocol::GeminiGenerateContent,
            Arc::new(|target, auth, transport, resolver, effects| {
                let adapter =
                    crate::providers::GeminiAdapter::new(target, auth, transport, resolver)?;
                Ok(Arc::new(match effects {
                    Some(e) => adapter.with_effects(e),
                    None => adapter,
                }))
            }),
        );
    }
}

/// The consumer-facing entry point: one transport, one credential port,
/// one effect port, one registry. `bind` produces live adapters.
pub struct Client {
    catalog: Catalog,
    transport: Arc<dyn Transport>,
    resolver: Arc<dyn CredentialResolver>,
    effects: Option<Arc<dyn EffectHost>>,
}

impl Client {
    pub fn new(
        catalog: Catalog,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
    ) -> Self {
        Self {
            catalog,
            transport,
            resolver,
            effects: None,
        }
    }

    pub fn with_effects(mut self, effects: Arc<dyn EffectHost>) -> Self {
        self.effects = Some(effects);
        self
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Bind an adapter from a provider profile: picks the profile's target
    /// for `protocol` (or its primary target when `protocol` is `None`)
    /// and constructs the adapter for that protocol family.
    pub fn bind_provider(
        &self,
        provider: &str,
        protocol: Option<Protocol>,
        region: Option<&str>,
        auth: AuthKind,
    ) -> Result<Arc<dyn Adapter>, Error> {
        let profile = self
            .catalog
            .resolve(provider)
            .ok_or_else(|| Error::Unsupported(format!("unknown provider {provider:?}")))?;
        let target = profile
            .target_for(&protocol.unwrap_or_else(|| profile.targets[0].protocol.clone()), region)
            .ok_or_else(|| {
                Error::Unsupported(format!("no service target for {provider}"))
            })?;
        self.catalog.bind(
            target,
            auth,
            self.transport.clone(),
            self.resolver.clone(),
            self.effects.clone(),
        )
    }

    /// Bind an adapter for an explicit service target (custom endpoints,
    /// self-hosted servers) without a profile.
    pub fn bind_target(&self, target: ServiceTarget, auth: AuthKind) -> Result<Arc<dyn Adapter>, Error> {
        self.catalog.bind(
            target,
            auth,
            self.transport.clone(),
            self.resolver.clone(),
            self.effects.clone(),
        )
    }
}
