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
    ///
    /// Per-resource providers (Azure resource names, Cloudflare account
    /// ids, ...) declare `base_url` templates with `{name}` placeholders;
    /// those must be materialized with [`Self::target_with`] instead —
    /// `target` fails rather than returning a URL containing placeholders.
    pub fn target(&self, provider: &ProviderId) -> Result<ServiceTarget, Error> {
        self.materialize(provider, self.base_url)
    }

    /// Materialize a templated endpoint by substituting `{name}`
    /// placeholders in `base_url` with values from `vars`. Unknown
    /// placeholder names make materialization fail with a protocol error
    /// naming the missing variable — never a silently literal URL.
    ///
    /// Values are host-supplied (account ids, resource names), so each
    /// must be a single URL atom: no `/ ? # @ :` or backslash (authority
    /// or path injection), no whitespace, no `..` (path escape). After
    /// substitution the materialized URL must still be `https` and its
    /// authority must match the template's declared host pattern —
    /// `{name}` spans may expand, literal segments must not move — so a
    /// host value can never redirect the request off the declared
    /// endpoint. Checks fail closed with `Error::Protocol`.
    pub fn target_with(
        &self,
        provider: &ProviderId,
        vars: &std::collections::BTreeMap<&str, &str>,
    ) -> Result<ServiceTarget, Error> {
        let mut resolved = self.base_url.to_owned();
        for (key, value) in vars {
            let needle = format!("{{{key}}}");
            // Variables meant for a sibling target stay inert: one vars
            // map may be shared across every target a profile offers.
            if !resolved.contains(&needle) {
                continue;
            }
            check_endpoint_var(provider, key, value, self.base_url)?;
            resolved = resolved.replace(&needle, value.trim_end_matches('/'));
        }
        if let Some(start) = resolved.find('{') {
            if let Some(end) = resolved[start..].find('}') {
                return Err(Error::Protocol(format!(
                    "profile {} endpoint requires variable {{{}}}",
                    provider.as_str(),
                    &resolved[start + 1..start + end],
                )));
            }
        }
        if !self.base_url.contains('{') {
            return self.materialize_checked(provider, &resolved);
        }
        if leading_var(self.base_url).is_some() {
            // A leading variable carries the URL itself: the caller chose
            // the endpoint, so there is no declared host to anchor. The
            // caller's own value already passed check_endpoint_var; parse
            // the substituted template once and build from it.
            let url = url::Url::parse(&resolved).map_err(|e| {
                Error::Protocol(format!("profile {} endpoint {e}", provider.as_str()))
            })?;
            return self.build_target(provider, url);
        }
        self.materialize_checked(provider, &resolved)
    }

    /// [`Self::materialize`] plus the template-materialization post
    /// checks: https only, and the resolved authority must fit the
    /// template host pattern.
    fn materialize_checked(
        &self,
        provider: &ProviderId,
        base_url: &str,
    ) -> Result<ServiceTarget, Error> {
        let url = url::Url::parse(base_url)
            .map_err(|e| Error::Protocol(format!("profile {} endpoint {e}", provider.as_str())))?;
        if url.scheme() != "https" {
            return Err(Error::Protocol(format!(
                "profile {} endpoint must be https after substitution",
                provider.as_str()
            )));
        }
        if !template_host_matches(self.base_url, &url) {
            return Err(Error::Protocol(format!(
                "profile {} endpoint authority left its declared host",
                provider.as_str()
            )));
        }
        self.build_target(provider, url)
    }

    fn materialize(&self, provider: &ProviderId, base_url: &str) -> Result<ServiceTarget, Error> {
        // An unmaterialized template must never surface as a literal URL:
        // `{`/`}` are legal to url::Url::parse, so leftover placeholders
        // are refused here rather than shipped as an endpoint.
        if base_url.contains(['{', '}']) {
            return Err(Error::Protocol(format!(
                "profile {} endpoint has unsubstituted placeholders",
                provider.as_str()
            )));
        }
        let url = url::Url::parse(base_url)
            .map_err(|e| Error::Protocol(format!("profile {} endpoint {e}", provider.as_str())))?;
        self.build_target(provider, url)
    }

    fn build_target(&self, provider: &ProviderId, url: url::Url) -> Result<ServiceTarget, Error> {
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

/// The variable that opens a template (`{name}…`) carries the endpoint
/// URL itself rather than an atom inside it.
fn leading_var(template: &str) -> Option<&str> {
    let rest = template.strip_prefix('{')?;
    let end = rest.find('}')?;
    Some(&rest[..end])
}

/// Is `url`'s host a private or loopback authority — a machine the
/// operator addresses directly rather than a public endpoint? Plain
/// `http` is confined to this class so a caller-supplied endpoint can
/// never put credential-bearing traffic in cleartext on a public path;
/// `https` is accepted everywhere.
fn endpoint_host_is_private(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(ip)) => {
            ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || (ip.octets()[0] == 100 && (ip.octets()[1] & 0xC0) == 0x40)
        }
        Some(url::Host::Ipv6(ip)) => {
            ip.is_loopback() || ip.is_unique_local() || ip.is_unicast_link_local()
        }
        Some(url::Host::Domain(domain)) => {
            domain == "localhost"
                || domain.ends_with(".localhost")
                || domain.ends_with(".local")
                || domain.ends_with(".internal")
                || domain.ends_with(".lan")
                || !domain.contains('.')
        }
        None => false,
    }
}

/// One endpoint variable's value class comes from its position in the
/// template. A leading `{name}` carries the endpoint URL itself (local
/// servers and host-managed gateways declare nothing else): the value
/// must be a complete http(s) URL with a host and no embedded
/// credentials, and `http` is confined to private or loopback hosts.
/// Any other variable is a single URL atom — an authority or
/// path-structure character would let a host-supplied value redirect
/// the endpoint, and `..` lets a path variable walk out of it.
fn check_endpoint_var(
    provider: &ProviderId,
    key: &str,
    value: &str,
    template: &str,
) -> Result<(), Error> {
    if leading_var(template) == Some(key) {
        let clean = url::Url::parse(value)
            .map(|url| {
                matches!(url.scheme(), "http" | "https")
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && (url.scheme() == "https" || endpoint_host_is_private(&url))
            })
            .unwrap_or(false);
        if !clean {
            return Err(Error::Protocol(format!(
                "profile {} endpoint variable {{{key}}} must be an http(s) URL without credentials",
                provider.as_str()
            )));
        }
        return Ok(());
    }
    let bad = value.is_empty()
        || value.contains("..")
        || value.bytes().any(|b| {
            matches!(b, b'/' | b'?' | b'#' | b'@' | b':' | b'\\') || b.is_ascii_whitespace()
        });
    if bad {
        return Err(Error::Protocol(format!(
            "profile {} endpoint variable {{{key}}} has an invalid value",
            provider.as_str()
        )));
    }
    Ok(())
}

/// Does `url`'s authority still fit the host portion of `template`?
/// The template's host is matched as an anchored glob: `{name}` spans
/// expand to any middle text, every literal segment must appear in
/// order, anchored at both ends. So `{resource}.openai.azure.com`
/// accepts `myres.openai.azure.com` and rejects `evil.example`.
fn template_host_matches(template: &str, url: &url::Url) -> bool {
    let Some(authority) = template.split("://").nth(1) else {
        return false;
    };
    let template_host = authority.split(['/', '?', '#']).next().unwrap_or("");
    let host = match url.port() {
        Some(port) => format!("{}:{port}", url.host_str().unwrap_or("")),
        None => url.host_str().unwrap_or("").to_owned(),
    };
    let host = host.to_ascii_lowercase();

    // Literal segments between `{name}` spans, in order.
    let mut segments = Vec::new();
    let mut rest = template_host;
    while let Some(open) = rest.find('{') {
        segments.push(&rest[..open]);
        match rest[open..].find('}') {
            Some(close) => rest = &rest[open + close + 1..],
            None => {
                segments.push(rest);
                rest = "";
            }
        }
    }
    segments.push(rest);

    let mut pos = 0usize;
    let last = segments.len() - 1;
    for (i, seg) in segments.iter().enumerate() {
        if seg.is_empty() {
            continue;
        }
        if i == 0 {
            if !host.starts_with(seg) {
                return false;
            }
            pos = seg.len();
        } else if i == last {
            return host[pos..].ends_with(seg);
        } else {
            match host[pos..].find(seg) {
                Some(found) => pos += found + seg.len(),
                None => return false,
            }
        }
    }
    true
}
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

    /// [`Self::target_for`] with endpoint variables for templated
    /// per-resource base URLs (`{name}` placeholders).
    pub fn target_for_with(
        &self,
        protocol: &Protocol,
        region: Option<&str>,
        vars: &std::collections::BTreeMap<&str, &str>,
    ) -> Option<ServiceTarget> {
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
        chosen.target_with(&self.id, vars).ok()
    }

    /// [`Self::target_for`] with endpoint variables, surfacing a missing
    /// placeholder variable as the error instead of `None`.
    pub fn try_target_for(
        &self,
        protocol: &Protocol,
        region: Option<&str>,
        vars: &std::collections::BTreeMap<&str, &str>,
    ) -> Result<ServiceTarget, Error> {
        let candidates: Vec<&ServiceDef> = self
            .targets
            .iter()
            .filter(|t| region.is_none() || t.region == region)
            .collect();
        let chosen = candidates
            .iter()
            .find(|t| t.protocol == *protocol)
            .or(candidates.first())
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("no service target for {}", self.id)))?;
        chosen.target_with(&self.id, vars)
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
        // Builtin profiles are crate data; a collision is a build bug
        // caught by the uniqueness tests, not a runtime condition.
        let mut c = Self::new(builtin_profiles()).expect("builtin profiles are unique");
        c.register_openai();
        c.register_anthropic();
        c.register_gemini();
        c.register_bedrock();
        c.register_cohere();
        c.register_ollama();
        c
    }

    /// Registry over an explicit profile set, no adapter factories.
    ///
    /// Registration is fail-closed: a duplicated canonical id, an alias
    /// claimed by two profiles, or an alias spelling another profile's
    /// canonical id is a `Error::Protocol` collision — never a silent
    /// overwrite that would misresolve a provider name.
    pub fn new(profiles: Vec<ProviderProfile>) -> Result<Self, Error> {
        let mut aliases: HashMap<String, ProviderId> = HashMap::new();
        let mut by_id: HashMap<ProviderId, ProviderProfile> = HashMap::new();
        for p in profiles {
            for a in p.aliases {
                let key = a.to_ascii_lowercase();
                match aliases.get(&key) {
                    Some(other) if *other != p.id => {
                        return Err(Error::Protocol(format!(
                            "alias {a:?} claimed by {} and {}",
                            other.as_str(),
                            p.id.as_str()
                        )));
                    }
                    _ => {
                        aliases.insert(key, p.id.clone());
                    }
                }
            }
            if by_id.insert(p.id.clone(), p.clone()).is_some() {
                return Err(Error::Protocol(format!(
                    "duplicate provider id {:?}",
                    p.id.as_str()
                )));
            }
        }
        for (key, owner) in &aliases {
            if let Some(other) = by_id.get(&ProviderId::new(key)) {
                if other.id != *owner {
                    return Err(Error::Protocol(format!(
                        "alias {key:?} of {} shadows provider id {}",
                        owner.as_str(),
                        other.id.as_str()
                    )));
                }
            }
        }
        Ok(Self {
            profiles: by_id,
            aliases,
            adapters: HashMap::new(),
        })
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
        self.bind_provider_with(provider, protocol, region, auth, &Default::default())
    }

    /// [`Self::bind_provider`] with endpoint variables for profiles whose
    /// `base_url` is a per-resource template (`{name}` placeholders, e.g.
    /// Azure resource names or Cloudflare account ids).
    pub fn bind_provider_with(
        &self,
        provider: &str,
        protocol: Option<Protocol>,
        region: Option<&str>,
        auth: AuthKind,
        vars: &std::collections::BTreeMap<&str, &str>,
    ) -> Result<Arc<dyn Adapter>, Error> {
        let profile = self
            .catalog
            .resolve(provider)
            .ok_or_else(|| Error::Unsupported(format!("unknown provider {provider:?}")))?;
        let protocol = match protocol {
            Some(p) => p,
            None => profile
                .targets
                .first()
                .ok_or_else(|| {
                    Error::Unsupported(format!("profile {} has no service targets", provider))
                })?
                .protocol
                .clone(),
        };
        let target = profile.try_target_for(&protocol, region, vars)?;
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
    pub fn bind_target(
        &self,
        target: ServiceTarget,
        auth: AuthKind,
    ) -> Result<Arc<dyn Adapter>, Error> {
        self.catalog.bind(
            target,
            auth,
            self.transport.clone(),
            self.resolver.clone(),
            self.effects.clone(),
        )
    }
}
