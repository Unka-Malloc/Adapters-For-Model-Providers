//! Catalog registry tests: profile resolution, alias handling, protocol-
//! dispatched binding, and live model discovery through mock transport.

mod common;

use std::sync::Arc;

use a4mp::*;
use common::{MockTransport, Response};

fn client() -> (Client, MockTransport) {
    let transport = MockTransport::new();
    let client = Client::new(
        Catalog::default(),
        Arc::new(transport.clone()),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new(
            common::run_token("api-key"),
        )))),
    );
    (client, transport)
}

#[test]
fn resolve_by_id_alias_and_case() {
    let catalog = Catalog::default();
    assert_eq!(catalog.resolve("openai").unwrap().name, "OpenAI");
    assert_eq!(catalog.resolve("OPENAI").unwrap().id.as_str(), "openai");
    assert_eq!(
        catalog.resolve("moonshot").unwrap().id.as_str(),
        "moonshotai"
    );
    assert_eq!(catalog.resolve("kimi").unwrap().id.as_str(), "moonshotai");
    assert!(catalog.resolve("nonexistent-provider").is_none());
}

#[test]
fn regional_targets_are_distinct() {
    let catalog = Catalog::default();
    let profile = catalog.resolve("moonshotai").unwrap();
    let cn = profile
        .target_for(&Protocol::OpenAiChatCompletions, Some("cn"))
        .unwrap();
    let intl = profile
        .target_for(&Protocol::OpenAiChatCompletions, Some("intl"))
        .unwrap();
    assert_eq!(cn.base_url.host_str(), Some("api.moonshot.cn"));
    assert_eq!(intl.base_url.host_str(), Some("api.moonshot.ai"));
    assert_eq!(cn.region.as_deref(), Some("cn"));
}

#[test]
fn anthropic_profile_carries_version_header() {
    let catalog = Catalog::default();
    let profile = catalog.resolve("anthropic").unwrap();
    let target = profile
        .target_for(&Protocol::AnthropicMessages, None)
        .unwrap();
    assert_eq!(
        target.headers.get("anthropic-version").unwrap(),
        "2023-06-01"
    );
}

#[tokio::test]
async fn bind_provider_constructs_adapter_and_discovers_models() {
    let (client, transport) = client();
    transport.push(Response::Json(
        200,
        serde_json::json!({"object": "list", "data": [
            {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"},
        ]}),
    ));

    let adapter = client
        .bind_provider("deepseek", None, None, AuthKind::ApiKey)
        .unwrap();
    assert_eq!(adapter.target().provider.as_str(), "deepseek");
    assert_eq!(adapter.target().protocol, Protocol::OpenAiChatCompletions);

    let models = adapter.models(&RequestControl::default()).await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "deepseek-chat");
    assert_eq!(models[0].owned_by.as_deref(), Some("deepseek"));

    let recorded = transport.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].method, http::Method::GET);
    assert_eq!(recorded[0].url.path(), "/models");
    assert_eq!(
        recorded[0]
            .headers
            .get(http::header::AUTHORIZATION)
            .unwrap(),
        format!("Bearer {}", common::run_token("api-key")).as_str()
    );
}

#[tokio::test]
async fn protocol_preference_selects_target() {
    let (client, _) = client();
    let adapter = client
        .bind_provider(
            "openai",
            Some(Protocol::OpenAiResponses),
            None,
            AuthKind::ApiKey,
        )
        .unwrap();
    assert_eq!(adapter.target().protocol, Protocol::OpenAiResponses);
}

#[tokio::test]
async fn anthropic_and_gemini_bind_through_profiles() {
    let (client, _) = client();
    let anthropic = client
        .bind_provider("anthropic", None, None, AuthKind::ApiKey)
        .unwrap();
    assert_eq!(anthropic.target().protocol, Protocol::AnthropicMessages);
    let gemini = client
        .bind_provider("gemini", None, None, AuthKind::ApiKey)
        .unwrap();
    assert_eq!(gemini.target().protocol, Protocol::GeminiGenerateContent);
}

#[tokio::test]
async fn unregistered_protocol_reports_unsupported() {
    let (client, _) = client();
    // A protocol family no adapter claims: binding must fail
    // explicitly, not fall through to another protocol family.
    let target = ServiceTarget::new(
        "bedrock",
        Protocol::Other("unregistered-family".into()),
        url::Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com").unwrap(),
    );
    let result = client.bind_target(target, AuthKind::Bearer);
    match result {
        Err(Error::Unsupported(_)) => {}
        Err(other) => panic!("expected Unsupported, got {other:?}"),
        Ok(_) => panic!("expected Unsupported"),
    }
}

#[tokio::test]
async fn unknown_provider_reports_unsupported() {
    let (client, _) = client();
    let result = client.bind_provider("acme-llm", None, None, AuthKind::ApiKey);
    match result {
        Err(Error::Unsupported(_)) => {}
        Err(other) => panic!("expected Unsupported, got {other:?}"),
        Ok(_) => panic!("expected Unsupported"),
    }
}

#[test]
fn profile_target_joins_endpoint_paths() {
    let catalog = Catalog::default();
    let profile = catalog.resolve("deepseek").unwrap();
    let target = profile
        .target_for(&Protocol::OpenAiChatCompletions, None)
        .unwrap();
    assert_eq!(
        target.endpoint("chat/completions").as_str(),
        "https://api.deepseek.com/chat/completions"
    );
}

// ---------------------------------------------------------------------------
// Registration collisions
// ---------------------------------------------------------------------------

fn test_profile(id: &'static str, aliases: &'static [&'static str]) -> ProviderProfile {
    ProviderProfile {
        id: id.into(),
        name: "Test",
        aliases,
        auth: vec![AuthSpec::ApiKey],
        targets: vec![ServiceDef {
            protocol: Protocol::OpenAiChatCompletions,
            base_url: "https://api.example.test",
            region: None,
            headers: &[],
        }],
        catalogue: None,
    }
}

#[test]
fn builtin_profiles_have_unique_ids_and_aliases() {
    // Registration is fail-closed: the builtin set must pass it.
    Catalog::new(a4mp::catalog::builtin_profiles()).unwrap();
}

#[test]
fn duplicate_ids_and_aliases_are_rejected() {
    // Duplicate canonical id.
    match Catalog::new(vec![test_profile("a", &[]), test_profile("a", &[])]) {
        Err(Error::Protocol(m)) => assert!(m.contains("a"), "{m}"),
        Err(e) => panic!("expected collision error, got {e:?}"),
        Ok(_) => panic!("expected collision error, got Ok"),
    }
    // Two profiles claiming one alias.
    match Catalog::new(vec![test_profile("a", &["x"]), test_profile("b", &["x"])]) {
        Err(Error::Protocol(_)) => {}
        Err(e) => panic!("expected collision error, got {e:?}"),
        Ok(_) => panic!("expected collision error, got Ok"),
    }
    // Same alias spelled with different case still collides.
    match Catalog::new(vec![test_profile("a", &["X"]), test_profile("b", &["x"])]) {
        Err(Error::Protocol(_)) => {}
        Err(e) => panic!("expected collision error, got {e:?}"),
        Ok(_) => panic!("expected collision error, got Ok"),
    }
    // An alias spelling another profile's canonical id would misresolve.
    match Catalog::new(vec![test_profile("a", &["b"]), test_profile("b", &[])]) {
        Err(Error::Protocol(_)) => {}
        Err(e) => panic!("expected collision error, got {e:?}"),
        Ok(_) => panic!("expected collision error, got Ok"),
    }
}

// ---------------------------------------------------------------------------
// Templated endpoints (per-resource base URLs)
// ---------------------------------------------------------------------------

#[test]
fn template_materializes_with_endpoint_variables() {
    let catalog = Catalog::default();
    let profile = catalog.resolve("azure-openai").unwrap();
    let vars = std::collections::BTreeMap::from([("resource", "myres")]);
    let target = profile
        .try_target_for(&Protocol::OpenAiChatCompletions, None, &vars)
        .unwrap();
    assert_eq!(
        target.base_url.as_str(),
        "https://myres.openai.azure.com/openai/v1"
    );

    // A path-scoped variable keeps the declared host and path shape.
    let cf = catalog.resolve("cloudflare-workers-ai").unwrap();
    let vars = std::collections::BTreeMap::from([("account_id", "acct9")]);
    let target = cf
        .try_target_for(&Protocol::OpenAiChatCompletions, None, &vars)
        .unwrap();
    assert_eq!(
        target.base_url.as_str(),
        "https://api.cloudflare.com/client/v4/accounts/acct9/ai/v1"
    );
}

#[test]
fn template_missing_variable_names_the_placeholder() {
    let catalog = Catalog::default();
    let profile = catalog.resolve("azure-openai").unwrap();
    match profile.try_target_for(&Protocol::OpenAiChatCompletions, None, &Default::default()) {
        Err(Error::Protocol(m)) => assert!(m.contains("resource"), "{m}"),
        Err(e) => panic!("expected Protocol error, got {e:?}"),
        Ok(_) => panic!("expected Protocol error, got Ok"),
    }
    // An unmaterialized template must never surface as a literal URL.
    assert!(
        profile
            .target_for(&Protocol::OpenAiChatCompletions, None)
            .is_none()
    );
}

#[test]
fn template_values_cannot_inject_authority_or_paths() {
    let catalog = Catalog::default();
    let profile = catalog.resolve("azure-openai").unwrap();
    for bad in [
        "evil.com/x",
        "a?x=1",
        "a#frag",
        "user@host",
        "a:443",
        "a\\b",
        "has space",
        "..",
        "a..b",
        "",
    ] {
        let vars = std::collections::BTreeMap::from([("resource", bad)]);
        match profile.try_target_for(&Protocol::OpenAiChatCompletions, None, &vars) {
            Err(Error::Protocol(_)) => {}
            other => panic!("value {bad:?} must fail closed, got {other:?}"),
        }
    }
    // A dotted value expands as a subdomain of the declared host only.
    let vars = std::collections::BTreeMap::from([("resource", "a.b")]);
    let target = profile
        .try_target_for(&Protocol::OpenAiChatCompletions, None, &vars)
        .unwrap();
    assert_eq!(target.base_url.host_str(), Some("a.b.openai.azure.com"));
}

#[test]
fn materialized_template_stays_https_and_on_host() {
    // A hand-rolled http template must fail closed too.
    let def = ServiceDef {
        protocol: Protocol::OpenAiChatCompletions,
        base_url: "http://{resource}.example.test",
        region: None,
        headers: &[],
    };
    let vars = std::collections::BTreeMap::from([("resource", "x")]);
    match def.target_with(&"test".into(), &vars) {
        Err(Error::Protocol(_)) => {}
        other => panic!("http template must fail, got {other:?}"),
    }
}

#[tokio::test]
async fn bind_provider_with_materializes_template() {
    let (client, _) = client();
    let vars = std::collections::BTreeMap::from([("resource", "acct42")]);
    let adapter = client
        .bind_provider_with("azure", None, None, AuthKind::ApiKey, &vars)
        .unwrap();
    assert_eq!(
        adapter.target().base_url.as_str(),
        "https://acct42.openai.azure.com/openai/v1"
    );
}

#[tokio::test]
async fn template_bind_rejects_bad_value_before_dispatch() {
    let (client, transport) = client();
    let vars = std::collections::BTreeMap::from([("resource", "a/b")]);
    match client.bind_provider_with("azure", None, None, AuthKind::ApiKey, &vars) {
        Err(Error::Protocol(_)) => {}
        Err(e) => panic!("expected Protocol error, got {e:?}"),
        Ok(_) => panic!("expected Protocol error, got Ok"),
    }
    assert!(transport.recorded().is_empty());
}

#[tokio::test]
async fn local_profiles_require_a_runtime_endpoint() {
    let (client, _) = client();
    for provider in ["ollama", "lmstudio", "omlx", "mlx-serve"] {
        match client.bind_provider(provider, None, None, AuthKind::None) {
            Err(Error::Protocol(msg)) => {
                assert!(msg.contains("{endpoint}"), "{provider}: {msg}")
            }
            Err(e) => panic!("{provider} without an endpoint must fail with Protocol, got {e:?}"),
            Ok(_) => panic!("{provider} without an endpoint must fail, got Ok"),
        }
    }
}

#[tokio::test]
async fn local_profiles_bind_the_runtime_endpoint() {
    let (client, _) = client();
    let vars = std::collections::BTreeMap::from([("endpoint", "http://models.internal:8000")]);
    // Every templated chat surface grows `/v1` on the caller's endpoint.
    for (provider, protocol) in [
        ("omlx", Protocol::OpenAiChatCompletions),
        ("omlx", Protocol::OpenAiResponses),
        ("mlx-serve", Protocol::OpenAiChatCompletions),
        ("mlx-serve", Protocol::OpenAiResponses),
        ("atomic-chat", Protocol::OpenAiChatCompletions),
        ("lmstudio", Protocol::OpenAiChatCompletions),
        ("ollama", Protocol::OpenAiChatCompletions),
    ] {
        let adapter = client
            .bind_provider_with(
                provider,
                Some(protocol.clone()),
                None,
                AuthKind::None,
                &vars,
            )
            .unwrap_or_else(|e| panic!("{provider}/{protocol:?} must bind: {e:?}"));
        let url = &adapter.target().base_url;
        assert_eq!(url.scheme(), "http", "{provider}");
        assert_eq!(url.host_str(), Some("models.internal"), "{provider}");
        assert_eq!(url.port(), Some(8000), "{provider}");
        assert_eq!(url.path(), "/v1", "{provider}");
    }
    // Ollama's native dialect binds the bare endpoint, no `/v1` suffix.
    let adapter = client
        .bind_provider_with(
            "ollama",
            Some(Protocol::Other("ollama".into())),
            None,
            AuthKind::None,
            &vars,
        )
        .unwrap();
    assert_eq!(
        adapter.target().base_url.as_str(),
        "http://models.internal:8000/"
    );
    // A region-selected template (privatemode's local proxy) binds too.
    let adapter = client
        .bind_provider_with(
            "privatemode",
            Some(Protocol::OpenAiChatCompletions),
            Some("local-proxy"),
            AuthKind::ApiKey,
            &vars,
        )
        .unwrap();
    assert_eq!(adapter.target().base_url.path(), "/v1");
    assert_eq!(adapter.target().region.as_deref(), Some("local-proxy"));
}

#[tokio::test]
async fn endpoint_trailing_slash_does_not_double_the_path() {
    let (client, _) = client();
    let vars = std::collections::BTreeMap::from([("endpoint", "http://10.0.0.8:8080/")]);
    let adapter = client
        .bind_provider_with("lmstudio", None, None, AuthKind::None, &vars)
        .unwrap();
    let url = &adapter.target().base_url;
    assert_eq!(url.path(), "/v1");
    assert!(!url.path().contains("//"), "{url}");
}

#[tokio::test]
async fn http_endpoints_stay_on_private_hosts() {
    let (client, _) = client();
    // Cleartext endpoints are confined to hosts the operator addresses
    // directly; TEST-NET-3 (203.0.113.0/24) stands in for a public IP.
    for bad in ["http://203.0.113.10:8080", "http://8.8.8.8"] {
        let vars = std::collections::BTreeMap::from([("endpoint", bad)]);
        match client.bind_provider_with("ollama", None, None, AuthKind::None, &vars) {
            Err(Error::Protocol(_)) => {}
            Err(e) => panic!("public http endpoint {bad:?} must fail with Protocol, got {e:?}"),
            Ok(_) => panic!("public http endpoint {bad:?} must fail, got Ok"),
        }
    }
    // Loopback, RFC1918, CGNAT, and operator-local names all bind.
    for good in [
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
        "http://192.168.1.5:11434",
        "http://100.64.1.2:8000",
        "http://mlx.local",
        "http://nas.internal",
        "http://server",
    ] {
        let vars = std::collections::BTreeMap::from([("endpoint", good)]);
        client
            .bind_provider_with("ollama", None, None, AuthKind::None, &vars)
            .unwrap_or_else(|e| panic!("private http endpoint {good:?} must bind: {e:?}"));
    }
    // https travels anywhere, public included.
    let vars = std::collections::BTreeMap::from([("endpoint", "https://203.0.113.10:8443")]);
    client
        .bind_provider_with("ollama", None, None, AuthKind::None, &vars)
        .unwrap();
}

#[tokio::test]
async fn endpoint_vars_map_may_carry_sibling_target_keys() {
    let (client, _) = client();
    // One vars map shared across targets: keys naming placeholders the
    // chosen target does not declare stay inert instead of failing.
    let vars = std::collections::BTreeMap::from([
        ("endpoint", "http://10.0.0.8:8080"),
        ("resource", "myres"),
        ("account_id", "acct9"),
    ]);
    let adapter = client
        .bind_provider_with("lmstudio", None, None, AuthKind::None, &vars)
        .unwrap();
    assert_eq!(adapter.target().base_url.path(), "/v1");
    // The reverse direction holds: an `endpoint` key does not disturb
    // an atom-template bind such as azure's `{resource}`.
    let adapter = client
        .bind_provider_with("azure", None, None, AuthKind::ApiKey, &vars)
        .unwrap();
    assert_eq!(
        adapter.target().base_url.as_str(),
        "https://myres.openai.azure.com/openai/v1"
    );
}

#[tokio::test]
async fn endpoint_variables_must_be_clean_http_urls() {
    let (client, _) = client();
    for bad in [
        "ftp://models.internal",
        "http://user:pw@models.internal",
        "not a url",
        "",
    ] {
        let vars = std::collections::BTreeMap::from([("endpoint", bad)]);
        match client.bind_provider_with("ollama", None, None, AuthKind::None, &vars) {
            Err(Error::Protocol(_)) => {}
            Err(e) => panic!("endpoint {bad:?} must fail with Protocol, got {e:?}"),
            Ok(_) => panic!("endpoint {bad:?} must fail, got Ok"),
        }
    }
}

// ---------------------------------------------------------------------------
// Target selection: regions, fallback, empty profiles
// ---------------------------------------------------------------------------

#[test]
fn region_filter_selects_or_misses() {
    let catalog = Catalog::default();
    let moonshot = catalog.resolve("moonshotai").unwrap();
    assert!(
        moonshot
            .target_for(&Protocol::OpenAiChatCompletions, Some("bogus-region"))
            .is_none()
    );
}

#[test]
fn unmatched_protocol_falls_back_to_primary_target() {
    let catalog = Catalog::default();
    let profile = catalog.resolve("deepseek").unwrap();
    let target = profile
        .target_for(&Protocol::Other("no-such-family".into()), None)
        .unwrap();
    assert_eq!(target.protocol, Protocol::OpenAiChatCompletions);
}

#[tokio::test]
async fn empty_target_profile_reports_unsupported_not_panic() {
    let mut empty = test_profile("empty", &[]);
    empty.targets.clear();
    let catalog = Catalog::new(vec![empty]).unwrap();
    let client = Client::new(
        catalog,
        Arc::new(MockTransport::new()),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new(
            common::run_token("api-key"),
        )))),
    );
    match client.bind_provider("empty", None, None, AuthKind::ApiKey) {
        Err(Error::Unsupported(_)) => {}
        Err(e) => panic!("expected Unsupported, got {e:?}"),
        Ok(_) => panic!("expected Unsupported, got Ok"),
    }
}
