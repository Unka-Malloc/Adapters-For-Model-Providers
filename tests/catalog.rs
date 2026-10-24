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
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new("sk-test")))),
    );
    (client, transport)
}

#[test]
fn resolve_by_id_alias_and_case() {
    let catalog = Catalog::default();
    assert_eq!(catalog.resolve("openai").unwrap().name, "OpenAI");
    assert_eq!(catalog.resolve("OPENAI").unwrap().id.as_str(), "openai");
    assert_eq!(catalog.resolve("moonshot").unwrap().id.as_str(), "moonshotai");
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
    assert_eq!(
        adapter.target().protocol,
        Protocol::OpenAiChatCompletions
    );

    let models = adapter.models(&RequestControl::default()).await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "deepseek-chat");
    assert_eq!(models[0].owned_by.as_deref(), Some("deepseek"));

    let recorded = transport.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].method, http::Method::GET);
    assert_eq!(recorded[0].url.path(), "/models");
    assert_eq!(
        recorded[0].headers.get(http::header::AUTHORIZATION).unwrap(),
        "Bearer sk-test"
    );
}

#[tokio::test]
async fn protocol_preference_selects_target() {
    let (client, _) = client();
    let adapter = client
        .bind_provider("openai", Some(Protocol::OpenAiResponses), None, AuthKind::ApiKey)
        .unwrap();
    assert_eq!(adapter.target().protocol, Protocol::OpenAiResponses);
}

#[tokio::test]
async fn unregistered_protocol_reports_unsupported() {
    let (client, _) = client();
    // Anthropic's adapter is not registered in the builtin catalog yet;
    // binding must fail explicitly, not fall through to another protocol.
    let result = client.bind_provider("anthropic", None, None, AuthKind::ApiKey);
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
