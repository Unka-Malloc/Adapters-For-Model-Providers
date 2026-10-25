//! Contract-level tests for the shared API surface.

use a4mp::*;

mod common;
use common::fixture_token;

#[test]
fn provider_id_is_open_and_transparent() {
    let id = ProviderId::new("deepseek");
    assert_eq!(id.as_str(), "deepseek");
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, "\"deepseek\"");
    let back: ProviderId = serde_json::from_str(&json).unwrap();
    assert_eq!(back, id);
}

#[test]
fn secret_never_leaks_in_debug_or_display() {
    let material = fixture_token("secret");
    let s = Secret::new(material.clone());
    assert!(!format!("{s:?}").contains(&material));
    assert!(!format!("{s}").contains(&material));
    assert_eq!(s.expose(), material);
}

#[test]
fn cloud_credential_debug_redacts_secret_context_at_any_depth() {
    let access_id = fixture_token("access-id");
    let session = fixture_token("session");
    let camel = fixture_token("camel");
    let private = fixture_token("private");
    let nested = fixture_token("nested");
    let token = fixture_token("token");
    let mut context = serde_json::Map::new();
    context.insert("access_key_id".into(), serde_json::json!(access_id));
    context.insert("session_token".into(), serde_json::json!(session));
    context.insert("region".into(), serde_json::json!("us-east-1"));
    context.insert("service".into(), serde_json::json!("bedrock"));
    context.insert("session_name".into(), serde_json::json!("nightly"));
    context.insert("secretAccessKey".into(), serde_json::json!(camel));
    context.insert("private_key".into(), serde_json::json!(private));
    context.insert(
        "sts".into(),
        serde_json::json!({"client_secret": nested, "label": "visible"}),
    );
    let cred = Credential::Cloud {
        token: Secret::new(token.clone()),
        expires_at: None,
        context,
    };

    let debug = format!("{cred:?}");
    for leaked in [&session, &access_id, &private, &nested, &camel, &token] {
        assert!(
            !debug.contains(leaked),
            "credential Debug leaked {leaked:?}: {debug}"
        );
    }
    assert!(debug.contains("<redacted>"), "{debug}");
    // Non-secret mechanism metadata remains inspectable for diagnostics.
    assert!(debug.contains("us-east-1"), "{debug}");
    assert!(debug.contains("bedrock"), "{debug}");
    assert!(debug.contains("nightly"), "{debug}");
    assert!(debug.contains("visible"), "{debug}");
}

#[test]
fn job_handle_roundtrips_and_carries_no_credentials() {
    let h = JobHandle {
        provider: "openai".into(),
        protocol: Protocol::OpenAiResponses,
        id: "job_123".into(),
        state: serde_json::json!({"cursor": "abc"}),
    };
    let json = serde_json::to_string(&h).unwrap();
    let back: JobHandle = serde_json::from_str(&json).unwrap();
    assert_eq!(back.id, "job_123");
    assert_eq!(back.state["cursor"], "abc");
}

#[test]
fn retry_safety_classification() {
    // Connect refused: provider never saw the request — safe.
    let e = Error::Transport(TransportError::Connect("refused".into()));
    assert_eq!(e.retry_safety(), RetrySafety::Safe);
    // Mid-body disconnect: upstream acceptance uncertain — unsafe.
    let e = Error::Transport(TransportError::Disconnected("closed".into()));
    assert_eq!(e.retry_safety(), RetrySafety::Unsafe);
    // Cancellation may race a dispatched request — unsafe to replay.
    let e = Error::Transport(TransportError::Cancelled);
    assert_eq!(e.retry_safety(), RetrySafety::Unsafe);
    // Provider rejection marked unsafe by the adapter stays unsafe.
    let e = Error::Provider {
        status: Some(500),
        code: None,
        message: "internal".into(),
        retry: RetrySafety::Unsafe,
        raw: None,
    };
    assert_eq!(e.retry_safety(), RetrySafety::Unsafe);
    // Local pre-send failures are safe to repeat.
    let e = Error::Unsupported("embed".into());
    assert_eq!(e.retry_safety(), RetrySafety::Safe);
    let e = Error::Conversion {
        report: Box::new(ConversionReport {
            source: Protocol::OpenAiChatCompletions,
            target: Protocol::AnthropicMessages,
            losses: vec![],
        }),
    };
    assert_eq!(e.retry_safety(), RetrySafety::Safe);
    let e = Error::Credential("entropy unavailable".into());
    assert_eq!(e.retry_safety(), RetrySafety::Safe);
    // Local protocol validation happens before dispatch and cannot bill.
    let e = Error::Protocol("request missing model".into());
    assert_eq!(e.retry_safety(), RetrySafety::Safe);
}

#[test]
fn post_dispatch_contract_errors_are_unsafe_to_retry() {
    // Caller cancellation can occur after the request reached the provider.
    assert_eq!(Error::Cancelled.retry_safety(), RetrySafety::Unsafe);
    // An upstream payload violation is observed after dispatch, so work may
    // already have been accepted upstream.
    let e = Error::UpstreamProtocol("response has no choices".into());
    assert_eq!(e.retry_safety(), RetrySafety::Unsafe);
}

#[test]
fn local_wire_decode_failure_is_a_safe_protocol_error() {
    // A real local, pre-dispatch producer: the request decoder rejects a
    // malformed body before any provider request exists.
    let err = a4mp::wire::decode_chat_request(
        a4mp::wire::WireDialect::OpenAiChatCompletions,
        br#"{"messages":[]}"#,
    )
    .unwrap_err();
    assert!(matches!(err, Error::Protocol(_)), "{err:?}");
    assert_eq!(err.retry_safety(), RetrySafety::Safe);
}

#[test]
fn control_carries_cancellation_only() {
    let ctrl = RequestControl::default();
    assert!(!ctrl.cancel.is_cancelled());
    ctrl.cancel.cancel();
    assert!(ctrl.cancel.is_cancelled());
}

#[tokio::test]
async fn adapter_defaults_report_unsupported() {
    struct Dummy(ServiceTarget);
    impl Adapter for Dummy {
        fn target(&self) -> &ServiceTarget {
            &self.0
        }
    }
    let target = ServiceTarget::new(
        "test-provider",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://example.test").unwrap(),
    );
    let a = Dummy(target);
    let err = a.chat(ChatRequest::default(), &RequestControl::default()).await;
    assert!(matches!(err, Err(Error::Unsupported(_))));
    let err = a
        .open_realtime(RealtimeConfig::default(), &RequestControl::default())
        .await;
    assert!(matches!(err, Err(Error::Unsupported(_))));
}
