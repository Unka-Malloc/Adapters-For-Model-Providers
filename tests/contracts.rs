//! Contract-level tests for the shared API surface.

use a4mp::*;

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
    let s = Secret::new("sk-live-material");
    assert!(!format!("{s:?}").contains("sk-live-material"));
    assert!(!format!("{s}").contains("sk-live-material"));
    assert_eq!(s.expose(), "sk-live-material");
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
}

#[test]
fn post_dispatch_contract_errors_are_unsafe_to_retry() {
    // Caller cancellation can occur after the request reached the provider.
    assert_eq!(Error::Cancelled.retry_safety(), RetrySafety::Unsafe);
    // Protocol violations are discovered after dispatch, so work may have
    // already been accepted upstream.
    let e = Error::Protocol("response has no choices".into());
    assert_eq!(e.retry_safety(), RetrySafety::Unsafe);
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
