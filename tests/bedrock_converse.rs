//! Original fixtures for the Amazon Bedrock Converse protocol family.
//! Payloads and event-stream frames are hand-authored from the wire
//! contract, not copied from reference projects. The frame encoder
//! below builds AWS event-stream binary messages byte by byte so tests
//! can split the stream across arbitrary chunk boundaries.

mod common;

use std::sync::Arc;

use a4mp::*;
use common::{MockTransport, Response, body_json, collect};
use serde_json::json;

fn adapter(transport: MockTransport) -> BedrockAdapter {
    let target = ServiceTarget::new(
        "aws-bedrock",
        Protocol::BedrockConverse,
        url::Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com").unwrap(),
    );
    BedrockAdapter::new(
        target,
        AuthKind::Bearer,
        Arc::new(transport),
        Arc::new(StaticCredential(Credential::Bearer {
            token: Secret::new("bedrock-token"),
            expires_at: None,
        })),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// AWS event-stream frame encoder (test-only)
// ---------------------------------------------------------------------------

/// CRC32 (IEEE, reflected), bitwise — no table, no dependency.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// One event-stream frame: prelude, string headers, payload, both CRCs.
/// Only string-typed (7) headers are emitted — those are the only ones
/// Converse uses.
fn frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut header_bytes = Vec::new();
    for (name, value) in headers {
        header_bytes.push(name.len() as u8);
        header_bytes.extend_from_slice(name.as_bytes());
        header_bytes.push(7u8); // value type: string
        header_bytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
        header_bytes.extend_from_slice(value.as_bytes());
    }
    let total = 12 + header_bytes.len() + payload.len() + 4;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(total as u32).to_be_bytes());
    out.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
    let prelude_crc = crc32(&out);
    out.extend_from_slice(&prelude_crc.to_be_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(payload);
    let message_crc = crc32(&out);
    out.extend_from_slice(&message_crc.to_be_bytes());
    debug_assert_eq!(out.len(), total);
    out
}

/// A frame whose bytes are all ASCII, so the scripted transport's
/// `&'static str` chunks can carry them. The CRCs are real; a `pad`
/// string grows one byte per attempt, which changes `total_len` and
/// therefore both CRCs until every byte lands under 0x80.
/// Deterministic: same payload → same frame.
fn frame_ascii(headers: &[(&str, &str)], payload: &serde_json::Value) -> Vec<u8> {
    for pad in 0usize.. {
        let mut p = payload.clone();
        if pad > 0 {
            p["pad"] = json!("x".repeat(pad));
        }
        let body = serde_json::to_vec(&p).unwrap();
        let bytes = frame(headers, &body);
        if bytes.iter().all(|b| b.is_ascii()) {
            return bytes;
        }
        assert!(pad < 200_000, "frame never became ASCII-only");
    }
    unreachable!()
}

/// A `message-type=event` frame carrying a Converse stream event.
fn event(event_type: &str, payload: serde_json::Value) -> Vec<u8> {
    frame_ascii(
        &[
            (":message-type", "event"),
            (":event-type", event_type),
            (":content-type", "application/json"),
        ],
        &payload,
    )
}

/// A `message-type=exception` frame carrying a provider exception.
fn exception(exception_type: &str, message: &str) -> Vec<u8> {
    frame_ascii(
        &[
            (":message-type", "exception"),
            (":exception-type", exception_type),
            (":content-type", "application/json"),
        ],
        &json!({ "message": message }),
    )
}

/// Leak an ASCII byte buffer into the `&'static str` the mock transport
/// wants. Test-only; the bytes are proven ASCII by the frame builder.
fn leak_ascii(bytes: Vec<u8>) -> &'static str {
    assert!(bytes.iter().all(|b| b.is_ascii()));
    Box::leak(String::from_utf8(bytes).unwrap().into_boxed_str())
}

// ---------------------------------------------------------------------------
// Request encoding
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_request_maps_system_tools_toolconfig_and_inference() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
        }),
    ));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "anthropic.claude-test-1:0".into(),
        ..Default::default()
    };
    req.messages.push(Message::text(Role::System, "be brief"));
    req.messages.push(Message::text(Role::User, "hello"));
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentPart::Text { text: "checking".into() },
            ContentPart::ToolCall(ToolCall {
                id: "tu_1".into(),
                name: "lookup".into(),
                arguments: json!({"q": "x"}),
            }),
        ],
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::ToolResult(ToolResult {
            call_id: "tu_1".into(),
            content: vec![ContentPart::Text { text: "42".into() }],
            is_error: true,
        })],
    });
    req.tools.push(ToolDef {
        name: "lookup".into(),
        description: "search".into(),
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        strict: true,
    });
    req.tool_choice = Some(ToolChoice::Named("lookup".into()));
    req.sampling.temperature = Some(0.5);
    req.sampling.top_p = Some(0.75);
    req.sampling.max_output_tokens = Some(64);
    req.sampling.stop = vec!["END".into()];
    req.extensions.insert(
        "additionalModelRequestFields".into(),
        json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}),
    );
    req.extensions.insert("serviceTier".into(), json!("priority"));

    let out = a.chat(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(
        req.url.path(),
        "/model/anthropic.claude-test-1%3A0/converse"
    );
    assert_eq!(
        req.headers.get("authorization").unwrap(),
        "Bearer bedrock-token"
    );
    let b = body_json(req);
    // The model id travels in the path, never in the body.
    assert!(b.get("model").is_none());
    assert_eq!(b["system"], json!([{"text": "be brief"}]));
    assert_eq!(b["messages"][0], json!({"role": "user", "content": [{"text": "hello"}]}));
    assert_eq!(
        b["messages"][1]["content"][1],
        json!({"toolUse": {"toolUseId": "tu_1", "name": "lookup", "input": {"q": "x"}}})
    );
    assert_eq!(
        b["messages"][2]["content"][0],
        json!({"toolResult": {"toolUseId": "tu_1", "content": [{"text": "42"}], "status": "error"}})
    );
    assert_eq!(
        b["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["type"],
        "object"
    );
    assert_eq!(b["toolConfig"]["toolChoice"], json!({"tool": {"name": "lookup"}}));
    assert_eq!(b["inferenceConfig"]["maxTokens"], 64);
    assert_eq!(b["inferenceConfig"]["temperature"], 0.5);
    assert_eq!(b["inferenceConfig"]["topP"], 0.75);
    assert_eq!(b["inferenceConfig"]["stopSequences"], json!(["END"]));
    assert_eq!(
        b["additionalModelRequestFields"]["thinking"]["budget_tokens"],
        1024
    );
    assert_eq!(b["serviceTier"], "priority");
    assert!(matches!(out.stop, Some(StopReason::Complete)));
}

#[tokio::test]
async fn chat_request_encodes_inline_images_and_infers_format() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"output": {"message": {"content": [{"text": "seen"}]}}}),
    ));
    let a = adapter(t.clone());

    // A declared webp image plus an octet-stream body whose PNG magic
    // must be sniffed (host-resolved resources arrive untyped).
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::User,
        content: vec![
            ContentPart::Media {
                kind: MediaKind::Image,
                source: MediaSource::Inline {
                    media_type: "image/webp".into(),
                    data: bytes::Bytes::from_static(b"RIFFxxxxWEBPxx"),
                },
                detail: None,
            },
            ContentPart::Media {
                kind: MediaKind::Image,
                source: MediaSource::Inline {
                    media_type: "application/octet-stream".into(),
                    data: bytes::Bytes::from_static(b"\x89PNG\r\n\x1a\nrest"),
                },
                detail: None,
            },
        ],
    });
    a.chat(req, &RequestControl::default()).await.unwrap();
    let b = body_json(&t.recorded()[0]);
    assert_eq!(b["messages"][0]["content"][0]["image"]["format"], "webp");
    assert_eq!(b["messages"][0]["content"][1]["image"]["format"], "png");
    assert_eq!(
        b["messages"][0]["content"][1]["image"]["source"]["bytes"],
        "iVBORw0KGgpyZXN0"
    );
}

#[tokio::test]
async fn chat_request_rejects_remote_media_and_tool_choice_none() {
    let t = MockTransport::new();
    let a = adapter(t);
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::Media {
            kind: MediaKind::Image,
            source: MediaSource::Url("https://img.test/a.png".into()),
            detail: None,
        }],
    });
    let err = a.chat(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));

    let t = MockTransport::new();
    let a = adapter(t);
    let mut req = ChatRequest::default();
    req.tools.push(ToolDef {
        name: "lookup".into(),
        description: "search".into(),
        parameters: json!({"type": "object"}),
        strict: false,
    });
    req.tool_choice = Some(ToolChoice::None);
    let err = a.chat(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[tokio::test]
async fn assistant_reasoning_replays_its_opaque_block() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"output": {"message": {"content": [{"text": "ok"}]}}}),
    ));
    let a = adapter(t.clone());
    let block = json!({"reasoningContent": {"reasoningText": {"text": "why", "signature": "sig9"}}});
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentPart::Reasoning(ReasoningBlock {
            protocol: Protocol::BedrockConverse,
            opaque: block.clone(),
            summary: Some("why".into()),
        })],
    });
    a.chat(req, &RequestControl::default()).await.unwrap();
    let b = body_json(&t.recorded()[0]);
    assert_eq!(b["messages"][0]["content"][0], block);
}

// ---------------------------------------------------------------------------
// Response decoding
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_response_decodes_blocks_usage_stop_and_metrics() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "why", "signature": "sig_abc"}}},
                {"text": "answer"},
                {"toolUse": {"toolUseId": "tu_7", "name": "lookup", "input": {"q": "x"}}},
            ]}},
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 21,
                "outputTokens": 9,
                "totalTokens": 30,
                "cacheReadInputTokens": 5,
            },
            "metrics": {"latencyMs": 812},
            "trace": {"guardrail": {"action": "NONE"}},
        }),
    ));
    let a = adapter(t);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    assert!(matches!(out.stop, Some(StopReason::ToolUse)));
    match &out.parts[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.protocol, Protocol::BedrockConverse);
            assert_eq!(rb.summary.as_deref(), Some("why"));
            assert_eq!(
                rb.opaque["reasoningContent"]["reasoningText"]["signature"],
                "sig_abc"
            );
        }
        other => panic!("expected reasoning block, got {other:?}"),
    }
    match &out.parts[1] {
        ContentPart::Text { text } => assert_eq!(text, "answer"),
        other => panic!("expected text, got {other:?}"),
    }
    match &out.parts[2] {
        ContentPart::ToolCall(tc) => {
            assert_eq!(tc.id, "tu_7");
            assert_eq!(tc.name, "lookup");
            assert_eq!(tc.arguments, json!({"q": "x"}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(21));
    assert_eq!(usage.output_tokens, Some(9));
    assert_eq!(usage.extensions["totalTokens"], 30);
    assert_eq!(usage.extensions["cacheReadInputTokens"], 5);
    assert_eq!(out.extensions["metrics"]["latencyMs"], 812);
    assert_eq!(out.extensions["trace"]["guardrail"]["action"], "NONE");
}

#[test]
fn stop_reasons_cover_the_wire_enum() {
    let cases = [
        ("end_turn", StopReason::Complete),
        ("stop_sequence", StopReason::Complete),
        ("max_tokens", StopReason::Length),
        ("tool_use", StopReason::ToolUse),
        ("content_filtered", StopReason::Filtered),
        ("guardrail_intervened", StopReason::Filtered),
    ];
    for (wire, want) in cases {
        let body = serde_json::to_vec(&json!({
            "output": {"message": {"content": []}},
            "stopReason": wire,
        }))
        .unwrap();
        let out = a4mp::protocols::bedrock_converse::decode_response(&body).unwrap();
        assert_eq!(out.stop, Some(want), "stopReason {wire}");
    }
}

// ---------------------------------------------------------------------------
// Streaming: AWS event-stream frames
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_decodes_frames_split_mid_frame() {
    let mut bytes = Vec::new();
    bytes.extend(event("messageStart", json!({"role": "assistant"})));
    // Text block: `start` is absent, so the first delta opens the part.
    bytes.extend(event("contentBlockStart", json!({"contentBlockIndex": 0})));
    bytes.extend(event(
        "contentBlockDelta",
        json!({"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
    ));
    bytes.extend(event(
        "contentBlockDelta",
        json!({"contentBlockIndex": 0, "delta": {"text": "lo"}}),
    ));
    bytes.extend(event("contentBlockStop", json!({"contentBlockIndex": 0})));
    // Tool-use block: the start carries the full call identity.
    bytes.extend(event(
        "contentBlockStart",
        json!({"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tu_1", "name": "lookup"}}}),
    ));
    bytes.extend(event(
        "contentBlockDelta",
        json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"q\":"}}}),
    ));
    bytes.extend(event(
        "contentBlockDelta",
        json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "\"x\"}"}}}),
    ));
    bytes.extend(event("contentBlockStop", json!({"contentBlockIndex": 1})));
    // Reasoning block: text deltas, then an opaque signature delta.
    bytes.extend(event("contentBlockStart", json!({"contentBlockIndex": 2})));
    bytes.extend(event(
        "contentBlockDelta",
        json!({"contentBlockIndex": 2, "delta": {"reasoningContent": {"text": "plan "}}}),
    ));
    bytes.extend(event(
        "contentBlockDelta",
        json!({"contentBlockIndex": 2, "delta": {"reasoningContent": {"signature": "sig_1"}}}),
    ));
    bytes.extend(event("contentBlockStop", json!({"contentBlockIndex": 2})));
    bytes.extend(event("messageStop", json!({"stopReason": "tool_use"})));
    bytes.extend(event(
        "metadata",
        json!({
            "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
            "metrics": {"latencyMs": 55},
        }),
    ));

    // Split the whole stream at two mid-frame positions; every byte is
    // ASCII so arbitrary byte offsets stay valid UTF-8 slices.
    let s = leak_ascii(bytes);
    let a = s.len() / 3;
    let b = 2 * s.len() / 3;
    let t = MockTransport::new();
    t.push(Response::Sse(vec![&s[..a], &s[a..b], &s[b..]]));
    let adapter = adapter(t.clone());

    let mut req = ChatRequest {
        model: "us.anthropic.claude-test:0".into(),
        ..Default::default()
    };
    req.messages.push(Message::text(Role::User, "hi"));
    let stream = adapter
        .chat_stream(req, &RequestControl::default())
        .await
        .unwrap();
    let events = collect(stream).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(
        req.url.path(),
        "/model/us.anthropic.claude-test%3A0/converse-stream"
    );
    assert_eq!(
        req.headers.get("accept").unwrap(),
        "application/vnd.amazon.eventstream"
    );
    // Same body shape as `converse`; no `stream` flag on this wire.
    assert!(body_json(req).get("stream").is_none());

    let mut summary = Vec::new();
    let mut text = String::new();
    let mut args = String::new();
    let mut reasoning = String::new();
    for e in &events {
        match e {
            StreamEvent::PartStarted { index, kind } => {
                summary.push(format!("start:{index}:{kind:?}"))
            }
            StreamEvent::TextDelta { text: d, .. } => text.push_str(d),
            StreamEvent::ToolCallStarted { index, id, name } => {
                summary.push(format!("tool:{index}:{id}:{name}"))
            }
            StreamEvent::ToolCallDelta { arguments, .. } => args.push_str(arguments),
            StreamEvent::ReasoningDelta { text: d, .. } => reasoning.push_str(d),
            StreamEvent::PartFinished { index } => summary.push(format!("stop:{index}")),
            StreamEvent::Native { part_index, value } => {
                summary.push(format!("native:{part_index:?}:{}", value))
            }
            StreamEvent::Usage(u) => {
                assert_eq!(u.input_tokens, Some(11));
                assert_eq!(u.output_tokens, Some(7));
                assert_eq!(u.extensions["totalTokens"], 18);
                assert_eq!(u.extensions["metrics"]["latencyMs"], 55);
                summary.push("usage".into());
            }
            StreamEvent::Finished { stop } => {
                assert!(matches!(stop, Some(StopReason::ToolUse)));
                summary.push("finished".into());
            }
            other => panic!("unexpected event {other:?}"),
        }
    }
    assert_eq!(text, "Hello");
    assert_eq!(args, "{\"q\":\"x\"}");
    assert_eq!(reasoning, "plan ");
    assert_eq!(
        summary,
        [
            "start:0:Text",
            "stop:0",
            "tool:1:tu_1:lookup",
            "stop:1",
            "start:2:Reasoning",
            "native:Some(2):{\"reasoningContent\":{\"signature\":\"sig_1\"}}",
            "stop:2",
            "finished",
            "usage",
        ]
    );
}

#[tokio::test]
async fn chat_stream_exception_frame_becomes_provider_error() {
    let mut bytes = Vec::new();
    bytes.extend(event("messageStart", json!({"role": "assistant"})));
    bytes.extend(event(
        "contentBlockDelta",
        json!({"contentBlockIndex": 0, "delta": {"text": "partial"}}),
    ));
    bytes.extend(exception("ThrottlingException", "slow down"));

    let s = leak_ascii(bytes);
    let t = MockTransport::new();
    t.push(Response::Sse(vec![s]));
    let mut stream = adapter(t)
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    // Events decoded before the exception must reach the caller first.
    let mut text = String::new();
    let mut err = None;
    while let Some(item) = std::future::poll_fn(|cx| stream.as_mut().poll_next(cx)).await {
        match item {
            Ok(StreamEvent::TextDelta { text: d, .. }) => text.push_str(&d),
            Ok(_) => {}
            Err(e) => {
                err = Some(e);
                break;
            }
        }
    }
    assert_eq!(text, "partial");
    match err.unwrap() {
        Error::Provider {
            status,
            code,
            message,
            retry,
            ..
        } => {
            assert_eq!(status, None);
            assert_eq!(code.as_deref(), Some("ThrottlingException"));
            assert_eq!(message, "slow down");
            assert_eq!(retry, RetrySafety::Unsafe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_stream_unknown_event_is_native_and_truncation_errors() {
    // Unknown event types are preserved verbatim as Native events.
    let mut bytes = Vec::new();
    bytes.extend(event("brandNewEvent", json!({"x": 1})));
    bytes.extend(event("messageStop", json!({"stopReason": "end_turn"})));
    let s = leak_ascii(bytes);
    let t = MockTransport::new();
    t.push(Response::Sse(vec![s]));
    let events = collect(
        adapter(t)
            .chat_stream(ChatRequest::default(), &RequestControl::default())
            .await
            .unwrap(),
    )
    .await
    .unwrap();
    match &events[0] {
        StreamEvent::Native { part_index, value } => {
            assert_eq!(*part_index, None);
            // `pad` is a fixture artifact from ASCII-safe framing.
            assert_eq!(value["x"], json!(1));
        }
        other => panic!("expected native event, got {other:?}"),
    }
    assert!(matches!(
        events[1],
        StreamEvent::Finished {
            stop: Some(StopReason::Complete)
        }
    ));

    // A stream that ends mid-frame is a protocol violation, not silence.
    let mut bytes = event("messageStart", json!({"role": "assistant"}));
    bytes.truncate(bytes.len() / 2);
    let s = leak_ascii(bytes);
    let t = MockTransport::new();
    t.push(Response::Sse(vec![s]));
    let err = collect(
        adapter(t)
            .chat_stream(ChatRequest::default(), &RequestControl::default())
            .await
            .unwrap(),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::Protocol(_)));
}

// ---------------------------------------------------------------------------
// Errors, auth and endpoints
// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_success_status_maps_to_provider_error() {
    let t = MockTransport::new();
    t.push(Response::Json(
        400,
        json!({
            "__type": "com.amazonaws.bedrock#ValidationException:1.0",
            "message": "malformed request",
        }),
    ));
    let err = adapter(t)
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap_err();
    match err {
        Error::Provider {
            status,
            code,
            message,
            retry,
            ..
        } => {
            assert_eq!(status, Some(400));
            assert_eq!(code.as_deref(), Some("ValidationException"));
            assert_eq!(message, "malformed request");
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_open_drains_error_body() {
    let t = MockTransport::new();
    t.push(Response::Json(
        503,
        json!({"__type": "ServiceUnavailableException", "message": "try later"}),
    ));
    let err = match adapter(t)
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
    {
        Err(e) => e,
        Ok(_) => panic!("expected provider error"),
    };
    match err {
        Error::Provider {
            status,
            code,
            retry,
            ..
        } => {
            assert_eq!(status, Some(503));
            assert_eq!(code.as_deref(), Some("ServiceUnavailableException"));
            assert_eq!(retry, RetrySafety::Unsafe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn api_key_credential_also_sends_bearer() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"output": {"message": {"content": [{"text": "ok"}]}}}),
    ));
    let target = ServiceTarget::new(
        "aws-bedrock",
        Protocol::BedrockConverse,
        url::Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com").unwrap(),
    );
    let a = BedrockAdapter::new(
        target,
        AuthKind::ApiKey,
        Arc::new(t.clone()),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new("key-1")))),
    )
    .unwrap();
    a.chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    assert_eq!(
        t.recorded()[0].headers.get("authorization").unwrap(),
        "Bearer key-1"
    );
}

#[tokio::test]
async fn cloud_credential_signs_with_sigv4() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        serde_json::json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1}
        }),
    ));
    let mut target = ServiceTarget::new(
        "aws-bedrock",
        Protocol::BedrockConverse,
        url::Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com").unwrap(),
    );
    target.region = Some("us-east-1".into());
    let mut context = serde_json::Map::new();
    context.insert("access_key_id".into(), "AKIDEXAMPLE".into());
    let a = BedrockAdapter::new(
        target,
        AuthKind::CloudIdentity {
            service: "bedrock".into(),
        },
        Arc::new(t.clone()),
        Arc::new(StaticCredential(Credential::Cloud {
            token: Secret::new("sigv4-material"),
            expires_at: None,
            context,
        })),
    )
    .unwrap();
    a.chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    let recorded = t.recorded();
    assert_eq!(recorded.len(), 1);
    let auth = recorded[0]
        .headers
        .get(http::header::AUTHORIZATION)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(auth.starts_with("AWS4-HMAC-SHA256 "), "{auth}");
    assert!(auth.contains("Credential=AKIDEXAMPLE/"));
    assert!(auth.contains("/bedrock/aws4_request"));
    assert!(recorded[0].headers.contains_key("x-amz-date"));
    assert!(recorded[0].headers.contains_key("x-amz-content-sha256"));
}

#[tokio::test]
async fn cloud_credential_missing_keys_names_the_gap() {
    let t = MockTransport::new();
    let target = ServiceTarget::new(
        "aws-bedrock",
        Protocol::BedrockConverse,
        url::Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com").unwrap(),
    );
    let a = BedrockAdapter::new(
        target,
        AuthKind::CloudIdentity {
            service: "bedrock".into(),
        },
        Arc::new(t.clone()),
        Arc::new(StaticCredential(Credential::Cloud {
            token: Secret::new("sigv4-material"),
            expires_at: None,
            context: serde_json::Map::new(),
        })),
    )
    .unwrap();
    let err = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap_err();
    match err {
        Error::Credential(m) => assert!(m.contains("access_key_id"), "{m}"),
        other => panic!("expected credential error, got {other:?}"),
    }
    // The credential must fail before any request is sent.
    assert!(t.recorded().is_empty());
}

#[tokio::test]
async fn models_listing_is_unsupported() {
    let t = MockTransport::new();
    let err = adapter(t)
        .models(&RequestControl::default())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[tokio::test]
async fn foreign_reasoning_replay_is_rejected() {
    let t = MockTransport::new();
    let a = adapter(t);
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentPart::Reasoning(ReasoningBlock {
            protocol: Protocol::AnthropicMessages,
            opaque: json!({"type": "thinking", "thinking": "x"}),
            summary: None,
        })],
    });
    let err = a.chat(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}
