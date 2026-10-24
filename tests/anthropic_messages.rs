//! Original fixtures for the Anthropic Messages protocol family.
//! Payloads are hand-authored from the wire contract, not copied
//! from reference projects.

mod common;

use std::sync::Arc;

use a4mp::protocols::anthropic_messages as am;
use a4mp::*;
use common::{MockTransport, Response, body_json, collect};
use serde_json::json;

fn adapter(transport: MockTransport) -> AnthropicAdapter {
    let target = ServiceTarget::new(
        "anthropic",
        Protocol::AnthropicMessages,
        url::Url::parse("https://api.anthropic.com").unwrap(),
    );
    AnthropicAdapter::new(
        target,
        AuthKind::ApiKey,
        Arc::new(transport),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new("sk-ant-test")))),
    )
    .unwrap()
}

fn ok_body() -> serde_json::Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test-1",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 9, "output_tokens": 3},
    })
}

#[tokio::test]
async fn chat_request_maps_system_media_thinking_tools_and_extensions() {
    let t = MockTransport::new();
    t.push(Response::Json(200, ok_body()));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "claude-test-1".into(),
        ..Default::default()
    };
    req.messages.push(Message::text(Role::System, "be brief"));
    req.messages.push(Message {
        role: Role::User,
        content: vec![
            ContentPart::Text { text: "hello".into() },
            ContentPart::Media {
                kind: MediaKind::Image,
                source: MediaSource::Url("https://img.test/a.png".into()),
                detail: None,
            },
            ContentPart::Media {
                kind: MediaKind::Image,
                source: MediaSource::Inline {
                    media_type: "image/png".into(),
                    data: bytes::Bytes::from_static(b"\x89PNG"),
                },
                detail: None,
            },
        ],
    });
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::AnthropicMessages,
                opaque: json!({
                    "type": "thinking",
                    "thinking": "prior reasoning",
                    "signature": "sig-1",
                }),
                summary: Some("prior reasoning".into()),
            }),
            ContentPart::Text { text: "checking".into() },
            ContentPart::ToolCall(ToolCall {
                id: "toolu_1".into(),
                name: "lookup".into(),
                arguments: json!({"q": "x"}),
            }),
        ],
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![
            ContentPart::ToolResult(ToolResult {
                call_id: "toolu_1".into(),
                content: vec![ContentPart::Text { text: "42".into() }],
                is_error: true,
            }),
            ContentPart::Text { text: "thanks".into() },
        ],
    });
    req.tools.push(ToolDef {
        name: "lookup".into(),
        description: "search".into(),
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        strict: true,
    });
    req.tool_choice = Some(ToolChoice::Named("lookup".into()));
    req.sampling.temperature = Some(0.5);
    req.sampling.max_output_tokens = Some(512);
    req.sampling.stop = vec!["END".into()];
    req.extensions
        .insert("metadata".into(), json!({"user_id": "u-9"}));
    req.extensions
        .insert("thinking".into(), json!({"type": "enabled", "budget_tokens": 1024}));

    let out = a.chat(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v1/messages");
    assert_eq!(req.headers.get("x-api-key").unwrap(), "sk-ant-test");
    assert_eq!(
        req.headers.get("anthropic-version").unwrap(),
        "2023-06-01"
    );
    assert!(req.headers.get("authorization").is_none());

    let b = body_json(req);
    assert_eq!(b["model"], "claude-test-1");
    assert_eq!(b["max_tokens"], 512);
    assert_eq!(b["system"], "be brief");
    assert_eq!(
        b["messages"][0],
        json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "source": {"type": "url", "url": "https://img.test/a.png"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw=="}},
            ],
        })
    );
    assert_eq!(
        b["messages"][1],
        json!({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "prior reasoning", "signature": "sig-1"},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
            ],
        })
    );
    assert_eq!(
        b["messages"][2],
        json!({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42", "is_error": true},
                {"type": "text", "text": "thanks"},
            ],
        })
    );
    assert_eq!(
        b["tools"][0],
        json!({
            "name": "lookup",
            "description": "search",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        })
    );
    assert_eq!(b["tool_choice"], json!({"type": "tool", "name": "lookup"}));
    assert_eq!(b["temperature"], 0.5);
    assert_eq!(b["stop_sequences"], json!(["END"]));
    assert_eq!(b["metadata"]["user_id"], "u-9");
    assert_eq!(b["thinking"]["type"], "enabled");

    assert!(matches!(out.stop, Some(StopReason::Complete)));
    assert_eq!(out.usage.unwrap().input_tokens, Some(9));
    match &out.parts[0] {
        ContentPart::Text { text } => assert_eq!(text, "hi"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_request_defaults_max_tokens_and_joins_system_text() {
    let t = MockTransport::new();
    t.push(Response::Json(200, ok_body()));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "claude-test-1".into(),
        ..Default::default()
    };
    req.messages.push(Message::text(Role::System, "rule one"));
    req.messages.push(Message::text(Role::System, "rule two"));
    req.messages.push(Message::text(Role::User, "hi"));

    a.chat(req, &RequestControl::default()).await.unwrap();

    let b = body_json(&t.recorded()[0]);
    // max_tokens is mandatory on this wire even without a caller bound.
    assert_eq!(b["max_tokens"], 4096);
    assert_eq!(b["system"], "rule one\nrule two");
    assert_eq!(b["messages"].as_array().unwrap().len(), 1);
}

#[test]
fn tool_choice_maps_to_wire_types() {
    for (choice, expect) in [
        (ToolChoice::Auto, json!({"type": "auto"})),
        (ToolChoice::None, json!({"type": "none"})),
        (ToolChoice::Required, json!({"type": "any"})),
        (ToolChoice::Named("f".into()), json!({"type": "tool", "name": "f"})),
    ] {
        let req = ChatRequest {
            tool_choice: Some(choice),
            ..Default::default()
        };
        let body = am::build_request(&req).unwrap();
        assert_eq!(body["tool_choice"], expect);
    }
}

#[tokio::test]
async fn chat_response_decodes_blocks_usage_and_extensions() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-test-1",
            "content": [
                {"type": "thinking", "thinking": "let me reason", "signature": "sig-abc"},
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "toolu_2", "name": "calc", "input": {"a": 1}},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 30,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 7,
            },
            "extra_field": "kept",
        }),
    ));
    let a = adapter(t);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    assert!(matches!(out.stop, Some(StopReason::ToolUse)));
    assert_eq!(out.parts.len(), 3);
    match &out.parts[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.protocol, Protocol::AnthropicMessages);
            assert_eq!(rb.summary.as_deref(), Some("let me reason"));
            // The whole block stays replayable as opaque state.
            assert_eq!(rb.opaque["signature"], "sig-abc");
            assert_eq!(rb.opaque["type"], "thinking");
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
    match &out.parts[1] {
        ContentPart::Text { text } => assert_eq!(text, "answer"),
        other => panic!("expected text, got {other:?}"),
    }
    match &out.parts[2] {
        ContentPart::ToolCall(tc) => {
            assert_eq!(tc.id, "toolu_2");
            assert_eq!(tc.name, "calc");
            assert_eq!(tc.arguments, json!({"a": 1}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(12));
    assert_eq!(usage.output_tokens, Some(30));
    assert_eq!(usage.extensions["cache_creation_input_tokens"], 5);
    assert_eq!(usage.extensions["cache_read_input_tokens"], 7);
    assert_eq!(out.extensions["extra_field"], "kept");
}

#[test]
fn chat_response_maps_every_stop_reason() {
    for (wire, expect) in [
        ("end_turn", StopReason::Complete),
        ("stop_sequence", StopReason::Complete),
        ("max_tokens", StopReason::Length),
        ("tool_use", StopReason::ToolUse),
        ("refusal", StopReason::Filtered),
        ("pause_turn", StopReason::Paused),
        ("mystery", StopReason::Other("mystery".into())),
    ] {
        let body = json!({
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": wire,
        })
        .to_string();
        let out = am::decode_response(body.as_bytes()).unwrap();
        assert_eq!(out.stop, Some(expect), "wire reason {wire}");
    }
}

#[tokio::test]
async fn chat_stream_decodes_blocks_deltas_signatures_and_usage() {
    let frames = [
        (
            "message_start",
            json!({"type": "message_start", "message": {
                "id": "msg_s1", "type": "message", "role": "assistant",
                "model": "claude-test-1", "content": [], "stop_reason": null,
                "usage": {"input_tokens": 15, "output_tokens": 2, "cache_read_input_tokens": 6},
            }}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": "first "}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": "principles"}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "signature_delta", "signature": "sig-xyz"}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 1,
                   "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "text_delta", "text": "Hello"}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "text_delta", "text": ", world"}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 1}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 2,
                   "content_block": {"type": "tool_use", "id": "toolu_9", "name": "calc", "input": {}}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 2,
                   "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 2,
                   "delta": {"type": "input_json_delta", "partial_json": "1}"}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 2}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta",
                   "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                   "usage": {"output_tokens": 64}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ];
    let sse: String = frames
        .iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
        .collect();
    // Split mid-frame to prove incremental decoding.
    let at = sse.len() / 3;
    let sse: &'static str = Box::leak(sse.into_boxed_str());

    let t = MockTransport::new();
    t.push(Response::Sse(vec![&sse[..at], &sse[at..]]));
    let a = adapter(t.clone());
    let stream = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    let events = collect(stream).await.unwrap();

    let b = body_json(&t.recorded()[0]);
    assert_eq!(b["stream"], true);
    assert_eq!(b["max_tokens"], 4096);

    let mut log = Vec::new();
    for e in &events {
        match e {
            StreamEvent::PartStarted { index, kind } => {
                log.push(format!("start:{kind:?}:{index}"))
            }
            StreamEvent::TextDelta { index, text } => log.push(format!("text:{index}:{text}")),
            StreamEvent::ToolCallStarted { index, id, name } => {
                log.push(format!("tool:{index}:{id}:{name}"))
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                log.push(format!("args:{index}:{arguments}"))
            }
            StreamEvent::ReasoningDelta { index, text } => {
                log.push(format!("reason:{index}:{text}"))
            }
            StreamEvent::PartFinished { index } => log.push(format!("finish:{index}")),
            StreamEvent::Native { part_index, value } => {
                log.push(format!("native:{part_index:?}:{}", value["signature"]))
            }
            StreamEvent::Usage(u) => log.push(format!(
                "usage:{:?}/{:?}/{}",
                u.input_tokens,
                u.output_tokens,
                u.extensions.get("cache_read_input_tokens").cloned().unwrap_or(json!(0))
            )),
            StreamEvent::Finished { stop } => log.push(format!("stop:{stop:?}")),
            other => log.push(format!("other:{other:?}")),
        }
    }
    assert_eq!(
        log,
        [
            "usage:Some(15)/Some(2)/6",
            "start:Reasoning:0",
            "reason:0:first ",
            "reason:0:principles",
            "native:Some(0):\"sig-xyz\"",
            "finish:0",
            "start:Text:1",
            "text:1:Hello",
            "text:1:, world",
            "finish:1",
            "tool:2:toolu_9:calc",
            "args:2:{\"a\":",
            "args:2:1}",
            "finish:2",
            "stop:Some(ToolUse)",
            "usage:None/Some(64)/0",
        ]
    );
}

#[tokio::test]
async fn chat_stream_surfaces_mid_stream_error_event() {
    let sse: &'static str = Box::leak(
        format!(
            "event: error\ndata: {}\n\n",
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
        )
        .into_boxed_str(),
    );
    let t = MockTransport::new();
    t.push(Response::Sse(vec![sse]));
    let a = adapter(t);
    let stream = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    let err = collect(stream).await.unwrap_err();
    match err {
        Error::Provider {
            status,
            code,
            message,
            retry,
            ..
        } => {
            assert_eq!(status, None);
            assert_eq!(code.as_deref(), Some("overloaded_error"));
            assert_eq!(message, "Overloaded");
            // No status attached mid-stream: acceptance is uncertain.
            assert_eq!(retry, RetrySafety::Unsafe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_maps_error_bodies_and_retry_safety() {
    let t = MockTransport::new();
    t.push(Response::Json(
        429,
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}),
    ));
    t.push(Response::Text(529, "service overloaded"));
    t.push(Response::Json(
        429,
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}),
    ));
    let a = adapter(t);

    let err = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap_err();
    match err {
        Error::Provider {
            status, code, retry, ..
        } => {
            assert_eq!(status, Some(429));
            assert_eq!(code.as_deref(), Some("rate_limit_error"));
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }

    let err = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap_err();
    match err {
        Error::Provider {
            status,
            message,
            retry,
            ..
        } => {
            assert_eq!(status, Some(529));
            assert_eq!(message, "service overloaded");
            assert_eq!(retry, RetrySafety::Unsafe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }

    // The streaming path drains the error body before failing.
    let err = match a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
    {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    };
    match err {
        Error::Provider {
            status, code, retry, ..
        } => {
            assert_eq!(status, Some(429));
            assert_eq!(code.as_deref(), Some("rate_limit_error"));
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn anthropic_rejects_foreign_reasoning_replay() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentPart::Reasoning(ReasoningBlock {
            protocol: Protocol::OpenAiChatCompletions,
            opaque: json!({"reasoning_content": "foreign"}),
            summary: None,
        })],
    });
    let err = a.chat(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    // The request is rejected before hitting the wire.
    assert!(t.recorded().is_empty());
}

#[test]
fn build_request_rejects_unrepresentable_parts() {
    // A role outside user/assistant/system has no wire form.
    let mut req = ChatRequest::default();
    req.messages.push(Message::text(Role::Other("tool".into()), "x"));
    assert!(matches!(
        am::build_request(&req),
        Err(Error::Unsupported(_))
    ));

    // Image detail hints have no Anthropic counterpart.
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::Media {
            kind: MediaKind::Image,
            source: MediaSource::Url("https://img.test/x.png".into()),
            detail: Some("high".into()),
        }],
    });
    assert!(matches!(
        am::build_request(&req),
        Err(Error::Unsupported(_))
    ));

    // Reasoning replay is assistant-only.
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::Reasoning(ReasoningBlock {
            protocol: Protocol::AnthropicMessages,
            opaque: json!({"type": "thinking", "thinking": "t", "signature": "s"}),
            summary: None,
        })],
    });
    assert!(matches!(
        am::build_request(&req),
        Err(Error::Unsupported(_))
    ));

    // Non-object opaque state cannot replay as a thinking block.
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentPart::Reasoning(ReasoningBlock {
            protocol: Protocol::AnthropicMessages,
            opaque: json!("sig-only"),
            summary: None,
        })],
    });
    assert!(matches!(
        am::build_request(&req),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn adapter_rejects_non_anthropic_protocol() {
    let target = ServiceTarget::new(
        "not-anthropic",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://api.vendor.test").unwrap(),
    );
    let result = AnthropicAdapter::new(
        target,
        AuthKind::ApiKey,
        Arc::new(MockTransport::new()),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new("k")))),
    );
    assert!(matches!(result, Err(Error::Unsupported(_))));
}

#[tokio::test]
async fn bearer_credential_uses_authorization_header() {
    let t = MockTransport::new();
    t.push(Response::Json(200, ok_body()));
    let target = ServiceTarget::new(
        "anthropic",
        Protocol::AnthropicMessages,
        url::Url::parse("https://api.anthropic.com").unwrap(),
    );
    let a = AnthropicAdapter::new(
        target,
        AuthKind::Bearer,
        Arc::new(t.clone()),
        Arc::new(StaticCredential(Credential::Bearer {
            token: Secret::new("oauth-tok"),
            expires_at: None,
        })),
    )
    .unwrap();

    a.chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    let req = &t.recorded()[0];
    assert_eq!(
        req.headers.get("authorization").unwrap(),
        "Bearer oauth-tok"
    );
    assert!(req.headers.get("x-api-key").is_none());
    assert_eq!(
        req.headers.get("anthropic-version").unwrap(),
        "2023-06-01"
    );
}

struct StubEffects;

#[async_trait::async_trait]
impl EffectHost for StubEffects {
    async fn read_resource(
        &self,
        resource: &mut ResourceRef,
        _ctrl: &RequestControl,
    ) -> Result<bytes::Bytes, Error> {
        assert_eq!(resource.id, "res-1");
        resource.media_type = Some("image/png".into());
        Ok(bytes::Bytes::from_static(b"\x89PNG"))
    }
}

#[tokio::test]
async fn media_resource_resolves_through_effects() {
    let t = MockTransport::new();
    t.push(Response::Json(200, ok_body()));
    let a = adapter(t.clone()).with_effects(Arc::new(StubEffects));

    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::Media {
            kind: MediaKind::Image,
            source: MediaSource::Resource("res-1".into()),
            detail: None,
        }],
    });
    a.chat(req, &RequestControl::default()).await.unwrap();

    let b = body_json(&t.recorded()[0]);
    assert_eq!(
        b["messages"][0]["content"][0],
        json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw==",
            },
        })
    );
}
