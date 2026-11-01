//! Original fixtures for the Chat Completions protocol family.
//! Payloads are hand-authored from the wire contract, not copied
//! from reference projects.

mod common;

use std::sync::Arc;

use a4mp::*;
use common::{MockTransport, Response, body_json, collect};
use serde_json::json;

fn adapter(transport: MockTransport) -> OpenAiAdapter {
    let target = ServiceTarget::new(
        "test-vendor",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://api.vendor.test/v1").unwrap(),
    );
    OpenAiAdapter::new(
        target,
        AuthKind::Bearer,
        Arc::new(transport),
        Arc::new(StaticCredential(Credential::Bearer {
            token: Secret::new(common::run_token("bearer")),
            expires_at: None,
        })),
    )
    .unwrap()
}

#[tokio::test]
async fn chat_request_maps_messages_tools_sampling_and_extensions() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "choices": [{"message": {"role":"assistant","content":"hi"}, "finish_reason":"stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        }),
    ));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "vendor-model".into(),
        ..Default::default()
    };
    req.messages.push(Message::text(Role::System, "be brief"));
    req.messages.push(Message::text(Role::User, "hello"));
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentPart::Text {
                text: "calling".into(),
            },
            ContentPart::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "lookup".into(),
                arguments: json!({"q": "x"}),
            }),
        ],
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::ToolResult(ToolResult {
            call_id: "call_1".into(),
            content: vec![ContentPart::Text { text: "42".into() }],
            is_error: false,
        })],
    });
    req.tools.push(ToolDef {
        name: "lookup".into(),
        description: "search".into(),
        parameters: json!({"type":"object"}),
        strict: true,
    });
    req.tool_choice = Some(ToolChoice::Named("lookup".into()));
    req.parallel_tool_calls = Some(false);
    req.sampling.temperature = Some(0.5);
    req.sampling.max_output_tokens = Some(64);
    req.extensions.insert("service_tier".into(), json!("flex"));

    let out = a.chat(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v1/chat/completions");
    assert_eq!(
        req.headers.get("authorization").unwrap(),
        format!("Bearer {}", common::run_token("bearer")).as_str()
    );
    let b = body_json(req);
    assert_eq!(b["model"], "vendor-model");
    assert_eq!(
        b["messages"][0],
        json!({"role":"system","content":"be brief"})
    );
    assert_eq!(b["messages"][1], json!({"role":"user","content":"hello"}));
    assert_eq!(
        b["messages"][2]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(
        b["messages"][3],
        json!({"role":"tool","tool_call_id":"call_1","content":"42"})
    );
    assert_eq!(b["tool_choice"]["function"]["name"], "lookup");
    assert_eq!(b["parallel_tool_calls"], false);
    assert_eq!(b["temperature"], 0.5);
    assert_eq!(b["max_completion_tokens"], 64);
    assert_eq!(b["service_tier"], "flex");

    assert!(matches!(out.stop, Some(StopReason::Complete)));
    assert_eq!(out.usage.unwrap().input_tokens, Some(3));
    match &out.parts[0] {
        ContentPart::Text { text } => assert_eq!(text, "hi"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_request_encodes_inline_file_as_file_data() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"choices": [{"message": {"role":"assistant","content":""}, "finish_reason":"stop"}]}),
    ));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "m".into(),
        ..Default::default()
    };
    req.messages.push(Message {
        role: Role::User,
        content: vec![
            ContentPart::Media {
                kind: MediaKind::File,
                source: MediaSource::Inline {
                    media_type: "application/pdf".into(),
                    data: bytes::Bytes::from_static(b"hi"),
                },
                detail: Some("brief.pdf".into()),
            },
            ContentPart::Media {
                kind: MediaKind::File,
                source: MediaSource::Inline {
                    media_type: "text/csv".into(),
                    data: bytes::Bytes::from_static(b"a,b"),
                },
                detail: None,
            },
        ],
    });
    a.chat(req, &RequestControl::default()).await.unwrap();

    let b = body_json(&t.recorded()[0]);
    // Inline bytes become `file_data`; `file_id` stays reserved for
    // provider file ids, which no canonical source can mint.
    assert_eq!(
        b["messages"][0]["content"][0]["file"],
        json!({"file_data": "data:application/pdf;base64,aGk=", "filename": "brief.pdf"})
    );
    assert_eq!(
        b["messages"][0]["content"][1]["file"],
        json!({"file_data": "data:text/csv;base64,YSxi", "filename": "file.csv"})
    );
    assert!(
        b["messages"][0]["content"][0]["file"]
            .get("file_id")
            .is_none()
    );
}

#[tokio::test]
async fn chat_request_rejects_unresolvable_file_sources() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    for source in [
        MediaSource::Url("https://x.test/a.pdf".into()),
        MediaSource::Resource("file-unresolved".into()),
    ] {
        let mut req = ChatRequest {
            model: "m".into(),
            ..Default::default()
        };
        req.messages.push(Message {
            role: Role::User,
            content: vec![ContentPart::Media {
                kind: MediaKind::File,
                source,
                detail: None,
            }],
        });
        let err = a.chat(req, &RequestControl::default()).await.unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    }
    assert!(t.recorded().is_empty(), "no request may be dispatched");
}

#[tokio::test]
async fn chat_response_preserves_reasoning_and_tool_calls() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_content": "thinking step",
                    "tool_calls": [{
                        "id": "call_9", "type": "function",
                        "function": {"name": "f", "arguments": "{\"x\":1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
    ));
    let a = adapter(t);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    assert!(matches!(out.stop, Some(StopReason::ToolUse)));
    let mut kinds = Vec::new();
    for p in &out.parts {
        kinds.push(match p {
            ContentPart::Reasoning(b) => {
                format!("reasoning:{}", b.summary.as_deref().unwrap_or(""))
            }
            ContentPart::ToolCall(tc) => format!("tool:{}:{}", tc.name, tc.id),
            ContentPart::Text { .. } => "text".into(),
            _ => "other".into(),
        });
    }
    assert_eq!(kinds, ["reasoning:thinking step", "text", "tool:f:call_9"]);
}

#[tokio::test]
async fn chat_stream_decodes_reasoning_text_toolcalls_and_usage() {
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step \"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"one\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\"\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":\\\"x\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9,\"total_tokens\":14}}\n\n",
        "data: [DONE]\n\n"
    );
    let t = MockTransport::new();
    // Split mid-frame to prove incremental decoding.
    let at = sse.len() / 3;
    t.push(Response::Sse(vec![&sse[..at], &sse[at..]]));
    let a = adapter(t.clone());
    let stream = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    let events = collect(stream).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(body_json(req)["stream"], true);
    assert_eq!(body_json(req)["stream_options"]["include_usage"], true);

    let mut reasoning = String::new();
    let mut text = String::new();
    let mut args = String::new();
    let mut started = Vec::new();
    let mut ended = Vec::new();
    let mut usage = false;
    let mut finish = false;
    for e in &events {
        match e {
            StreamEvent::PartStarted { kind, .. } => started.push(*kind),
            StreamEvent::PartFinished { index } => ended.push(*index),
            StreamEvent::ReasoningDelta { text: d, .. } => reasoning.push_str(d),
            StreamEvent::TextDelta { text: d, .. } => text.push_str(d),
            StreamEvent::ToolCallStarted { id, name, .. } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "lookup");
            }
            StreamEvent::ToolCallDelta { arguments, .. } => args.push_str(arguments),
            StreamEvent::Usage(u) => usage = u.output_tokens == Some(9),
            StreamEvent::Finished { stop } => finish = matches!(stop, Some(StopReason::ToolUse)),
            _ => {}
        }
    }
    assert_eq!(started, [PartKind::Reasoning, PartKind::Text]);
    assert_eq!(reasoning, "step one");
    assert_eq!(text, "Hello");
    assert_eq!(args, "{\"q\":\"x\"}");
    // Every open part — reasoning, text and the tool call — is closed
    // before Finished; per-part consumers must not leak state.
    assert_eq!(ended, [0, 1, 2]);
    assert!(usage);
    assert!(finish);
}

#[tokio::test]
async fn chat_stream_surfaces_provider_error_body() {
    let t = MockTransport::new();
    t.push(Response::Json(
        429,
        json!({"error": {"message": "rate limited", "type": "rate_limit_error"}}),
    ));
    let a = adapter(t);
    let err = match a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
    {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    };
    match err {
        Error::Provider {
            status,
            code,
            retry,
            ..
        } => {
            assert_eq!(status, Some(429));
            assert_eq!(code.as_deref(), Some("rate_limit_error"));
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_completions_rejects_foreign_reasoning_replay() {
    let t = MockTransport::new();
    t.push(Response::Json(200, json!({})));
    let a = adapter(t);
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentPart::Reasoning(ReasoningBlock {
            protocol: Protocol::AnthropicMessages,
            opaque: json!({"signature": "abc"}),
            summary: None,
        })],
    });
    let err = a.chat(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[test]
fn sse_decoder_handles_crlf_and_split_frames() {
    let mut dec = SseDecoder::new();
    // CRLF frame boundary followed by a partial frame in one chunk.
    let ev = dec.feed(b"data: a\r\n\r\nda").unwrap().pop().unwrap();
    assert_eq!(ev.data, "a");
    let ev = dec.feed(b"ta: b\n\n").unwrap().pop().unwrap();
    assert_eq!(ev.data, "b");
    // Comment/heartbeat frames produce nothing.
    assert!(dec.feed(b": keep-alive\n\n").unwrap().is_empty());
    // Multi-line data joins with newline.
    let ev = dec.feed(b"data: x\ndata: y\n\n").unwrap().pop().unwrap();
    assert_eq!(ev.data, "x\ny");
    // A trailing unterminated frame flushes on finish.
    assert!(dec.feed(b"data: tail").unwrap().is_empty());
    let ev = dec.finish().pop().unwrap();
    assert_eq!(ev.data, "tail");
}

#[test]
fn sse_decoder_bounds_frame_memory_and_resumes_scan() {
    let mut dec = SseDecoder::new();
    // An unterminated blob past the 8 MiB cap is a broken peer, not
    // more buffering.
    let big = vec![b'x'; 9 * 1024 * 1024];
    match dec.feed(&big) {
        Err(Error::Protocol(m)) => assert!(m.contains("cap"), "{m}"),
        other => panic!("expected Protocol error, got {other:?}"),
    }

    // A separator split across three feeds still frames correctly; the
    // scan cursor backs up to the incomplete `\n` lookahead.
    let mut dec = SseDecoder::new();
    assert!(dec.feed(b"data: a\r\n\r").unwrap().is_empty());
    assert_eq!(dec.feed(b"\ndata: b\n").unwrap().pop().unwrap().data, "a");
    assert_eq!(dec.feed(b"\n").unwrap().pop().unwrap().data, "b");
    assert_eq!(dec.feed(b"data: c\n\n").unwrap().pop().unwrap().data, "c");
}
