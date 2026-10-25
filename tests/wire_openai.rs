//! Original fixtures for the inbound Chat Completions dialect:
//! client-shaped request bodies decode to canonical requests, and
//! canonical outputs/streams encode back to the OpenAI wire shape.
//! Payloads are hand-authored from the wire contract, not copied
//! from reference projects.

use a4mp::wire::*;
use a4mp::*;
use serde_json::{Map, json};

fn decode(body: &serde_json::Value) -> (ChatRequest, ConversionReport) {
    decode_chat_request(
        WireDialect::OpenAiChatCompletions,
        &serde_json::to_vec(body).unwrap(),
    )
    .unwrap()
}

fn loss_paths(report: &ConversionReport) -> Vec<String> {
    report.losses.iter().map(|l| l.path.clone()).collect()
}

#[test]
fn decodes_messages_tools_sampling_and_extensions() {
    let body = json!({
        "model": "client-model",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "checking", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"}
        ],
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "description": "search",
                         "parameters": {"type": "object"}, "strict": true}
        }],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "parallel_tool_calls": false,
        "temperature": 0.5,
        "max_completion_tokens": 64,
        "stop": ["END", "\n\n"]
    });
    let (req, report) = decode(&body);

    assert_eq!(req.model, "client-model");
    assert!(matches!(req.messages[0].role, Role::System));
    assert!(matches!(req.messages[1].role, Role::User));
    assert!(matches!(req.messages[2].role, Role::Assistant));
    match &req.messages[2].content[1] {
        ContentPart::ToolCall(tc) => {
            assert_eq!(tc.id, "call_1");
            assert_eq!(tc.name, "lookup");
            assert_eq!(tc.arguments, json!({"q": "x"}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    // A `tool` message lands as a ToolResult part on a user message.
    assert!(matches!(req.messages[3].role, Role::User));
    match &req.messages[3].content[0] {
        ContentPart::ToolResult(tr) => {
            assert_eq!(tr.call_id, "call_1");
            assert!(!tr.is_error);
            match &tr.content[0] {
                ContentPart::Text { text } => assert_eq!(text, "42"),
                other => panic!("expected text, got {other:?}"),
            }
        }
        other => panic!("expected tool result, got {other:?}"),
    }

    assert_eq!(req.tools[0].name, "lookup");
    assert!(req.tools[0].strict);
    assert!(matches!(req.tool_choice, Some(ToolChoice::Named(ref n)) if n == "lookup"));
    assert_eq!(req.parallel_tool_calls, Some(false));
    assert_eq!(req.sampling.temperature, Some(0.5));
    assert_eq!(req.sampling.max_output_tokens, Some(64));
    assert_eq!(req.sampling.stop, ["END", "\n\n"]);
    assert!(report.losses.is_empty(), "unexpected losses: {:?}", loss_paths(&report));
}

#[test]
fn decodes_media_parts_and_reasoning() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk=", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": "https://img.test/pic.png"}},
                {"type": "input_audio", "input_audio": {"data": "aGk=", "format": "mp3"}},
                {"type": "file", "file": {"file_id": "file-9"}}
            ]},
            {"role": "assistant", "content": null, "reasoning_content": "ponder"}
        ]
    });
    let (req, report) = decode(&body);
    assert!(report.losses.is_empty(), "losses: {:?}", loss_paths(&report));

    let parts = &req.messages[0].content;
    assert_eq!(parts.len(), 5);
    match &parts[1] {
        ContentPart::Media { kind, source, detail } => {
            assert_eq!(*kind, MediaKind::Image);
            match source {
                MediaSource::Inline { media_type, data } => {
                    assert_eq!(media_type, "image/png");
                    assert_eq!(&data[..], b"hi");
                }
                other => panic!("expected inline media, got {other:?}"),
            }
            assert_eq!(detail.as_deref(), Some("high"));
        }
        other => panic!("expected image, got {other:?}"),
    }
    match &parts[2] {
        ContentPart::Media { source: MediaSource::Url(u), .. } => {
            assert_eq!(u, "https://img.test/pic.png");
        }
        other => panic!("expected url image, got {other:?}"),
    }
    match &parts[3] {
        ContentPart::Media { kind, source, .. } => {
            assert_eq!(*kind, MediaKind::Audio);
            match source {
                MediaSource::Inline { media_type, .. } => assert_eq!(media_type, "audio/mp3"),
                other => panic!("expected inline audio, got {other:?}"),
            }
        }
        other => panic!("expected audio, got {other:?}"),
    }
    match &parts[4] {
        ContentPart::Media { kind, source: MediaSource::Resource(id), .. } => {
            assert_eq!(*kind, MediaKind::File);
            assert_eq!(id, "file-9");
        }
        other => panic!("expected file resource, got {other:?}"),
    }
    match &req.messages[1].content[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.protocol, Protocol::OpenAiChatCompletions);
            assert_eq!(rb.summary.as_deref(), Some("ponder"));
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
}

#[test]
fn unclaimed_fields_preserve_into_extensions_with_losses() {
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "seed": 7,
        "response_format": {"type": "json_object"},
        "presence_penalty": 0.3,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let (req, report) = decode(&body);
    // `stream` is hop-local control: consumed, not forwarded.
    assert!(!req.extensions.contains_key("stream"));
    assert_eq!(req.extensions["seed"], 7);
    assert_eq!(req.extensions["response_format"]["type"], "json_object");
    assert_eq!(req.extensions["stream_options"]["include_usage"], true);
    let paths = loss_paths(&report);
    for expected in [
        "request.seed",
        "request.response_format",
        "request.presence_penalty",
        "request.stream_options",
    ] {
        assert!(paths.iter().any(|p| p == expected), "missing loss {expected} in {paths:?}");
    }
    assert_eq!(report.source, Protocol::OpenAiChatCompletions);
}

#[test]
fn malformed_and_unrepresentable_parts_report_not_silent() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": [
                {"type": "mystery", "mystery": {}},
                {"type": "refusal", "refusal": "no"}
            ]},
            {"role": "user", "content": "ok", "name": "bob"}
        ],
        "tools": [{"type": "custom_tool", "custom_tool": {"name": "x"}}]
    });
    let (req, report) = decode(&body);
    // Text survives; unrepresentable parts are gone but reported.
    assert_eq!(req.messages[0].content.len(), 0);
    let paths = loss_paths(&report);
    assert!(paths.iter().any(|p| p.contains("messages[0].content[0]")));
    assert!(paths.iter().any(|p| p.contains("messages[0].content[1]")));
    assert!(paths.iter().any(|p| p == "messages[1].name"));
    assert!(paths.iter().any(|p| p == "tools[0]"));
}

#[test]
fn decode_rejects_malformed_bodies() {
    assert!(decode_chat_request(WireDialect::OpenAiChatCompletions, b"not json").is_err());
    // `model` is required on this wire.
    let missing_model = decode_chat_request(
        WireDialect::OpenAiChatCompletions,
        &serde_json::to_vec(&json!({"messages": []})).unwrap(),
    );
    assert!(matches!(missing_model, Err(Error::Protocol(_))));
}

#[test]
fn encodes_output_as_chat_completion() {
    let out = Output {
        parts: vec![
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::OpenAiChatCompletions,
                opaque: json!({"reasoning_content": "steps"}),
                summary: Some("steps".into()),
            }),
            ContentPart::Text { text: "done".into() },
            ContentPart::ToolCall(ToolCall {
                id: "call_9".into(),
                name: "f".into(),
                arguments: json!({"x": 1}),
            }),
        ],
        stop: Some(StopReason::ToolUse),
        usage: Some(Usage {
            input_tokens: Some(5),
            output_tokens: Some(9),
            extensions: Map::from_iter([(
                "completion_tokens_details".into(),
                json!({"reasoning_tokens": 4}),
            )]),
        }),
        extensions: Map::from_iter([("system_fingerprint".into(), json!("fp_1"))]),
    };
    let (v, report) = encode_chat_output(WireDialect::OpenAiChatCompletions, &out, "m");
    assert!(report.losses.is_empty(), "losses: {:?}", loss_paths(&report));

    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "m");
    assert_eq!(v["system_fingerprint"], "fp_1");
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["role"], "assistant");
    assert_eq!(msg["content"], "done");
    assert_eq!(msg["reasoning_content"], "steps");
    assert_eq!(msg["tool_calls"][0]["id"], "call_9");
    assert_eq!(msg["tool_calls"][0]["type"], "function");
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "f");
    assert_eq!(msg["tool_calls"][0]["function"]["arguments"], "{\"x\":1}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(v["usage"]["prompt_tokens"], 5);
    assert_eq!(v["usage"]["completion_tokens"], 9);
    assert_eq!(v["usage"]["total_tokens"], 14);
    assert_eq!(v["usage"]["completion_tokens_details"]["reasoning_tokens"], 4);
}

#[test]
fn encode_reports_media_and_foreign_reasoning() {
    let out = Output {
        parts: vec![
            ContentPart::Media {
                kind: MediaKind::Image,
                source: MediaSource::Url("https://img.test/o.png".into()),
                detail: None,
            },
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::AnthropicMessages,
                opaque: json!({"type": "thinking", "signature": "sig"}),
                summary: Some("thought".into()),
            }),
            ContentPart::Text { text: "answer".into() },
        ],
        stop: Some(StopReason::Paused),
        ..Default::default()
    };
    let (v, report) = encode_chat_output(WireDialect::OpenAiChatCompletions, &out, "m");
    let paths = loss_paths(&report);
    assert!(paths.iter().any(|p| p.starts_with("parts[0]")), "{paths:?}");
    assert!(paths.iter().any(|p| p.starts_with("parts[1]")), "{paths:?}");
    assert!(paths.iter().any(|p| p.starts_with("stop")), "{paths:?}");
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"], "answer");
    assert_eq!(msg["reasoning_content"], "thought");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[test]
fn finish_reason_maps_stop_variants() {
    for (stop, want, lossy) in [
        (StopReason::Complete, "stop", false),
        (StopReason::Length, "length", false),
        (StopReason::ToolUse, "tool_calls", false),
        (StopReason::Filtered, "content_filter", false),
        (StopReason::Paused, "stop", true),
    ] {
        let out = Output {
            parts: vec![ContentPart::Text { text: "x".into() }],
            stop: Some(stop),
            ..Default::default()
        };
        let (v, report) = encode_chat_output(WireDialect::OpenAiChatCompletions, &out, "m");
        assert_eq!(v["choices"][0]["finish_reason"], want);
        assert_eq!(!report.losses.is_empty(), lossy, "{want}");
    }
}

#[test]
fn stream_encodes_chunks_and_done() {
    let dialect = WireDialect::OpenAiChatCompletions;
    assert!(encode_stream_start(dialect, "m").is_empty());
    let mut enc = stream_encoder(dialect, "m");

    let events = [
        StreamEvent::PartStarted { index: 0, kind: PartKind::Reasoning },
        StreamEvent::ReasoningDelta { index: 0, text: "th".into() },
        StreamEvent::PartStarted { index: 1, kind: PartKind::Text },
        StreamEvent::TextDelta { index: 1, text: "Hel".into() },
        StreamEvent::TextDelta { index: 1, text: "lo".into() },
        StreamEvent::ToolCallStarted { index: 2, id: "call_1".into(), name: "lookup".into() },
        StreamEvent::ToolCallDelta { index: 2, arguments: "{\"q\":".into() },
        StreamEvent::ToolCallDelta { index: 2, arguments: "\"x\"}".into() },
        StreamEvent::Usage(Usage {
            input_tokens: Some(3),
            output_tokens: Some(7),
            ..Default::default()
        }),
        StreamEvent::Finished { stop: Some(StopReason::ToolUse) },
    ];
    let mut frames = Vec::new();
    for e in &events {
        frames.extend(enc.encode(e));
    }
    frames.extend(enc.finish());
    assert!(enc.drain_losses().is_empty());

    // First frame opens with the assistant role.
    let first: serde_json::Value = serde_json::from_str(&frames[0].data).unwrap();
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(first["object"], "chat.completion.chunk");
    assert!(first["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(first["model"], "m");

    let chunk = |f: &SseEvent| serde_json::from_str::<serde_json::Value>(&f.data).unwrap();
    assert_eq!(chunk(&frames[1])["choices"][0]["delta"]["reasoning_content"], "th");
    assert_eq!(chunk(&frames[2])["choices"][0]["delta"]["content"], "Hel");
    assert_eq!(chunk(&frames[3])["choices"][0]["delta"]["content"], "lo");
    let tc = chunk(&frames[4])["choices"][0]["delta"]["tool_calls"][0].clone();
    assert_eq!(tc["index"], 0);
    assert_eq!(tc["id"], "call_1");
    assert_eq!(tc["type"], "function");
    assert_eq!(tc["function"]["name"], "lookup");
    assert_eq!(chunk(&frames[5])["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], "{\"q\":");
    assert_eq!(chunk(&frames[6])["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], "\"x\"}");
    // Usage rides a chunk with empty choices.
    let usage = chunk(&frames[7]);
    assert!(usage["choices"].as_array().unwrap().is_empty());
    assert_eq!(usage["usage"]["total_tokens"], 10);
    // The finish chunk carries the mapped reason, then [DONE].
    assert_eq!(chunk(&frames[8])["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(frames[9].data, "[DONE]");
    assert_eq!(frames[9].event, None);
    // finish is idempotent.
    assert!(enc.finish().is_empty());
}

#[test]
fn round_trip_request_is_identity_on_representable_inputs() {
    let messages = json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "checking", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        {"role": "user", "content": "thanks"}
    ]);
    let body = json!({
        "model": "client-model",
        "messages": messages,
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "description": "search",
                         "parameters": {"type": "object"}, "strict": false}
        }],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "temperature": 0.5,
        "max_tokens": 64,
        "stop": ["END"]
    });
    let (req, report) = decode(&body);
    assert!(report.losses.is_empty(), "losses: {:?}", loss_paths(&report));

    // Re-encode through the outbound protocol codec: the round trip is
    // identity on every representable field.
    let out = protocols::openai_chat::build_request(&req).unwrap();
    assert_eq!(out["model"], body["model"]);
    assert_eq!(out["messages"], messages);
    assert_eq!(out["tools"], body["tools"]);
    assert_eq!(out["tool_choice"], "auto");
    assert_eq!(out["parallel_tool_calls"], true);
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_completion_tokens"], 64);
    assert_eq!(out["stop"], json!(["END"]));
}
