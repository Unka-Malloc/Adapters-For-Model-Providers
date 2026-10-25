//! Original fixtures for the inbound Messages dialect: client-shaped
//! request bodies decode to canonical requests, and canonical
//! outputs/streams encode back to the Anthropic wire shape.
//! Payloads are hand-authored from the wire contract, not copied
//! from reference projects.

use a4mp::wire::*;
use a4mp::*;
use serde_json::{Map, json};

fn decode(body: &serde_json::Value) -> (ChatRequest, ConversionReport) {
    decode_chat_request(
        WireDialect::AnthropicMessages,
        &serde_json::to_vec(body).unwrap(),
    )
    .unwrap()
}

fn loss_paths(report: &ConversionReport) -> Vec<String> {
    report.losses.iter().map(|l| l.path.clone()).collect()
}

#[test]
fn decodes_system_blocks_tools_and_sampling() {
    let body = json!({
        "model": "claude-x",
        "max_tokens": 256,
        "system": "be brief",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "see"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
                {"type": "image", "source": {"type": "url", "url": "https://img.test/pic.png"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "ponder", "signature": "sig1"},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true,
                 "content": [{"type": "text", "text": "boom"}]}
            ]}
        ],
        "tools": [{"name": "lookup", "description": "search", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "lookup"},
        "temperature": 0.5,
        "stop_sequences": ["END"]
    });
    let (req, report) = decode(&body);
    assert!(report.losses.is_empty(), "losses: {:?}", loss_paths(&report));

    assert_eq!(req.model, "claude-x");
    assert_eq!(req.sampling.max_output_tokens, Some(256));
    assert_eq!(req.sampling.temperature, Some(0.5));
    assert_eq!(req.sampling.stop, ["END"]);
    assert!(matches!(req.tool_choice, Some(ToolChoice::Named(ref n)) if n == "lookup"));
    assert_eq!(req.tools[0].parameters, json!({"type": "object"}));

    // `system` lands as a leading system message.
    assert!(matches!(req.messages[0].role, Role::System));
    match &req.messages[0].content[0] {
        ContentPart::Text { text } => assert_eq!(text, "be brief"),
        other => panic!("expected text, got {other:?}"),
    }

    let user = &req.messages[1].content;
    match &user[1] {
        ContentPart::Media { kind, source, .. } => {
            assert_eq!(*kind, MediaKind::Image);
            match source {
                MediaSource::Inline { media_type, data } => {
                    assert_eq!(media_type, "image/png");
                    assert_eq!(&data[..], b"hi");
                }
                other => panic!("expected inline image, got {other:?}"),
            }
        }
        other => panic!("expected image, got {other:?}"),
    }
    match &user[2] {
        ContentPart::Media { source: MediaSource::Url(u), .. } => {
            assert_eq!(u, "https://img.test/pic.png")
        }
        other => panic!("expected url image, got {other:?}"),
    }

    let assistant = &req.messages[2].content;
    match &assistant[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.protocol, Protocol::AnthropicMessages);
            assert_eq!(rb.opaque["signature"], "sig1");
            assert_eq!(rb.summary.as_deref(), Some("ponder"));
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
    match &assistant[2] {
        ContentPart::ToolCall(tc) => {
            assert_eq!(tc.id, "toolu_1");
            assert_eq!(tc.arguments, json!({"q": "x"}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }

    match &req.messages[3].content[0] {
        ContentPart::ToolResult(tr) => {
            assert_eq!(tr.call_id, "toolu_1");
            assert!(tr.is_error);
            match &tr.content[0] {
                ContentPart::Text { text } => assert_eq!(text, "boom"),
                other => panic!("expected text, got {other:?}"),
            }
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn server_tools_and_unknown_fields_report_not_silent() {
    let body = json!({
        "model": "m",
        "max_tokens": 8,
        "system": [{"type": "text", "text": "s1"}, {"type": "image", "source": {"type": "url", "url": "https://x/i.png"}}],
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "t", "input_schema": {"type": "object"}},
            {"type": "web_search_v1", "name": "web_search"}
        ],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
        "top_k": 40,
        "metadata": {"user_id": "u1"},
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    });
    let (req, report) = decode(&body);

    assert_eq!(req.tools.len(), 1);
    assert!(matches!(req.tool_choice, Some(ToolChoice::Auto)));
    // `disable_parallel_tool_use` is canonical, not a loss.
    assert_eq!(req.parallel_tool_calls, Some(false));
    assert_eq!(req.extensions["top_k"], 40);
    assert_eq!(req.extensions["metadata"]["user_id"], "u1");
    assert_eq!(req.extensions["thinking"]["budget_tokens"], 1024);
    // The non-text system block is reported, not silently dropped.
    assert!(matches!(req.messages[0].role, Role::System));
    assert_eq!(req.messages[0].content.len(), 1);

    let paths = loss_paths(&report);
    for expected in [
        "system[1]",
        "tools[1]",
        "request.top_k",
        "request.metadata",
        "request.thinking",
    ] {
        assert!(paths.iter().any(|p| p == expected), "missing loss {expected} in {paths:?}");
    }
    assert_eq!(report.source, Protocol::AnthropicMessages);
}

#[test]
fn decode_rejects_malformed_bodies() {
    assert!(decode_chat_request(WireDialect::AnthropicMessages, b"not json").is_err());
    let missing_messages = decode_chat_request(
        WireDialect::AnthropicMessages,
        &serde_json::to_vec(&json!({"model": "m"})).unwrap(),
    );
    assert!(matches!(missing_messages, Err(Error::Protocol(_))));
}

#[test]
fn encodes_output_as_message() {
    let out = Output {
        parts: vec![
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::AnthropicMessages,
                opaque: json!({"type": "thinking", "thinking": "ponder", "signature": "sig1"}),
                summary: Some("ponder".into()),
            }),
            ContentPart::Text { text: "done".into() },
            ContentPart::ToolCall(ToolCall {
                id: "toolu_9".into(),
                name: "f".into(),
                arguments: json!({"x": 1}),
            }),
        ],
        stop: Some(StopReason::ToolUse),
        usage: Some(Usage {
            input_tokens: Some(11),
            output_tokens: Some(7),
            extensions: Map::from_iter([("cache_read_input_tokens".into(), json!(3))]),
        }),
        extensions: Map::new(),
    };
    let (v, report) = encode_chat_output(WireDialect::AnthropicMessages, &out, "m");
    assert!(report.losses.is_empty(), "losses: {:?}", loss_paths(&report));

    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], "m");
    // Same-dialect thinking replays verbatim, signature intact.
    assert_eq!(v["content"][0], json!({"type": "thinking", "thinking": "ponder", "signature": "sig1"}));
    assert_eq!(v["content"][1], json!({"type": "text", "text": "done"}));
    assert_eq!(
        v["content"][2],
        json!({"type": "tool_use", "id": "toolu_9", "name": "f", "input": {"x": 1}})
    );
    assert_eq!(v["stop_reason"], "tool_use");
    assert_eq!(v["usage"]["input_tokens"], 11);
    assert_eq!(v["usage"]["output_tokens"], 7);
    assert_eq!(v["usage"]["cache_read_input_tokens"], 3);
}

#[test]
fn encode_reports_media_and_unsigned_reasoning() {
    let out = Output {
        parts: vec![
            ContentPart::Media {
                kind: MediaKind::Audio,
                source: MediaSource::Inline {
                    media_type: "audio/wav".into(),
                    data: bytes::Bytes::from_static(b"x"),
                },
                detail: None,
            },
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::OpenAiChatCompletions,
                opaque: json!({"reasoning_content": "chain"}),
                summary: Some("chain".into()),
            }),
        ],
        stop: Some(StopReason::Complete),
        ..Default::default()
    };
    let (v, report) = encode_chat_output(WireDialect::AnthropicMessages, &out, "m");
    let paths = loss_paths(&report);
    assert!(paths.iter().any(|p| p.starts_with("parts[0]")), "{paths:?}");
    assert!(paths.iter().any(|p| p.starts_with("parts[1]")), "{paths:?}");
    // Foreign reasoning renders as a signature-less thinking block —
    // reported, never forged.
    assert_eq!(v["content"][0], json!({"type": "thinking", "thinking": "chain"}));
    assert!(v["content"][0].get("signature").is_none());
    assert_eq!(v["stop_reason"], "end_turn");
}

#[test]
fn stop_reason_maps_variants() {
    for (stop, want) in [
        (StopReason::Complete, "end_turn"),
        (StopReason::Length, "max_tokens"),
        (StopReason::ToolUse, "tool_use"),
        (StopReason::Filtered, "refusal"),
        (StopReason::Paused, "pause_turn"),
    ] {
        let out = Output {
            parts: vec![ContentPart::Text { text: "x".into() }],
            stop: Some(stop),
            ..Default::default()
        };
        let (v, report) = encode_chat_output(WireDialect::AnthropicMessages, &out, "m");
        assert_eq!(v["stop_reason"], want);
        assert!(report.losses.is_empty(), "{want}");
    }
}

#[test]
fn stream_encodes_message_blocks_and_stop() {
    let dialect = WireDialect::AnthropicMessages;
    let start = encode_stream_start(dialect, "m");
    assert_eq!(start.len(), 1);
    assert_eq!(start[0].event.as_deref(), Some("message_start"));
    let envelope: serde_json::Value = serde_json::from_str(&start[0].data).unwrap();
    assert_eq!(envelope["type"], "message_start");
    assert_eq!(envelope["message"]["role"], "assistant");
    assert_eq!(envelope["message"]["model"], "m");
    assert!(envelope["message"]["id"].as_str().unwrap().starts_with("msg_"));

    let mut enc = stream_encoder(dialect, "m");
    let events = [
        StreamEvent::PartStarted { index: 0, kind: PartKind::Reasoning },
        StreamEvent::ReasoningDelta { index: 0, text: "th".into() },
        // A signature rides through verbatim as a provider-native delta.
        StreamEvent::Native {
            part_index: Some(0),
            value: json!({"type": "signature_delta", "signature": "sig1"}),
        },
        StreamEvent::PartFinished { index: 0 },
        StreamEvent::PartStarted { index: 1, kind: PartKind::Text },
        StreamEvent::TextDelta { index: 1, text: "Hello".into() },
        StreamEvent::PartFinished { index: 1 },
        StreamEvent::ToolCallStarted { index: 2, id: "toolu_1".into(), name: "lookup".into() },
        StreamEvent::ToolCallDelta { index: 2, arguments: "{\"q\":\"x\"}".into() },
        StreamEvent::PartFinished { index: 2 },
        StreamEvent::Usage(Usage { output_tokens: Some(9), ..Default::default() }),
        StreamEvent::Finished { stop: Some(StopReason::ToolUse) },
    ];
    let mut frames = Vec::new();
    for e in &events {
        frames.extend(enc.encode(e));
    }
    frames.extend(enc.finish());
    assert!(enc.drain_losses().is_empty());

    let data = |f: &SseEvent| serde_json::from_str::<serde_json::Value>(&f.data).unwrap();
    let mut names = Vec::new();
    for f in &frames {
        names.push(f.event.clone().unwrap_or_default());
    }
    assert_eq!(
        names,
        [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_delta",
            "message_stop",
        ]
    );

    assert_eq!(data(&frames[0])["content_block"]["type"], "thinking");
    assert_eq!(data(&frames[1])["delta"], json!({"type": "thinking_delta", "thinking": "th"}));
    assert_eq!(data(&frames[2])["index"], 0);
    assert_eq!(data(&frames[2])["delta"]["type"], "signature_delta");
    assert_eq!(data(&frames[2])["delta"]["signature"], "sig1");
    assert_eq!(data(&frames[4])["index"], 1);
    assert_eq!(data(&frames[4])["content_block"], json!({"type": "text", "text": ""}));
    assert_eq!(data(&frames[5])["delta"]["text"], "Hello");
    let tool = data(&frames[7]);
    assert_eq!(tool["index"], 2);
    assert_eq!(tool["content_block"]["type"], "tool_use");
    assert_eq!(tool["content_block"]["id"], "toolu_1");
    assert_eq!(tool["content_block"]["input"], json!({}));
    assert_eq!(data(&frames[8])["delta"]["partial_json"], "{\"q\":\"x\"}");
    assert_eq!(data(&frames[10])["usage"]["output_tokens"], 9);
    assert_eq!(data(&frames[11])["delta"]["stop_reason"], "tool_use");
    assert_eq!(data(&frames[12])["type"], "message_stop");
    // finish is idempotent.
    assert!(enc.finish().is_empty());
}

#[test]
fn stream_relays_native_blocks_and_events() {
    let dialect = WireDialect::AnthropicMessages;
    let mut enc = stream_encoder(dialect, "m");
    // An unknown part kind waits for its provider-native block object.
    let mut frames = enc.encode(&StreamEvent::PartStarted { index: 0, kind: PartKind::Other });
    assert!(frames.is_empty());
    frames.extend(enc.encode(&StreamEvent::Native {
        part_index: Some(0),
        value: json!({"type": "server_tool_use", "id": "srv_1", "name": "web_search"}),
    }));
    assert_eq!(frames[0].event.as_deref(), Some("content_block_start"));
    let start: serde_json::Value = serde_json::from_str(&frames[0].data).unwrap();
    assert_eq!(start["content_block"]["type"], "server_tool_use");
    // A partless event-shaped native relays as its own named frame.
    frames.extend(enc.encode(&StreamEvent::Native {
        part_index: None,
        value: json!({"type": "ping"}),
    }));
    assert_eq!(frames[1].event.as_deref(), Some("ping"));
    // A shapeless native is reported, not emitted.
    frames.extend(enc.encode(&StreamEvent::Native {
        part_index: Some(0),
        value: json!(42),
    }));
    assert_eq!(frames.len(), 2);
    assert!(!enc.drain_losses().is_empty());
}

#[test]
fn round_trip_request_is_identity_on_representable_inputs() {
    let messages = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "see"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}
        ]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "ponder", "signature": "sig1"},
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}
        ]}
    ]);
    let body = json!({
        "model": "claude-x",
        "max_tokens": 256,
        "system": "be brief",
        "messages": messages,
        "tools": [{"name": "lookup", "description": "search", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
        "temperature": 0.5,
        "stop_sequences": ["END"]
    });
    let (req, report) = decode(&body);
    assert!(report.losses.is_empty(), "losses: {:?}", loss_paths(&report));

    // Re-encode through the outbound protocol codec: the round trip is
    // identity on every representable field.
    let out = protocols::anthropic_messages::build_request(&req).unwrap();
    assert_eq!(out["model"], body["model"]);
    assert_eq!(out["system"], "be brief");
    assert_eq!(out["max_tokens"], 256);
    assert_eq!(out["messages"], messages);
    assert_eq!(out["tools"], body["tools"]);
    assert_eq!(out["tool_choice"], json!({"type": "any"}));
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["stop_sequences"], json!(["END"]));
}
