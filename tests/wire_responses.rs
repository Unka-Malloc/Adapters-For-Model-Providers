//! Original fixtures for the inbound Responses dialect: client-shaped
//! request bodies decode to canonical requests, and canonical
//! outputs/streams encode back to the Responses wire shape.
//! Payloads are hand-authored from the wire contract.

use a4mp::protocols::openai_responses::ResponsesStreamDecoder;
use a4mp::transport::SseEventDecoder;
use a4mp::wire::*;
use a4mp::*;
use serde_json::{Map, json};

fn decode(body: &serde_json::Value) -> (ChatRequest, ConversionReport) {
    decode_chat_request(
        WireDialect::OpenAiResponses,
        &serde_json::to_vec(body).unwrap(),
    )
    .unwrap()
}

fn loss_paths(report: &ConversionReport) -> Vec<String> {
    report.losses.iter().map(|l| l.path.clone()).collect()
}

#[test]
fn dialect_maps_to_responses_protocol() {
    assert_eq!(
        WireDialect::OpenAiResponses.protocol(),
        Protocol::OpenAiResponses
    );
}

#[test]
fn decodes_instructions_items_media_tools_and_sampling() {
    let body = json!({
        "model": "gpt-x",
        "instructions": "be brief",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "see"},
                {"type": "input_image", "image_url": "data:image/png;base64,aGk=", "detail": "high"},
                {"type": "input_image", "image_url": "https://img.test/pic.png"},
                {"type": "input_file", "file_id": "file-9"},
                {"type": "input_audio", "input_audio": {"data": "aGk=", "format": "mp3"}}
            ]},
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc",
             "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "function_call", "call_id": "call_1", "name": "lookup",
             "arguments": "{\"q\":\"x\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "42"}
        ],
        "tools": [{"type": "function", "name": "lookup", "description": "search",
                   "parameters": {"type": "object"}, "strict": true}],
        "tool_choice": {"type": "function", "name": "lookup"},
        "parallel_tool_calls": false,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_output_tokens": 64,
        "stream": true,
        "previous_response_id": "resp_prev"
    });
    let (req, report) = decode(&body);

    assert_eq!(req.model, "gpt-x");
    assert_eq!(req.parallel_tool_calls, Some(false));

    // `instructions` is a leading system turn.
    assert!(matches!(req.messages[0].role, Role::System));
    match &req.messages[0].content[0] {
        ContentPart::Text { text } => assert_eq!(text, "be brief"),
        other => panic!("expected instruction text, got {other:?}"),
    }

    let user = &req.messages[1].content;
    assert_eq!(user.len(), 5);
    match &user[1] {
        ContentPart::Media {
            kind,
            source,
            detail,
        } => {
            assert_eq!(*kind, MediaKind::Image);
            match source {
                MediaSource::Inline { media_type, data } => {
                    assert_eq!(media_type, "image/png");
                    assert_eq!(&data[..], b"hi");
                }
                other => panic!("expected inline image, got {other:?}"),
            }
            assert_eq!(detail.as_deref(), Some("high"));
        }
        other => panic!("expected image, got {other:?}"),
    }
    match &user[2] {
        ContentPart::Media {
            source: MediaSource::Url(u),
            ..
        } => {
            assert_eq!(u, "https://img.test/pic.png");
        }
        other => panic!("expected url image, got {other:?}"),
    }
    match &user[3] {
        ContentPart::Media {
            kind,
            source: MediaSource::Resource(id),
            ..
        } => {
            assert_eq!(*kind, MediaKind::File);
            assert_eq!(id, "file-9");
        }
        other => panic!("expected file resource, got {other:?}"),
    }
    match &user[4] {
        ContentPart::Media { kind, source, .. } => {
            assert_eq!(*kind, MediaKind::Audio);
            match source {
                MediaSource::Inline { media_type, .. } => assert_eq!(media_type, "audio/mp3"),
                other => panic!("expected inline audio, got {other:?}"),
            }
        }
        other => panic!("expected audio, got {other:?}"),
    }

    // Reasoning and the tool call merge into one assistant turn, in order.
    assert!(matches!(req.messages[2].role, Role::Assistant));
    match &req.messages[2].content[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.protocol, Protocol::OpenAiResponses);
            assert_eq!(rb.summary.as_deref(), Some("think"));
            assert_eq!(rb.opaque["id"], "rs_1");
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
    match &req.messages[2].content[1] {
        ContentPart::ToolCall(tc) => {
            assert_eq!(tc.id, "call_1");
            assert_eq!(tc.name, "lookup");
            assert_eq!(tc.arguments, json!({"q": "x"}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }

    assert!(matches!(req.messages[3].role, Role::User));
    match &req.messages[3].content[0] {
        ContentPart::ToolResult(tr) => {
            assert_eq!(tr.call_id, "call_1");
            match &tr.content[0] {
                ContentPart::Text { text } => assert_eq!(text, "42"),
                other => panic!("expected result text, got {other:?}"),
            }
        }
        other => panic!("expected tool result, got {other:?}"),
    }

    assert_eq!(req.tools[0].name, "lookup");
    assert!(req.tools[0].strict);
    assert!(matches!(req.tool_choice, Some(ToolChoice::Named(ref n)) if n == "lookup"));
    assert_eq!(req.sampling.temperature, Some(0.5));
    assert_eq!(req.sampling.top_p, Some(0.9));
    assert_eq!(req.sampling.max_output_tokens, Some(64));
    // `stream` is hop-local; server-state fields are preserved as dialect-bound.
    assert!(!req.extensions.contains_key("stream"));
    assert_eq!(req.extensions["previous_response_id"], "resp_prev");
    let paths = loss_paths(&report);
    assert!(
        paths.iter().any(|p| p == "request.previous_response_id"),
        "{paths:?}"
    );
    assert_eq!(report.source, Protocol::OpenAiResponses);
}

#[test]
fn accepts_string_input_and_plain_role_objects() {
    let (req, report) = decode(&json!({"model": "m", "input": "hello"}));
    assert!(
        report.losses.is_empty(),
        "losses: {:?}",
        loss_paths(&report)
    );
    assert!(matches!(req.messages[0].role, Role::User));
    match &req.messages[0].content[0] {
        ContentPart::Text { text } => assert_eq!(text, "hello"),
        other => panic!("expected text, got {other:?}"),
    }

    let (req, _) = decode(&json!({
        "model": "m",
        "input": [{"role": "assistant", "content": "replayed"}]
    }));
    assert!(matches!(req.messages[0].role, Role::Assistant));

    // An instructions-only request is valid: no input items.
    let (req, report) = decode(&json!({"model": "m", "instructions": "rule"}));
    assert!(
        report.losses.is_empty(),
        "losses: {:?}",
        loss_paths(&report)
    );
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn decode_rejects_malformed_bodies() {
    assert!(decode_chat_request(WireDialect::OpenAiResponses, b"not json").is_err());
    let missing_model = decode_chat_request(
        WireDialect::OpenAiResponses,
        &serde_json::to_vec(&json!({"input": "hi"})).unwrap(),
    );
    assert!(matches!(missing_model, Err(Error::Protocol(_))));
    let bad_input = decode_chat_request(
        WireDialect::OpenAiResponses,
        &serde_json::to_vec(&json!({"model": "m", "input": 5})).unwrap(),
    );
    assert!(matches!(bad_input, Err(Error::Protocol(_))));
}

#[test]
fn unknown_items_and_parts_report_not_silent() {
    let body = json!({
        "model": "m",
        "input": [
            {"role": "user", "content": [
                {"type": "mystery", "mystery": {}},
                {"type": "refusal", "refusal": "no"}
            ]},
            {"type": "item_reference", "id": "item_1"},
            {"type": "web_search_call", "id": "ws_1"}
        ]
    });
    let (req, report) = decode(&body);
    let paths = loss_paths(&report);
    assert!(
        paths.iter().any(|p| p.starts_with("input[0].content[0]")),
        "{paths:?}"
    );
    assert!(
        paths
            .iter()
            .any(|p| p == "input[0].content[1].type (refusal)"),
        "{paths:?}"
    );
    assert!(paths.iter().any(|p| p == "input[1]"), "{paths:?}");
    assert!(paths.iter().any(|p| p == "input[2]"), "{paths:?}");
    // Refusal text survives; the server item stays replayable as opaque state.
    assert!(matches!(&req.messages[0].content[0], ContentPart::Text { text } if text == "no"));
    assert!(
        matches!(&req.messages[1].content[0], ContentPart::Reasoning(rb)
        if rb.opaque["type"] == "web_search_call")
    );
}

#[test]
fn encodes_output_as_response_object() {
    let out = Output {
        parts: vec![
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::OpenAiResponses,
                opaque: json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}),
                summary: None,
            }),
            ContentPart::Text {
                text: "done".into(),
            },
            ContentPart::ToolCall(ToolCall {
                id: "call_9".into(),
                name: "f".into(),
                arguments: json!({"x": 1}),
            }),
        ],
        stop: Some(StopReason::Complete),
        usage: Some(Usage {
            input_tokens: Some(5),
            output_tokens: Some(9),
            extensions: Map::from_iter([(
                "output_tokens_details".into(),
                json!({"reasoning_tokens": 4}),
            )]),
        }),
        extensions: Map::from_iter([("store".into(), json!(false))]),
    };
    let (v, report) = encode_chat_output(WireDialect::OpenAiResponses, &out, "m");
    assert!(
        report.losses.is_empty(),
        "losses: {:?}",
        loss_paths(&report)
    );

    assert!(v["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(v["object"], "response");
    assert_eq!(v["status"], "completed");
    assert_eq!(v["model"], "m");
    assert_eq!(v["store"], false);
    assert_eq!(
        v["output"][0],
        json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"})
    );
    assert_eq!(v["output"][1]["type"], "message");
    assert_eq!(v["output"][1]["content"][0]["type"], "output_text");
    assert_eq!(v["output"][1]["content"][0]["text"], "done");
    assert_eq!(v["output"][2]["type"], "function_call");
    assert_eq!(v["output"][2]["call_id"], "call_9");
    assert_eq!(v["output"][2]["arguments"], "{\"x\":1}");
    assert_eq!(v["usage"]["input_tokens"], 5);
    assert_eq!(v["usage"]["output_tokens"], 9);
    assert_eq!(v["usage"]["total_tokens"], 14);
    assert_eq!(v["usage"]["output_tokens_details"]["reasoning_tokens"], 4);

    // Buffered round trip: canonical output survives encode → decode.
    let back =
        protocols::openai_responses::decode_response(&serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(matches!(back.stop, Some(StopReason::Complete)));
    assert!(matches!(&back.parts[0], ContentPart::Reasoning(rb) if rb.opaque["id"] == "rs_1"));
    assert!(matches!(&back.parts[1], ContentPart::Text { text } if text == "done"));
    assert!(matches!(&back.parts[2], ContentPart::ToolCall(tc)
        if tc.id == "call_9" && tc.arguments["x"] == 1));
    assert_eq!(back.usage.unwrap().input_tokens, Some(5));
}

#[test]
fn stop_and_unrepresentable_parts_map_without_silent_drops() {
    for (stop, status, reason, lossy) in [
        (StopReason::Complete, "completed", None, false),
        (StopReason::ToolUse, "completed", None, false),
        (
            StopReason::Length,
            "incomplete",
            Some("max_output_tokens"),
            false,
        ),
        (
            StopReason::Filtered,
            "incomplete",
            Some("content_filter"),
            false,
        ),
        (StopReason::Paused, "incomplete", Some("paused"), true),
        (
            StopReason::Other("cancelled".into()),
            "incomplete",
            Some("cancelled"),
            true,
        ),
    ] {
        let out = Output {
            parts: vec![ContentPart::Text { text: "x".into() }],
            stop: Some(stop),
            ..Default::default()
        };
        let (v, report) = encode_chat_output(WireDialect::OpenAiResponses, &out, "m");
        assert_eq!(v["status"], status, "{status}");
        match reason {
            Some(r) => assert_eq!(v["incomplete_details"]["reason"], r),
            None => assert!(v["incomplete_details"].is_null()),
        }
        assert_eq!(!report.losses.is_empty(), lossy);
    }

    let (v, _) = encode_chat_output(
        WireDialect::OpenAiResponses,
        &Output {
            stop: None,
            ..Default::default()
        },
        "m",
    );
    assert!(v["status"].is_null());

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
            ContentPart::Text {
                text: "answer".into(),
            },
        ],
        ..Default::default()
    };
    let (v, report) = encode_chat_output(WireDialect::OpenAiResponses, &out, "m");
    let paths = loss_paths(&report);
    assert!(paths.iter().any(|p| p.starts_with("parts[0]")), "{paths:?}");
    assert!(paths.iter().any(|p| p.starts_with("parts[1]")), "{paths:?}");
    // Foreign reasoning is rendered as a summary-only reasoning item.
    assert_eq!(v["output"][0]["type"], "reasoning");
    assert_eq!(v["output"][0]["summary"][0]["text"], "thought");
    assert_eq!(v["output"][1]["content"][0]["text"], "answer");
}

fn summarize(event: &StreamEvent) -> String {
    match event {
        StreamEvent::PartStarted { index, kind } => format!("start:{index}:{kind:?}"),
        StreamEvent::TextDelta { index, text } => format!("text:{index}:{text}"),
        StreamEvent::ReasoningDelta { index, text } => format!("reasoning:{index}:{text}"),
        StreamEvent::ToolCallStarted { index, id, name } => {
            format!("tool-start:{index}:{id}:{name}")
        }
        StreamEvent::ToolCallDelta { index, arguments } => {
            format!("tool-args:{index}:{arguments}")
        }
        StreamEvent::PartFinished { index } => format!("finish:{index}"),
        StreamEvent::Usage(u) => format!("usage:{:?}:{:?}", u.input_tokens, u.output_tokens),
        StreamEvent::Finished { stop } => format!("done:{stop:?}"),
        StreamEvent::Native { part_index, value } => {
            format!("native:{part_index:?}:{value}")
        }
        StreamEvent::MediaDelta { index, .. } => format!("media:{index}"),
        _ => "other".into(),
    }
}

fn decode_frames(frames: &[SseEvent]) -> Vec<StreamEvent> {
    let mut dec = ResponsesStreamDecoder::new();
    let mut events = Vec::new();
    for f in frames {
        events.extend(dec.decode(f).unwrap());
    }
    events
}

#[test]
fn stream_round_trips_through_canonical_events() {
    let events = vec![
        StreamEvent::PartStarted {
            index: 0,
            kind: PartKind::Text,
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "Hel".into(),
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "lo".into(),
        },
        StreamEvent::PartFinished { index: 0 },
        StreamEvent::PartStarted {
            index: 1,
            kind: PartKind::Reasoning,
        },
        StreamEvent::ReasoningDelta {
            index: 1,
            text: "ponder".into(),
        },
        StreamEvent::PartFinished { index: 1 },
        StreamEvent::PartStarted {
            index: 2,
            kind: PartKind::ToolCall,
        },
        StreamEvent::ToolCallStarted {
            index: 2,
            id: "call_1".into(),
            name: "lookup".into(),
        },
        StreamEvent::ToolCallDelta {
            index: 2,
            arguments: "{\"q\":".into(),
        },
        StreamEvent::ToolCallDelta {
            index: 2,
            arguments: "\"x\"}".into(),
        },
        StreamEvent::PartFinished { index: 2 },
        StreamEvent::Usage(Usage {
            input_tokens: Some(3),
            output_tokens: Some(7),
            ..Default::default()
        }),
        StreamEvent::Finished {
            stop: Some(StopReason::Complete),
        },
    ];

    let mut enc = stream_encoder(WireDialect::OpenAiResponses, "m");
    let preamble = enc.preamble();
    assert_eq!(preamble[0].event.as_deref(), Some("response.created"));
    assert_eq!(preamble[1].event.as_deref(), Some("response.in_progress"));
    assert!(decode_frames(&preamble).is_empty());

    let mut frames = Vec::new();
    for e in &events {
        frames.extend(enc.encode(e));
    }
    frames.extend(enc.finish());
    assert!(enc.drain_losses().is_empty());
    assert!(enc.finish().is_empty());

    let back = decode_frames(&frames);
    let want: Vec<String> = events.iter().map(summarize).collect();
    let got: Vec<String> = back.iter().map(summarize).collect();
    assert_eq!(got, want);
}

#[test]
fn native_responses_frames_round_trip_through_canonical() {
    // A wire-first stream: decode to canonical, re-encode, decode again.
    let wire = vec![
        SseEvent {
            event: Some("response.created".into()),
            data: json!({"type": "response.created", "response": {"id": "resp_1"}}).to_string(),
            id: None,
        },
        SseEvent {
            event: Some("response.output_item.added".into()),
            data: json!({"type": "response.output_item.added", "output_index": 0,
                         "item": {"type": "message", "id": "msg_1", "role": "assistant"}})
            .to_string(),
            id: None,
        },
        SseEvent {
            event: Some("response.output_text.delta".into()),
            data: json!({"type": "response.output_text.delta", "output_index": 0,
                         "delta": "Hel"})
            .to_string(),
            id: None,
        },
        SseEvent {
            event: Some("response.output_text.done".into()),
            data: json!({"type": "response.output_text.done", "output_index": 0,
                         "text": "Hel"})
            .to_string(),
            id: None,
        },
        SseEvent {
            event: Some("response.output_item.done".into()),
            data: json!({"type": "response.output_item.done", "output_index": 0,
                         "item": {"type": "message", "id": "msg_1", "content": [
                             {"type": "output_text", "text": "Hel"}]}})
            .to_string(),
            id: None,
        },
        SseEvent {
            event: Some("response.output_item.added".into()),
            data: json!({"type": "response.output_item.added", "output_index": 1,
                         "item": {"type": "function_call", "call_id": "call_3",
                                  "name": "lookup"}})
            .to_string(),
            id: None,
        },
        SseEvent {
            event: Some("response.function_call_arguments.delta".into()),
            data: json!({"type": "response.function_call_arguments.delta",
                         "output_index": 1, "delta": "{\"q\":\"x\"}"})
            .to_string(),
            id: None,
        },
        SseEvent {
            event: Some("response.output_item.done".into()),
            data: json!({"type": "response.output_item.done", "output_index": 1,
                         "item": {"type": "function_call"}})
            .to_string(),
            id: None,
        },
        SseEvent {
            event: Some("response.completed".into()),
            data: json!({"type": "response.completed", "response": {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 5, "total_tokens": 6}
            }})
            .to_string(),
            id: None,
        },
    ];
    let canonical = decode_frames(&wire);
    assert!(
        canonical
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "Hel"))
    );
    assert!(
        canonical
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallStarted { id, .. } if id == "call_3"))
    );

    let mut enc = stream_encoder(WireDialect::OpenAiResponses, "m");
    let mut frames = enc.preamble();
    for e in &canonical {
        frames.extend(enc.encode(e));
    }
    frames.extend(enc.finish());

    let back = decode_frames(&frames);
    let want: Vec<String> = canonical.iter().map(summarize).collect();
    let got: Vec<String> = back.iter().map(summarize).collect();
    assert_eq!(got, want);
}
