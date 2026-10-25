//! Original fixtures for the OpenAI Responses protocol family.

mod common;

use std::sync::Arc;

use a4mp::*;
use common::{MockTransport, Response, body_json, collect};
use serde_json::json;

fn adapter(transport: MockTransport) -> OpenAiAdapter {
    let target = ServiceTarget::new(
        "openai",
        Protocol::OpenAiResponses,
        url::Url::parse("https://api.openai.test/v1").unwrap(),
    );
    OpenAiAdapter::new(
        target,
        AuthKind::Bearer,
        Arc::new(transport),
        Arc::new(StaticCredential(Credential::Bearer {
            token: Secret::new("sk-test"),
            expires_at: None,
        })),
    )
    .unwrap()
}

#[tokio::test]
async fn responses_request_builds_input_items_and_tools() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"status": "completed", "output": [], "usage": {"input_tokens": 2, "output_tokens": 0, "total_tokens": 2}}),
    ));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "o-test".into(),
        ..Default::default()
    };
    req.messages.push(Message::text(Role::System, "rule"));
    req.messages.push(Message::text(Role::User, "hi"));
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::OpenAiResponses,
                opaque: json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}),
                summary: None,
            }),
            ContentPart::Text { text: "answer".into() },
            ContentPart::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "f".into(),
                arguments: json!({"a": 1}),
            }),
        ],
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::ToolResult(ToolResult {
            call_id: "call_1".into(),
            content: vec![ContentPart::Text { text: "ok".into() }],
            is_error: false,
        })],
    });
    req.tools.push(ToolDef {
        name: "f".into(),
        description: "d".into(),
        parameters: json!({}),
        strict: false,
    });
    req.parallel_tool_calls = Some(false);
    req.extensions.insert("reasoning".into(), json!({"effort": "high"}));

    a.chat(req, &RequestControl::default()).await.unwrap();

    let b = body_json(&t.recorded()[0]);
    let input = b["input"].as_array().unwrap();
    assert_eq!(input[0], json!({"type":"message","role":"system","content":[{"type":"input_text","text":"rule"}]}));
    assert_eq!(input[1]["role"], "user");
    // Assistant turn: reasoning item replayed verbatim + message + function_call item.
    assert_eq!(input[2], json!({"type":"reasoning","id":"rs_1","encrypted_content":"enc"}));
    assert_eq!(input[3]["content"][0]["type"], "output_text");
    assert_eq!(input[4], json!({"type":"function_call","call_id":"call_1","name":"f","arguments":"{\"a\":1}"}));
    assert_eq!(input[5], json!({"type":"function_call_output","call_id":"call_1","output":"ok"}));
    assert_eq!(b["tools"][0]["type"], "function");
    assert_eq!(b["parallel_tool_calls"], false);
    assert_eq!(b["reasoning"], json!({"effort": "high"}));
}

#[tokio::test]
async fn responses_decode_maps_items_usage_and_stop() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [{"type":"summary_text","text":"considered"}]},
                {"type": "message", "role": "assistant", "content": [{"type":"output_text","text":"partial"}]},
                {"type": "function_call", "call_id": "call_7", "name": "f", "arguments": "{\"z\":2}"}
            ],
            "usage": {"input_tokens": 11, "output_tokens": 64, "total_tokens": 75,
                      "output_tokens_details": {"reasoning_tokens": 30}}
        }),
    ));
    let a = adapter(t);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    assert!(matches!(out.stop, Some(StopReason::Length)));
    assert_eq!(out.usage.as_ref().unwrap().input_tokens, Some(11));
    assert_eq!(
        out.usage.as_ref().unwrap().extensions["output_tokens_details"]["reasoning_tokens"],
        30
    );
    assert!(matches!(out.parts[0], ContentPart::Reasoning(_)));
    assert!(matches!(out.parts[1], ContentPart::Text { .. }));
    match &out.parts[2] {
        ContentPart::ToolCall(tc) => {
            assert_eq!(tc.id, "call_7");
            assert_eq!(tc.arguments["z"], 2);
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    // Reasoning blocks must replay back into a Responses request verbatim.
    let mut req = ChatRequest::default();
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![out.parts[0].clone()],
    });
    let body = a4mp::protocols::openai_responses::build_request(&req).unwrap();
    assert_eq!(body["input"][0]["type"], "reasoning");
    assert_eq!(body["input"][0]["id"], "rs_1");
}

#[tokio::test]
async fn responses_stream_decodes_named_events() {
    let sse = concat!(
        "event: response.output_item.added\n",
        "data: {\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n",
        "event: response.output_item.added\n",
        "data: {\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"output_index\":1,\"delta\":\"Hel\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"output_index\":1,\"delta\":\"lo\"}\n\n",
        "event: response.output_item.added\n",
        "data: {\"output_index\":2,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_3\",\"name\":\"lookup\"}}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"output_index\":2,\"delta\":\"{\\\"q\\\":\"}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"output_index\":2,\"delta\":\"\\\"x\\\"}\"}\n\n",
        "event: response.output_item.done\n",
        "data: {\"output_index\":2,\"item\":{\"type\":\"function_call\"}}\n\n",
        "event: response.completed\n",
        "data: {\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":5,\"total_tokens\":6}}}\n\n"
    );
    let t = MockTransport::new();
    t.push(Response::Sse(vec![sse]));
    let a = adapter(t);
    let stream = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    let events = collect(stream).await.unwrap();

    let mut text = String::new();
    let mut args = String::new();
    let mut tool = None;
    let mut usage = false;
    let mut finish = false;
    for e in &events {
        match e {
            StreamEvent::TextDelta { text: d, .. } => text.push_str(d),
            StreamEvent::ToolCallStarted { id, name, .. } => tool = Some((id.clone(), name.clone())),
            StreamEvent::ToolCallDelta { arguments, .. } => args.push_str(arguments),
            StreamEvent::Usage(u) => usage = u.output_tokens == Some(5),
            StreamEvent::Finished { stop } => finish = matches!(stop, Some(StopReason::Complete)),
            _ => {}
        }
    }
    assert_eq!(text, "Hello");
    assert_eq!(args, "{\"q\":\"x\"}");
    assert_eq!(
        tool,
        Some(("call_3".to_string(), "lookup".to_string()))
    );
    assert!(usage);
    assert!(finish);
}

fn frame(event: &str, data: serde_json::Value) -> a4mp::SseEvent {
    a4mp::SseEvent {
        event: Some(event.to_owned()),
        data: data.to_string(),
        id: None,
    }
}

fn decode_frames(frames: &[a4mp::SseEvent]) -> Vec<StreamEvent> {
    let mut decoder =
        a4mp::protocols::openai_responses::ResponsesStreamDecoder::new();
    let mut events = Vec::new();
    for f in frames {
        events.extend(decoder.decode(f).unwrap());
    }
    events
}

#[test]
fn stream_done_only_text_frames_backfill_missing_deltas() {
    let frames = vec![
        frame(
            "response.output_item.added",
            json!({"output_index": 0, "item": {"type": "message", "id": "msg_1"}}),
        ),
        frame(
            "response.output_text.done",
            json!({"output_index": 0, "text": "complete-only"}),
        ),
        frame(
            "response.output_item.done",
            json!({"output_index": 0, "item": {"type": "message", "content": [
                {"type": "output_text", "text": "complete-only"}
            ]}}),
        ),
        frame(
            "response.completed",
            json!({"response": {"status": "completed"}}),
        ),
    ];
    let events = decode_frames(&frames);
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "complete-only");
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::TextDelta { .. }))
            .count(),
        1
    );
    assert!(events.iter().any(|e| matches!(e, StreamEvent::PartFinished { .. })));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::Finished { .. })));
    // Envelopes and done events are consumed, not leaked as Native.
    assert!(!events.iter().any(|e| matches!(e, StreamEvent::Native { .. })));
}

#[test]
fn stream_done_only_tool_arguments_are_preserved() {
    let frames = vec![
        frame(
            "response.output_item.added",
            json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_9", "name": "f"}}),
        ),
        frame(
            "response.function_call_arguments.done",
            json!({"output_index": 0, "arguments": "{\"x\":1}"}),
        ),
        frame(
            "response.output_item.done",
            json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_9", "name": "f",
                "arguments": "{\"x\":1}"}}),
        ),
        frame(
            "response.completed",
            json!({"response": {"status": "completed"}}),
        ),
    ];
    let events = decode_frames(&frames);
    let args: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolCallDelta { arguments, .. } => Some(arguments.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(args, "{\"x\":1}");
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolCallStarted { .. }))
            .count(),
        1
    );
}

#[test]
fn stream_done_only_refusal_text_is_preserved() {
    let frames = vec![
        frame(
            "response.output_item.added",
            json!({"output_index": 0, "item": {"type": "message", "id": "msg_1"}}),
        ),
        frame(
            "response.refusal.done",
            json!({"output_index": 0, "refusal": "cannot help"}),
        ),
        frame(
            "response.output_item.done",
            json!({"output_index": 0, "item": {"type": "message", "content": [
                {"type": "refusal", "refusal": "cannot help"}
            ]}}),
        ),
        frame(
            "response.completed",
            json!({"response": {"status": "completed"}}),
        ),
    ];
    let events = decode_frames(&frames);
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "cannot help");
}

#[test]
fn stream_empty_delta_does_not_suppress_done_payload() {
    let frames = vec![
        frame(
            "response.output_item.added",
            json!({"output_index": 0, "item": {"type": "message", "id": "msg_1"}}),
        ),
        frame(
            "response.output_text.delta",
            json!({"output_index": 0, "delta": ""}),
        ),
        frame(
            "response.output_text.done",
            json!({"output_index": 0, "text": "real payload"}),
        ),
        frame(
            "response.completed",
            json!({"response": {"status": "completed"}}),
        ),
    ];
    let events = decode_frames(&frames);
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "real payload");
}

#[test]
fn stream_item_done_only_function_call_keeps_identity_and_arguments() {
    let frames = vec![
        frame(
            "response.output_item.done",
            json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_7", "name": "lookup",
                "arguments": "{\"a\":1}"}}),
        ),
        frame(
            "response.completed",
            json!({"response": {"status": "completed"}}),
        ),
    ];
    let events = decode_frames(&frames);

    assert!(matches!(
        &events[0],
        StreamEvent::PartStarted { index: 0, kind: PartKind::ToolCall }
    ));
    assert!(matches!(
        &events[1],
        StreamEvent::ToolCallStarted { index: 0, id, name }
            if id == "call_7" && name == "lookup"
    ));
    let args: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolCallDelta { arguments, .. } => Some(arguments.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(args, "{\"a\":1}");
    assert!(events.iter().any(|e| matches!(e, StreamEvent::PartFinished { index: 0 })));
}

#[test]
fn stream_reasoning_done_only_keeps_text() {
    let frames = vec![
        frame(
            "response.output_item.done",
            json!({"output_index": 0, "item": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought"}]}}),
        ),
        frame(
            "response.completed",
            json!({"response": {"status": "completed"}}),
        ),
    ];
    let events = decode_frames(&frames);

    assert!(matches!(
        &events[0],
        StreamEvent::PartStarted { index: 0, kind: PartKind::Reasoning }
    ));
    let reasoning: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "thought");
    assert!(events.iter().any(|e| matches!(e, StreamEvent::PartFinished { index: 0 })));
}
