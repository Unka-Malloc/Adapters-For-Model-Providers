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
