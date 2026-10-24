//! Original fixtures for the Gemini generateContent protocol family.
//! Payloads are hand-authored from the v1beta wire contract, not copied
//! from reference projects.

mod common;

use std::sync::Arc;

use a4mp::protocols::gemini_generate as gg;
use a4mp::*;
use common::{MockTransport, Response, body_json, collect};
use serde_json::json;

fn adapter_with(transport: MockTransport, auth: AuthKind, cred: Credential) -> GeminiAdapter {
    let target = ServiceTarget::new(
        "test-gemini",
        Protocol::GeminiGenerateContent,
        url::Url::parse("https://generativelanguage.test").unwrap(),
    );
    GeminiAdapter::new(target, auth, Arc::new(transport), Arc::new(StaticCredential(cred)))
        .unwrap()
}

fn adapter(transport: MockTransport) -> GeminiAdapter {
    adapter_with(
        transport,
        AuthKind::ApiKey,
        Credential::ApiKey(Secret::new("key-abc")),
    )
}

fn ok_body() -> serde_json::Value {
    json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP",
        }]
    })
}

#[tokio::test]
async fn chat_request_maps_contents_system_tools_config_and_extensions() {
    let t = MockTransport::new();
    t.push(Response::Json(200, ok_body()));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "gemini-test-1".into(),
        ..Default::default()
    };
    // System messages hoist into one systemInstruction, parts merged.
    req.messages.push(Message::text(Role::System, "be terse"));
    req.messages.push(Message::text(Role::System, "reply in json"));
    req.messages.push(Message::text(Role::User, "hello"));
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::GeminiGenerateContent,
                opaque: json!({"thought": true, "text": "checking"}),
                summary: Some("checking".into()),
            }),
            ContentPart::ToolCall(ToolCall {
                id: "fc_0".into(),
                name: "lookup".into(),
                arguments: json!({"q": "x"}),
            }),
            ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::GeminiGenerateContent,
                opaque: json!({"thoughtSignature": "sig-9"}),
                summary: None,
            }),
        ],
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::ToolResult(ToolResult {
            call_id: "fc_0".into(),
            content: vec![ContentPart::Text { text: "42".into() }],
            is_error: false,
        })],
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::Media {
            kind: MediaKind::Image,
            source: MediaSource::Inline {
                media_type: "image/png".into(),
                data: bytes::Bytes::from_static(b"hi"),
            },
            detail: None,
        }],
    });
    req.tools.push(ToolDef {
        name: "lookup".into(),
        description: "search".into(),
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        strict: false,
    });
    req.tool_choice = Some(ToolChoice::Named("lookup".into()));
    req.sampling.temperature = Some(0.25);
    req.sampling.top_p = Some(0.5);
    req.sampling.max_output_tokens = Some(128);
    req.sampling.stop = vec!["END".into()];
    req.extensions.insert(
        "safetySettings".into(),
        json!([{"category": "HARM_CATEGORY_TEST", "threshold": "BLOCK_NONE"}]),
    );

    let out = a.chat(req, &RequestControl::default()).await.unwrap();

    let rec = &t.recorded()[0];
    // Model id rides the URL path, not the body.
    assert_eq!(rec.url.path(), "/v1beta/models/gemini-test-1:generateContent");
    assert_eq!(rec.headers.get("x-goog-api-key").unwrap(), "key-abc");
    assert!(rec.headers.get("authorization").is_none());
    let b = body_json(rec);
    assert!(b.get("model").is_none());
    assert_eq!(
        b["systemInstruction"]["parts"],
        json!([{"text": "be terse"}, {"text": "reply in json"}])
    );
    let contents = b["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 4);
    assert_eq!(
        contents[0],
        json!({"role": "user", "parts": [{"text": "hello"}]})
    );
    // Assistant → model; thought parts replay verbatim, a signature-only
    // reasoning block merges onto the functionCall part it decorated.
    assert_eq!(contents[1]["role"], "model");
    assert_eq!(
        contents[1]["parts"],
        json!([
            {"thought": true, "text": "checking"},
            {"functionCall": {"name": "lookup", "args": {"q": "x"}}, "thoughtSignature": "sig-9"},
        ])
    );
    // functionResponse correlates by function name, resolved from the
    // earlier functionCall via the call id.
    assert_eq!(
        contents[2],
        json!({"role": "user", "parts": [{"functionResponse": {
            "name": "lookup", "response": {"result": "42"}}}]})
    );
    assert_eq!(
        contents[3]["parts"][0],
        json!({"inlineData": {"mimeType": "image/png", "data": "aGk="}})
    );
    assert_eq!(
        b["tools"][0]["functionDeclarations"][0],
        json!({
            "name": "lookup",
            "description": "search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        })
    );
    assert_eq!(b["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
    assert_eq!(
        b["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"],
        json!(["lookup"])
    );
    assert_eq!(b["generationConfig"]["temperature"], 0.25);
    assert_eq!(b["generationConfig"]["topP"], 0.5);
    assert_eq!(b["generationConfig"]["maxOutputTokens"], 128);
    assert_eq!(b["generationConfig"]["stopSequences"], json!(["END"]));
    assert_eq!(b["safetySettings"][0]["threshold"], "BLOCK_NONE");
    assert!(matches!(out.stop, Some(StopReason::Complete)));
}

#[test]
fn decode_response_maps_parts_usage_finish_and_extensions() {
    let body = json!({
        "candidates": [{
            "content": {"role": "model", "parts": [
                {"thought": true, "text": "pondering"},
                {"text": "the answer"},
                {"functionCall": {"name": "lookup", "args": {"q": "x"}},
                 "thoughtSignature": "sig-1"},
                {"inlineData": {"mimeType": "image/png", "data": "aGk="}},
            ]},
            "finishReason": "STOP",
            "safetyRatings": [{"category": "HARM_CATEGORY_TEST", "probability": "NEGLIGIBLE"}],
        }],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "totalTokenCount": 18,
            "thoughtsTokenCount": 4,
            "cachedContentTokenCount": 3,
            "toolUsePromptTokenCount": 2,
        },
        "modelVersion": "gemini-test-1-001",
    });
    let out = gg::decode_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(out.parts.len(), 5);
    match &out.parts[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.protocol, Protocol::GeminiGenerateContent);
            assert_eq!(rb.summary.as_deref(), Some("pondering"));
            assert_eq!(rb.opaque["thought"], true);
            assert_eq!(rb.opaque["text"], "pondering");
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
    match &out.parts[1] {
        ContentPart::Text { text } => assert_eq!(text, "the answer"),
        other => panic!("expected text, got {other:?}"),
    }
    match &out.parts[2] {
        ContentPart::ToolCall(tc) => {
            // No call id on the wire: a stable one is synthesized.
            assert_eq!(tc.id, "fc_0");
            assert_eq!(tc.name, "lookup");
            assert_eq!(tc.arguments, json!({"q": "x"}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    match &out.parts[3] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.opaque, json!({"thoughtSignature": "sig-1"}));
            assert_eq!(rb.summary, None);
        }
        other => panic!("expected signature reasoning, got {other:?}"),
    }
    match &out.parts[4] {
        ContentPart::Media { kind, source, .. } => {
            assert_eq!(*kind, MediaKind::Image);
            match source {
                MediaSource::Inline { media_type, data } => {
                    assert_eq!(media_type, "image/png");
                    assert_eq!(data.as_ref(), b"hi");
                }
                other => panic!("expected inline media, got {other:?}"),
            }
        }
        other => panic!("expected media, got {other:?}"),
    }
    // STOP plus a functionCall means the model paused for tool use.
    assert!(matches!(out.stop, Some(StopReason::ToolUse)));
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(11));
    assert_eq!(usage.output_tokens, Some(7));
    assert_eq!(usage.extensions["thoughtsTokenCount"], 4);
    assert_eq!(usage.extensions["cachedContentTokenCount"], 3);
    assert_eq!(usage.extensions["toolUsePromptTokenCount"], 2);
    assert_eq!(usage.extensions["totalTokenCount"], 18);
    assert_eq!(out.extensions["modelVersion"], "gemini-test-1-001");
    assert_eq!(
        out.extensions["candidate"]["safetyRatings"][0]["probability"],
        "NEGLIGIBLE"
    );
}

#[test]
fn decode_response_maps_finish_reasons() {
    for (wire, want) in [
        ("STOP", StopReason::Complete),
        ("MAX_TOKENS", StopReason::Length),
        ("SAFETY", StopReason::Filtered),
        ("BLOCKLIST", StopReason::Filtered),
        ("PROHIBITED_CONTENT", StopReason::Filtered),
        ("RECITATION", StopReason::Filtered),
    ] {
        let body = json!({
            "candidates": [{
                "content": {"parts": [{"text": "x"}]},
                "finishReason": wire,
            }]
        });
        let out = gg::decode_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(out.stop, Some(want), "finishReason {wire}");
    }
    let body = json!({
        "candidates": [{
            "content": {"parts": []},
            "finishReason": "MALFORMED_FUNCTION_CALL",
        }]
    });
    let out = gg::decode_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(matches!(out.stop, Some(StopReason::Other(ref s)) if s == "MALFORMED_FUNCTION_CALL"));
}

#[test]
fn decode_response_block_reason_is_provider_error() {
    let body = json!({
        "promptFeedback": {
            "blockReason": "SAFETY",
            "blockReasonMessage": "prompt tripped a filter",
        },
        "usageMetadata": {"promptTokenCount": 5},
    });
    match gg::decode_response(&serde_json::to_vec(&body).unwrap()) {
        Err(Error::Provider {
            status,
            code,
            message,
            retry,
            ..
        }) => {
            assert_eq!(status, None);
            assert_eq!(code.as_deref(), Some("SAFETY"));
            assert_eq!(message, "prompt tripped a filter");
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_stream_decodes_response_frames_incrementally() {
    // Each SSE frame carries a complete GenerateContentResponse object.
    let sse = concat!(
        "data: {\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[{\"thought\":true,\"text\":\"let me \"}]}}]}\n\n",
        "data: {\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[{\"thought\":true,\"text\":\"think\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"},{\"text\":\"lo\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"q\":\"x\"}},\"thoughtSignature\":\"sig-7\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":8,\"candidatesTokenCount\":6,\"thoughtsTokenCount\":3}}\n\n"
    );
    let t = MockTransport::new();
    // Split mid-frame to prove incremental decoding.
    let at = sse.len() / 2;
    t.push(Response::Sse(vec![&sse[..at], &sse[at..]]));
    let a = adapter(t.clone());
    let req = ChatRequest {
        model: "gemini-test-1".into(),
        ..Default::default()
    };
    let stream = a.chat_stream(req, &RequestControl::default()).await.unwrap();
    let events = collect(stream).await.unwrap();

    let rec = &t.recorded()[0];
    assert_eq!(
        rec.url.path(),
        "/v1beta/models/gemini-test-1:streamGenerateContent"
    );
    assert_eq!(rec.url.query(), Some("alt=sse"));

    let mut reasoning = String::new();
    let mut text = String::new();
    let mut args = String::new();
    let mut log = Vec::new();
    for e in &events {
        match e {
            StreamEvent::PartStarted { kind, .. } => log.push(format!("start:{kind:?}")),
            StreamEvent::ReasoningDelta { text: d, .. } => {
                reasoning.push_str(d);
                log.push("reason".into());
            }
            StreamEvent::TextDelta { text: d, .. } => text.push_str(d),
            StreamEvent::ToolCallStarted { id, name, .. } => {
                assert_eq!(id, "fc_0");
                assert_eq!(name, "lookup");
                log.push("tool".into());
            }
            StreamEvent::ToolCallDelta { arguments, .. } => args.push_str(arguments),
            StreamEvent::PartFinished { .. } => log.push("fin".into()),
            StreamEvent::Native { value, .. } => {
                log.push(format!("native:{}", value["thoughtSignature"].as_str().unwrap_or("?")))
            }
            StreamEvent::Usage(u) => {
                assert_eq!(u.input_tokens, Some(8));
                assert_eq!(u.output_tokens, Some(6));
                assert_eq!(u.extensions["thoughtsTokenCount"], 3);
                log.push("usage".into());
            }
            StreamEvent::Finished { stop } => {
                assert!(matches!(stop, Some(StopReason::ToolUse)));
                log.push("done".into());
            }
            _ => {}
        }
    }
    assert_eq!(reasoning, "let me think");
    assert_eq!(text, "Hello");
    assert_eq!(args, "{\"q\":\"x\"}");
    assert_eq!(
        log,
        [
            "start:Reasoning",
            "reason",
            "reason",
            "fin",
            "start:Text",
            "fin",
            "tool",
            "native:sig-7",
            "fin",
            "usage",
            "done",
        ]
    );
}

#[tokio::test]
async fn bearer_auth_uses_authorization_header() {
    let t = MockTransport::new();
    t.push(Response::Json(200, ok_body()));
    let a = adapter_with(
        t.clone(),
        AuthKind::Bearer,
        Credential::Bearer {
            token: Secret::new("tok-1"),
            expires_at: None,
        },
    );
    a.chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    let rec = &t.recorded()[0];
    assert_eq!(rec.headers.get("authorization").unwrap(), "Bearer tok-1");
    assert!(rec.headers.get("x-goog-api-key").is_none());
}

#[tokio::test]
async fn http_error_maps_gemini_error_envelope() {
    let t = MockTransport::new();
    t.push(Response::Json(
        400,
        json!({"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}),
    ));
    let a = adapter(t);
    match a.chat(ChatRequest::default(), &RequestControl::default()).await {
        Err(Error::Provider {
            status,
            code,
            message,
            retry,
            ..
        }) => {
            assert_eq!(status, Some(400));
            assert_eq!(code.as_deref(), Some("INVALID_ARGUMENT"));
            assert_eq!(message, "bad request");
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }

    let t = MockTransport::new();
    t.push(Response::Text(500, "upstream exploded"));
    let a = adapter(t);
    match a.chat(ChatRequest::default(), &RequestControl::default()).await {
        Err(Error::Provider {
            status, retry, ..
        }) => {
            assert_eq!(status, Some(500));
            assert_eq!(retry, RetrySafety::Unsafe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_http_error_drains_body_into_provider_error() {
    let t = MockTransport::new();
    t.push(Response::Json(
        429,
        json!({"error": {"code": 429, "message": "quota exhausted", "status": "RESOURCE_EXHAUSTED"}}),
    ));
    let a = adapter(t);
    match a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
    {
        Err(Error::Provider {
            status,
            code,
            retry,
            ..
        }) => {
            assert_eq!(status, Some(429));
            assert_eq!(code.as_deref(), Some("RESOURCE_EXHAUSTED"));
            assert_eq!(retry, RetrySafety::Safe);
        }
        Err(other) => panic!("expected provider error, got {other:?}"),
        Ok(_) => panic!("expected provider error, got a stream"),
    }
}

#[tokio::test]
async fn foreign_reasoning_replay_is_rejected() {
    let t = MockTransport::new();
    t.push(Response::Json(200, ok_body()));
    let a = adapter(t.clone());
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
    // Rejected locally: the transport never saw a request.
    assert!(t.recorded().is_empty());
}

#[tokio::test]
async fn strict_tool_flag_has_no_gemini_encoding() {
    let t = MockTransport::new();
    t.push(Response::Json(200, ok_body()));
    let a = adapter(t);
    let mut req = ChatRequest::default();
    req.messages.push(Message::text(Role::User, "hi"));
    req.tools.push(ToolDef {
        name: "lookup".into(),
        description: "search".into(),
        parameters: json!({"type": "object"}),
        strict: true,
    });
    let err = a.chat(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[tokio::test]
async fn wrong_protocol_target_is_rejected() {
    let t = MockTransport::new();
    let target = ServiceTarget::new(
        "test-other",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://example.test").unwrap(),
    );
    let res = GeminiAdapter::new(
        target,
        AuthKind::None,
        Arc::new(t),
        Arc::new(NoCredentials),
    );
    assert!(matches!(res, Err(Error::Unsupported(_))));
}
