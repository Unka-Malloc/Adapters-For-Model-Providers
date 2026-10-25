//! Original fixtures for the Ollama protocol family (`/api/chat`,
//! `/api/embed`, `/api/tags`). Payloads are hand-authored from the wire
//! contract, not copied from reference projects.

mod common;

use std::sync::Arc;

use a4mp::*;
use common::{MockTransport, Response, body_json, collect};
use serde_json::json;

// The ollama modules are not yet declared in the crate's mod.rs files;
// integration wiring lands separately. Compiling them in place here
// resolves their `crate::` paths and exercises the same sources the
// library ships — the shim can be dropped once they are wired.
mod api {
    pub use a4mp::api::*;
}
mod auth {
    pub use a4mp::auth::*;
}
mod effects {
    pub use a4mp::effects::*;
}
mod transport {
    pub use a4mp::transport::*;
}
mod protocols {
    pub mod ollama {
        pub use crate::ollama_codec::*;
    }

    /// Stand-in for the crate-private media helper while the codec is
    /// compiled in place.
    pub(crate) fn base64_encode(data: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
        for chunk in data.chunks(3) {
            let mut n = 0u32;
            for &b in chunk {
                n = n << 8 | b as u32;
            }
            n <<= (3 - chunk.len()) * 8;
            out.push(ALPHABET[(n >> 18) as usize & 63] as char);
            out.push(ALPHABET[(n >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[n as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }
}

#[allow(dead_code)]
#[path = "../src/protocols/ollama.rs"]
mod ollama_codec;
#[allow(dead_code)]
#[path = "../src/providers/ollama.rs"]
mod ollama_provider;

use ollama_codec::NdjsonDecoder;
use ollama_provider::OllamaAdapter;

/// Local Ollama server: no credential material at all.
fn adapter(transport: MockTransport) -> OllamaAdapter {
    let target = ServiceTarget::new(
        "ollama",
        Protocol::Other("ollama".into()),
        url::Url::parse("http://localhost:11434").unwrap(),
    );
    OllamaAdapter::new(
        target,
        AuthKind::None,
        Arc::new(transport),
        Arc::new(NoCredentials),
    )
    .unwrap()
}

#[tokio::test]
async fn chat_request_maps_messages_images_tools_and_options() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "model": "qwen3",
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "sure"},
            "done": true,
            "done_reason": "stop",
            "total_duration": 500,
            "load_duration": 40,
            "prompt_eval_count": 12,
            "prompt_eval_duration": 60,
            "eval_count": 4,
            "eval_duration": 300,
            "context": [7, 8]
        }),
    ));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "qwen3".into(),
        ..Default::default()
    };
    // System stays an ordinary message on this wire — no hoisting.
    req.messages.push(Message::text(Role::System, "be brief"));
    req.messages.push(Message {
        role: Role::User,
        content: vec![
            ContentPart::Text { text: "look at this".into() },
            ContentPart::Media {
                kind: MediaKind::Image,
                source: MediaSource::Inline {
                    media_type: "image/png".into(),
                    data: bytes::Bytes::from_static(&[10, 20, 30]),
                },
                detail: None,
            },
        ],
    });
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentPart::ToolCall(ToolCall {
            id: "ignored-by-ollama".into(),
            name: "get_weather".into(),
            arguments: json!({"city": "Paris"}),
        })],
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::ToolResult(ToolResult {
            call_id: "ignored-by-ollama".into(),
            content: vec![ContentPart::Text { text: "sunny".into() }],
            is_error: false,
        })],
    });
    req.tools.push(ToolDef {
        name: "get_weather".into(),
        description: "weather lookup".into(),
        parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        strict: false,
    });
    req.tool_choice = Some(ToolChoice::Auto);
    req.sampling.temperature = Some(0.5);
    req.sampling.top_p = Some(0.75);
    req.sampling.max_output_tokens = Some(64);
    req.sampling.stop = vec!["\n\n".into()];
    req.extensions.insert("keep_alive".into(), json!("5m"));

    let out = a.chat(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/api/chat");
    // Local servers carry no credential material.
    assert!(req.headers.get("authorization").is_none());
    let b = body_json(req);
    assert_eq!(b["model"], "qwen3");
    // Non-streaming is explicit: the wire defaults to streaming.
    assert_eq!(b["stream"], false);
    assert_eq!(b["messages"][0], json!({"role": "system", "content": "be brief"}));
    assert_eq!(b["messages"][1]["role"], "user");
    assert_eq!(b["messages"][1]["content"], "look at this");
    assert_eq!(b["messages"][1]["images"], json!(["ChQe"]));
    // Arguments travel as a JSON object, not a serialized string.
    assert_eq!(
        b["messages"][2]["tool_calls"][0]["function"]["arguments"],
        json!({"city": "Paris"})
    );
    assert_eq!(
        b["messages"][3],
        json!({"role": "tool", "content": "sunny"})
    );
    assert_eq!(b["tools"][0]["function"]["name"], "get_weather");
    // Sampling lives under `options` with wire-native names.
    assert_eq!(b["options"]["temperature"], 0.5);
    assert_eq!(b["options"]["top_p"], 0.75);
    assert_eq!(b["options"]["num_predict"], 64);
    assert_eq!(b["options"]["stop"], json!(["\n\n"]));
    assert!(b.get("tool_choice").is_none());
    assert_eq!(b["keep_alive"], "5m");

    assert!(matches!(out.stop, Some(StopReason::Complete)));
    match &out.parts[0] {
        ContentPart::Text { text } => assert_eq!(text, "sure"),
        other => panic!("expected text, got {other:?}"),
    }
    // `*_eval_count` fields are the usage surface; durations ride
    // usage extensions, unclaimed wire fields ride output extensions.
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(12));
    assert_eq!(usage.output_tokens, Some(4));
    assert_eq!(usage.extensions["total_duration"], 500);
    assert_eq!(out.extensions["context"], json!([7, 8]));
}

#[tokio::test]
async fn chat_response_decodes_object_tool_calls() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "model": "qwen3",
            "created_at": "2026-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}
                ]
            },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 30,
            "eval_count": 6
        }),
    ));
    let a = adapter(t);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    assert_eq!(out.parts.len(), 1);
    match &out.parts[0] {
        ContentPart::ToolCall(tc) => {
            // Ollama tool calls carry no id; matching is positional.
            assert_eq!(tc.id, "");
            assert_eq!(tc.name, "get_weather");
            assert_eq!(tc.arguments, json!({"city": "Paris", "days": 3}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    assert!(matches!(out.stop, Some(StopReason::Complete)));
}

#[tokio::test]
async fn chat_response_decodes_thinking_into_reasoning() {
    // Live-verified shape: thinking models (qwen3, deepseek-r1) return
    // `message.thinking` alongside `message.content`.
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "model": "qwen3",
            "created_at": "2026-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "pong",
                "thinking": "the user asked to reply exactly"
            },
            "done": true,
            "done_reason": "stop"
        }),
    ));
    let a = adapter(t);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    assert_eq!(out.parts.len(), 2);
    match &out.parts[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.summary.as_deref(), Some("the user asked to reply exactly"));
            assert_eq!(rb.opaque["thinking"], "the user asked to reply exactly");
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
    match &out.parts[1] {
        ContentPart::Text { text } => assert_eq!(text, "pong"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_stream_decodes_thinking_deltas() {
    const BODY: &str = concat!(
        r#"{"model":"q","created_at":"t0","message":{"role":"assistant","thinking":"th"},"done":false}"#,
        "\n",
        r#"{"model":"q","created_at":"t1","message":{"role":"assistant","content":"ok"},"done":false}"#,
        "\n",
        r#"{"model":"q","created_at":"t2","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}"#,
        "\n",
    );
    let t = MockTransport::new();
    t.push(Response::Sse(vec![BODY.into()]));
    let a = adapter(t);
    let st = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    let kinds = collect(st).await.unwrap();
    assert!(kinds.iter().any(|e| matches!(
        e,
        StreamEvent::PartStarted { kind: PartKind::Reasoning, .. }
    )));
    assert!(kinds.iter().any(|e| matches!(
        e,
        StreamEvent::ReasoningDelta { text, .. } if text == "th"
    )));
    assert!(kinds.iter().any(|e| matches!(
        e,
        StreamEvent::TextDelta { text, .. } if text == "ok"
    )));
}

#[tokio::test]
async fn chat_stream_decodes_ndjson_split_mid_line() {
    const L1: &str = r#"{"model":"qwen3","created_at":"t0","message":{"role":"assistant","content":"Hel"},"done":false}"#;
    const L2: &str = r#"{"model":"qwen3","created_at":"t1","message":{"role":"assistant","content":"lo"},"done":false}"#;
    const BODY: &str = concat!(
        r#"{"model":"qwen3","created_at":"t0","message":{"role":"assistant","content":"Hel"},"done":false}"#,
        "\n",
        r#"{"model":"qwen3","created_at":"t1","message":{"role":"assistant","content":"lo"},"done":false}"#,
        "\n",
        r#"{"model":"qwen3","created_at":"t2","message":{"role":"assistant","tool_calls":[{"function":{"name":"lookup","arguments":{"q":"x"}}}]},"done":false}"#,
        "\n",
        r#"{"model":"qwen3","created_at":"t3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":9,"eval_count":3,"total_duration":42}"#,
        "\n"
    );
    // Split inside a JSON object to prove incremental line framing.
    let at = L1.len() + 1 + L2.len() / 2;
    let (chunk_a, chunk_b) = BODY.split_at(at);
    let t = MockTransport::new();
    t.push(Response::Sse(vec![chunk_a, chunk_b]));
    let a = adapter(t.clone());
    let stream = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    let events = collect(stream).await.unwrap();

    assert_eq!(body_json(&t.recorded()[0])["stream"], true);

    let mut text = String::new();
    let mut tool_started = false;
    let mut args = String::new();
    let mut usage = None;
    let mut finished = false;
    for e in &events {
        match e {
            StreamEvent::PartStarted { index, kind } => {
                assert_eq!(*index, 0);
                assert_eq!(*kind, PartKind::Text);
            }
            StreamEvent::TextDelta { text: d, .. } => text.push_str(d),
            StreamEvent::ToolCallStarted { index, name, .. } => {
                assert_eq!(*index, 1);
                assert_eq!(name, "lookup");
                tool_started = true;
            }
            StreamEvent::ToolCallDelta { arguments, .. } => args.push_str(arguments),
            StreamEvent::Usage(u) => usage = Some((u.input_tokens, u.output_tokens)),
            StreamEvent::Finished { stop } => {
                finished = matches!(stop, Some(StopReason::Complete))
            }
            _ => {}
        }
    }
    assert_eq!(text, "Hello");
    assert!(tool_started);
    assert_eq!(args, "{\"q\":\"x\"}");
    assert_eq!(usage, Some((Some(9), Some(3))));
    assert!(finished);
}

#[test]
fn ndjson_decoder_handles_crlf_partial_and_empty_lines() {
    let mut dec = NdjsonDecoder::new();
    // A CRLF-terminated line followed by a partial line in one chunk.
    let lines = dec.feed(b"{\"a\":1}\r\n{\"b\":");
    assert_eq!(lines, ["{\"a\":1}"]);
    // Completing the partial line; an empty line is skipped.
    let lines = dec.feed(b"2}\n\n{\"c\":3}");
    assert_eq!(lines, ["{\"b\":2}"]);
    // Trailing unterminated line flushes on finish.
    let lines = dec.finish();
    assert_eq!(lines, ["{\"c\":3}"]);
    assert!(dec.finish().is_empty());
}

#[tokio::test]
async fn models_lists_local_tags() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "models": [
                {
                    "name": "qwen3:latest",
                    "model": "qwen3:latest",
                    "size": 5200000000u64,
                    "digest": "sha256:abc123",
                    "modified_at": "2026-01-01T00:00:00Z",
                    "details": {
                        "family": "qwen3",
                        "parameter_size": "8B",
                        "quantization_level": "Q4_K_M"
                    }
                },
                {
                    "name": "nomic-embed-text:latest",
                    "model": "nomic-embed-text:latest",
                    "size": 274000000u64,
                    "digest": "sha256:def456",
                    "modified_at": "2026-01-02T00:00:00Z",
                    "details": {"family": "nomic-bert", "parameter_size": "137M"}
                }
            ]
        }),
    ));
    let a = adapter(t.clone());
    let models = a.models(&RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.method, http::Method::GET);
    assert_eq!(req.url.path(), "/api/tags");

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "qwen3:latest");
    assert_eq!(models[0].extensions["details"]["family"], "qwen3");
    assert_eq!(models[1].id, "nomic-embed-text:latest");
}

#[tokio::test]
async fn embed_request_maps_input_array_and_usage() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "total_duration": 88,
            "load_duration": 30,
            "prompt_eval_count": 6
        }),
    ));
    let a = adapter(t.clone());

    let mut req = EmbedRequest {
        model: "nomic-embed-text".into(),
        ..Default::default()
    };
    req.inputs.push(EmbedInput::Text("first".into()));
    req.inputs.push(EmbedInput::Text("second".into()));

    let out = a.embed(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/api/embed");
    let b = body_json(req);
    assert_eq!(b["model"], "nomic-embed-text");
    assert_eq!(b["input"], json!(["first", "second"]));

    assert_eq!(out.embeddings.len(), 2);
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(6));
    assert_eq!(usage.extensions["load_duration"], 30);
}

#[tokio::test]
async fn chat_rejects_url_images_and_named_tool_choice() {
    let t = MockTransport::new();
    let a = adapter(t.clone());

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

    let mut req = ChatRequest::default();
    req.tool_choice = Some(ToolChoice::Required);
    let err = a.chat(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));

    assert!(t.recorded().is_empty());
}

#[tokio::test]
async fn provider_error_maps_error_string_and_status() {
    let t = MockTransport::new();
    t.push(Response::Json(
        404,
        json!({"error": "model 'missing:latest' not found"}),
    ));
    let a = adapter(t);
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
            assert_eq!(status, Some(404));
            assert!(message.contains("missing:latest"));
            // Pre-flight rejections never bill: a retry is safe.
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn bearer_credential_sets_authorization_header() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "message": {"role": "assistant", "content": "ok"},
            "done": true
        }),
    ));
    let target = ServiceTarget::new(
        "ollama",
        Protocol::Other("ollama".into()),
        url::Url::parse("https://ollama.com").unwrap(),
    );
    let a = OllamaAdapter::new(
        target,
        AuthKind::Bearer,
        Arc::new(t.clone()),
        Arc::new(StaticCredential(Credential::Bearer {
            token: Secret::new("cloud-token"),
            expires_at: None,
        })),
    )
    .unwrap();

    a.chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    assert_eq!(
        t.recorded()[0].headers.get("authorization").unwrap(),
        "Bearer cloud-token"
    );
}

#[test]
fn adapter_rejects_foreign_protocols() {
    let target = ServiceTarget::new(
        "not-ollama",
        Protocol::AnthropicMessages,
        url::Url::parse("http://localhost:11434").unwrap(),
    );
    let result = OllamaAdapter::new(
        target,
        AuthKind::None,
        Arc::new(MockTransport::new()),
        Arc::new(NoCredentials),
    );
    match result {
        Err(Error::Unsupported(_)) => {}
        _ => panic!("expected Unsupported"),
    }
}
