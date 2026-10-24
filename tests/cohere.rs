//! Original fixtures for the Cohere v2 protocol family (chat, embed,
//! rerank) and the v1 classify surface. Payloads are hand-authored from
//! the wire contract, not copied from reference projects.

mod common;

use std::sync::Arc;

use a4mp::*;
use common::{MockTransport, Response, body_json, collect};
use serde_json::json;

// The cohere modules are not yet declared in the crate's mod.rs files;
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
    pub mod cohere {
        pub use crate::cohere_codec::*;
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
#[path = "../src/protocols/cohere.rs"]
mod cohere_codec;
#[allow(dead_code)]
#[path = "../src/providers/cohere.rs"]
mod cohere_provider;

use cohere_provider::CohereAdapter;

fn adapter(transport: MockTransport) -> CohereAdapter {
    let target = ServiceTarget::new(
        "cohere",
        Protocol::Other("cohere".into()),
        url::Url::parse("https://api.cohere.com").unwrap(),
    );
    CohereAdapter::new(
        target,
        AuthKind::ApiKey,
        Arc::new(transport),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new(
            "cohere-test-key",
        )))),
    )
    .unwrap()
}

#[tokio::test]
async fn chat_request_maps_messages_tools_sampling_and_extensions() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "id": "rsp_1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}]
            },
            "finish_reason": "COMPLETE",
            "usage": {
                "billed_units": {"input_tokens": 14, "output_tokens": 3},
                "tokens": {"input_tokens": 15, "output_tokens": 4}
            }
        }),
    ));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "command-r-plus".into(),
        ..Default::default()
    };
    req.messages.push(Message::text(Role::System, "be brief"));
    req.messages.push(Message::text(Role::User, "hi"));
    req.messages.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentPart::Text { text: "checking".into() },
            ContentPart::ToolCall(ToolCall {
                id: "tc_1".into(),
                name: "web_search".into(),
                arguments: json!({"q": "rust"}),
            }),
        ],
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![ContentPart::ToolResult(ToolResult {
            call_id: "tc_1".into(),
            content: vec![ContentPart::Text { text: "42 hits".into() }],
            is_error: false,
        })],
    });
    req.tools.push(ToolDef {
        name: "web_search".into(),
        description: "search the web".into(),
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        strict: false,
    });
    req.tool_choice = Some(ToolChoice::Required);
    req.sampling.temperature = Some(0.5);
    req.sampling.top_p = Some(0.25);
    req.sampling.max_output_tokens = Some(128);
    req.sampling.stop = vec!["END".into()];
    req.extensions.insert("seed".into(), json!(42));

    let out = a.chat(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v2/chat");
    assert_eq!(
        req.headers.get("authorization").unwrap(),
        "Bearer cohere-test-key"
    );
    let b = body_json(req);
    assert_eq!(b["model"], "command-r-plus");
    assert_eq!(b["messages"][0], json!({"role": "system", "content": "be brief"}));
    assert_eq!(b["messages"][1], json!({"role": "user", "content": "hi"}));
    assert_eq!(b["messages"][2]["role"], "assistant");
    assert_eq!(b["messages"][2]["content"], "checking");
    assert_eq!(
        b["messages"][2]["tool_calls"][0],
        json!({
            "id": "tc_1",
            "type": "function",
            "function": {"name": "web_search", "arguments": "{\"q\":\"rust\"}"}
        })
    );
    assert_eq!(
        b["messages"][3],
        json!({"role": "tool", "tool_call_id": "tc_1", "content": "42 hits"})
    );
    assert_eq!(b["tools"][0]["function"]["name"], "web_search");
    assert_eq!(b["tool_choice"], "REQUIRED");
    assert_eq!(b["temperature"], 0.5);
    assert_eq!(b["p"], 0.25);
    assert_eq!(b["max_tokens"], 128);
    assert_eq!(b["stop_sequences"], json!(["END"]));
    assert_eq!(b["seed"], 42);

    assert!(matches!(out.stop, Some(StopReason::Complete)));
    // `tokens.*` wins over `billed_units.*` when both are present.
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(15));
    assert_eq!(usage.output_tokens, Some(4));
    match &out.parts[0] {
        ContentPart::Text { text } => assert_eq!(text, "done"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_response_decodes_tool_plan_and_tool_calls() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "id": "rsp_2",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_plan": "call search then summarize",
                "tool_calls": [{
                    "id": "tc_7",
                    "type": "function",
                    "function": {"name": "summarize", "arguments": "{\"doc\":9}"}
                }]
            },
            "finish_reason": "TOOL_CALLS",
            "usage": {"billed_units": {"input_tokens": 20, "output_tokens": 8}}
        }),
    ));
    let a = adapter(t);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    assert!(matches!(out.stop, Some(StopReason::ToolUse)));
    // `billed_units` supplies the counts when `tokens` is absent.
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(20));
    assert_eq!(usage.output_tokens, Some(8));

    assert_eq!(out.parts.len(), 2);
    match &out.parts[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.protocol, Protocol::Other("cohere".into()));
            assert_eq!(rb.summary.as_deref(), Some("call search then summarize"));
            assert_eq!(rb.opaque, json!({"tool_plan": "call search then summarize"}));
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
    match &out.parts[1] {
        ContentPart::ToolCall(tc) => {
            assert_eq!(tc.id, "tc_7");
            assert_eq!(tc.name, "summarize");
            assert_eq!(tc.arguments, json!({"doc": 9}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_rejects_foreign_reasoning_replay() {
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

#[tokio::test]
async fn chat_stream_decodes_sse_parts_tools_and_usage() {
    let sse = concat!(
        "data: {\"id\":\"r1\",\"type\":\"message-start\"}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"content-start\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"\"}}}}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"Hel\"}}}}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"lo\"}}}}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"content-end\",\"index\":0}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"tool-plan-delta\",\"index\":1,\"delta\":{\"message\":{\"tool_plan\":\"search \"}}}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"tool-plan-delta\",\"index\":1,\"delta\":{\"message\":{\"tool_plan\":\"first\"}}}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"tool-call-start\",\"index\":2,\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"tc_9\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"q\\\"\"}}}}}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"tool-call-delta\",\"index\":2,\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\":\\\"docs\\\"}\"}}}}}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"tool-call-end\",\"index\":2}\n\n",
        "data: {\"id\":\"r1\",\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"TOOL_CALLS\",\"usage\":{\"billed_units\":{\"input_tokens\":11,\"output_tokens\":6}}}}\n\n"
    );
    let t = MockTransport::new();
    // Split across arbitrary byte boundaries mid-frame.
    let a_at = sse.len() / 3;
    let b_at = sse.len() * 2 / 3;
    t.push(Response::Sse(vec![
        &sse[..a_at],
        &sse[a_at..b_at],
        &sse[b_at..],
    ]));
    let a = adapter(t.clone());
    let stream = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    let events = collect(stream).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(body_json(req)["stream"], true);

    let mut started = Vec::new();
    let mut finished_parts = Vec::new();
    let mut text = String::new();
    let mut plan = String::new();
    let mut args = String::new();
    let mut usage_input = None;
    let mut finished = false;
    for e in &events {
        match e {
            StreamEvent::PartStarted { index, kind } => started.push((*index, *kind)),
            StreamEvent::PartFinished { index } => finished_parts.push(*index),
            StreamEvent::TextDelta { text: d, .. } => text.push_str(d),
            StreamEvent::ReasoningDelta { text: d, .. } => plan.push_str(d),
            StreamEvent::ToolCallStarted { index, id, name } => {
                assert_eq!(*index, 2);
                assert_eq!(id, "tc_9");
                assert_eq!(name, "web_search");
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                assert_eq!(*index, 2);
                args.push_str(arguments);
            }
            StreamEvent::Usage(u) => usage_input = u.input_tokens,
            StreamEvent::Finished { stop } => {
                finished = matches!(stop, Some(StopReason::ToolUse))
            }
            _ => {}
        }
    }
    assert_eq!(
        started,
        [(0, PartKind::Text), (1, PartKind::Reasoning)]
    );
    assert_eq!(finished_parts, [0, 2]);
    assert_eq!(text, "Hello");
    assert_eq!(plan, "search first");
    assert_eq!(args, "{\"q\":\"docs\"}");
    assert_eq!(usage_input, Some(11));
    assert!(finished);
}

#[tokio::test]
async fn embed_request_maps_texts_and_meta_usage() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "id": "emb_1",
            "response_type": "embeddings_floats",
            "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
            "meta": {
                "api_version": {"version": "2"},
                "billed_units": {"input_tokens": 5}
            }
        }),
    ));
    let a = adapter(t.clone());

    let mut req = EmbedRequest {
        model: "embed-v4".into(),
        ..Default::default()
    };
    req.inputs.push(EmbedInput::Text("alpha".into()));
    req.inputs.push(EmbedInput::Text("beta".into()));
    req.dimensions = Some(256);
    req.extensions
        .insert("input_type".into(), json!("search_query"));
    req.extensions.insert("truncate".into(), json!("END"));

    let out = a.embed(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v2/embed");
    let b = body_json(req);
    assert_eq!(b["model"], "embed-v4");
    assert_eq!(b["texts"], json!(["alpha", "beta"]));
    // Caller's extension wins over the text default.
    assert_eq!(b["input_type"], "search_query");
    assert_eq!(b["embedding_types"], json!(["float"]));
    assert_eq!(b["output_dimension"], 256);
    assert_eq!(b["truncate"], "END");

    assert_eq!(out.embeddings.len(), 2);
    assert_eq!(out.embeddings[0], vec![0.1, 0.2]);
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(5));
    assert_eq!(usage.extensions["api_version"]["version"], "2");
}

#[tokio::test]
async fn embed_maps_media_inputs_to_images() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "embeddings": {"float": [[0.5, 0.6]]},
            "meta": {"billed_units": {"image_tokens": 2}}
        }),
    ));
    let a = adapter(t.clone());

    let mut req = EmbedRequest {
        model: "embed-v4".into(),
        ..Default::default()
    };
    req.inputs.push(EmbedInput::Media(MediaSource::Inline {
        media_type: "image/png".into(),
        data: bytes::Bytes::from_static(&[1, 2, 3]),
    }));

    let out = a.embed(req, &RequestControl::default()).await.unwrap();

    let b = body_json(&t.recorded()[0]);
    assert_eq!(b["input_type"], "image");
    assert_eq!(b["images"], json!(["data:image/png;base64,AQID"]));
    assert!(b.get("texts").is_none());

    // Non-token billed units stay visible through usage extensions.
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, None);
    assert_eq!(usage.extensions["billed_units"]["image_tokens"], 2);
}

#[tokio::test]
async fn rerank_request_and_hits_preserve_document_echoes() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "id": "rr_1",
            "results": [
                {"index": 1, "relevance_score": 0.91, "document": {"text": "d1"}},
                {"index": 2, "relevance_score": 0.42, "document": {"text": "d2"}}
            ],
            "meta": {
                "api_version": {"version": "2"},
                "billed_units": {"search_units": 1}
            }
        }),
    ));
    let a = adapter(t.clone());

    let mut req = RerankRequest {
        model: "rerank-v3".into(),
        query: "rust async".into(),
        documents: vec!["d0".into(), "d1".into(), "d2".into()],
        top_n: Some(2),
        extensions: Default::default(),
    };
    req.extensions
        .insert("return_documents".into(), json!(true));

    let out = a.rerank(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v2/rerank");
    let b = body_json(req);
    assert_eq!(b["model"], "rerank-v3");
    assert_eq!(b["query"], "rust async");
    assert_eq!(b["documents"], json!(["d0", "d1", "d2"]));
    assert_eq!(b["top_n"], 2);
    assert_eq!(b["return_documents"], true);

    assert_eq!(out.results.len(), 2);
    assert_eq!(out.results[0].index, 1);
    assert_eq!(out.results[0].score, 0.91);
    assert_eq!(out.results[1].index, 2);
    // Per-hit fields with no canonical slot stay verbatim in extensions.
    assert_eq!(
        out.extensions["results"][0]["document"]["text"],
        "d1"
    );
    let usage = out.usage.unwrap();
    assert_eq!(usage.extensions["billed_units"]["search_units"], 1);
}

#[tokio::test]
async fn classify_roundtrips_inputs_examples_and_confidences() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "id": "cls_1",
            "classifications": [
                {
                    "id": "c1",
                    "input": "a great day",
                    "prediction": "positive",
                    "confidence": 0.98,
                    "labels": {
                        "positive": {"confidence": 0.98},
                        "negative": {"confidence": 0.02}
                    }
                },
                {
                    "id": "c2",
                    "input": "terrible news",
                    "prediction": "negative",
                    "confidence": 0.87,
                    "labels": {
                        "positive": {"confidence": 0.13},
                        "negative": {"confidence": 0.87}
                    }
                }
            ]
        }),
    ));
    let a = adapter(t.clone());

    let mut req = ClassifyRequest {
        model: "classify-small".into(),
        inputs: vec!["a great day".into(), "terrible news".into()],
        labels: vec![],
        extensions: Default::default(),
    };
    req.extensions.insert(
        "examples".into(),
        json!([
            {"text": "sunny", "label": "positive"},
            {"text": "rain", "label": "negative"}
        ]),
    );

    let out = a.classify(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    // Classify lives on the v1 surface even for v2-capable targets.
    assert_eq!(req.url.path(), "/v1/classify");
    let b = body_json(req);
    assert_eq!(b["model"], "classify-small");
    assert_eq!(b["inputs"], json!(["a great day", "terrible news"]));
    assert_eq!(b["examples"][1]["label"], "negative");

    assert_eq!(out.predictions.len(), 2);
    assert_eq!(out.predictions[0].input, 0);
    assert_eq!(out.predictions[0].label, "positive");
    assert_eq!(out.predictions[0].confidence, Some(0.98));
    assert_eq!(out.predictions[1].input, 1);
    assert_eq!(out.predictions[1].label, "negative");
    // Per-label confidence maps have no canonical slot; the raw list
    // stays visible in extensions.
    assert_eq!(
        out.extensions["classifications"][0]["labels"]["negative"]["confidence"],
        0.02
    );
}

#[tokio::test]
async fn classify_rejects_closed_label_lists() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    let req = ClassifyRequest {
        model: "classify-small".into(),
        inputs: vec!["x".into()],
        labels: vec!["positive".into(), "negative".into()],
        extensions: Default::default(),
    };
    let err = a.classify(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(t.recorded().is_empty());
}

#[tokio::test]
async fn provider_error_maps_message_and_status() {
    let t = MockTransport::new();
    t.push(Response::Json(
        429,
        json!({"message": "you are being rate limited"}),
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
            assert_eq!(status, Some(429));
            assert_eq!(message, "you are being rate limited");
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[test]
fn adapter_rejects_foreign_protocols() {
    let target = ServiceTarget::new(
        "not-cohere",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://api.cohere.com").unwrap(),
    );
    let result = CohereAdapter::new(
        target,
        AuthKind::ApiKey,
        Arc::new(MockTransport::new()),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new("k")))),
    );
    match result {
        Err(Error::Unsupported(_)) => {}
        _ => panic!("expected Unsupported"),
    }
}
