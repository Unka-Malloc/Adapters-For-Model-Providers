//! Original fixtures for the OpenAI non-chat operation families:
//! embeddings, moderations, image generation, audio (speech +
//! transcription) and batch jobs. Payloads are hand-authored from the
//! wire contract, not copied from reference projects.

#[allow(dead_code)]
mod common;

use std::sync::Arc;

use a4mp::*;
use common::{MockTransport, Response, body_json};
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
            token: Secret::new("sk-test"),
            expires_at: None,
        })),
    )
    .unwrap()
}

fn handle(id: &str) -> JobHandle {
    JobHandle {
        provider: ProviderId::new("test-vendor"),
        protocol: Protocol::OpenAiChatCompletions,
        id: id.into(),
        state: json!({}),
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_maps_inputs_dimensions_and_extensions() {
    let t = MockTransport::new();
    // Rows arrive out of order; the canonical output follows `index`.
    t.push(Response::Json(
        200,
        json!({
            "object": "list",
            "model": "vendor-embed",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.25, -0.5]},
                {"object": "embedding", "index": 0, "embedding": [0.75, 1.5]}
            ],
            "usage": {"prompt_tokens": 9, "total_tokens": 9}
        }),
    ));
    let a = adapter(t.clone());

    let mut req = EmbedRequest {
        model: "vendor-embed".into(),
        inputs: vec![
            EmbedInput::Text("first".into()),
            EmbedInput::Text("second".into()),
        ],
        dimensions: Some(2),
        ..Default::default()
    };
    req.extensions
        .insert("encoding_format".into(), json!("float"));

    let out = a.embed(req, &RequestControl::default()).await.unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v1/embeddings");
    assert_eq!(req.headers.get("authorization").unwrap(), "Bearer sk-test");
    let b = body_json(req);
    assert_eq!(b["model"], "vendor-embed");
    assert_eq!(b["input"], json!(["first", "second"]));
    assert_eq!(b["dimensions"], 2);
    assert_eq!(b["encoding_format"], "float");

    assert_eq!(out.embeddings, vec![vec![0.75, 1.5], vec![0.25, -0.5]]);
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(9));
    assert_eq!(usage.output_tokens, None);
    // prompt_tokens/total_tokens are consumed; nothing left over.
    assert!(usage.extensions.is_empty());
}

#[tokio::test]
async fn embed_single_input_travels_as_bare_string() {
    let t = MockTransport::new();
    t.push(Response::Json(200, json!({"data": [{"index":0,"embedding":[1.0]}]})));
    let a = adapter(t.clone());
    let req = EmbedRequest {
        model: "m".into(),
        inputs: vec![EmbedInput::Text("lone".into())],
        ..Default::default()
    };
    a.embed(req, &RequestControl::default()).await.unwrap();
    assert_eq!(body_json(&t.recorded()[0])["input"], "lone");
}

#[tokio::test]
async fn embed_decodes_base64_rows_when_requested() {
    let t = MockTransport::new();
    // f32 little-endian: [1.0, 0.5] and [2.0].
    t.push(Response::Json(
        200,
        json!({
            "data": [
                {"index": 0, "embedding": "AACAPwAAAD8="},
                {"index": 1, "embedding": "AAAAQA=="}
            ]
        }),
    ));
    let a = adapter(t);
    let mut req = EmbedRequest {
        model: "m".into(),
        inputs: vec![
            EmbedInput::Text("a".into()),
            EmbedInput::Text("b".into()),
        ],
        ..Default::default()
    };
    req.extensions
        .insert("encoding_format".into(), json!("base64"));
    let out = a.embed(req, &RequestControl::default()).await.unwrap();
    assert_eq!(out.embeddings, vec![vec![1.0, 0.5], vec![2.0]]);
}

#[tokio::test]
async fn embed_rejects_media_inputs() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    let req = EmbedRequest {
        model: "m".into(),
        inputs: vec![EmbedInput::Media(MediaSource::Url(
            "https://x.test/pic.png".into(),
        ))],
        ..Default::default()
    };
    let err = a.embed(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(t.recorded().is_empty(), "no request should be sent");
}

// ---------------------------------------------------------------------------
// Moderations
// ---------------------------------------------------------------------------

#[tokio::test]
async fn moderate_merges_category_flags_scores_and_input_types() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "id": "modr-1",
            "model": "omni-moderation",
            "vendor_flag": "pilot",
            "results": [{
                "flagged": true,
                "categories": {
                    "harassment": true,
                    "self-harm": false
                },
                "category_scores": {
                    "harassment": 0.91,
                    "self-harm": 0.002
                },
                "category_applied_input_types": {
                    "harassment": ["text"]
                }
            }]
        }),
    ));
    let a = adapter(t.clone());

    let out = a
        .moderate(
            ModerateRequest {
                model: "omni-moderation".into(),
                inputs: vec!["say hi".into(), "say bye".into()],
                extensions: Default::default(),
            },
            &RequestControl::default(),
        )
        .await
        .unwrap();

    let b = body_json(&t.recorded()[0]);
    assert_eq!(b["model"], "omni-moderation");
    assert_eq!(b["input"], json!(["say hi", "say bye"]));

    assert_eq!(out.results.len(), 1);
    let r = &out.results[0];
    assert!(r.flagged);
    assert_eq!(r.categories["harassment"]["flagged"], true);
    assert_eq!(r.categories["harassment"]["score"], 0.91);
    assert_eq!(
        r.categories["harassment"]["applied_input_types"],
        json!(["text"])
    );
    assert_eq!(r.categories["self-harm"]["flagged"], false);
    // Consumed keys leave the map; unknown ones stay as extensions.
    assert_eq!(out.extensions["vendor_flag"], "pilot");
    assert!(out.extensions.get("id").is_none());
    assert!(out.extensions.get("results").is_none());
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

#[tokio::test]
async fn image_requests_b64_and_decodes_inline_media() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "created": 1700000000,
            "data": [{
                "b64_json": "iVBORw==",
                "revised_prompt": "a fixed prompt"
            }],
            "usage": {"input_tokens": 12, "output_tokens": 1, "total_tokens": 13}
        }),
    ));
    let a = adapter(t.clone());

    let mut req = ImageRequest {
        model: "paint-2".into(),
        prompt: "a fox".into(),
        count: Some(1),
        size: Some("1024x1024".into()),
        extensions: Default::default(),
    };
    req.extensions.insert("quality".into(), json!("hd"));
    req.extensions.insert("style".into(), json!("vivid"));

    let out = a.image(req, &RequestControl::default()).await.unwrap();

    let b = body_json(&t.recorded()[0]);
    assert_eq!(t.recorded()[0].url.path(), "/v1/images/generations");
    assert_eq!(b["model"], "paint-2");
    assert_eq!(b["prompt"], "a fox");
    assert_eq!(b["n"], 1);
    assert_eq!(b["size"], "1024x1024");
    assert_eq!(b["response_format"], "b64_json");
    assert_eq!(b["quality"], "hd");
    assert_eq!(b["style"], "vivid");

    assert_eq!(out.media.len(), 1);
    match &out.media[0] {
        MediaSource::Inline { media_type, data } => {
            assert_eq!(media_type, "image/png");
            assert_eq!(data.as_ref(), &[0x89, 0x50, 0x4e, 0x47]);
        }
        other => panic!("expected inline media, got {other:?}"),
    }
    assert_eq!(out.text, ["a fixed prompt"]);
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(12));
    assert_eq!(usage.output_tokens, Some(1));
}

#[tokio::test]
async fn image_extension_can_request_url_output() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"data": [{"url": "https://cdn.test/img.png"}]}),
    ));
    let a = adapter(t.clone());
    let mut req = ImageRequest {
        model: "paint-2".into(),
        prompt: "a fox".into(),
        count: None,
        size: None,
        extensions: Default::default(),
    };
    req.extensions
        .insert("response_format".into(), json!("url"));
    let out = a.image(req, &RequestControl::default()).await.unwrap();
    assert_eq!(body_json(&t.recorded()[0])["response_format"], "url");
    match &out.media[0] {
        MediaSource::Url(u) => assert_eq!(u, "https://cdn.test/img.png"),
        other => panic!("expected url media, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

#[tokio::test]
async fn audio_speech_maps_fields_and_passes_bytes_through() {
    let t = MockTransport::new();
    t.push(Response::Text(200, "raw-mp3-bytes"));
    let a = adapter(t.clone());

    let mut extensions = Extensions::new();
    extensions.insert("speed".into(), json!(1.25));
    let out = a
        .audio(
            AudioRequest::Synthesize {
                model: "voice-1".into(),
                input: "hello there".into(),
                voice: Some("alloy".into()),
                format: Some("mp3".into()),
                extensions,
            },
            &RequestControl::default(),
        )
        .await
        .unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v1/audio/speech");
    let b = body_json(req);
    assert_eq!(b["model"], "voice-1");
    assert_eq!(b["input"], "hello there");
    assert_eq!(b["voice"], "alloy");
    assert_eq!(b["response_format"], "mp3");
    assert_eq!(b["speed"], 1.25);

    // Raw audio bytes survive undecoded.
    assert_eq!(out.media.len(), 1);
    match &out.media[0] {
        MediaSource::Inline { media_type, data } => {
            assert_eq!(media_type, "audio/mpeg");
            assert_eq!(data.as_ref(), b"raw-mp3-bytes");
        }
        other => panic!("expected inline media, got {other:?}"),
    }
}

#[tokio::test]
async fn audio_transcribe_uploads_multipart_and_decodes_text() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"text": "hello world", "language": "en", "duration": 1.4}),
    ));
    let a = adapter(t.clone());

    let mut extensions = Extensions::new();
    extensions.insert("prompt".into(), json!("glossary words"));
    let out = a
        .audio(
            AudioRequest::Transcribe {
                model: "ear-2".into(),
                audio: MediaSource::Inline {
                    media_type: "audio/mpeg".into(),
                    data: bytes::Bytes::from_static(b"FAKE-AUDIO"),
                },
                language: Some("en".into()),
                extensions,
            },
            &RequestControl::default(),
        )
        .await
        .unwrap();

    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v1/audio/transcriptions");
    let content_type = req
        .headers
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let boundary = content_type
        .strip_prefix("multipart/form-data; boundary=")
        .expect("multipart content type");
    let body = String::from_utf8_lossy(&req.body).into_owned();
    // Fields then the file part, framed by the boundary.
    assert!(body.contains(&format!("--{boundary}\r\n")));
    assert!(body.contains("Content-Disposition: form-data; name=\"model\"\r\n\r\near-2\r\n"));
    assert!(body.contains("Content-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n"));
    assert!(body.contains(
        "Content-Disposition: form-data; name=\"prompt\"\r\n\r\nglossary words\r\n"
    ));
    assert!(body.contains(
        "Content-Disposition: form-data; name=\"file\"; filename=\"input.mp3\"\r\nContent-Type: audio/mpeg\r\n\r\n"
    ));
    assert!(body.contains("FAKE-AUDIO"));
    assert!(body.ends_with(&format!("--{boundary}--\r\n")));

    assert_eq!(out.text, ["hello world"]);
    assert_eq!(out.extensions["language"], "en");
    assert_eq!(out.extensions["duration"], 1.4);
    assert!(out.media.is_empty());
}

#[tokio::test]
async fn audio_transcribe_text_format_passes_document_through() {
    let t = MockTransport::new();
    t.push(Response::Text(200, "transcript as plain text"));
    let a = adapter(t.clone());
    let mut extensions = Extensions::new();
    extensions.insert("response_format".into(), json!("text"));
    let out = a
        .audio(
            AudioRequest::Transcribe {
                model: "ear-2".into(),
                audio: MediaSource::Inline {
                    media_type: "audio/wav".into(),
                    data: bytes::Bytes::from_static(b"RIFF...."),
                },
                language: None,
                extensions,
            },
            &RequestControl::default(),
        )
        .await
        .unwrap();
    assert_eq!(out.text, ["transcript as plain text"]);
    let body = String::from_utf8_lossy(&t.recorded()[0].body).into_owned();
    assert!(body.contains("filename=\"input.wav\""));
}

#[tokio::test]
async fn audio_transcribe_rejects_url_source() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    let err = a
        .audio(
            AudioRequest::Transcribe {
                model: "ear-2".into(),
                audio: MediaSource::Url("https://x.test/clip.mp3".into()),
                language: None,
                extensions: Extensions::new(),
            },
            &RequestControl::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(t.recorded().is_empty());
}

// ---------------------------------------------------------------------------
// Errors + compatible targets
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_http_429_maps_to_provider_error() {
    let t = MockTransport::new();
    t.push(Response::Json(
        429,
        json!({"error": {"message": "slow down", "type": "rate_limit_error"}}),
    ));
    let a = adapter(t);
    let err = a
        .embed(
            EmbedRequest {
                model: "m".into(),
                inputs: vec![EmbedInput::Text("x".into())],
                ..Default::default()
            },
            &RequestControl::default(),
        )
        .await
        .unwrap_err();
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
async fn openai_compatible_target_reuses_same_paths() {
    // A DeepSeek-style target speaks the same wire shapes.
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"data": [{"index":0,"embedding":[0.5]}]}),
    ));
    let target = ServiceTarget::new(
        "deepseek",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://api.deepseek.test/v1").unwrap(),
    );
    let a = OpenAiAdapter::new(
        target,
        AuthKind::Bearer,
        Arc::new(t.clone()),
        Arc::new(StaticCredential(Credential::Bearer {
            token: Secret::new("sk-ds"),
            expires_at: None,
        })),
    )
    .unwrap();
    let out = a
        .embed(
            EmbedRequest {
                model: "deepseek-embed".into(),
                inputs: vec![EmbedInput::Text("hi".into())],
                ..Default::default()
            },
            &RequestControl::default(),
        )
        .await
        .unwrap();
    assert_eq!(out.embeddings, vec![vec![0.5]]);
    let req = &t.recorded()[0];
    assert_eq!(req.url.path(), "/v1/embeddings");
    assert_eq!(req.headers.get("authorization").unwrap(), "Bearer sk-ds");
}

// ---------------------------------------------------------------------------
// Batch jobs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn submit_job_uploads_jsonl_file_then_creates_batch() {
    let t = MockTransport::new();
    t.push(Response::Json(200, json!({"id": "file-9", "purpose": "batch"})));
    t.push(Response::Json(
        200,
        json!({"id": "batch_7", "object": "batch", "status": "validating",
               "endpoint": "/v1/chat/completions"}),
    ));
    let a = adapter(t.clone());

    let mut req = ChatRequest {
        model: "vendor-model".into(),
        ..Default::default()
    };
    req.messages.push(Message::text(Role::User, "hi"));
    let h = a
        .submit_job(JobOperation::Chat(req), &RequestControl::default())
        .await
        .unwrap();

    // Request 1: multipart upload to /files.
    let upload = &t.recorded()[0];
    assert_eq!(upload.url.path(), "/v1/files");
    let content_type = upload
        .headers
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let boundary = content_type
        .strip_prefix("multipart/form-data; boundary=")
        .unwrap();
    let body = String::from_utf8_lossy(&upload.body).into_owned();
    assert!(body.contains("Content-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"));
    assert!(body.contains("filename=\"input.jsonl\""));
    // The JSONL line carries the endpoint and the encoded chat body.
    assert!(body.contains(r#""method":"POST""#));
    assert!(body.contains(r#""url":"/v1/chat/completions""#));
    assert!(body.contains(r#""model":"vendor-model""#));
    assert!(body.ends_with(&format!("--{boundary}--\r\n")));

    // Request 2: batch creation.
    let create = &t.recorded()[1];
    assert_eq!(create.url.path(), "/v1/batches");
    let b = body_json(create);
    assert_eq!(b["input_file_id"], "file-9");
    assert_eq!(b["endpoint"], "/v1/chat/completions");
    assert_eq!(b["completion_window"], "24h");

    assert_eq!(h.id, "batch_7");
    assert_eq!(h.provider.as_str(), "test-vendor");
    assert_eq!(h.state["endpoint"], "/v1/chat/completions");
    assert_eq!(h.state["input_file_id"], "file-9");
}

#[tokio::test]
async fn job_state_maps_lifecycle_and_fetches_output() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    let h = handle("batch_7");
    let ctrl = RequestControl::default();

    t.push(Response::Json(200, json!({"id":"batch_7","status":"validating"})));
    t.push(Response::Json(200, json!({"id":"batch_7","status":"in_progress"})));
    t.push(Response::Json(
        200,
        json!({"id":"batch_7","status":"completed","output_file_id":"file-out"}),
    ));
    t.push(Response::Text(
        200,
        "{\"custom_id\":\"a4mp-0\",\"response\":{\"status_code\":200,\"body\":{\"choices\":[]}}}\n",
    ));
    t.push(Response::Json(200, json!({"id":"batch_7","status":"cancelled"})));

    assert!(matches!(
        a.job_state(&h, &ctrl).await.unwrap(),
        JobState::Submitted
    ));
    assert!(matches!(
        a.job_state(&h, &ctrl).await.unwrap(),
        JobState::Running
    ));
    match a.job_state(&h, &ctrl).await.unwrap() {
        JobState::Succeeded(result) => {
            assert_eq!(result.output[0]["custom_id"], "a4mp-0");
            assert_eq!(result.output[0]["response"]["status_code"], 200);
            assert_eq!(result.files[0].id, "file-out");
            assert_eq!(result.files[0].purpose.as_deref(), Some("batch_output"));
        }
        other => panic!("expected Succeeded, got {other:?}"),
    }
    // The completion poll fetched the output file.
    let fetch = &t.recorded()[3];
    assert_eq!(fetch.url.path(), "/v1/files/file-out/content");
    assert_eq!(fetch.method, http::Method::GET);
    assert!(matches!(
        a.job_state(&h, &ctrl).await.unwrap(),
        JobState::Cancelled
    ));
}

#[tokio::test]
async fn job_state_failed_surfaces_batch_errors() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "id": "batch_8",
            "status": "failed",
            "errors": {"data": [{"code": "invalid_file", "message": "bad jsonl line 1"}]}
        }),
    ));
    let a = adapter(t);
    match a
        .job_state(&handle("batch_8"), &RequestControl::default())
        .await
        .unwrap()
    {
        JobState::Failed { code, message } => {
            assert_eq!(code.as_deref(), Some("invalid_file"));
            assert_eq!(message, "bad jsonl line 1");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_job_posts_cancel_endpoint() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"id": "batch_7", "status": "cancelling"}),
    ));
    let a = adapter(t.clone());
    a.cancel_job(&handle("batch_7"), &RequestControl::default())
        .await
        .unwrap();
    let req = &t.recorded()[0];
    assert_eq!(req.method, http::Method::POST);
    assert_eq!(req.url.path(), "/v1/batches/batch_7/cancel");
}

#[tokio::test]
async fn submit_job_rejects_unsupported_operation() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    let err = a
        .submit_job(
            JobOperation::Image(ImageRequest {
                model: "paint".into(),
                prompt: "x".into(),
                count: None,
                size: None,
                extensions: Extensions::new(),
            }),
            &RequestControl::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(t.recorded().is_empty());
}

// ---------------------------------------------------------------------------
// Still-unsupported families stay explicit
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unsupported_families_report_cleanly() {
    let t = MockTransport::new();
    let a = adapter(t);
    let err = a
        .video(
            VideoRequest {
                model: "v".into(),
                prompt: "x".into(),
                duration_secs: None,
                extensions: Extensions::new(),
            },
            &RequestControl::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    let err = a
        .rerank(
            RerankRequest {
                model: "r".into(),
                query: "q".into(),
                documents: vec![],
                top_n: None,
                extensions: Extensions::new(),
            },
            &RequestControl::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}
