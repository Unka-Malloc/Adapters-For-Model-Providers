//! Original fixtures for the Gemini media surfaces: the Imagen-style
//! `:predict` image operation and the Veo-style `:predictLongRunning`
//! job lifecycle (submit, poll, cancel). Payloads are hand-authored
//! from the v1beta wire contract, not copied from reference projects.

#[allow(dead_code)]
mod common;

use std::sync::Arc;

use a4mp::*;
use common::{MockTransport, Response, body_json};
use serde_json::json;

use a4mp::protocols::gemini_media as gemini_media_codec;
use a4mp::providers::GeminiAdapter;

fn adapter(transport: MockTransport) -> GeminiAdapter {
    let target = ServiceTarget::new(
        "test-gemini",
        Protocol::GeminiGenerateContent,
        url::Url::parse("https://generativelanguage.test").unwrap(),
    );
    GeminiAdapter::new(
        target,
        AuthKind::ApiKey,
        Arc::new(transport),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new(
            "key-abc",
        )))),
    )
    .unwrap()
}

fn handle(id: &str) -> JobHandle {
    JobHandle {
        provider: ProviderId::new("test-gemini"),
        protocol: Protocol::GeminiGenerateContent,
        id: id.into(),
        state: json!({}),
    }
}

// ---------------------------------------------------------------------------
// Images — :predict
// ---------------------------------------------------------------------------

#[tokio::test]
async fn image_predict_maps_instance_parameters_and_decodes_media() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "predictions": [
                {"bytesBase64Encoded": "iVBORw==", "mimeType": "image/png"},
                {"bytesBase64Encoded": "aGk=", "mimeType": "image/png",
                 "raiFilteredReason": "none"}
            ],
            "modelVersion": "imagen-3.0-001"
        }),
    ));
    let a = adapter(t.clone());

    let mut req = ImageRequest {
        model: "imagen-3.0-generate".into(),
        prompt: "a lighthouse".into(),
        count: Some(2),
        size: Some("16:9".into()),
        extensions: Extensions::new(),
    };
    req.extensions
        .insert("negativePrompt".into(), json!("blur"));
    req.extensions
        .insert("personGeneration".into(), json!("allow_adult"));
    // Per-instance knobs ride the `instance` extension object.
    req.extensions.insert("instance".into(), json!({"seed": 7}));

    let out = a.image(req, &RequestControl::default()).await.unwrap();

    let rec = &t.recorded()[0];
    assert_eq!(
        rec.url.path(),
        "/v1beta/models/imagen-3.0-generate:predict"
    );
    assert_eq!(rec.headers.get("x-goog-api-key").unwrap(), "key-abc");
    let b = body_json(rec);
    // The prompt is the single instance; `instance` merges into it.
    assert_eq!(
        b["instances"],
        json!([{"prompt": "a lighthouse", "seed": 7}])
    );
    assert_eq!(b["parameters"]["sampleCount"], 2);
    assert_eq!(b["parameters"]["aspectRatio"], "16:9");
    assert_eq!(b["parameters"]["negativePrompt"], "blur");
    assert_eq!(b["parameters"]["personGeneration"], "allow_adult");

    assert_eq!(out.media.len(), 2);
    match &out.media[0] {
        MediaSource::Inline { media_type, data } => {
            assert_eq!(media_type, "image/png");
            assert_eq!(data.as_ref(), &[0x89, 0x50, 0x4e, 0x47]);
        }
        other => panic!("expected inline media, got {other:?}"),
    }
    match &out.media[1] {
        MediaSource::Inline { data, .. } => assert_eq!(data.as_ref(), b"hi"),
        other => panic!("expected inline media, got {other:?}"),
    }
    // Unclaimed wire fields stay inspectable.
    assert_eq!(out.extensions["modelVersion"], "imagen-3.0-001");
    assert_eq!(out.extensions["predictions"][1]["raiFilteredReason"], "none");
}

#[tokio::test]
async fn image_bare_request_omits_parameters() {
    let t = MockTransport::new();
    t.push(Response::Json(200, json!({"predictions": []})));
    let a = adapter(t.clone());
    a.image(
        ImageRequest {
            model: "imagen-3.0-generate".into(),
            prompt: "x".into(),
            count: None,
            size: None,
            extensions: Extensions::new(),
        },
        &RequestControl::default(),
    )
    .await
    .unwrap();
    let b = body_json(&t.recorded()[0]);
    assert_eq!(b["instances"], json!([{"prompt": "x"}]));
    assert!(b.get("parameters").is_none());
}

#[test]
fn decode_image_output_roundtrips_predictions() {
    let body = json!({
        "predictions": [
            {"bytesBase64Encoded": "aGk=", "mimeType": "image/jpeg"},
        ],
    });
    let out =
        gemini_media_codec::decode_image_output(&serde_json::to_vec(&body).unwrap()).unwrap();
    match &out.media[0] {
        MediaSource::Inline { media_type, data } => {
            assert_eq!(media_type, "image/jpeg");
            assert_eq!(data.as_ref(), b"hi");
        }
        other => panic!("expected inline media, got {other:?}"),
    }
    // No extras: the predictions extension stays absent.
    assert!(out.extensions.get("predictions").is_none());
}

#[test]
fn decode_image_output_rejects_missing_bytes() {
    let body = json!({"predictions": [{"mimeType": "image/png"}]});
    match gemini_media_codec::decode_image_output(&serde_json::to_vec(&body).unwrap()) {
        Err(Error::UpstreamProtocol(m)) => assert!(m.contains("bytesBase64Encoded"), "{m}"),
        other => panic!("expected protocol error, got {other:?}"),
    }
}

#[tokio::test]
async fn image_http_error_maps_gemini_error_envelope() {
    let t = MockTransport::new();
    t.push(Response::Json(
        400,
        json!({"error": {"code": 400, "message": "bad prompt", "status": "INVALID_ARGUMENT"}}),
    ));
    let a = adapter(t);
    match a
        .image(
            ImageRequest {
                model: "imagen-3.0-generate".into(),
                prompt: "x".into(),
                count: None,
                size: None,
                extensions: Extensions::new(),
            },
            &RequestControl::default(),
        )
        .await
    {
        Err(Error::Provider {
            status,
            code,
            message,
            retry,
            ..
        }) => {
            assert_eq!(status, Some(400));
            assert_eq!(code.as_deref(), Some("INVALID_ARGUMENT"));
            assert_eq!(message, "bad prompt");
            assert_eq!(retry, RetrySafety::Safe);
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn non_object_instance_extension_is_rejected() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    let mut req = ImageRequest {
        model: "imagen-3.0-generate".into(),
        prompt: "x".into(),
        count: None,
        size: None,
        extensions: Extensions::new(),
    };
    req.extensions.insert("instance".into(), json!("not-an-object"));
    let err = a.image(req, &RequestControl::default()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(t.recorded().is_empty(), "no request should be sent");
}

// ---------------------------------------------------------------------------
// Video — :predictLongRunning via the job contract
// ---------------------------------------------------------------------------

#[tokio::test]
async fn submit_job_posts_predict_long_running_and_returns_handle() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({
            "name": "models/veo-3/operations/op-42",
            "metadata": {"createTime": "2026-01-01T00:00:00Z"}
        }),
    ));
    let a = adapter(t.clone());

    let mut req = VideoRequest {
        model: "veo-3".into(),
        prompt: "waves at dusk".into(),
        duration_secs: Some(8.0),
        extensions: Extensions::new(),
    };
    req.extensions.insert("aspectRatio".into(), json!("9:16"));
    req.extensions.insert("negativePrompt".into(), json!("people"));
    // A starting frame rides `instance.image` in the wire's inlineData
    // blob shape.
    req.extensions.insert(
        "instance".into(),
        json!({"image": {"inlineData": {"mimeType": "image/png", "data": "aGk="}}}),
    );

    let h = a
        .submit_job(JobOperation::Video(req), &RequestControl::default())
        .await
        .unwrap();

    let rec = &t.recorded()[0];
    assert_eq!(rec.method, http::Method::POST);
    assert_eq!(
        rec.url.path(),
        "/v1beta/models/veo-3:predictLongRunning"
    );
    let b = body_json(rec);
    assert_eq!(b["instances"][0]["prompt"], "waves at dusk");
    assert_eq!(
        b["instances"][0]["image"]["inlineData"],
        json!({"mimeType": "image/png", "data": "aGk="})
    );
    // Whole durations travel as integers on this integral wire field.
    assert_eq!(b["parameters"]["durationSeconds"], 8);
    assert_eq!(b["parameters"]["aspectRatio"], "9:16");
    assert_eq!(b["parameters"]["negativePrompt"], "people");

    // The provider operation name is the resumable job id; submit
    // metadata rides the opaque state token.
    assert_eq!(h.provider.as_str(), "test-gemini");
    assert_eq!(h.protocol, Protocol::GeminiGenerateContent);
    assert_eq!(h.id, "models/veo-3/operations/op-42");
    assert_eq!(h.state["createTime"], "2026-01-01T00:00:00Z");
}

#[tokio::test]
async fn job_state_polls_operation_and_maps_lifecycle() {
    let t = MockTransport::new();
    t.push(Response::Json(
        200,
        json!({"name": "models/veo-3/operations/op-42", "done": false}),
    ));
    t.push(Response::Json(
        200,
        json!({
            "name": "models/veo-3/operations/op-42",
            "done": true,
            "response": {
                "@type": "type.googleapis.com/test.GenerateVideoResponse",
                "generateVideoResponse": {
                    "generatedSamples": [
                        {"video": {"uri": "https://files.test/v1beta/files/9:download?alt=media"}},
                        {"video": {"uri": "https://files.test/v1beta/files/8:download?alt=media"}}
                    ],
                    "raiMediaFilteredCount": 0
                }
            }
        }),
    ));
    t.push(Response::Json(
        200,
        json!({
            "name": "models/veo-3/operations/op-43",
            "done": true,
            "error": {"code": 3, "message": "prompt rejected by policy"}
        }),
    ));
    let a = adapter(t.clone());
    let ctrl = RequestControl::default();

    let h = handle("models/veo-3/operations/op-42");
    assert!(matches!(
        a.job_state(&h, &ctrl).await.unwrap(),
        JobState::Running
    ));
    match a.job_state(&h, &ctrl).await.unwrap() {
        JobState::Succeeded(result) => {
            assert_eq!(result.files.len(), 2);
            assert_eq!(
                result.files[0].url.as_deref(),
                Some("https://files.test/v1beta/files/9:download?alt=media")
            );
            assert_eq!(
                result.files[0].id,
                "https://files.test/v1beta/files/9:download?alt=media"
            );
            assert_eq!(
                result.files[0].purpose.as_deref(),
                Some("generated_video")
            );
            // The native response payload stays inspectable verbatim.
            assert_eq!(
                result.output["generateVideoResponse"]["generatedSamples"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            assert_eq!(
                result.output["generateVideoResponse"]["raiMediaFilteredCount"],
                0
            );
        }
        other => panic!("expected Succeeded, got {other:?}"),
    }

    match a
        .job_state(&handle("models/veo-3/operations/op-43"), &ctrl)
        .await
        .unwrap()
    {
        JobState::Failed { code, message } => {
            assert_eq!(code.as_deref(), Some("3"));
            assert_eq!(message, "prompt rejected by policy");
        }
        other => panic!("expected Failed, got {other:?}"),
    }

    // Every poll GETs the operation resource under v1beta.
    let recorded = t.recorded();
    assert_eq!(recorded.len(), 3);
    for rec in &recorded {
        assert_eq!(rec.method, http::Method::GET);
        assert!(
            rec.url
                .path()
                .starts_with("/v1beta/models/veo-3/operations/"),
            "{}",
            rec.url
        );
    }
}

#[tokio::test]
async fn job_state_encodes_operation_name_segments() {
    let t = MockTransport::new();
    t.push(Response::Json(200, json!({"done": false})));
    let a = adapter(t.clone());
    a.job_state(
        &handle("models/veo 3/operations/op?1"),
        &RequestControl::default(),
    )
    .await
    .unwrap();
    // Reserved characters inside one segment are percent-encoded; the
    // `/` separators stay path separators.
    assert_eq!(
        t.recorded()[0].url.path(),
        "/v1beta/models/veo%203/operations/op%3F1"
    );
}

#[tokio::test]
async fn cancel_job_posts_cancel_verb() {
    let t = MockTransport::new();
    t.push(Response::Json(200, json!({})));
    let a = adapter(t.clone());
    a.cancel_job(
        &handle("models/veo-3/operations/op-42"),
        &RequestControl::default(),
    )
    .await
    .unwrap();
    let rec = &t.recorded()[0];
    assert_eq!(rec.method, http::Method::POST);
    assert_eq!(
        rec.url.path(),
        "/v1beta/models/veo-3/operations/op-42:cancel"
    );
}

#[tokio::test]
async fn unsupported_job_operations_and_sync_video_send_nothing() {
    let t = MockTransport::new();
    let a = adapter(t.clone());
    let ctrl = RequestControl::default();

    // Chat, embed and image are synchronous surfaces on Gemini — no LRO
    // exists for them on this adapter.
    let ops = [
        JobOperation::Chat(ChatRequest::default()),
        JobOperation::Embed(EmbedRequest::default()),
        JobOperation::Image(ImageRequest {
            model: "imagen-3.0-generate".into(),
            prompt: "x".into(),
            count: None,
            size: None,
            extensions: Extensions::new(),
        }),
        JobOperation::Custom {
            kind: "anything".into(),
            payload: json!({}),
        },
    ];
    for op in ops {
        let err = a.submit_job(op, &ctrl).await.unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    }

    // The synchronous video contract cannot express an LRO; the job
    // contract is the only route.
    let err = a
        .video(
            VideoRequest {
                model: "veo-3".into(),
                prompt: "x".into(),
                duration_secs: None,
                extensions: Extensions::new(),
            },
            &ctrl,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));

    assert!(t.recorded().is_empty(), "no request should be sent");
}
