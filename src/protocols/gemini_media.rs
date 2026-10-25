//! Gemini media wire codecs: the Imagen-style `:predict` image surface
//! and the Veo-style `:predictLongRunning` long-running-operation (LRO)
//! surface for video generation.
//!
//! Pure functions; no transport here. Original implementation against
//! the public `v1beta` wire contract.

use serde_json::{Map, Value, json};

use crate::api::*;
use crate::protocols::base64_decode;
use crate::protocols::gemini_generate::provider_error;

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

/// Parse a buffered JSON body; `what` names the surface in protocol
/// errors. The `error` key is *not* consumed here — each decoder decides
/// whether it is a hard provider error or a failed job result.
fn parse(body: &[u8], what: &str) -> Result<Value, Error> {
    serde_json::from_slice(body).map_err(|e| Error::UpstreamProtocol(format!("{what}: {e}")))
}

/// Fields not claimed by the canonical mapping stay inspectable verbatim.
fn unclaimed(v: &Value, claimed: &[&str]) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                out.insert(k.clone(), val.clone());
            }
        }
    }
    out
}

/// Merge media-request extensions into `instances[0]`/`parameters`. An
/// `instance` extension must be an object and merges into the single
/// request instance (per-instance knobs — a starting frame rides
/// `instance.image` in the wire's own shape); every other extension key
/// lands flat inside `parameters`. Extensions merge last and may
/// explicitly override canonical fields.
fn merge_extensions(
    instance: &mut Map<String, Value>,
    parameters: &mut Map<String, Value>,
    extensions: &Extensions,
) -> Result<(), Error> {
    for (k, v) in extensions {
        if k == "instance" {
            let Value::Object(fields) = v else {
                return Err(Error::Unsupported(
                    "extension `instance` must be a JSON object".into(),
                ));
            };
            for (ik, iv) in fields {
                instance.insert(ik.clone(), iv.clone());
            }
        } else {
            parameters.insert(k.clone(), v.clone());
        }
    }
    Ok(())
}

/// `{instances:[instance]}` plus `parameters` when it carries anything.
fn predict_body(instance: Map<String, Value>, parameters: Map<String, Value>) -> Value {
    let mut body = Map::new();
    body.insert("instances".into(), json!([Value::Object(instance)]));
    if !parameters.is_empty() {
        body.insert("parameters".into(), parameters.into());
    }
    body.into()
}

// ---------------------------------------------------------------------------
// Images — POST v1beta/models/{model}:predict
// ---------------------------------------------------------------------------

/// Build an Imagen-style predict request. `prompt` is the single
/// instance's prompt; `count` → `parameters.sampleCount`; `size` →
/// `parameters.aspectRatio` (the provider aspect string, verbatim).
/// Extensions merge per `merge_extensions` (`negativePrompt`, `seed`,
/// `personGeneration`, `outputOptions`, `addWatermark`, …).
pub fn build_image_request(req: &ImageRequest) -> Result<Value, Error> {
    let mut instance = Map::new();
    instance.insert("prompt".into(), req.prompt.clone().into());
    let mut parameters = Map::new();
    if let Some(count) = req.count {
        parameters.insert("sampleCount".into(), count.into());
    }
    if let Some(size) = &req.size {
        parameters.insert("aspectRatio".into(), size.clone().into());
    }
    merge_extensions(&mut instance, &mut parameters, &req.extensions)?;
    Ok(predict_body(instance, parameters))
}

/// Decode a predict response: `predictions[].bytesBase64Encoded` →
/// inline media carrying the prediction's `mimeType`. Per-prediction
/// fields outside those two keys stay inspectable under
/// `extensions["predictions"]`; unclaimed top-level fields ride
/// `extensions` verbatim.
pub fn decode_image_output(body: &[u8]) -> Result<MediaOutput, Error> {
    let v = parse(body, "predict response")?;
    if v.get("error").filter(|e| !e.is_null()).is_some() {
        return Err(provider_error(None, &v));
    }
    let predictions = v["predictions"]
        .as_array()
        .ok_or_else(|| Error::UpstreamProtocol("predict response missing predictions[]".into()))?;
    let mut media = Vec::with_capacity(predictions.len());
    let mut extras = Vec::with_capacity(predictions.len());
    let mut any_extra = false;
    for p in predictions {
        let b64 = p["bytesBase64Encoded"]
            .as_str()
            .ok_or_else(|| Error::UpstreamProtocol("prediction carries no bytesBase64Encoded".into()))?;
        let media_type = p["mimeType"]
            .as_str()
            .unwrap_or("application/octet-stream")
            .to_owned();
        media.push(MediaSource::Inline {
            media_type,
            data: base64_decode(b64)
                .map_err(|_| Error::UpstreamProtocol("invalid base64 in prediction".into()))?,
        });
        let extra = unclaimed(p, &["bytesBase64Encoded", "mimeType"]);
        any_extra |= !extra.is_empty();
        extras.push(Value::Object(extra));
    }
    let mut extensions = unclaimed(&v, &["predictions"]);
    if any_extra {
        extensions.insert("predictions".into(), extras.into());
    }
    Ok(MediaOutput {
        media,
        text: Vec::new(),
        usage: None,
        extensions,
    })
}

// ---------------------------------------------------------------------------
// Video — POST v1beta/models/{model}:predictLongRunning
// ---------------------------------------------------------------------------

/// Build a Veo-style predictLongRunning request. `duration_secs` →
/// `parameters.durationSeconds` (the wire field is integral: whole
/// values travel as integers, fractional hints stay fractional for the
/// provider to approximate). Extensions merge per `merge_extensions`
/// (`aspectRatio`, `negativePrompt`, `resolution`, `personGeneration`,
/// `sampleCount`, …); a starting frame rides
/// `extensions["instance"]["image"]` in the wire's own shape (an
/// `inlineData` blob, `{bytesBase64Encoded, mimeType}` or a uri ref).
pub fn build_video_request(req: &VideoRequest) -> Result<Value, Error> {
    let mut instance = Map::new();
    instance.insert("prompt".into(), req.prompt.clone().into());
    let mut parameters = Map::new();
    if let Some(d) = req.duration_secs {
        let seconds = if d.fract() == 0.0 {
            json!(d as i64)
        } else {
            json!(d)
        };
        parameters.insert("durationSeconds".into(), seconds);
    }
    merge_extensions(&mut instance, &mut parameters, &req.extensions)?;
    Ok(predict_body(instance, parameters))
}

/// Decode a predictLongRunning submit response into a resumable
/// `JobHandle`: the operation `name` (`models/{model}/operations/{id}`)
/// is the job id; `metadata` (when the provider reports any) rides
/// `state` as the opaque, credential-free resume token.
pub fn decode_operation_handle(target: &ServiceTarget, body: &[u8]) -> Result<JobHandle, Error> {
    let v = parse(body, "operation submit response")?;
    if v.get("error").filter(|e| !e.is_null()).is_some() {
        return Err(provider_error(None, &v));
    }
    let name = v["name"]
        .as_str()
        .ok_or_else(|| Error::UpstreamProtocol("operation response missing name".into()))?;
    Ok(JobHandle {
        provider: target.provider.clone(),
        protocol: target.protocol.clone(),
        id: name.to_owned(),
        state: v.get("metadata").cloned().unwrap_or_else(|| json!({})),
    })
}

/// Decode a polled operation into `JobState`: `error`
/// (google.rpc.Status shape) → `Failed`; `done` → `Succeeded` — video
/// `uri` samples become `RemoteFile`s while inline samples and every
/// response detail stay native inside `JobResult::output`; absent or
/// false `done` → `Running`.
pub fn decode_operation_state(body: &[u8]) -> Result<JobState, Error> {
    let v = parse(body, "operation poll response")?;
    if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
        let code = err["status"]
            .as_str()
            .or_else(|| err["code"].as_str())
            .map(str::to_owned)
            .or_else(|| err["code"].as_i64().map(|c| c.to_string()));
        return Ok(JobState::Failed {
            code,
            message: err["message"]
                .as_str()
                .unwrap_or("operation failed")
                .to_owned(),
        });
    }
    if v["done"].as_bool() != Some(true) {
        return Ok(JobState::Running);
    }
    let response = v
        .get("response")
        .filter(|r| !r.is_null())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut files = Vec::new();
    if let Some(samples) = response["generateVideoResponse"]["generatedSamples"].as_array() {
        for sample in samples {
            if let Some(uri) = sample["video"]["uri"].as_str() {
                files.push(RemoteFile {
                    id: uri.to_owned(),
                    purpose: Some("generated_video".into()),
                    url: Some(uri.to_owned()),
                });
            }
        }
    }
    Ok(JobState::Succeeded(JobResult {
        output: response,
        files,
    }))
}
