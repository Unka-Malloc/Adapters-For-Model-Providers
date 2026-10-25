//! OpenAI data-plane wire codecs: embeddings, moderations and the
//! batch-job file format. Pure functions; no transport here. Every
//! OpenAI-compatible target shares this codec.
//!
//! Original implementation against the public wire contract.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use crate::api::*;
use crate::protocols::base64_decode;

use super::openai_chat::{decode_usage, provider_error};

/// Top-level response fields not consumed by a codec become output
/// `extensions` verbatim.
pub(crate) fn unclaimed(v: &Value, claimed: &[&str]) -> Map<String, Value> {
    let mut extensions = Map::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                extensions.insert(k.clone(), val.clone());
            }
        }
    }
    extensions
}

fn parse_body(body: &[u8], what: &str) -> Result<Value, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::UpstreamProtocol(format!("{what} response: {e}")))?;
    if let Some(err) = v.get("error") {
        return Err(provider_error(None, err));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Embeddings — POST /embeddings
// ---------------------------------------------------------------------------

/// Build an embeddings request body. OpenAI embeddings carry text only:
/// `EmbedInput::Media` is rejected rather than silently dropped.
/// `extensions` merge last (encoding_format, user, …).
pub fn build_embed_request(req: &EmbedRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    let mut inputs = Vec::with_capacity(req.inputs.len());
    for input in &req.inputs {
        match input {
            EmbedInput::Text(text) => inputs.push(Value::from(text.as_str())),
            EmbedInput::Media(_) => {
                return Err(Error::Unsupported(
                    "OpenAI embeddings accept text inputs only".into(),
                ));
            }
        }
    }
    // A single input travels as a bare string — both shapes are legal
    // and the string form is the most widely compatible.
    body.insert(
        "input".into(),
        match inputs.len() {
            1 => inputs.pop().unwrap(),
            _ => Value::Array(inputs),
        },
    );
    if let Some(d) = req.dimensions {
        body.insert("dimensions".into(), d.into());
    }
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// Decode an embeddings response. Rows are ordered by their `index`;
/// an `encoding_format:"base64"` answer decodes little-endian f32s.
pub fn decode_embed_output(body: &[u8]) -> Result<EmbedOutput, Error> {
    let v = parse_body(body, "embeddings")?;
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::UpstreamProtocol("embeddings response missing data[]".into()))?;
    let mut rows = Vec::with_capacity(data.len());
    for (position, item) in data.iter().enumerate() {
        let index = item["index"].as_u64().unwrap_or(position as u64);
        rows.push((index, decode_embedding(&item["embedding"])?));
    }
    rows.sort_by_key(|(index, _)| *index);
    Ok(EmbedOutput {
        embeddings: rows.into_iter().map(|(_, e)| e).collect(),
        usage: v.get("usage").filter(|u| !u.is_null()).map(decode_usage),
        extensions: unclaimed(&v, &["object", "model", "data", "usage"]),
    })
}

/// One embedding row: a number array, or a base64 little-endian f32
/// blob when `encoding_format:"base64"` was requested.
fn decode_embedding(v: &Value) -> Result<Vec<f32>, Error> {
    match v {
        Value::Array(items) => items
            .iter()
            .map(|n| {
                n.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| Error::UpstreamProtocol("embedding entry is not a number".into()))
            })
            .collect(),
        Value::String(s) => {
            let bytes = base64_decode(s)?;
            if bytes.len() % 4 != 0 {
                return Err(Error::UpstreamProtocol(
                    "base64 embedding is not f32-aligned".into(),
                ));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        _ => Err(Error::UpstreamProtocol(
            "embedding row is neither a number array nor base64".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Moderations — POST /moderations
// ---------------------------------------------------------------------------

/// Build a moderations request body. `extensions` merge last.
pub fn build_moderate_request(req: &ModerateRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    let mut inputs: Vec<Value> = req.inputs.iter().map(|s| s.as_str().into()).collect();
    body.insert(
        "input".into(),
        match inputs.len() {
            1 => inputs.pop().unwrap(),
            _ => Value::Array(inputs),
        },
    );
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// Decode a moderations response. Each result merges `categories`
/// (flags), `category_scores` (confidences) and
/// `category_applied_input_types` into one per-category entry so the
/// provider-native detail survives in the canonical map.
pub fn decode_moderate_output(body: &[u8]) -> Result<ModerateOutput, Error> {
    let v = parse_body(body, "moderations")?;
    let results = v
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::UpstreamProtocol("moderations response missing results[]".into()))?;
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        let mut names = BTreeSet::new();
        for key in ["categories", "category_scores", "category_applied_input_types"] {
            if let Some(obj) = r[key].as_object() {
                names.extend(obj.keys().cloned());
            }
        }
        let mut categories = BTreeMap::new();
        for name in names {
            let mut entry = Map::new();
            if let Some(f) = r["categories"].get(&name).and_then(Value::as_bool) {
                entry.insert("flagged".into(), f.into());
            }
            if let Some(s) = r["category_scores"].get(&name).and_then(Value::as_f64) {
                entry.insert("score".into(), s.into());
            }
            if let Some(t) = r["category_applied_input_types"].get(&name) {
                entry.insert("applied_input_types".into(), t.clone());
            }
            categories.insert(name, entry.into());
        }
        out.push(ModerationResult {
            flagged: r["flagged"].as_bool().unwrap_or(false),
            categories,
        });
    }
    Ok(ModerateOutput {
        results: out,
        usage: None,
        extensions: unclaimed(&v, &["id", "model", "results"]),
    })
}

// ---------------------------------------------------------------------------
// Batch jobs — POST /files + /batches, GET /batches/{id},
//              POST /batches/{id}/cancel, GET /files/{id}/content
// ---------------------------------------------------------------------------

/// One request line of a batch JSONL input file.
pub fn batch_line(custom_id: &str, endpoint: &str, body: Value) -> Result<Vec<u8>, Error> {
    let mut line = serde_json::to_vec(&json!({
        "custom_id": custom_id,
        "method": "POST",
        "url": endpoint,
        "body": body,
    }))
    .map_err(|e| Error::Protocol(format!("batch line encode: {e}")))?;
    line.push(b'\n');
    Ok(line)
}

/// The `id` of a `/files` upload response.
pub fn decode_file_id(body: &[u8]) -> Result<String, Error> {
    let v = parse_body(body, "file upload")?;
    v.get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| Error::UpstreamProtocol("file upload response missing id".into()))
}

/// Batch status snapshot from `POST /batches` or `GET /batches/{id}`.
#[derive(Clone, Debug)]
pub struct BatchInfo {
    pub id: String,
    pub status: String,
    pub output_file_id: Option<String>,
    /// `(code, message)` pairs from `errors.data[]`.
    pub errors: Vec<(Option<String>, String)>,
}

/// Decode a batch object.
pub fn decode_batch(body: &[u8]) -> Result<BatchInfo, Error> {
    let v = parse_body(body, "batch")?;
    let mut errors = Vec::new();
    if let Some(data) = v["errors"]["data"].as_array() {
        for e in data {
            errors.push((
                e["code"].as_str().map(str::to_owned),
                e["message"]
                    .as_str()
                    .unwrap_or("batch error")
                    .to_owned(),
            ));
        }
    }
    Ok(BatchInfo {
        id: v["id"].as_str().unwrap_or_default().to_owned(),
        status: v["status"].as_str().unwrap_or_default().to_owned(),
        output_file_id: v["output_file_id"].as_str().map(str::to_owned),
        errors,
    })
}

/// Parse a JSONL document (batch output file) into per-line values.
/// Empty lines are skipped; a malformed line is a protocol error.
pub fn decode_jsonl(body: &[u8]) -> Result<Vec<Value>, Error> {
    let mut out = Vec::new();
    for (n, line) in body.split(|&b| b == b'\n').enumerate() {
        let text = String::from_utf8_lossy(line);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(text)
                .map_err(|e| Error::UpstreamProtocol(format!("jsonl line {}: {e}", n + 1)))?,
        );
    }
    Ok(out)
}
