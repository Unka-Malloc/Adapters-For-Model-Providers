//! OpenAI media wire codecs: image generation and audio (speech
//! synthesis + transcription). Includes the multipart/form-data
//! encoding the file-upload endpoints require — built by hand, no
//! extra dependencies.
//!
//! Pure functions; no transport here. Original implementation against
//! the public wire contract.

use serde_json::{Map, Value};

use crate::api::*;
use crate::protocols::base64_decode;

use super::openai_chat::provider_error;
use super::openai_data::unclaimed;

// ---------------------------------------------------------------------------
// Images — POST /images/generations
// ---------------------------------------------------------------------------

/// Build an image-generation request body. `response_format` defaults
/// to `b64_json` so the output carries bytes rather than expiring URLs;
/// extensions merge last and may override it (plus `quality`, `style`, …).
pub fn build_image_request(req: &ImageRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    body.insert("prompt".into(), req.prompt.clone().into());
    if let Some(n) = req.count {
        body.insert("n".into(), n.into());
    }
    if let Some(size) = &req.size {
        body.insert("size".into(), size.clone().into());
    }
    body.insert("response_format".into(), "b64_json".into());
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// Decode an image-generation response: `b64_json` items become inline
/// sources, `url` items stay remote, `revised_prompt` rides the text
/// channel.
pub fn decode_image_output(body: &[u8]) -> Result<MediaOutput, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::UpstreamProtocol(format!("images response: {e}")))?;
    if let Some(err) = v.get("error") {
        return Err(provider_error(None, err));
    }
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::UpstreamProtocol("images response missing data[]".into()))?;
    let mut media = Vec::with_capacity(data.len());
    let mut text = Vec::new();
    for item in data {
        if let Some(b64) = item["b64_json"].as_str() {
            media.push(MediaSource::Inline {
                media_type: "image/png".into(),
                data: base64_decode(b64)?,
            });
        } else if let Some(url) = item["url"].as_str() {
            media.push(MediaSource::Url(url.to_owned()));
        } else {
            return Err(Error::UpstreamProtocol(
                "image item carries neither b64_json nor url".into(),
            ));
        }
        if let Some(revised) = item["revised_prompt"].as_str() {
            text.push(revised.to_owned());
        }
    }
    Ok(MediaOutput {
        media,
        text,
        usage: v.get("usage").filter(|u| !u.is_null()).map(media_usage),
        extensions: unclaimed(&v, &["created", "data", "usage"]),
    })
}

/// Usage on media responses (`gpt-image-1`, audio models) names tokens
/// `input_tokens`/`output_tokens`; chat-style `prompt_tokens`/
/// `completion_tokens` are accepted as aliases. The rest rides
/// `usage.extensions` verbatim.
fn media_usage(u: &Value) -> Usage {
    let claimed = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ];
    let mut extensions = Map::new();
    if let Some(obj) = u.as_object() {
        for (k, v) in obj {
            if !claimed.contains(&k.as_str()) {
                extensions.insert(k.clone(), v.clone());
            }
        }
    }
    Usage {
        input_tokens: u["input_tokens"]
            .as_u64()
            .or_else(|| u["prompt_tokens"].as_u64()),
        output_tokens: u["output_tokens"]
            .as_u64()
            .or_else(|| u["completion_tokens"].as_u64()),
        extensions,
    }
}

// ---------------------------------------------------------------------------
// Speech — POST /audio/speech
// ---------------------------------------------------------------------------

/// Build a speech request body. `format` maps verbatim onto
/// `response_format` (mp3/opus/aac/flac/wav/pcm); `voice` when present
/// and any provider fields ride `extensions` merged last.
pub fn build_speech_request(
    model: &str,
    input: &str,
    voice: Option<&str>,
    format: Option<&str>,
    extensions: &Extensions,
) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), model.into());
    body.insert("input".into(), input.into());
    if let Some(v) = voice {
        body.insert("voice".into(), v.into());
    }
    if let Some(f) = format {
        body.insert("response_format".into(), f.into());
    }
    for (k, v) in extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// Media type for synthesized audio bytes, from the requested format
/// (the wire default is mp3).
pub fn speech_media_type(format: Option<&str>) -> String {
    match format.unwrap_or("mp3") {
        "mp3" => "audio/mpeg".into(),
        "opus" => "audio/opus".into(),
        "aac" => "audio/aac".into(),
        "flac" => "audio/flac".into(),
        "wav" => "audio/wav".into(),
        "pcm" => "audio/pcm".into(),
        other => format!("audio/{other}"),
    }
}

// ---------------------------------------------------------------------------
// Transcription — POST /audio/transcriptions (multipart/form-data)
// ---------------------------------------------------------------------------

/// One file part inside a multipart/form-data body.
pub struct MultipartFile<'a> {
    /// Form field name (`file` for uploads).
    pub field: &'a str,
    pub filename: String,
    pub content_type: String,
    pub data: &'a [u8],
}

/// Encode `multipart/form-data`: text fields in order, then the file
/// part, then the closing boundary. Returns the full Content-Type
/// header value (with boundary) and the body bytes.
pub fn encode_multipart(
    fields: &[(String, String)],
    file: &MultipartFile<'_>,
) -> (String, Vec<u8>) {
    let boundary = format!("a4mp{:016x}", rand::random::<u64>());
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\nContent-Type: {}\r\n\r\n",
            file.field, file.filename, file.content_type
        )
        .as_bytes(),
    );
    body.extend_from_slice(file.data);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// Text fields for a transcription upload: `model`, then `language`
/// when present, then extensions. Non-string extension values
/// serialize to their JSON text (`true`, `0.5`, `["word"]`, …).
pub fn transcription_fields(
    model: &str,
    language: Option<&str>,
    extensions: &Extensions,
) -> Vec<(String, String)> {
    let mut fields = vec![("model".to_owned(), model.to_owned())];
    if let Some(language) = language {
        fields.push(("language".to_owned(), language.to_owned()));
    }
    for (k, v) in extensions {
        let text = match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        fields.push((k.clone(), text));
    }
    fields
}

/// Upload filename for an audio payload: `input.<ext>` where the
/// extension follows the payload's media type.
pub fn audio_filename(media_type: &str) -> String {
    let ext = match media_type {
        "audio/mpeg" | "audio/mp3" | "audio/mpga" => "mp3",
        "audio/wav" | "audio/x-wav" | "audio/wave" | "audio/vnd.wave" => "wav",
        "audio/flac" | "audio/x-flac" => "flac",
        "audio/ogg" | "application/ogg" => "ogg",
        "audio/oga" => "oga",
        "audio/opus" => "opus",
        "audio/aac" | "audio/aacp" => "aac",
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" => "m4a",
        "audio/webm" | "video/webm" => "webm",
        "video/mp4" => "mp4",
        _ => "bin",
    };
    format!("input.{ext}")
}

/// Decode a transcription response. JSON answers carry `text` plus
/// detail fields (`language`, `duration`, `segments`, `words`) which
/// ride `extensions`; `response_format` text/srt/vtt answers are the
/// document itself and pass through as text.
pub fn decode_transcription(body: &[u8]) -> Result<MediaOutput, Error> {
    match serde_json::from_slice::<Value>(body) {
        Ok(v) => {
            if let Some(err) = v.get("error") {
                return Err(provider_error(None, err));
            }
            let text = v["text"]
                .as_str()
                .ok_or_else(|| Error::UpstreamProtocol("transcription response missing text".into()))?
                .to_owned();
            Ok(MediaOutput {
                media: Vec::new(),
                text: vec![text],
                usage: None,
                extensions: unclaimed(&v, &["text"]),
            })
        }
        Err(_) => Ok(MediaOutput {
            media: Vec::new(),
            text: vec![String::from_utf8_lossy(body).into_owned()],
            usage: None,
            extensions: Map::new(),
        }),
    }
}
