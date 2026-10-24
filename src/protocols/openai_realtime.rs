//! OpenAI Realtime wire codec: canonical `RealtimeInput` / `RealtimeEvent`
//! ↔ the JSON event frames OpenAI's Realtime API exchanges over a duplex
//! channel (`wss://{base}/realtime?model=…`, `OpenAI-Beta: realtime=v1`).
//!
//! Pure JSON mapping — no transport, no credentials. The adapter owns the
//! channel; this codec owns the frame shapes. Beta event names are primary;
//! the GA aliases (`response.output_text.*`, `response.output_audio.*`)
//! carry the same payload shape and decode identically.

use serde_json::{Map, Value, json};

use crate::api::*;
use crate::protocols::{base64_decode, base64_encode};
use crate::protocols::openai_chat::provider_error;

/// The realtime handshake URL: `{base}/realtime?model=…` with the scheme
/// mapped `https`→`wss`, `http`→`ws` (an already-`ws(s)` base is kept).
pub fn session_url(target: &ServiceTarget, model: &str) -> url::Url {
    let mut url = target.endpoint("realtime");
    let scheme = url.scheme().to_owned();
    let ws_scheme = match scheme.as_str() {
        "https" => "wss",
        "http" => "ws",
        other => other,
    };
    // Special-scheme swap cannot fail for http(s) ↔ ws(s).
    let _ = url.set_scheme(ws_scheme);
    url.query_pairs_mut().append_pair("model", model);
    url
}

/// `session.update` carrying the configured session fields, or `None`
/// when the config declares nothing to change. `extensions` merge into
/// the `session` object verbatim — provider-native fields (voice,
/// turn_detection, input/output audio formats, tools, …) ride them.
pub fn encode_session_update(config: &RealtimeConfig) -> Option<Value> {
    let mut session = Map::new();
    if !config.modalities.is_empty() {
        session.insert("modalities".into(), config.modalities.clone().into());
    }
    for (k, v) in &config.extensions {
        session.insert(k.clone(), v.clone());
    }
    if session.is_empty() {
        None
    } else {
        Some(json!({ "type": "session.update", "session": session }))
    }
}

/// Stateful canonical-input → client-event encoder.
///
/// `Commit` means "the caller's turn is finished": appended audio is
/// committed first (only when audio is actually pending — committing an
/// empty buffer would create an empty input item), then `response.create`
/// asks for the model's turn. A bare buffer commit without a response is
/// still reachable through `RealtimeInput::Native`.
#[derive(Default)]
pub struct RealtimeEncoder {
    appended_audio: bool,
}

impl RealtimeEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode one canonical input into the client event frame(s) to send,
    /// in send order.
    pub fn encode(&mut self, input: &RealtimeInput) -> Result<Vec<Value>, Error> {
        match input {
            RealtimeInput::Text(text) => Ok(vec![json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": text }],
                },
            })]),
            RealtimeInput::Audio { data, media_type: _ } => {
                // The wire append carries only base64 audio; the format is
                // session-level (`input_audio_format`), so `media_type`
                // describes the caller's bytes but has no frame field.
                self.appended_audio = true;
                Ok(vec![json!({
                    "type": "input_audio_buffer.append",
                    "audio": base64_encode(data),
                })])
            }
            RealtimeInput::ToolResult(result) => Ok(vec![json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": tool_result_text(result),
                },
            })]),
            RealtimeInput::Native(frame) => {
                if frame.get("type").and_then(Value::as_str).is_none() {
                    return Err(Error::Protocol(
                        "native realtime client event must be an object with a string type".into(),
                    ));
                }
                Ok(vec![frame.clone()])
            }
            RealtimeInput::Commit => {
                let mut out = Vec::new();
                if self.appended_audio {
                    self.appended_audio = false;
                    out.push(json!({ "type": "input_audio_buffer.commit" }));
                }
                out.push(json!({ "type": "response.create" }));
                Ok(out)
            }
            RealtimeInput::Cancel => Ok(vec![json!({ "type": "response.cancel" })]),
        }
    }
}

/// Stateful server-event → canonical-event decoder. Frames carry no
/// audio format, so `RealtimeEvent::Audio.media_type` is synthesized from
/// the session's configured `output_audio_format` as `audio/{format}`.
pub struct RealtimeDecoder {
    output_media_type: String,
}

impl RealtimeDecoder {
    pub fn new(config: &RealtimeConfig) -> Self {
        let format = config
            .extensions
            .get("output_audio_format")
            .and_then(Value::as_str)
            .unwrap_or("pcm16");
        Self {
            output_media_type: format!("audio/{format}"),
        }
    }

    /// Decode one inbound JSON text frame into zero or more canonical
    /// events. Provider `error` frames surface as `Err(Error::Provider)`
    /// like the request/response codecs; the channel itself stays open.
    ///
    /// Lifecycle and bookkeeping frames (`session.*`, `conversation.item.*`,
    /// `response.output_item.*`, `response.content_part.*`, `*.done`
    /// completions, `rate_limits.updated`, …) have no canonical variant —
    /// they arrive as `RealtimeEvent::Native` verbatim. A
    /// `function_call_arguments.delta` is also `Native`: the canonical
    /// event set has no partial tool-call shape, and `done` carries the
    /// complete call.
    pub fn decode(&mut self, frame: &[u8]) -> Result<Vec<RealtimeEvent>, Error> {
        let v: Value = serde_json::from_slice(frame)
            .map_err(|e| Error::Protocol(format!("realtime server event: {e}")))?;
        let ty = v["type"].as_str().unwrap_or_default();
        match ty {
            "error" => Err(provider_error(None, &v)),
            "response.text.delta" | "response.output_text.delta" => Ok(vec![
                RealtimeEvent::Text(delta_str(&v).to_owned()),
            ]),
            "response.audio.delta" | "response.output_audio.delta" => {
                Ok(vec![RealtimeEvent::Audio {
                    data: base64_decode(delta_str(&v))?,
                    media_type: self.output_media_type.clone(),
                }])
            }
            "response.audio_transcript.delta"
            | "response.output_audio_transcript.delta"
            | "conversation.item.input_audio_transcription.delta" => Ok(vec![
                RealtimeEvent::Transcript {
                    text: delta_str(&v).to_owned(),
                    final_: false,
                },
            ]),
            "response.audio_transcript.done"
            | "response.output_audio_transcript.done"
            | "conversation.item.input_audio_transcription.completed" => Ok(vec![
                RealtimeEvent::Transcript {
                    text: v["transcript"].as_str().unwrap_or_default().to_owned(),
                    final_: true,
                },
            ]),
            "conversation.item.input_audio_transcription.failed" => {
                Err(provider_error(None, &v))
            }
            "response.function_call_arguments.delta" => Ok(vec![RealtimeEvent::Native(v)]),
            "response.function_call_arguments.done" => {
                let arguments = match &v["arguments"] {
                    Value::String(s) => serde_json::from_str(s).map_err(|e| {
                        Error::Protocol(format!("function call arguments not JSON: {e}"))
                    })?,
                    other => other.clone(),
                };
                Ok(vec![RealtimeEvent::ToolCall(ToolCall {
                    id: v["call_id"].as_str().unwrap_or_default().to_owned(),
                    name: v["name"].as_str().unwrap_or_default().to_owned(),
                    arguments,
                })])
            }
            "response.done" => {
                let mut out = Vec::new();
                if let Some(usage) = v["response"]["usage"].as_object() {
                    out.push(RealtimeEvent::Usage(decode_usage(usage)));
                }
                out.push(RealtimeEvent::Native(v));
                Ok(out)
            }
            _ => Ok(vec![RealtimeEvent::Native(v)]),
        }
    }
}

fn delta_str(v: &Value) -> &str {
    v["delta"].as_str().unwrap_or_default()
}

/// Realtime usage shape: `input_tokens`/`output_tokens`/`total_tokens`
/// plus token-detail objects; unclaimed fields ride `extensions`.
fn decode_usage(u: &Map<String, Value>) -> Usage {
    let mut extensions = Map::new();
    for (k, v) in u {
        if !matches!(k.as_str(), "input_tokens" | "output_tokens" | "total_tokens") {
            extensions.insert(k.clone(), v.clone());
        }
    }
    Usage {
        input_tokens: u.get("input_tokens").and_then(Value::as_u64),
        output_tokens: u.get("output_tokens").and_then(Value::as_u64),
        extensions,
    }
}

/// Tool outputs on this wire are plain strings; text parts join.
fn tool_result_text(result: &ToolResult) -> String {
    let mut text = String::new();
    for part in &result.content {
        if let ContentPart::Text { text: t } = part {
            text.push_str(t);
        }
    }
    text
}
