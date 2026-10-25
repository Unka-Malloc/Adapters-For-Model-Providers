//! Gemini generateContent wire codec: canonical request → native JSON,
//! native JSON/SSE → canonical output and stream events.
//!
//! Pure functions and a stateful stream decoder; no transport here.
//! Covers the `v1beta` `models/{model}:generateContent` and
//! `models/{model}:streamGenerateContent` surfaces; profile-owned
//! differences ride `extensions`.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::api::*;
use crate::protocols::base64_encode;
use crate::transport::SseEvent;

/// Part-level keys that make a reasoning opaque payload a standalone
/// wire part rather than attributes decorating a sibling part.
const PART_KEYS: [&str; 7] = [
    "text",
    "functionCall",
    "functionResponse",
    "inlineData",
    "fileData",
    "executableCode",
    "codeExecutionResult",
];

/// Build a generateContent request body. The model id travels in the
/// URL path, not the body. `extensions` merge last and may add or
/// explicitly override fields (`safetySettings`, `cachedContent`, …).
pub fn build_request(req: &ChatRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    let (contents, system) = encode_contents(&req.messages)?;
    body.insert("contents".into(), contents);
    if let Some(system) = system {
        body.insert("systemInstruction".into(), system);
    }
    if !req.tools.is_empty() {
        body.insert("tools".into(), encode_tools(&req.tools)?);
    }
    if let Some(choice) = &req.tool_choice {
        body.insert("toolConfig".into(), encode_tool_config(choice));
    }
    let s = &req.sampling;
    let mut config = Map::new();
    if let Some(t) = s.temperature {
        config.insert("temperature".into(), t.into());
    }
    if let Some(p) = s.top_p {
        config.insert("topP".into(), p.into());
    }
    if let Some(m) = s.max_output_tokens {
        config.insert("maxOutputTokens".into(), m.into());
    }
    if !s.stop.is_empty() {
        config.insert("stopSequences".into(), s.stop.clone().into());
    }
    if !config.is_empty() {
        body.insert("generationConfig".into(), config.into());
    }
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// Encode the conversation into `contents` plus a hoisted
/// `systemInstruction`. `Role::Assistant` maps to `model`; every system
/// message's parts merge into one instruction. `call_names` records
/// each `functionCall` name under its call id so a later `ToolResult`
/// can fill the name-correlated `functionResponse`.
fn encode_contents(messages: &[Message]) -> Result<(Value, Option<Value>), Error> {
    let mut contents = Vec::new();
    let mut system_parts = Vec::new();
    let mut call_names: HashMap<String, String> = HashMap::new();
    for m in messages {
        if m.role == Role::System {
            for part in &m.content {
                system_parts.push(encode_data_part(part)?);
            }
            continue;
        }
        let role = match &m.role {
            Role::User => "user",
            Role::Assistant => "model",
            Role::Other(r) => r.as_str(),
            Role::System => unreachable!(),
        };
        let mut parts = Vec::new();
        for part in &m.content {
            match part {
                ContentPart::ToolResult(tr) => {
                    if role != "user" {
                        return Err(Error::Unsupported(
                            "function responses belong to user-role contents".into(),
                        ));
                    }
                    parts.push(encode_function_response(tr, &call_names)?);
                }
                ContentPart::ToolCall(tc) => {
                    if role != "model" {
                        return Err(Error::Unsupported(
                            "function calls belong to model contents".into(),
                        ));
                    }
                    call_names.insert(tc.id.clone(), tc.name.clone());
                    parts.push(json!({
                        "functionCall": { "name": tc.name, "args": tc.arguments },
                    }));
                }
                ContentPart::Reasoning(rb) => {
                    if role != "model" {
                        return Err(Error::Unsupported(
                            "reasoning parts belong to model contents".into(),
                        ));
                    }
                    encode_reasoning_replay(rb, &mut parts)?;
                }
                _ => parts.push(encode_data_part(part)?),
            }
        }
        if !parts.is_empty() {
            contents.push(json!({ "role": role, "parts": parts }));
        }
    }
    let system = (!system_parts.is_empty()).then(|| json!({ "parts": system_parts }));
    Ok((contents.into(), system))
}

/// Text and media parts share one shape across `contents` and
/// `systemInstruction`. For referenced files, `detail` carries the
/// declared MIME type of `fileData`.
fn encode_data_part(part: &ContentPart) -> Result<Value, Error> {
    match part {
        ContentPart::Text { text } => Ok(json!({ "text": text })),
        ContentPart::Media { source, detail, .. } => match source {
            MediaSource::Inline { media_type, data } => Ok(json!({
                "inlineData": { "mimeType": media_type, "data": base64_encode(data) },
            })),
            MediaSource::Url(uri) => {
                let mut file_data = json!({ "fileUri": uri });
                if let Some(d) = detail {
                    file_data["mimeType"] = d.clone().into();
                }
                Ok(json!({ "fileData": file_data }))
            }
            MediaSource::Resource(id) => Err(Error::Unsupported(format!(
                "unresolved resource {id:?}; the adapter resolves resources first"
            ))),
        },
        other => Err(Error::Unsupported(format!(
            "{other:?} is not representable as a generateContent data part"
        ))),
    }
}

/// Replay provider-native reasoning state. Opaque payloads carrying
/// part keys (`text`, `functionCall`, …) replay verbatim as their own
/// part; attribute-only objects such as `{"thoughtSignature": …}`
/// merge back onto the part they decorated.
fn encode_reasoning_replay(rb: &ReasoningBlock, parts: &mut Vec<Value>) -> Result<(), Error> {
    if rb.protocol != Protocol::GeminiGenerateContent {
        return Err(Error::Unsupported(
            "reasoning block was not issued by generateContent".into(),
        ));
    }
    let Value::Object(fields) = &rb.opaque else {
        return Err(Error::Unsupported(
            "malformed reasoning block for generateContent replay".into(),
        ));
    };
    let standalone = fields.keys().any(|k| PART_KEYS.contains(&k.as_str()));
    match (standalone, parts.last_mut()) {
        (false, Some(last)) => {
            for (k, v) in fields {
                last[k.clone()] = v.clone();
            }
        }
        _ => parts.push(rb.opaque.clone()),
    }
    Ok(())
}

/// `functionResponse` correlates by function *name*: this wire has no
/// call ids. Names recorded from earlier `functionCall` parts resolve
/// synthesized ids back; an unrecognized id passes through as the name
/// itself, which is also the right encoding for callers that carry the
/// function name in `call_id`.
fn encode_function_response(
    tr: &ToolResult,
    call_names: &HashMap<String, String>,
) -> Result<Value, Error> {
    let name = call_names
        .get(&tr.call_id)
        .cloned()
        .unwrap_or_else(|| tr.call_id.clone());
    let mut text = String::new();
    for part in &tr.content {
        match part {
            ContentPart::Text { text: t } => text.push_str(t),
            _ => {
                return Err(Error::Unsupported(
                    "non-text tool result parts are not representable in functionResponse"
                        .into(),
                ));
            }
        }
    }
    let key = if tr.is_error { "error" } else { "result" };
    Ok(json!({
        "functionResponse": {
            "name": name,
            "response": { key: text },
        }
    }))
}

fn encode_tools(tools: &[ToolDef]) -> Result<Value, Error> {
    if tools.iter().any(|t| t.strict) {
        return Err(Error::Unsupported(
            "generateContent has no strict-schema flag for function declarations".into(),
        ));
    }
    Ok(json!([{
        "functionDeclarations": tools
            .iter()
            .map(|t| json!({
                "name": t.name,
                "description": t.description,
                // OpenAPI-ish schema, passed through verbatim.
                "parameters": t.parameters,
            }))
            .collect::<Vec<_>>(),
    }]))
}

fn encode_tool_config(choice: &ToolChoice) -> Value {
    let calling = match choice {
        ToolChoice::Auto => json!({ "mode": "AUTO" }),
        ToolChoice::None => json!({ "mode": "NONE" }),
        ToolChoice::Required => json!({ "mode": "ANY" }),
        ToolChoice::Named(name) => json!({
            "mode": "ANY",
            "allowedFunctionNames": [name],
        }),
    };
    json!({ "functionCallingConfig": calling })
}

/// Decode a buffered generateContent response.
pub fn decode_response(body: &[u8]) -> Result<Output, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::UpstreamProtocol(format!("generateContent response: {e}")))?;
    if v.get("error").filter(|e| !e.is_null()).is_some() {
        return Err(provider_error(None, &v));
    }
    if let Some(reason) = v["promptFeedback"]["blockReason"].as_str() {
        return Err(blocked_prompt(reason, &v));
    }
    let candidate = v["candidates"]
        .as_array()
        .and_then(|c| c.first())
        .ok_or_else(|| Error::UpstreamProtocol("response has no candidates".into()))?;
    let mut calls = 0u32;
    let mut parts = Vec::new();
    if let Some(ps) = candidate["content"]["parts"].as_array() {
        for p in ps {
            decode_part(p, &mut calls, &mut parts)?;
        }
    }
    let tool_use = parts.iter().any(|p| matches!(p, ContentPart::ToolCall(_)));
    let claimed = ["candidates", "usageMetadata"];
    let mut extensions = Map::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                extensions.insert(k.clone(), val.clone());
            }
        }
    }
    // Unclaimed candidate-level fields (safetyRatings, citationMetadata,
    // groundingMetadata, …) stay inspectable under one extension key.
    let candidate_claimed = ["content", "finishReason", "index"];
    let mut candidate_extra = Map::new();
    if let Some(obj) = candidate.as_object() {
        for (k, val) in obj {
            if !candidate_claimed.contains(&k.as_str()) {
                candidate_extra.insert(k.clone(), val.clone());
            }
        }
    }
    if !candidate_extra.is_empty() {
        extensions.insert("candidate".into(), candidate_extra.into());
    }
    Ok(Output {
        parts,
        stop: candidate["finishReason"]
            .as_str()
            .map(|r| stop_reason(r, tool_use)),
        usage: v
            .get("usageMetadata")
            .filter(|u| !u.is_null())
            .map(decode_usage),
        extensions,
    })
}

/// One wire part → canonical part(s). A `thoughtSignature` travelling
/// on a payload part becomes a trailing reasoning block carrying only
/// the signature, so replay merges it back onto that part. Unknown
/// shapes stay replayable as opaque reasoning state.
fn decode_part(part: &Value, calls: &mut u32, parts: &mut Vec<ContentPart>) -> Result<(), Error> {
    if part["thought"].as_bool().unwrap_or(false) {
        parts.push(ContentPart::Reasoning(ReasoningBlock {
            protocol: Protocol::GeminiGenerateContent,
            opaque: part.clone(),
            summary: part["text"].as_str().map(str::to_owned),
        }));
        return Ok(());
    }
    let signature = part
        .get("thoughtSignature")
        .filter(|s| !s.is_null())
        .cloned();
    let push_signature = |parts: &mut Vec<ContentPart>| {
        if let Some(sig) = &signature {
            parts.push(ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::GeminiGenerateContent,
                opaque: json!({ "thoughtSignature": sig }),
                summary: None,
            }));
        }
    };
    if let Some(call) = part.get("functionCall") {
        // The wire has no call ids; `fc_{n}` is stable within the output.
        parts.push(ContentPart::ToolCall(ToolCall {
            id: format!("fc_{}", *calls),
            name: call["name"].as_str().unwrap_or_default().to_owned(),
            arguments: call.get("args").cloned().unwrap_or_else(|| json!({})),
        }));
        *calls += 1;
        push_signature(parts);
        return Ok(());
    }
    if let Some(text) = part.get("text") {
        if let Some(t) = text.as_str() {
            parts.push(ContentPart::Text { text: t.to_owned() });
            push_signature(parts);
        } else {
            parts.push(opaque_part(part));
        }
        return Ok(());
    }
    if let Some(inline) = part.get("inlineData") {
        let media_type = inline["mimeType"]
            .as_str()
            .unwrap_or("application/octet-stream")
            .to_owned();
        let data = base64_decode(inline["data"].as_str().unwrap_or_default())?;
        parts.push(ContentPart::Media {
            kind: media_kind(&media_type),
            source: MediaSource::Inline { media_type, data },
            detail: None,
        });
        push_signature(parts);
        return Ok(());
    }
    if let Some(file) = part.get("fileData") {
        let media_type = file["mimeType"].as_str().map(str::to_owned);
        parts.push(ContentPart::Media {
            kind: media_type.as_deref().map(media_kind).unwrap_or(MediaKind::File),
            source: MediaSource::Url(file["fileUri"].as_str().unwrap_or_default().to_owned()),
            detail: media_type,
        });
        push_signature(parts);
        return Ok(());
    }
    parts.push(opaque_part(part));
    Ok(())
}

/// Signature-only and unrecognized part shapes stay replayable as
/// opaque reasoning state.
fn opaque_part(part: &Value) -> ContentPart {
    ContentPart::Reasoning(ReasoningBlock {
        protocol: Protocol::GeminiGenerateContent,
        opaque: part.clone(),
        summary: None,
    })
}

fn media_kind(mime: &str) -> MediaKind {
    if mime.starts_with("image/") {
        MediaKind::Image
    } else if mime.starts_with("audio/") {
        MediaKind::Audio
    } else if mime.starts_with("video/") {
        MediaKind::Video
    } else {
        MediaKind::File
    }
}

fn stop_reason(reason: &str, tool_use: bool) -> StopReason {
    match reason {
        // The wire reports STOP for tool-call turns; the presence of a
        // functionCall part carries the real semantic.
        "STOP" if tool_use => StopReason::ToolUse,
        "STOP" => StopReason::Complete,
        "MAX_TOKENS" => StopReason::Length,
        "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "RECITATION" | "IMAGE_SAFETY"
        | "SPII" | "LANGUAGE" => StopReason::Filtered,
        other => StopReason::Other(other.to_owned()),
    }
}

pub(crate) fn decode_usage(u: &Value) -> Usage {
    let mut extensions = Map::new();
    if let Some(obj) = u.as_object() {
        for (k, v) in obj {
            if !["promptTokenCount", "candidatesTokenCount"].contains(&k.as_str()) {
                extensions.insert(k.clone(), v.clone());
            }
        }
    }
    Usage {
        input_tokens: u["promptTokenCount"].as_u64(),
        output_tokens: u["candidatesTokenCount"].as_u64(),
        extensions,
    }
}

/// A blocked prompt is a provider rejection, not a filtered completion.
fn blocked_prompt(reason: &str, body: &Value) -> Error {
    let message = body["promptFeedback"]["blockReasonMessage"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("prompt blocked ({reason})"));
    Error::Provider {
        status: None,
        code: Some(reason.to_owned()),
        message,
        retry: RetrySafety::Safe,
        raw: Some(body.clone()),
    }
}

/// Map a Gemini error envelope `{"error":{"code":int,"message","status"}}`
/// into `Error::Provider`. `status` is the HTTP status when known.
pub fn provider_error(status: Option<u16>, body: &Value) -> Error {
    let err = body.get("error").filter(|e| !e.is_null()).unwrap_or(body);
    let status = status.or_else(|| err["code"].as_u64().map(|c| c as u16));
    Error::Provider {
        status,
        code: err["status"]
            .as_str()
            .or_else(|| err["code"].as_str())
            .map(str::to_owned),
        message: err["message"]
            .as_str()
            .unwrap_or("provider error")
            .to_owned(),
        retry: match status {
            // Rejected before accepting work: 4xx never bills.
            Some(400..=499) => RetrySafety::Safe,
            _ => RetrySafety::Unsafe,
        },
        raw: Some(body.clone()),
    }
}

/// Status-code error for non-2xx responses whose body may be JSON or text.
pub fn http_error(status: u16, body: &[u8]) -> Error {
    match serde_json::from_slice::<Value>(body) {
        Ok(v) => provider_error(Some(status), &v),
        Err(_) => Error::Provider {
            status: Some(status),
            code: None,
            message: String::from_utf8_lossy(body).chars().take(512).collect(),
            retry: match status {
                400..=499 => RetrySafety::Safe,
                _ => RetrySafety::Unsafe,
            },
            raw: None,
        },
    }
}

/// Standard base64 (with padding) → bytes for `inlineData` payloads.
/// Shared with the other codecs to keep the alphabet consistent.
fn base64_decode(s: &str) -> Result<bytes::Bytes, Error> {
    super::base64_decode(s)
        .map_err(|_| Error::UpstreamProtocol("invalid base64 in inlineData part".into()))
}

/// Stateful decoder: streamGenerateContent SSE frames → canonical
/// events. Every frame's `data` is a complete GenerateContentResponse;
/// there is no `[DONE]` sentinel — the body simply ends.
///
/// Part indices are assigned lazily in arrival order. Consecutive
/// text/thought chunks continue the open part of that kind; any other
/// part closes it. A `functionCall` arrives complete, so it emits
/// `ToolCallStarted` + one arguments `ToolCallDelta` + `PartFinished`.
#[derive(Default)]
pub struct GenerateStreamDecoder {
    next_part: u32,
    calls: u32,
    open: Option<u32>,
    open_kind: Option<PartKind>,
    saw_tool_call: bool,
}

impl GenerateStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reuse the open part of `kind`, or close whatever is open and
    /// start a new part.
    fn continue_part(&mut self, kind: PartKind, out: &mut Vec<StreamEvent>) -> u32 {
        if let (Some(index), Some(open_kind)) = (self.open, self.open_kind) {
            if open_kind == kind {
                return index;
            }
            out.push(StreamEvent::PartFinished { index });
        }
        let index = self.next_part;
        self.next_part += 1;
        self.open = Some(index);
        self.open_kind = Some(kind);
        out.push(StreamEvent::PartStarted { index, kind });
        index
    }

    fn close_open(&mut self, out: &mut Vec<StreamEvent>) {
        if let Some(index) = self.open.take() {
            self.open_kind = None;
            out.push(StreamEvent::PartFinished { index });
        }
    }

    fn stream_part(&mut self, part: &Value, out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        let signature = part
            .get("thoughtSignature")
            .filter(|s| !s.is_null())
            .cloned();
        if part["thought"].as_bool().unwrap_or(false) {
            let mut index = None;
            if let Some(t) = part["text"].as_str().filter(|t| !t.is_empty()) {
                let i = self.continue_part(PartKind::Reasoning, out);
                out.push(StreamEvent::ReasoningDelta {
                    index: i,
                    text: t.to_owned(),
                });
                index = Some(i);
            }
            if let Some(sig) = signature {
                out.push(StreamEvent::Native {
                    part_index: index.or(self.open),
                    value: json!({ "thoughtSignature": sig }),
                });
            } else if index.is_none() {
                out.push(StreamEvent::Native {
                    part_index: self.open,
                    value: part.clone(),
                });
            }
            return Ok(());
        }
        if let Some(call) = part.get("functionCall") {
            self.close_open(out);
            let index = self.next_part;
            self.next_part += 1;
            let id = format!("fc_{}", self.calls);
            self.calls += 1;
            self.saw_tool_call = true;
            out.push(StreamEvent::ToolCallStarted {
                index,
                id,
                name: call["name"].as_str().unwrap_or_default().to_owned(),
            });
            let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
            out.push(StreamEvent::ToolCallDelta {
                index,
                arguments: args.to_string(),
            });
            if let Some(sig) = signature {
                out.push(StreamEvent::Native {
                    part_index: Some(index),
                    value: json!({ "thoughtSignature": sig }),
                });
            }
            out.push(StreamEvent::PartFinished { index });
            return Ok(());
        }
        if let Some(text) = part.get("text") {
            match text.as_str() {
                Some(t) if !t.is_empty() => {
                    let index = self.continue_part(PartKind::Text, out);
                    out.push(StreamEvent::TextDelta {
                        index,
                        text: t.to_owned(),
                    });
                    if let Some(sig) = signature {
                        out.push(StreamEvent::Native {
                            part_index: Some(index),
                            value: json!({ "thoughtSignature": sig }),
                        });
                    }
                }
                _ => {
                    out.push(StreamEvent::Native {
                        part_index: self.open,
                        value: part.clone(),
                    });
                }
            }
            return Ok(());
        }
        if let Some(inline) = part.get("inlineData") {
            self.close_open(out);
            let index = self.next_part;
            self.next_part += 1;
            out.push(StreamEvent::PartStarted {
                index,
                kind: PartKind::Media,
            });
            let data = base64_decode(inline["data"].as_str().unwrap_or_default())?;
            out.push(StreamEvent::MediaDelta {
                index,
                data,
                media_type: inline["mimeType"].as_str().map(str::to_owned),
            });
            if let Some(sig) = signature {
                out.push(StreamEvent::Native {
                    part_index: Some(index),
                    value: json!({ "thoughtSignature": sig }),
                });
            }
            out.push(StreamEvent::PartFinished { index });
            return Ok(());
        }
        // fileData references, executableCode/codeExecutionResult,
        // signature-only and unknown shapes surface verbatim.
        out.push(StreamEvent::Native {
            part_index: self.open,
            value: part.clone(),
        });
        Ok(())
    }
}

impl crate::transport::SseEventDecoder for GenerateStreamDecoder {
    fn decode(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let data = event.data.trim();
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let v: Value = serde_json::from_str(data)
            .map_err(|e| Error::UpstreamProtocol(format!("stream chunk: {e}")))?;
        if v.get("error").filter(|e| !e.is_null()).is_some() {
            return Err(provider_error(None, &v));
        }
        if let Some(reason) = v["promptFeedback"]["blockReason"].as_str() {
            return Err(blocked_prompt(reason, &v));
        }
        let mut out = Vec::new();
        if let Some(candidate) = v["candidates"].as_array().and_then(|c| c.first()) {
            if let Some(parts) = candidate["content"]["parts"].as_array() {
                for part in parts {
                    self.stream_part(part, &mut out)?;
                }
            }
        }
        if let Some(u) = v.get("usageMetadata").filter(|u| !u.is_null()) {
            out.push(StreamEvent::Usage(decode_usage(u)));
        }
        if let Some(candidate) = v["candidates"].as_array().and_then(|c| c.first()) {
            if let Some(reason) = candidate["finishReason"].as_str() {
                self.close_open(&mut out);
                out.push(StreamEvent::Finished {
                    stop: Some(stop_reason(reason, self.saw_tool_call)),
                });
            }
        }
        Ok(out)
    }
}
