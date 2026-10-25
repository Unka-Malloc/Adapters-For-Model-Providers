//! OpenAI Chat Completions wire codec: canonical request → native JSON,
//! native JSON/SSE → canonical output and stream events.
//!
//! Pure functions and a stateful stream decoder; no transport here.
//! OpenAI-compatible providers (DeepSeek, Moonshot, xAI, routers, …)
//! share this codec; profile-owned differences ride `extensions`.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::api::*;
use crate::protocols::base64_encode;
use crate::transport::SseEvent;

/// Build a Chat Completions request body. `extensions` merge last and may
/// add or explicitly override fields — that is their purpose.
pub fn build_request(req: &ChatRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    body.insert("messages".into(), encode_messages(&req.messages)?);
    if !req.tools.is_empty() {
        body.insert("tools".into(), encode_tools(&req.tools));
    }
    if let Some(choice) = &req.tool_choice {
        body.insert("tool_choice".into(), encode_tool_choice(choice));
    }
    if let Some(parallel) = req.parallel_tool_calls {
        body.insert("parallel_tool_calls".into(), parallel.into());
    }
    let s = &req.sampling;
    if let Some(t) = s.temperature {
        body.insert("temperature".into(), t.into());
    }
    if let Some(p) = s.top_p {
        body.insert("top_p".into(), p.into());
    }
    if let Some(m) = s.max_output_tokens {
        body.insert("max_completion_tokens".into(), m.into());
    }
    if !s.stop.is_empty() {
        body.insert("stop".into(), s.stop.clone().into());
    }
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

fn encode_messages(messages: &[Message]) -> Result<Value, Error> {
    let mut out = Vec::new();
    for m in messages {
        // Tool results are their own `tool` messages on this wire.
        let mut content_parts = Vec::new();
        for part in &m.content {
            if let ContentPart::ToolResult(tr) = part {
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tr.call_id,
                    "content": tool_result_text(tr),
                }));
            } else {
                content_parts.push(part);
            }
        }
        if content_parts.is_empty() {
            continue;
        }
        match m.role {
            Role::Assistant => out.push(encode_assistant(&content_parts)?),
            _ => {
                let role = match &m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Other(r) => r.as_str(),
                    Role::Assistant => unreachable!(),
                };
                out.push(json!({ "role": role, "content": encode_content(&content_parts, role)? }));
            }
        }
    }
    Ok(out.into())
}

fn encode_assistant(parts: &[&ContentPart]) -> Result<Value, Error> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut msg = Map::new();
    msg.insert("role".into(), "assistant".into());
    for part in parts {
        match part {
            ContentPart::Text { text: t } => text.push_str(t),
            ContentPart::ToolCall(tc) => tool_calls.push(json!({
                "id": tc.id,
                "type": "function",
                "function": { "name": tc.name, "arguments": tc.arguments.to_string() },
            })),
            ContentPart::Reasoning(rb) => {
                // Reasoning replay: only fields the provider itself emitted
                // (e.g. DeepSeek's `reasoning_content`) may go back.
                match &rb.opaque {
                    Value::Object(fields) if rb.protocol == Protocol::OpenAiChatCompletions => {
                        for (k, v) in fields {
                            msg.insert(k.clone(), v.clone());
                        }
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "reasoning block cannot be replayed to chat completions".into(),
                        ));
                    }
                }
            }
            ContentPart::Media { .. } => {
                return Err(Error::Unsupported(
                    "assistant media parts are not representable in chat completions".into(),
                ));
            }
            ContentPart::ToolResult(_) => unreachable!("tool results are split earlier"),
        }
    }
    msg.insert(
        "content".into(),
        if text.is_empty() { Value::Null } else { text.into() },
    );
    if !tool_calls.is_empty() {
        msg.insert("tool_calls".into(), tool_calls.into());
    }
    Ok(msg.into())
}

fn encode_content(parts: &[&ContentPart], role: &str) -> Result<Value, Error> {
    // Collapse to a plain string when everything is text — both shapes are
    // legal, and the string form is the most widely compatible.
    if parts
        .iter()
        .all(|p| matches!(p, ContentPart::Text { .. }))
    {
        let mut text = String::new();
        for p in parts {
            if let ContentPart::Text { text: t } = p {
                text.push_str(t);
            }
        }
        return Ok(text.into());
    }
    let mut out = Vec::new();
    for part in parts {
        out.push(encode_part(part, role)?);
    }
    Ok(out.into())
}

fn encode_part(part: &ContentPart, role: &str) -> Result<Value, Error> {
    match part {
        ContentPart::Text { text } => Ok(json!({ "type": "text", "text": text })),
        ContentPart::Media { kind, source, detail } => {
            let url = match source {
                MediaSource::Url(u) => u.clone(),
                MediaSource::Inline { media_type, data } => format!(
                    "data:{media_type};base64,{}",
                    base64_encode(data)
                ),
                MediaSource::Resource(id) => {
                    return Err(Error::Unsupported(format!(
                        "unresolved resource {id:?}; the adapter resolves resources first"
                    )));
                }
            };
            match kind {
                MediaKind::Image => {
                    let mut image = json!({ "url": url });
                    if let Some(d) = detail {
                        image["detail"] = d.clone().into();
                    }
                    Ok(json!({ "type": "image_url", "image_url": image }))
                }
                MediaKind::Audio => {
                    let format = match source {
                        MediaSource::Inline { media_type, .. } => {
                            media_type.rsplit('/').next().unwrap_or("wav").to_owned()
                        }
                        _ => "wav".into(),
                    };
                    let data = match source {
                        MediaSource::Inline { data, .. } => base64_encode(data),
                        _ => {
                            return Err(Error::Unsupported(
                                "audio parts must be inline data for chat completions".into(),
                            ));
                        }
                    };
                    Ok(json!({ "type": "input_audio", "input_audio": { "data": data, "format": format } }))
                }
                MediaKind::File => Ok(json!({ "type": "file", "file": { "file_id": url } })),
                MediaKind::Video => Err(Error::Unsupported(
                    "video parts are not representable in chat completions".into(),
                )),
            }
        }
        ContentPart::Reasoning(_) => Err(Error::Unsupported(format!(
            "reasoning parts are not representable for role {role}"
        ))),
        ContentPart::ToolCall(_) => Err(Error::Unsupported(
            "tool calls belong to assistant messages only".into(),
        )),
        ContentPart::ToolResult(_) => unreachable!("tool results are split earlier"),
    }
}

fn tool_result_text(tr: &ToolResult) -> String {
    let mut text = String::new();
    for part in &tr.content {
        if let ContentPart::Text { text: t } = part {
            text.push_str(t);
        }
    }
    text
}

fn encode_tools(tools: &[ToolDef]) -> Value {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": t.strict,
                },
            })
        })
        .collect()
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => "auto".into(),
        ToolChoice::None => "none".into(),
        ToolChoice::Required => "required".into(),
        ToolChoice::Named(name) => json!({
            "type": "function",
            "function": { "name": name },
        }),
    }
}

/// Decode a buffered Chat Completions response.
pub fn decode_response(body: &[u8]) -> Result<Output, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("chat completion response: {e}")))?;
    if let Some(err) = v.get("error") {
        return Err(provider_error(None, err));
    }
    let choice = v["choices"]
        .as_array()
        .and_then(|c| c.first())
        .ok_or_else(|| Error::Protocol("response has no choices".into()))?;
    let message = &choice["message"];
    let mut parts = Vec::new();
    if let Some(reasoning) = reasoning_text(message) {
        parts.push(ContentPart::Reasoning(reasoning_block(&reasoning)));
    }
    if let Some(text) = message["content"].as_str() {
        if !text.is_empty() {
            parts.push(ContentPart::Text { text: text.to_owned() });
        }
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for tc in calls {
            parts.push(ContentPart::ToolCall(decode_tool_call(tc)?));
        }
    }
    let claimed = ["id", "object", "created", "model", "choices", "usage", "system_fingerprint", "service_tier"];
    let mut extensions = Map::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                extensions.insert(k.clone(), val.clone());
            }
        }
    }
    Ok(Output {
        parts,
        stop: choice["finish_reason"].as_str().map(stop_reason),
        usage: v.get("usage").filter(|u| !u.is_null()).map(decode_usage),
        extensions,
    })
}

/// The visible reasoning text a provider emits (`reasoning_content`,
/// `reasoning`): kept both displayable and replayable.
fn reasoning_text(message: &Value) -> Option<String> {
    for key in ["reasoning_content", "reasoning"] {
        if let Some(text) = message.get(key).and_then(|v| v.as_str()) {
            if !text.is_empty() {
                return Some(text.to_owned());
            }
        }
    }
    None
}

fn reasoning_block(text: &str) -> ReasoningBlock {
    ReasoningBlock {
        protocol: Protocol::OpenAiChatCompletions,
        opaque: json!({ "reasoning_content": text }),
        summary: Some(text.to_owned()),
    }
}

fn decode_tool_call(v: &Value) -> Result<ToolCall, Error> {
    let function = &v["function"];
    let arguments = match &function["arguments"] {
        Value::String(s) => serde_json::from_str(s)
            .map_err(|e| Error::Protocol(format!("tool call arguments not JSON: {e}")))?,
        other => other.clone(),
    };
    Ok(ToolCall {
        id: v["id"].as_str().unwrap_or_default().to_owned(),
        name: function["name"].as_str().unwrap_or_default().to_owned(),
        arguments,
    })
}

pub(crate) fn stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Complete,
        "length" | "max_tokens" | "max_output_tokens" => StopReason::Length,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "content_filter" => StopReason::Filtered,
        other => StopReason::Other(other.to_owned()),
    }
}

pub(crate) fn decode_usage(u: &Value) -> Usage {
    let mut extensions = Map::new();
    if let Some(obj) = u.as_object() {
        for (k, v) in obj {
            if !["prompt_tokens", "completion_tokens", "total_tokens"].contains(&k.as_str()) {
                extensions.insert(k.clone(), v.clone());
            }
        }
    }
    Usage {
        input_tokens: u["prompt_tokens"].as_u64(),
        output_tokens: u["completion_tokens"].as_u64(),
        extensions,
    }
}

/// Map a provider error body into `Error::Provider`.
/// `status` is the HTTP status when known (stream errors may lack one).
pub fn provider_error(status: Option<u16>, body: &Value) -> Error {
    let err = body.get("error").unwrap_or(body);
    Error::Provider {
        status,
        code: err["type"]
            .as_str()
            .or_else(|| err["code"].as_str())
            .map(str::to_owned),
        message: err["message"]
            .as_str()
            .unwrap_or("provider error")
            .to_owned(),
        retry: match status {
            // Rejected before accepting work: 4xx and rate limits never bill.
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

/// Decode a `GET /models` listing into canonical `ModelInfo` entries.
/// Unknown per-model fields ride `extensions` verbatim.
pub fn decode_models(body: &[u8]) -> Result<Vec<ModelInfo>, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("models response is not JSON: {e}")))?;
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Protocol("models response missing data[]".into()))?;
    data.iter()
        .map(|m| {
            let id = m
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Protocol("model entry missing id".into()))?
                .to_owned();
            let owned_by = m
                .get("owned_by")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let mut extensions = m.as_object().cloned().unwrap_or_default();
            extensions.remove("id");
            extensions.remove("owned_by");
            Ok(ModelInfo {
                id,
                owned_by,
                extensions,
            })
        })
        .collect()
}

/// Stateful decoder: Chat Completions SSE chunks → canonical events.
///
/// Part indices are assigned lazily in arrival order; OpenAI's tool-call
/// `index` maps onto our tool part index, so interleaved calls keep
/// their identities.
#[derive(Default)]
pub struct ChatStreamDecoder {
    next_part: u32,
    text_part: Option<u32>,
    reasoning_part: Option<u32>,
    tool_parts: HashMap<u64, u32>,
}

impl ChatStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn open(&mut self, kind: Slot) -> (u32, StreamEvent) {
        self.next_part += 1;
        let index = self.next_part - 1;
        let event = match kind {
            Slot::Text => StreamEvent::PartStarted { index, kind: PartKind::Text },
            Slot::Reasoning => StreamEvent::PartStarted { index, kind: PartKind::Reasoning },
        };
        (index, event)
    }
}

enum Slot {
    Text,
    Reasoning,
}

impl crate::transport::SseEventDecoder for ChatStreamDecoder {
    fn decode(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, Error> {
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(Vec::new());
        }
        let v: Value = serde_json::from_str(data)
            .map_err(|e| Error::Protocol(format!("stream chunk: {e}")))?;
        if v.get("error").is_some() {
            return Err(provider_error(None, &v));
        }
        let mut out = Vec::new();
        if let Some(choice) = v["choices"].as_array().and_then(|c| c.first()) {
            let delta = &choice["delta"];
            // Reasoning text first: providers emit it before content.
            if let Some(text) = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(|r| r.as_str())
            {
                if !text.is_empty() {
                    let index = match self.reasoning_part {
                        Some(i) => i,
                        None => {
                            let (i, ev) = self.open(Slot::Reasoning);
                            self.reasoning_part = Some(i);
                            out.push(ev);
                            i
                        }
                    };
                    out.push(StreamEvent::ReasoningDelta {
                        index,
                        text: text.to_owned(),
                    });
                }
            }
            if let Some(text) = delta["content"].as_str() {
                if !text.is_empty() {
                    let index = match self.text_part {
                        Some(i) => i,
                        None => {
                            let (i, ev) = self.open(Slot::Text);
                            self.text_part = Some(i);
                            out.push(ev);
                            i
                        }
                    };
                    out.push(StreamEvent::TextDelta {
                        index,
                        text: text.to_owned(),
                    });
                }
            }
            if let Some(calls) = delta["tool_calls"].as_array() {
                for call in calls {
                    let wire_index = call["index"].as_u64().unwrap_or(0);
                    let index = match self.tool_parts.get(&wire_index) {
                        Some(i) => *i,
                        None => {
                            let i = self.next_part;
                            self.next_part += 1;
                            self.tool_parts.insert(wire_index, i);
                            out.push(StreamEvent::ToolCallStarted {
                                index: i,
                                id: call["id"].as_str().unwrap_or_default().to_owned(),
                                name: call["function"]["name"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_owned(),
                            });
                            i
                        }
                    };
                    if let Some(args) = call["function"]["arguments"].as_str() {
                        if !args.is_empty() {
                            out.push(StreamEvent::ToolCallDelta {
                                index,
                                arguments: args.to_owned(),
                            });
                        }
                    }
                }
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                out.push(StreamEvent::Finished {
                    stop: Some(stop_reason(reason)),
                });
            }
        }
        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
            out.push(StreamEvent::Usage(decode_usage(u)));
        }
        Ok(out)
    }
}

