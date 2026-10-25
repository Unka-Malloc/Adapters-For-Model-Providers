//! OpenAI Responses wire codec: canonical request → `input` items,
//! `output` items → canonical output, named SSE events → stream events.
//!
//! Pure functions and a stateful stream decoder; no transport here.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

use crate::api::*;
use crate::protocols::base64_encode;
use crate::transport::SseEvent;

use super::openai_chat::{decode_usage, provider_error};

/// Build a Responses request body. `extensions` merge last and may add
/// or explicitly override fields (e.g. `reasoning`, `include`,
/// `previous_response_id`, `instructions`).
pub fn build_request(req: &ChatRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    body.insert("input".into(), encode_input(&req.messages)?);
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
        body.insert("max_output_tokens".into(), m.into());
    }
    if !s.stop.is_empty() {
        return Err(Error::Unsupported(
            "Responses API has no stop-sequences field; use extensions for a provider variant".into(),
        ));
    }
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

fn encode_input(messages: &[Message]) -> Result<Value, Error> {
    let mut out = Vec::new();
    for m in messages {
        let role = match &m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Other(r) => r.as_str(),
        };
        // Item order mirrors part order: a standalone item (tool call,
        // tool result, reasoning) flushes the accumulated message first.
        let mut content: Vec<Value> = Vec::new();
        let flush = |content: &mut Vec<Value>, out: &mut Vec<Value>| {
            if !content.is_empty() {
                out.push(json!({
                    "type": "message",
                    "role": role,
                    "content": std::mem::take(content),
                }));
            }
        };
        for part in &m.content {
            match part {
                ContentPart::ToolResult(tr) => {
                    flush(&mut content, &mut out);
                    out.push(json!({
                        "type": "function_call_output",
                        "call_id": tr.call_id,
                        "output": tool_result_text(tr),
                    }));
                }
                ContentPart::ToolCall(tc) => {
                    flush(&mut content, &mut out);
                    out.push(json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments.to_string(),
                    }));
                }
                ContentPart::Reasoning(rb) => {
                    if rb.protocol != Protocol::OpenAiResponses {
                        return Err(Error::Unsupported(
                            "reasoning block was not issued by the Responses protocol".into(),
                        ));
                    }
                    // Replay provider-native reasoning items verbatim.
                    match &rb.opaque {
                        Value::Object(_) | Value::Array(_) => {
                            flush(&mut content, &mut out);
                            out.push(rb.opaque.clone());
                        }
                        _ => {
                            return Err(Error::Unsupported(
                                "malformed reasoning block for Responses replay".into(),
                            ));
                        }
                    }
                }
                ContentPart::Text { text } => {
                    let ty = match m.role {
                        Role::Assistant => "output_text",
                        _ => "input_text",
                    };
                    content.push(json!({ "type": ty, "text": text }));
                }
                ContentPart::Media { kind, source, .. } => {
                    content.push(encode_media(kind, source)?);
                }
            }
        }
        flush(&mut content, &mut out);
    }
    Ok(out.into())
}

fn encode_media(kind: &MediaKind, source: &MediaSource) -> Result<Value, Error> {
    match (kind, source) {
        (MediaKind::Image, MediaSource::Url(u)) => {
            Ok(json!({ "type": "input_image", "image_url": u }))
        }
        (MediaKind::Image, MediaSource::Inline { media_type, data }) => Ok(json!({
            "type": "input_image",
            "image_url": format!("data:{media_type};base64,{}", base64_encode(data)),
        })),
        (MediaKind::Audio, MediaSource::Inline { media_type, data }) => Ok(json!({
            "type": "input_audio",
            "input_audio": {
                "data": base64_encode(data),
                "format": media_type.rsplit('/').next().unwrap_or("wav"),
            },
        })),
        (MediaKind::File, MediaSource::Url(u)) => {
            Ok(json!({ "type": "input_file", "file_url": u }))
        }
        (MediaKind::File, MediaSource::Inline { media_type, data }) => Ok(json!({
            "type": "input_file",
            "file_data": format!("data:{media_type};base64,{}", base64_encode(data)),
        })),
        (_, MediaSource::Resource(id)) => Err(Error::Unsupported(format!(
            "unresolved resource {id:?}; the adapter resolves resources first"
        ))),
        _ => Err(Error::Unsupported("media kind not representable in Responses input".into())),
    }
}

fn tool_result_text(tr: &ToolResult) -> String {
    tr.content
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn encode_tools(tools: &[ToolDef]) -> Value {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": t.strict,
            })
        })
        .collect()
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => "auto".into(),
        ToolChoice::None => "none".into(),
        ToolChoice::Required => "required".into(),
        ToolChoice::Named(name) => json!({ "type": "function", "name": name }),
    }
}

/// Decode a buffered Responses object.
pub fn decode_response(body: &[u8]) -> Result<Output, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("responses object: {e}")))?;
    if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
        return Err(provider_error(None, err));
    }
    let mut parts = Vec::new();
    if let Some(items) = v["output"].as_array() {
        for item in items {
            decode_item(item, &mut parts)?;
        }
    }
    let stop = match v["status"].as_str() {
        Some("completed") => Some(StopReason::Complete),
        Some("incomplete") => match v["incomplete_details"]["reason"].as_str() {
            Some("max_output_tokens") | Some("max_tokens") => Some(StopReason::Length),
            Some(other) => Some(StopReason::Other(other.to_owned())),
            None => Some(StopReason::Length),
        },
        Some("failed") => {
            return Err(Error::Provider {
                status: None,
                code: None,
                message: v["error"]["message"].as_str().unwrap_or("response failed").to_owned(),
                retry: RetrySafety::Unsafe,
                raw: Some(v.clone()),
            });
        }
        Some("cancelled") => Some(StopReason::Other("cancelled".into())),
        Some(other) => Some(StopReason::Other(other.to_owned())),
        None => None,
    };
    let claimed = [
        "id", "object", "created_at", "model", "output", "usage", "status",
        "incomplete_details", "error", "metadata",
    ];
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
        stop,
        usage: v.get("usage").filter(|u| !u.is_null()).map(decode_usage_responses),
        extensions,
    })
}

/// One output item → canonical parts.
fn decode_item(item: &Value, parts: &mut Vec<ContentPart>) -> Result<(), Error> {
    match item["type"].as_str() {
        Some("message") => {
            if let Some(content) = item["content"].as_array() {
                for c in content {
                    match c["type"].as_str() {
                        Some("output_text") | Some("text") => {
                            if let Some(t) = c["text"].as_str() {
                                parts.push(ContentPart::Text { text: t.to_owned() });
                            }
                        }
                        Some("refusal") => {
                            if let Some(t) = c["refusal"].as_str() {
                                parts.push(ContentPart::Text { text: t.to_owned() });
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Some("function_call") => {
            parts.push(ContentPart::ToolCall(ToolCall {
                id: item["call_id"].as_str().or(item["id"].as_str()).unwrap_or_default().to_owned(),
                name: item["name"].as_str().unwrap_or_default().to_owned(),
                arguments: serde_json::from_str(item["arguments"].as_str().unwrap_or("{}"))
                    .unwrap_or_else(|_| item["arguments"].clone()),
            }));
        }
        Some("reasoning") => {
            // The whole item is opaque replay state; its summary array is
            // displayable text when present.
            let summary = item["summary"].as_array().and_then(|s| {
                let joined = s
                    .iter()
                    .filter_map(|p| p["text"].as_str())
                    .collect::<String>();
                (!joined.is_empty()).then_some(joined)
            });
            parts.push(ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::OpenAiResponses,
                opaque: item.clone(),
                summary,
            }));
        }
        _ => {
            // Unknown provider item (server tools, file search, …): keep it.
            parts.push(ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::OpenAiResponses,
                opaque: item.clone(),
                summary: None,
            }));
        }
    }
    Ok(())
}

fn decode_usage_responses(u: &Value) -> Usage {
    let mut usage = decode_usage(&json!({
        "prompt_tokens": u["input_tokens"],
        "completion_tokens": u["output_tokens"],
    }));
    if let Some(obj) = u.as_object() {
        for (k, v) in obj {
            if !["input_tokens", "output_tokens", "total_tokens"].contains(&k.as_str()) {
                usage.extensions.insert(k.clone(), v.clone());
            }
        }
    }
    usage
}

/// Stateful decoder: named Responses SSE events → canonical events.
///
/// `output_index` on wire events maps to our part index per output item.
#[derive(Default)]
pub struct ResponsesStreamDecoder {
    parts: HashMap<u64, u32>,
    next_part: u32,
    /// Parts that already received incremental deltas. Terminal `.done`
    /// events only backfill a part when no deltas were observed.
    text_seen: HashSet<u32>,
    reasoning_seen: HashSet<u32>,
    args_seen: HashSet<u32>,
    /// Tool parts whose identity (`ToolCallStarted`) was already emitted.
    tools_started: HashSet<u32>,
}

impl ResponsesStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn part(&mut self, wire_index: u64, kind: PartKind) -> (u32, Option<StreamEvent>) {
        match self.parts.get(&wire_index) {
            Some(i) => (*i, None),
            None => {
                let i = self.next_part;
                self.next_part += 1;
                self.parts.insert(wire_index, i);
                (i, Some(StreamEvent::PartStarted { index: i, kind }))
            }
        }
    }

    /// Existing part for a wire index, or a newly announced one.
    fn part_or_start(
        &mut self,
        wire_index: u64,
        kind: PartKind,
        out: &mut Vec<StreamEvent>,
    ) -> u32 {
        let (part, started) = self.part(wire_index, kind);
        if let Some(ev) = started {
            out.push(ev);
        }
        part
    }

    /// Backfill text from a `.done`/item payload when no delta arrived.
    fn text_fallback(&mut self, part: u32, text: &str, out: &mut Vec<StreamEvent>) {
        if !text.is_empty() && self.text_seen.insert(part) {
            out.push(StreamEvent::TextDelta {
                index: part,
                text: text.to_owned(),
            });
        }
    }

    /// Backfill reasoning text when no delta arrived.
    fn reasoning_fallback(&mut self, part: u32, text: &str, out: &mut Vec<StreamEvent>) {
        if !text.is_empty() && self.reasoning_seen.insert(part) {
            out.push(StreamEvent::ReasoningDelta {
                index: part,
                text: text.to_owned(),
            });
        }
    }

    /// Backfill tool arguments when no delta arrived.
    fn args_fallback(&mut self, part: u32, args: &str, out: &mut Vec<StreamEvent>) {
        if !args.is_empty() && self.args_seen.insert(part) {
            out.push(StreamEvent::ToolCallDelta {
                index: part,
                arguments: args.to_owned(),
            });
        }
    }
}

impl crate::transport::SseEventDecoder for ResponsesStreamDecoder {
    fn decode(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, Error> {
        if event.data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let v: Value = serde_json::from_str(&event.data)
            .map_err(|e| Error::Protocol(format!("responses event: {e}")))?;
        let ty = event
            .event
            .as_deref()
            .or_else(|| v["type"].as_str())
            .unwrap_or_default();
        let mut out = Vec::new();
        match ty {
            // Envelopes and content-part announcements carry no new output.
            "response.created"
            | "response.in_progress"
            | "response.queued"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done" => {}
            "response.output_item.added" => {
                let idx = v["output_index"].as_u64().unwrap_or(0);
                let kind = match v["item"]["type"].as_str() {
                    Some("function_call") => PartKind::ToolCall,
                    Some("reasoning") => PartKind::Reasoning,
                    Some("message") => PartKind::Text,
                    _ => PartKind::Other,
                };
                let (part, started) = self.part(idx, kind);
                if let Some(ev) = started {
                    out.push(ev);
                }
                if let (PartKind::ToolCall, Some(item)) = (kind, v.get("item")) {
                    self.tools_started.insert(part);
                    out.push(StreamEvent::ToolCallStarted {
                        index: part,
                        id: item["call_id"]
                            .as_str()
                            .or(item["id"].as_str())
                            .unwrap_or_default()
                            .to_owned(),
                        name: item["name"].as_str().unwrap_or_default().to_owned(),
                    });
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let idx = v["output_index"].as_u64().unwrap_or(0);
                let part = self.part_or_start(idx, PartKind::Text, &mut out);
                if let Some(text) = v["delta"].as_str() {
                    self.text_seen.insert(part);
                    out.push(StreamEvent::TextDelta {
                        index: part,
                        text: text.to_owned(),
                    });
                }
            }
            "response.output_text.done" | "response.refusal.done" => {
                let idx = v["output_index"].as_u64().unwrap_or(0);
                let part = self.part_or_start(idx, PartKind::Text, &mut out);
                let field = if ty == "response.refusal.done" {
                    "refusal"
                } else {
                    "text"
                };
                if let Some(text) = v[field].as_str() {
                    self.text_fallback(part, text, &mut out);
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let idx = v["output_index"].as_u64().unwrap_or(0);
                let part = self.part_or_start(idx, PartKind::Reasoning, &mut out);
                if let Some(text) = v["delta"].as_str() {
                    self.reasoning_seen.insert(part);
                    out.push(StreamEvent::ReasoningDelta {
                        index: part,
                        text: text.to_owned(),
                    });
                }
            }
            "response.reasoning_summary_text.done" | "response.reasoning_text.done" => {
                let idx = v["output_index"].as_u64().unwrap_or(0);
                let part = self.part_or_start(idx, PartKind::Reasoning, &mut out);
                if let Some(text) = v["text"].as_str() {
                    self.reasoning_fallback(part, text, &mut out);
                }
            }
            "response.function_call_arguments.delta" => {
                let idx = v["output_index"].as_u64().unwrap_or(0);
                let part = self.part_or_start(idx, PartKind::ToolCall, &mut out);
                if let Some(args) = v["delta"].as_str() {
                    self.args_seen.insert(part);
                    out.push(StreamEvent::ToolCallDelta {
                        index: part,
                        arguments: args.to_owned(),
                    });
                }
            }
            "response.function_call_arguments.done" => {
                let idx = v["output_index"].as_u64().unwrap_or(0);
                let part = self.part_or_start(idx, PartKind::ToolCall, &mut out);
                if let Some(args) = v["arguments"].as_str() {
                    self.args_fallback(part, args, &mut out);
                }
            }
            "response.output_item.done" => {
                let idx = v["output_index"].as_u64().unwrap_or(0);
                let item = &v["item"];
                let kind = match item["type"].as_str() {
                    Some("function_call") => PartKind::ToolCall,
                    Some("reasoning") => PartKind::Reasoning,
                    Some("message") => PartKind::Text,
                    _ => PartKind::Other,
                };
                let part = match self.parts.get(&idx).copied() {
                    Some(part) => part,
                    None => self.part_or_start(idx, kind, &mut out),
                };
                match item["type"].as_str() {
                    Some("message") => {
                        let mut text = String::new();
                        if let Some(content) = item["content"].as_array() {
                            for c in content {
                                match c["type"].as_str() {
                                    Some("output_text" | "text") => {
                                        if let Some(t) = c["text"].as_str() {
                                            text.push_str(t);
                                        }
                                    }
                                    Some("refusal") => {
                                        if let Some(t) = c["refusal"].as_str() {
                                            text.push_str(t);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        self.text_fallback(part, &text, &mut out);
                    }
                    Some("reasoning") => {
                        let joined = item["summary"]
                            .as_array()
                            .map(|summary| {
                                summary
                                    .iter()
                                    .filter_map(|p| p["text"].as_str())
                                    .collect::<String>()
                            })
                            .unwrap_or_default();
                        self.reasoning_fallback(part, &joined, &mut out);
                    }
                    Some("function_call") => {
                        // A terminal item may be the only place identity or
                        // arguments appear (no `added`/delta frames).
                        if self.tools_started.insert(part) {
                            out.push(StreamEvent::ToolCallStarted {
                                index: part,
                                id: item
                                    .get("call_id")
                                    .and_then(Value::as_str)
                                    .or_else(|| item.get("id").and_then(Value::as_str))
                                    .unwrap_or_default()
                                    .to_owned(),
                                name: item
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_owned(),
                            });
                        }
                        if let Some(args) = item["arguments"].as_str() {
                            self.args_fallback(part, args, &mut out);
                        }
                    }
                    _ => {}
                }
                out.push(StreamEvent::PartFinished { index: part });
                // Preserve provider-native items (server tools, …) opaque.
                if !matches!(
                    item["type"].as_str(),
                    Some("message" | "reasoning" | "function_call")
                ) {
                    out.push(StreamEvent::Native {
                        part_index: Some(part),
                        value: item.clone(),
                    });
                }
            }
            "response.completed" | "response.incomplete" => {
                if let Some(u) = v["response"]["usage"].as_object().map(|_| &v["response"]["usage"]) {
                    out.push(StreamEvent::Usage(decode_usage_responses(u)));
                }
                let stop = match ty {
                    "response.completed" => Some(StopReason::Complete),
                    _ => match v["response"]["incomplete_details"]["reason"].as_str() {
                        Some("max_output_tokens") | Some("max_tokens") => Some(StopReason::Length),
                        Some(other) => Some(StopReason::Other(other.to_owned())),
                        None => Some(StopReason::Length),
                    },
                };
                out.push(StreamEvent::Finished { stop });
            }
            "response.failed" => {
                return Err(provider_error(None, &v["response"]["error"]));
            }
            "error" => {
                return Err(provider_error(None, &v));
            }
            _ => {
                // Unknown event types are preserved verbatim.
                out.push(StreamEvent::Native {
                    part_index: v["output_index"]
                        .as_u64()
                        .and_then(|w| self.parts.get(&w).copied()),
                    value: v.clone(),
                });
            }
        }
        Ok(out)
    }
}
