//! OpenAI Chat Completions as an inbound dialect: a client POST body
//! becomes a canonical `ChatRequest`, and canonical output/events are
//! rendered back as `chat.completion` / `chat.completion.chunk` frames.
//!
//! Pure functions plus a stateful stream encoder; no transport here.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

use super::*;
use crate::api::*;
use crate::transport::SseEvent;

// ---------------------------------------------------------------------------
// Request decode: wire → canonical
// ---------------------------------------------------------------------------

/// Decode a Chat Completions request body into a canonical request.
pub fn decode_request(body: &[u8]) -> Result<(ChatRequest, ConversionReport), Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("chat completions request: {e}")))?;
    let obj = v
        .as_object()
        .ok_or_else(|| Error::Protocol("chat completions request is not an object".into()))?;
    let mut report = request_report(WireDialect::OpenAiChatCompletions);

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Protocol("request missing string field \"model\"".into()))?
        .to_owned();
    let messages = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Protocol("request missing messages[]".into()))?;

    let mut req = ChatRequest {
        model,
        ..Default::default()
    };
    for (i, m) in messages.iter().enumerate() {
        decode_message(i, m, &mut req, &mut report)?;
    }
    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        for (i, t) in tools.iter().enumerate() {
            decode_tool(i, t, &mut req, &mut report);
        }
    }
    // Legacy `functions` predates `tools`; map with a note.
    if let Some(fns) = obj.get("functions").and_then(Value::as_array) {
        for (i, f) in fns.iter().enumerate() {
            let path = format!("functions[{i}]");
            match f.as_object() {
                Some(f) if f.get("name").and_then(Value::as_str).is_some() => {
                    req.tools.push(ToolDef {
                        name: f["name"].as_str().unwrap_or_default().to_owned(),
                        description: f["description"].as_str().unwrap_or_default().to_owned(),
                        parameters: f.get("parameters").cloned().unwrap_or(json!({})),
                        strict: false,
                    });
                    loss(&mut report, LossKind::Lossy, path);
                }
                _ => loss(&mut report, LossKind::Unsupported, path),
            }
        }
    }
    decode_tool_choice(obj.get("tool_choice"), &mut req, &mut report);
    if let Some(parallel) = obj.get("parallel_tool_calls").and_then(Value::as_bool) {
        req.parallel_tool_calls = Some(parallel);
    }
    if req.tool_choice.is_none() {
        // Legacy `function_call` is the pre-tools tool choice.
        if let Some(fc) = obj.get("function_call") {
            let named = fc
                .as_str()
                .filter(|s| *s != "none" && *s != "auto")
                .or_else(|| fc["name"].as_str());
            req.tool_choice = Some(match named {
                Some(name) => ToolChoice::Named(name.to_owned()),
                None if fc.as_str() == Some("none") => ToolChoice::None,
                _ => ToolChoice::Auto,
            });
            loss(&mut report, LossKind::Lossy, "request.function_call");
        }
    }
    decode_sampling(obj, &mut req, &mut report);
    // `stream` is hop-local control: the gateway reproduces it by choosing
    // its own response mode, so it is claimed but not forwarded.
    const CLAIMED: &[&str] = &[
        "model",
        "messages",
        "tools",
        "functions",
        "tool_choice",
        "function_call",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "stream",
        "parallel_tool_calls",
    ];
    for (k, val) in obj {
        if !CLAIMED.contains(&k.as_str()) {
            preserve_extension(&mut req.extensions, &mut report, k, val.clone());
        }
    }
    Ok((req, report))
}

fn decode_message(
    index: usize,
    m: &Value,
    req: &mut ChatRequest,
    report: &mut ConversionReport,
) -> Result<(), Error> {
    let path = || format!("messages[{index}]");
    let obj = m
        .as_object()
        .ok_or_else(|| Error::Protocol(format!("{}: not an object", path())))?;
    let role = obj
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Protocol(format!("{}: missing role", path())))?;

    // `tool` and legacy `function` roles carry tool results, not content.
    if role == "tool" || role == "function" {
        let call_id = match obj.get("tool_call_id").and_then(Value::as_str) {
            Some(id) => id.to_owned(),
            None => {
                let name = obj.get("name").and_then(Value::as_str).unwrap_or_default();
                loss(report, LossKind::Lossy, format!("{}.tool_call_id", path()));
                name.to_owned()
            }
        };
        let content = decode_content(&format!("{}.content", path()), obj.get("content"), report)?;
        if role == "function" {
            loss(report, LossKind::Lossy, format!("{}.role", path()));
        }
        req.messages.push(Message {
            role: Role::User,
            content: vec![ContentPart::ToolResult(ToolResult {
                call_id,
                content,
                is_error: false,
            })],
        });
        return Ok(());
    }

    let role = match role {
        // `developer` is a system-priority instruction channel.
        "system" | "developer" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        other => {
            loss(report, LossKind::Lossy, format!("{}.role", path()));
            Role::Other(other.to_owned())
        }
    };

    let mut content = Vec::new();
    if role == Role::Assistant {
        content.extend(decode_reasoning(obj));
    }
    content.extend(decode_content(
        &format!("{}.content", path()),
        obj.get("content"),
        report,
    )?);
    if role == Role::Assistant {
        if let Some(calls) = obj.get("tool_calls").and_then(Value::as_array) {
            for (j, tc) in calls.iter().enumerate() {
                if let Some(part) = decode_tool_call(
                    &format!("{}.tool_calls[{j}]", path()),
                    tc,
                    report,
                ) {
                    content.push(part);
                }
            }
        }
    } else if obj.get("tool_calls").is_some() {
        loss(report, LossKind::Unsupported, format!("{}.tool_calls", path()));
    }

    const CLAIMED: &[&str] = &[
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "reasoning_content",
        "reasoning",
        "reasoning_details",
    ];
    for key in extra_keys(obj, CLAIMED) {
        loss(report, LossKind::Lossy, format!("{}.{key}", path()));
    }
    req.messages.push(Message { role, content });
    Ok(())
}

/// Visible reasoning fields a client echoed back (`reasoning_content`,
/// `reasoning`, `reasoning_details`) ride the same opaque shape the
/// outbound codec replays into assistant messages.
fn decode_reasoning(obj: &Map<String, Value>) -> Vec<ContentPart> {
    let mut opaque = Map::new();
    let mut summary = None;
    for key in ["reasoning_content", "reasoning", "reasoning_details"] {
        if let Some(val) = obj.get(key).filter(|v| !v.is_null()) {
            if summary.is_none() {
                summary = val.as_str().filter(|s| !s.is_empty()).map(str::to_owned);
            }
            opaque.insert(key.to_owned(), val.clone());
        }
    }
    if opaque.is_empty() {
        return Vec::new();
    }
    vec![ContentPart::Reasoning(ReasoningBlock {
        protocol: Protocol::OpenAiChatCompletions,
        opaque: Value::Object(opaque),
        summary,
    })]
}

/// A message `content` field: a plain string, a part array, or null.
fn decode_content(
    path: &str,
    content: Option<&Value>,
    report: &mut ConversionReport,
) -> Result<Vec<ContentPart>, Error> {
    match content {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![ContentPart::Text { text: s.clone() }]),
        Some(Value::Array(items)) => {
            let mut out = Vec::new();
            for (j, part) in items.iter().enumerate() {
                if let Some(p) = decode_part(&format!("{path}[{j}]"), part, report)? {
                    out.push(p);
                }
            }
            Ok(out)
        }
        Some(_) => Err(Error::Protocol(format!("{path}: unsupported content shape"))),
    }
}

fn decode_part(
    path: &str,
    part: &Value,
    report: &mut ConversionReport,
) -> Result<Option<ContentPart>, Error> {
    let obj = match part.as_object() {
        Some(o) => o,
        None => {
            loss(report, LossKind::Unsupported, path.to_owned());
            return Ok(None);
        }
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("text") => Ok(Some(ContentPart::Text {
            text: obj["text"].as_str().unwrap_or_default().to_owned(),
        })),
        Some("image_url") => {
            let image = &obj["image_url"];
            let url = image["url"].as_str().unwrap_or_default();
            let source = parse_media_source(path, url)?;
            Ok(Some(ContentPart::Media {
                kind: MediaKind::Image,
                source,
                detail: image.get("detail").and_then(Value::as_str).map(str::to_owned),
            }))
        }
        Some("input_audio") => {
            let audio = &obj["input_audio"];
            let data = audio["data"].as_str().unwrap_or_default();
            let format = audio["format"].as_str().unwrap_or("wav");
            Ok(Some(ContentPart::Media {
                kind: MediaKind::Audio,
                source: MediaSource::Inline {
                    media_type: format!("audio/{format}"),
                    data: base64_decode(data, path)?,
                },
                detail: None,
            }))
        }
        Some("file") => {
            let file = &obj["file"];
            let source = if let Some(id) = file.get("file_id").and_then(Value::as_str) {
                MediaSource::Resource(id.to_owned())
            } else if let Some(data) = file.get("file_data").and_then(Value::as_str) {
                MediaSource::Inline {
                    media_type: "application/octet-stream".into(),
                    data: base64_decode(data, path)?,
                }
            } else {
                loss(report, LossKind::Unsupported, path.to_owned());
                return Ok(None);
            };
            Ok(Some(ContentPart::Media {
                kind: MediaKind::File,
                source,
                detail: None,
            }))
        }
        Some(other) => {
            loss(report, LossKind::Unsupported, format!("{path} (type {other:?})"));
            Ok(None)
        }
        None => {
            loss(report, LossKind::Unsupported, path.to_owned());
            Ok(None)
        }
    }
}

fn decode_tool_call(
    path: &str,
    tc: &Value,
    report: &mut ConversionReport,
) -> Option<ContentPart> {
    if tc.get("type").and_then(Value::as_str).is_some_and(|t| t != "function") {
        loss(report, LossKind::Unsupported, format!("{path}.type"));
        return None;
    }
    let function = &tc["function"];
    let arguments = match &function["arguments"] {
        Value::String(s) => match serde_json::from_str(s) {
            Ok(v) => v,
            Err(_) => {
                loss(report, LossKind::Lossy, format!("{path}.arguments"));
                Value::String(s.clone())
            }
        },
        other => other.clone(),
    };
    Some(ContentPart::ToolCall(ToolCall {
        id: tc["id"].as_str().unwrap_or_default().to_owned(),
        name: function["name"].as_str().unwrap_or_default().to_owned(),
        arguments,
    }))
}

fn decode_tool(index: usize, t: &Value, req: &mut ChatRequest, report: &mut ConversionReport) {
    let path = format!("tools[{index}]");
    let function = t.get("function").filter(|f| f.is_object());
    match function {
        Some(f) => req.tools.push(ToolDef {
            name: f["name"].as_str().unwrap_or_default().to_owned(),
            description: f["description"].as_str().unwrap_or_default().to_owned(),
            parameters: f.get("parameters").cloned().unwrap_or(json!({})),
            strict: f.get("strict").and_then(Value::as_bool).unwrap_or(false),
        }),
        None => loss(report, LossKind::Unsupported, path),
    }
}

fn decode_tool_choice(choice: Option<&Value>, req: &mut ChatRequest, report: &mut ConversionReport) {
    let Some(choice) = choice else { return };
    let parsed = match choice {
        Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        },
        Value::Object(o) if o["type"].as_str() == Some("function") => o["function"]["name"]
            .as_str()
            .map(|n| ToolChoice::Named(n.to_owned())),
        _ => None,
    };
    match parsed {
        Some(c) => req.tool_choice = Some(c),
        None if !choice.is_null() => loss(report, LossKind::Unsupported, "request.tool_choice"),
        None => {}
    }
}

fn decode_sampling(obj: &Map<String, Value>, req: &mut ChatRequest, report: &mut ConversionReport) {
    for (key, slot) in [
        ("temperature", &mut req.sampling.temperature as &mut Option<f32>),
        ("top_p", &mut req.sampling.top_p),
    ] {
        if let Some(v) = obj.get(key) {
            match v.as_f64() {
                Some(n) => *slot = Some(n as f32),
                None => {
                    preserve_extension(&mut req.extensions, report, key, v.clone());
                }
            }
        }
    }
    // `max_tokens` and `max_completion_tokens` name the same ceiling.
    let limit = obj
        .get("max_completion_tokens")
        .and_then(Value::as_u64)
        .or_else(|| obj.get("max_tokens").and_then(Value::as_u64));
    match limit {
        Some(m) => req.sampling.max_output_tokens = Some(m as u32),
        None => {
            for key in ["max_completion_tokens", "max_tokens"] {
                if let Some(v) = obj.get(key).filter(|v| !v.is_null()) {
                    preserve_extension(&mut req.extensions, report, key, v.clone());
                }
            }
        }
    }
    if let Some(v) = obj.get("stop") {
        match string_list(v) {
            Some(stop) => req.sampling.stop = stop,
            None if !v.is_null() => {
                preserve_extension(&mut req.extensions, report, "stop", v.clone());
            }
            None => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Response encode: canonical → wire
// ---------------------------------------------------------------------------

/// Encode a canonical `Output` as a `chat.completion` body.
pub fn encode_output(output: &Output, model: &str) -> (Value, ConversionReport) {
    let mut report = response_report(WireDialect::OpenAiChatCompletions);
    let mut message = Map::new();
    message.insert("role".into(), "assistant".into());

    let mut text = String::new();
    let mut reasoning_text = String::new();
    let mut tool_calls = Vec::new();
    for (i, part) in output.parts.iter().enumerate() {
        let path = format!("parts[{i}]");
        match part {
            ContentPart::Text { text: t } => text.push_str(t),
            ContentPart::ToolCall(tc) => tool_calls.push(json!({
                "id": tc.id,
                "type": "function",
                "function": { "name": tc.name, "arguments": tc.arguments.to_string() },
            })),
            ContentPart::Reasoning(rb) => match &rb.opaque {
                // Native fields replay verbatim; foreign reasoning keeps
                // only its visible summary as `reasoning_content`.
                Value::Object(fields) if rb.protocol == Protocol::OpenAiChatCompletions => {
                    for (k, v) in fields {
                        message.insert(k.clone(), v.clone());
                    }
                }
                _ => match &rb.summary {
                    Some(s) => {
                        reasoning_text.push_str(s);
                        loss(
                            &mut report,
                            LossKind::Lossy,
                            format!("{path} (reasoning state reduced to summary)"),
                        );
                    }
                    None => loss(&mut report, LossKind::Unsupported, path),
                },
            },
            ContentPart::Media { .. } | ContentPart::ToolResult(_) => {
                loss(&mut report, LossKind::Unsupported, path)
            }
        }
    }
    message.insert(
        "content".into(),
        if text.is_empty() { Value::Null } else { text.into() },
    );
    if !reasoning_text.is_empty() {
        message
            .entry("reasoning_content".to_owned())
            .or_insert_with(|| reasoning_text.into());
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), tool_calls.into());
    }

    let mut body = json!({
        "id": wire_id("chatcmpl-"),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": finish_reason(output.stop.as_ref(), &mut report),
        }],
    });
    if let Some(u) = &output.usage {
        body["usage"] = encode_usage(u);
    }
    if let Some(obj) = body.as_object_mut() {
        for (k, v) in &output.extensions {
            obj.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    (body, report)
}

/// OpenAI `usage` object: the canonical counts plus provider extras.
fn encode_usage(u: &Usage) -> Value {
    let mut usage = Map::new();
    if let Some(n) = u.input_tokens {
        usage.insert("prompt_tokens".into(), n.into());
    }
    if let Some(n) = u.output_tokens {
        usage.insert("completion_tokens".into(), n.into());
    }
    if let (Some(i), Some(o)) = (u.input_tokens, u.output_tokens) {
        usage.insert("total_tokens".into(), (i + o).into());
    }
    for (k, v) in &u.extensions {
        usage.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(usage)
}

/// `stop` → `finish_reason`; reasons outside the wire vocabulary are
/// emitted verbatim (or approximated for `Paused`) and reported.
fn finish_reason(stop: Option<&StopReason>, report: &mut ConversionReport) -> Value {
    match stop {
        None => Value::Null,
        Some(StopReason::Complete) => "stop".into(),
        Some(StopReason::Length) => "length".into(),
        Some(StopReason::ToolUse) => "tool_calls".into(),
        Some(StopReason::Filtered) => "content_filter".into(),
        Some(StopReason::Paused) => {
            loss(report, LossKind::Lossy, "stop (paused → \"stop\")");
            "stop".into()
        }
        Some(StopReason::Other(s)) => {
            loss(report, LossKind::Lossy, "stop");
            s.clone().into()
        }
    }
}

// ---------------------------------------------------------------------------
// Stream encode: canonical events → `chat.completion.chunk` frames
// ---------------------------------------------------------------------------

/// Chat Completions stream encoder.
///
/// Assigns each tool-call part a dense wire `index`, emits the leading
/// `delta.role` chunk once, and terminates the stream with `[DONE]`.
pub struct OpenAiStreamEncoder {
    id: String,
    created: u64,
    model: String,
    opened: bool,
    tool_indices: HashMap<u32, u64>,
    media_parts: HashSet<u32>,
    next_tool: u64,
    losses: Vec<ConversionLoss>,
    done_sent: bool,
}

impl OpenAiStreamEncoder {
    pub fn new(model: &str) -> Self {
        Self {
            id: wire_id("chatcmpl-"),
            created: now_secs(),
            model: model.to_owned(),
            opened: false,
            tool_indices: HashMap::new(),
            media_parts: HashSet::new(),
            next_tool: 0,
            losses: Vec::new(),
            done_sent: false,
        }
    }

    fn chunk(&self, delta: Value) -> SseEvent {
        sse(
            None,
            json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{ "index": 0, "delta": delta }],
            }),
        )
    }

    /// The first real chunk carries `delta.role`; emit it lazily so an
    /// empty stream produces nothing.
    fn ensure_opened(&mut self, out: &mut Vec<SseEvent>) {
        if !self.opened {
            self.opened = true;
            out.push(self.chunk(json!({ "role": "assistant" })));
        }
    }
}

impl StreamEncoder for OpenAiStreamEncoder {
    /// No leading frame: the first data chunk carries `delta.role`.
    fn preamble(&mut self) -> Vec<SseEvent> {
        Vec::new()
    }

    fn encode(&mut self, event: &StreamEvent) -> Vec<SseEvent> {
        let mut out = Vec::new();
        match event {
            StreamEvent::PartStarted { index, kind } => match kind {
                PartKind::Media => {
                    self.media_parts.insert(*index);
                    self.losses.push(ConversionLoss {
                        path: format!("parts[{index}] (media part)"),
                        kind: LossKind::Unsupported,
                    });
                }
                PartKind::Other => {
                    self.losses.push(ConversionLoss {
                        path: format!("parts[{index}] (provider-native part)"),
                        kind: LossKind::Unsupported,
                    });
                }
                _ => self.ensure_opened(&mut out),
            },
            StreamEvent::TextDelta { text, .. } => {
                if !text.is_empty() {
                    self.ensure_opened(&mut out);
                    out.push(self.chunk(json!({ "content": text })));
                }
            }
            StreamEvent::ReasoningDelta { text, .. } => {
                if !text.is_empty() {
                    self.ensure_opened(&mut out);
                    out.push(self.chunk(json!({ "reasoning_content": text })));
                }
            }
            StreamEvent::ToolCallStarted { index, id, name } => {
                let wire = *self.tool_indices.entry(*index).or_insert_with(|| {
                    let i = self.next_tool;
                    self.next_tool += 1;
                    i
                });
                self.ensure_opened(&mut out);
                out.push(self.chunk(json!({
                    "tool_calls": [{
                        "index": wire,
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": "" },
                    }],
                })));
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                if arguments.is_empty() {
                    return out;
                }
                let wire = *self.tool_indices.entry(*index).or_insert_with(|| {
                    let i = self.next_tool;
                    self.next_tool += 1;
                    i
                });
                self.ensure_opened(&mut out);
                out.push(self.chunk(json!({
                    "tool_calls": [{
                        "index": wire,
                        "function": { "arguments": arguments },
                    }],
                })));
            }
            StreamEvent::MediaDelta { index, .. } => {
                if self.media_parts.insert(*index) {
                    self.losses.push(ConversionLoss {
                        path: format!("parts[{index}] (media deltas)"),
                        kind: LossKind::Unsupported,
                    });
                }
            }
            StreamEvent::PartFinished { .. } => {}
            StreamEvent::Native { part_index, .. } => {
                self.losses.push(ConversionLoss {
                    path: match part_index {
                        Some(i) => format!("parts[{i}] (provider-native value)"),
                        None => "stream (provider-native value)".into(),
                    },
                    kind: LossKind::Unsupported,
                });
            }
            StreamEvent::Usage(u) => {
                self.ensure_opened(&mut out);
                out.push(sse(
                    None,
                    json!({
                        "id": self.id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [],
                        "usage": encode_usage(u),
                    }),
                ));
            }
            StreamEvent::Finished { stop } => {
                self.ensure_opened(&mut out);
                let mut report = response_report(WireDialect::OpenAiChatCompletions);
                let reason = finish_reason(stop.as_ref(), &mut report);
                self.losses.extend(report.losses);
                out.push(sse(
                    None,
                    json!({
                        "id": self.id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }],
                    }),
                ));
            }
        }
        out
    }

    /// Chat Completions terminates the SSE body with a literal `[DONE]`.
    fn finish(&mut self) -> Vec<SseEvent> {
        if self.done_sent {
            return Vec::new();
        }
        self.done_sent = true;
        vec![SseEvent {
            event: None,
            data: "[DONE]".into(),
            id: None,
        }]
    }

    fn drain_losses(&mut self) -> Vec<ConversionLoss> {
        std::mem::take(&mut self.losses)
    }
}
