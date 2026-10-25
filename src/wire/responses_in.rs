//! OpenAI Responses as an inbound dialect: a client POST body becomes a
//! canonical `ChatRequest`, and canonical output/events are rendered back
//! as `response` objects and named `response.*` SSE frames.
//!
//! Pure functions plus a stateful stream encoder; no transport here.
//! Items the canonical model cannot name (`item_reference`, server-tool
//! calls, …) keep their Responses shape as opaque reasoning state so a
//! same-dialect backend still receives them verbatim.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

use super::*;
use crate::api::*;
use crate::transport::SseEvent;

// ---------------------------------------------------------------------------
// Request decode: wire → canonical
// ---------------------------------------------------------------------------

/// Decode a Responses request body into a canonical request.
///
/// `input` is either a plain string or an ordered item array; `instructions`
/// becomes a leading system message. Standalone `function_call` and
/// `reasoning` items merge into the preceding assistant message so the
/// canonical turn boundaries match the outbound codec.
pub fn decode_request(body: &[u8]) -> Result<(ChatRequest, ConversionReport), Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("responses request: {e}")))?;
    let obj = v
        .as_object()
        .ok_or_else(|| Error::Protocol("responses request is not an object".into()))?;
    let mut report = request_report(WireDialect::OpenAiResponses);

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Protocol("request missing string field \"model\"".into()))?
        .to_owned();

    let mut req = ChatRequest {
        model,
        ..Default::default()
    };

    if let Some(instructions) = obj.get("instructions") {
        match instructions {
            Value::String(s) if !s.is_empty() => {
                req.messages.push(Message::text(Role::System, s));
            }
            Value::Null => {}
            Value::String(_) => {}
            _ => {
                return Err(Error::Protocol("instructions: expected a string".into()));
            }
        }
    }

    match obj.get("input") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => req.messages.push(Message::text(Role::User, s)),
        Some(Value::Array(items)) => {
            for (i, item) in items.iter().enumerate() {
                decode_input_item(i, item, &mut req, &mut report)?;
            }
        }
        Some(_) => {
            return Err(Error::Protocol(
                "input: expected a string or item array".into(),
            ));
        }
    }

    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        for (i, t) in tools.iter().enumerate() {
            decode_tool(i, t, &mut req, &mut report);
        }
    }
    decode_tool_choice(obj.get("tool_choice"), &mut req, &mut report);
    if let Some(parallel) = obj.get("parallel_tool_calls").and_then(Value::as_bool) {
        req.parallel_tool_calls = Some(parallel);
    }
    decode_sampling(obj, &mut req, &mut report);

    // `stream` is hop-local control: the gateway reproduces it by choosing
    // its own response mode, so it is claimed but not forwarded.
    const CLAIMED: &[&str] = &[
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "max_output_tokens",
        "stop",
        "stream",
    ];
    for (k, val) in obj {
        if !CLAIMED.contains(&k.as_str()) {
            preserve_extension(&mut req.extensions, &mut report, k, val.clone());
        }
    }
    Ok((req, report))
}

fn decode_input_item(
    index: usize,
    item: &Value,
    req: &mut ChatRequest,
    report: &mut ConversionReport,
) -> Result<(), Error> {
    let path = format!("input[{index}]");
    let obj = match item.as_object() {
        Some(o) => o,
        None => {
            loss(report, LossKind::Unsupported, path);
            return Ok(());
        }
    };
    // Be permissive about a missing `type`: a role/content object is a
    // message item on every Responses-shaped client seen in the wild.
    match obj.get("type").and_then(Value::as_str) {
        None if obj.get("role").is_some() => decode_message_item(&path, obj, req, report),
        Some("message") => decode_message_item(&path, obj, req, report),
        Some("function_call") => {
            let arguments = match obj.get("arguments") {
                Some(Value::String(s)) => match serde_json::from_str(s) {
                    Ok(v) => v,
                    Err(_) => {
                        loss(report, LossKind::Lossy, format!("{path}.arguments"));
                        Value::String(s.clone())
                    }
                },
                Some(other) => other.clone(),
                None => json!({}),
            };
            let call = ContentPart::ToolCall(ToolCall {
                id: obj
                    .get("call_id")
                    .and_then(Value::as_str)
                    .or_else(|| obj.get("id").and_then(Value::as_str))
                    .unwrap_or_default()
                    .to_owned(),
                name: obj
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                arguments,
            });
            push_assistant(req, vec![call]);
            for key in extra_keys(
                obj,
                &["type", "id", "call_id", "name", "arguments", "status"],
            ) {
                loss(report, LossKind::Lossy, format!("{path}.{key}"));
            }
            Ok(())
        }
        Some("function_call_output") => {
            let content = match obj.get("output") {
                None | Some(Value::Null) => Vec::new(),
                Some(Value::String(s)) => vec![ContentPart::Text { text: s.clone() }],
                Some(Value::Array(items)) => {
                    let mut out = Vec::new();
                    for (j, part) in items.iter().enumerate() {
                        if let Some(p) =
                            decode_content_part(&format!("{path}.output[{j}]"), part, report)?
                        {
                            out.push(p);
                        }
                    }
                    out
                }
                Some(_) => {
                    loss(report, LossKind::Unsupported, format!("{path}.output"));
                    Vec::new()
                }
            };
            req.messages.push(Message {
                role: Role::User,
                content: vec![ContentPart::ToolResult(ToolResult {
                    call_id: obj
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    content,
                    is_error: false,
                })],
            });
            for key in extra_keys(obj, &["type", "call_id", "output"]) {
                loss(report, LossKind::Lossy, format!("{path}.{key}"));
            }
            Ok(())
        }
        Some("reasoning") => {
            push_assistant(
                req,
                vec![ContentPart::Reasoning(ReasoningBlock {
                    protocol: Protocol::OpenAiResponses,
                    opaque: item.clone(),
                    summary: reasoning_summary(obj),
                })],
            );
            Ok(())
        }
        Some("item_reference") => {
            // A server-side item id cannot be resolved outside its response.
            loss(report, LossKind::Unsupported, path);
            Ok(())
        }
        Some(_) | None => {
            // Provider-native items (server tools, hosted search, …) stay
            // replayable in their Responses shape.
            push_assistant(
                req,
                vec![ContentPart::Reasoning(ReasoningBlock {
                    protocol: Protocol::OpenAiResponses,
                    opaque: item.clone(),
                    summary: None,
                })],
            );
            loss(report, LossKind::Lossy, path);
            Ok(())
        }
    }
}

fn decode_message_item(
    path: &str,
    obj: &Map<String, Value>,
    req: &mut ChatRequest,
    report: &mut ConversionReport,
) -> Result<(), Error> {
    let role = match obj.get("role").and_then(Value::as_str) {
        Some("system" | "developer") => Role::System,
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some(other) => {
            loss(report, LossKind::Lossy, format!("{path}.role"));
            Role::Other(other.to_owned())
        }
        None => return Err(Error::Protocol(format!("{path}: missing role"))),
    };
    let content = decode_content(&format!("{path}.content"), obj.get("content"), report)?;
    for key in extra_keys(obj, &["type", "id", "role", "content", "status"]) {
        loss(report, LossKind::Lossy, format!("{path}.{key}"));
    }
    req.messages.push(Message { role, content });
    Ok(())
}

/// Append a standalone item's parts to the previous assistant turn when
/// possible: the Responses item stream has no message boundary around
/// reasoning/function-call items, while canonical turns do.
fn push_assistant(req: &mut ChatRequest, parts: Vec<ContentPart>) {
    match req.messages.last_mut() {
        Some(m) if matches!(m.role, Role::Assistant) => m.content.extend(parts),
        _ => req.messages.push(Message {
            role: Role::Assistant,
            content: parts,
        }),
    }
}

fn reasoning_summary(obj: &Map<String, Value>) -> Option<String> {
    let summary = obj.get("summary")?.as_array()?;
    let joined = summary
        .iter()
        .filter_map(|p| p["text"].as_str())
        .collect::<String>();
    (!joined.is_empty()).then_some(joined)
}

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
                if let Some(p) = decode_content_part(&format!("{path}[{j}]"), part, report)? {
                    out.push(p);
                }
            }
            Ok(out)
        }
        Some(_) => Err(Error::Protocol(format!(
            "{path}: unsupported content shape"
        ))),
    }
}

fn decode_content_part(
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
        Some("input_text" | "output_text" | "text") => Ok(Some(ContentPart::Text {
            text: obj
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        })),
        Some("refusal") => {
            // Refusal wording is representable; the refusal distinction is not.
            loss(report, LossKind::Lossy, format!("{path}.type (refusal)"));
            Ok(Some(ContentPart::Text {
                text: obj
                    .get("refusal")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }))
        }
        Some("input_image") => {
            let source = match obj.get("file_id").and_then(Value::as_str) {
                Some(id) => MediaSource::Resource(id.to_owned()),
                None => {
                    let url = obj
                        .get("image_url")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    parse_media_source(path, url)?
                }
            };
            Ok(Some(ContentPart::Media {
                kind: MediaKind::Image,
                source,
                detail: obj.get("detail").and_then(Value::as_str).map(str::to_owned),
            }))
        }
        Some("input_file") => {
            let source = if let Some(id) = obj.get("file_id").and_then(Value::as_str) {
                MediaSource::Resource(id.to_owned())
            } else if let Some(url) = obj.get("file_url").and_then(Value::as_str) {
                MediaSource::Url(url.to_owned())
            } else if let Some(data) = obj.get("file_data").and_then(Value::as_str) {
                parse_media_source(path, data)?
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
        Some("input_audio") => {
            let audio = obj.get("input_audio");
            let data = audio
                .and_then(|a| a.get("data"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let format = audio
                .and_then(|a| a.get("format"))
                .and_then(Value::as_str)
                .unwrap_or("wav");
            Ok(Some(ContentPart::Media {
                kind: MediaKind::Audio,
                source: MediaSource::Inline {
                    media_type: format!("audio/{format}"),
                    data: base64_decode(data, path)?,
                },
                detail: None,
            }))
        }
        Some(other) => {
            loss(
                report,
                LossKind::Unsupported,
                format!("{path} (type {other:?})"),
            );
            Ok(None)
        }
        None => {
            loss(report, LossKind::Unsupported, path.to_owned());
            Ok(None)
        }
    }
}

fn decode_tool(index: usize, t: &Value, req: &mut ChatRequest, report: &mut ConversionReport) {
    let path = format!("tools[{index}]");
    let function = t.get("function").filter(|f| f.is_object()).unwrap_or(t);
    let is_function =
        t.get("type").and_then(Value::as_str) == Some("function") || t.get("function").is_some();
    if !is_function {
        loss(report, LossKind::Unsupported, path.clone());
    } else if function.get("name").and_then(Value::as_str).is_some() {
        req.tools.push(ToolDef {
            name: function["name"].as_str().unwrap_or_default().to_owned(),
            description: function["description"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            parameters: function.get("parameters").cloned().unwrap_or(json!({})),
            strict: function
                .get("strict")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        });
    } else {
        loss(report, LossKind::Unsupported, path.clone());
    }
    if let Some(obj) = t.as_object() {
        for key in extra_keys(
            obj,
            &[
                "type",
                "function",
                "name",
                "description",
                "parameters",
                "strict",
            ],
        ) {
            loss(report, LossKind::Lossy, format!("{path}.{key}"));
        }
    }
}

fn decode_tool_choice(
    choice: Option<&Value>,
    req: &mut ChatRequest,
    report: &mut ConversionReport,
) {
    let Some(choice) = choice else { return };
    let parsed = match choice {
        Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        },
        Value::Object(o) => match o.get("type").and_then(Value::as_str) {
            // Responses names the function flat; tolerate the Chat-shaped
            // nested form as well.
            Some("function") => o["name"]
                .as_str()
                .or_else(|| o["function"]["name"].as_str())
                .map(|n| ToolChoice::Named(n.to_owned())),
            Some("auto") => Some(ToolChoice::Auto),
            Some("none") => Some(ToolChoice::None),
            Some("required") => Some(ToolChoice::Required),
            _ => None,
        },
        _ => None,
    };
    match parsed {
        Some(c) => req.tool_choice = Some(c),
        None if !choice.is_null() => loss(report, LossKind::Unsupported, "request.tool_choice"),
        None => {}
    }
    if let Some(obj) = choice.as_object() {
        for key in extra_keys(obj, &["type", "name", "function"]) {
            loss(report, LossKind::Lossy, format!("tool_choice.{key}"));
        }
    }
}

fn decode_sampling(obj: &Map<String, Value>, req: &mut ChatRequest, report: &mut ConversionReport) {
    for (key, slot) in [
        (
            "temperature",
            &mut req.sampling.temperature as &mut Option<f32>,
        ),
        ("top_p", &mut req.sampling.top_p),
    ] {
        if let Some(v) = obj.get(key) {
            match v.as_f64() {
                Some(n) => *slot = Some(n as f32),
                None if !v.is_null() => {
                    preserve_extension(&mut req.extensions, report, key, v.clone())
                }
                None => {}
            }
        }
    }
    if let Some(v) = obj.get("max_output_tokens") {
        match v.as_u64() {
            Some(m) => req.sampling.max_output_tokens = Some(m as u32),
            None if !v.is_null() => {
                preserve_extension(&mut req.extensions, report, "max_output_tokens", v.clone())
            }
            None => {}
        }
    }
    if let Some(v) = obj.get("stop") {
        // Responses has no stop-sequences field.
        preserve_extension(&mut req.extensions, report, "stop", v.clone());
    }
}

// ---------------------------------------------------------------------------
// Response encode: canonical → wire
// ---------------------------------------------------------------------------

/// Encode a canonical `Output` as a Responses `response` body.
///
/// Contiguous text parts form one `message` item; tool calls and reasoning
/// state are separate items in part order. Native Responses reasoning items
/// replay verbatim.
pub fn encode_output(output: &Output, model: &str) -> (Value, ConversionReport) {
    let mut report = response_report(WireDialect::OpenAiResponses);
    let mut items: Vec<Value> = Vec::new();
    let mut message_content: Vec<Value> = Vec::new();

    fn flush(items: &mut Vec<Value>, content: &mut Vec<Value>) {
        if content.is_empty() {
            return;
        }
        items.push(json!({
            "type": "message",
            "id": wire_id("msg_"),
            "status": "completed",
            "role": "assistant",
            "content": std::mem::take(content),
        }));
    }

    for (i, part) in output.parts.iter().enumerate() {
        let path = format!("parts[{i}]");
        match part {
            ContentPart::Text { text } => message_content.push(json!({
                "type": "output_text",
                "text": text,
                "annotations": [],
            })),
            ContentPart::ToolCall(tc) => {
                flush(&mut items, &mut message_content);
                items.push(json!({
                    "type": "function_call",
                    "id": wire_id("fc_"),
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments.to_string(),
                    "status": "completed",
                }));
            }
            ContentPart::Reasoning(rb) => {
                flush(&mut items, &mut message_content);
                match &rb.opaque {
                    Value::Object(_) if rb.protocol == Protocol::OpenAiResponses => {
                        items.push(rb.opaque.clone());
                    }
                    _ => match &rb.summary {
                        Some(s) => {
                            items.push(json!({
                                "type": "reasoning",
                                "id": wire_id("rs_"),
                                "summary": [{ "type": "summary_text", "text": s }],
                            }));
                            loss(
                                &mut report,
                                LossKind::Lossy,
                                format!("{path} (reasoning rendered without native state)"),
                            );
                        }
                        None => loss(&mut report, LossKind::Unsupported, path),
                    },
                }
            }
            ContentPart::Media { .. } | ContentPart::ToolResult(_) => {
                loss(&mut report, LossKind::Unsupported, path)
            }
        }
    }
    flush(&mut items, &mut message_content);

    let (status, incomplete) = response_status(output.stop.as_ref(), &mut report);
    let mut body = json!({
        "id": wire_id("resp_"),
        "object": "response",
        "created_at": now_secs(),
        "status": status,
        "model": model,
        "output": items,
        "error": Value::Null,
        "incomplete_details": incomplete,
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

/// `stop` → (`status`, `incomplete_details`). Responses has no stop reason
/// field; `incomplete_details.reason` is the wire vocabulary.
fn response_status(stop: Option<&StopReason>, report: &mut ConversionReport) -> (Value, Value) {
    match stop {
        None => (Value::Null, Value::Null),
        Some(StopReason::Complete | StopReason::ToolUse) => ("completed".into(), Value::Null),
        Some(StopReason::Length) => (
            "incomplete".into(),
            json!({ "reason": "max_output_tokens" }),
        ),
        Some(StopReason::Filtered) => ("incomplete".into(), json!({ "reason": "content_filter" })),
        Some(StopReason::Paused) => {
            loss(report, LossKind::Lossy, "stop (paused → incomplete)");
            ("incomplete".into(), json!({ "reason": "paused" }))
        }
        Some(StopReason::Other(s)) => {
            loss(report, LossKind::Lossy, "stop");
            ("incomplete".into(), json!({ "reason": s }))
        }
    }
}

/// Responses `usage` object: canonical counts plus provider extras.
fn encode_usage(u: &Usage) -> Value {
    let mut usage = Map::new();
    if let Some(n) = u.input_tokens {
        usage.insert("input_tokens".into(), n.into());
    }
    if let Some(n) = u.output_tokens {
        usage.insert("output_tokens".into(), n.into());
    }
    if let (Some(i), Some(o)) = (u.input_tokens, u.output_tokens) {
        usage.insert("total_tokens".into(), (i + o).into());
    }
    for (k, v) in &u.extensions {
        usage.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(usage)
}

// ---------------------------------------------------------------------------
// Stream encode: canonical events → named `response.*` frames
// ---------------------------------------------------------------------------

/// One open wire output item, keyed by canonical part index.
struct EncodedItem {
    /// Wire `output_index`, assigned densely in first-seen order.
    index: u64,
    id: String,
    kind: PartKind,
    added: bool,
    done: bool,
    text: String,
    reasoning: String,
    tool_id: String,
    tool_name: String,
    arguments: String,
    /// Provider-native item object, when one arrived.
    native: Option<Value>,
}

/// Responses stream encoder.
///
/// Assigns each canonical part a dense `output_index`, opens one output
/// item per part, and terminates with `response.completed` /
/// `response.incomplete`. Usage rides the terminal response object, as it
/// does on the native wire.
pub struct ResponsesStreamEncoder {
    response_id: String,
    created_at: u64,
    model: String,
    items: Vec<EncodedItem>,
    indices: HashMap<u32, usize>,
    media_parts: HashSet<u32>,
    usage: Option<Usage>,
    losses: Vec<ConversionLoss>,
    preambled: bool,
    finished: bool,
}

impl ResponsesStreamEncoder {
    pub fn new(model: &str) -> Self {
        Self {
            response_id: wire_id("resp_"),
            created_at: now_secs(),
            model: model.to_owned(),
            items: Vec::new(),
            indices: HashMap::new(),
            media_parts: HashSet::new(),
            usage: None,
            losses: Vec::new(),
            preambled: false,
            finished: false,
        }
    }

    fn frame(&self, ty: &str, mut data: Value) -> SseEvent {
        if let Some(obj) = data.as_object_mut() {
            obj.entry("type".to_owned()).or_insert_with(|| ty.into());
        }
        sse(Some(ty), data)
    }

    fn item(&mut self, canonical: u32, kind: PartKind) -> usize {
        if let Some(i) = self.indices.get(&canonical) {
            return *i;
        }
        let index = self.items.len() as u64;
        let id = match kind {
            PartKind::Text => wire_id("msg_"),
            PartKind::Reasoning => wire_id("rs_"),
            PartKind::ToolCall => wire_id("fc_"),
            _ => wire_id("item_"),
        };
        self.items.push(EncodedItem {
            index,
            id,
            kind,
            added: false,
            done: false,
            text: String::new(),
            reasoning: String::new(),
            tool_id: String::new(),
            tool_name: String::new(),
            arguments: String::new(),
            native: None,
        });
        self.indices.insert(canonical, index as usize);
        index as usize
    }

    /// The completed/started shape of an item.
    fn item_json(&self, i: usize, status: &str) -> Value {
        let item = &self.items[i];
        match item.kind {
            PartKind::Text => json!({
                "type": "message",
                "id": item.id,
                "status": status,
                "role": "assistant",
                "content": if status == "completed" {
                    vec![json!({
                        "type": "output_text",
                        "text": item.text,
                        "annotations": [],
                    })]
                } else {
                    Vec::<Value>::new()
                },
            }),
            PartKind::Reasoning => json!({
                "type": "reasoning",
                "id": item.id,
                "summary": if status == "completed" {
                    vec![json!({ "type": "summary_text", "text": item.reasoning })]
                } else {
                    Vec::<Value>::new()
                },
            }),
            PartKind::ToolCall => json!({
                "type": "function_call",
                "id": item.id,
                "call_id": item.tool_id,
                "name": item.tool_name,
                "arguments": item.arguments,
                "status": status,
            }),
            _ => item.native.clone().unwrap_or_else(|| {
                json!({
                    "type": "item",
                    "id": item.id,
                    "status": status,
                })
            }),
        }
    }

    /// `response.output_item.added`, once per item.
    fn open(&mut self, i: usize) -> Option<SseEvent> {
        if self.items[i].added {
            return None;
        }
        self.items[i].added = true;
        let item = self.item_json(i, "in_progress");
        Some(self.frame(
            "response.output_item.added",
            json!({
                "output_index": self.items[i].index,
                "item": item,
            }),
        ))
    }

    /// `response.output_item.done`, once per open item.
    fn close(&mut self, i: usize) -> Option<SseEvent> {
        if !self.items[i].added || self.items[i].done {
            return None;
        }
        self.items[i].done = true;
        let item = self.item_json(i, "completed");
        Some(self.frame(
            "response.output_item.done",
            json!({
                "output_index": self.items[i].index,
                "item": item,
            }),
        ))
    }

    fn delta(&self, i: usize, suffix: &str, delta: &str, content_index: bool) -> SseEvent {
        let item = &self.items[i];
        let mut data = json!({
            "item_id": item.id,
            "output_index": item.index,
            "delta": delta,
        });
        if content_index {
            data["content_index"] = 0.into();
        } else {
            data["summary_index"] = 0.into();
        }
        self.frame(suffix, data)
    }

    fn close_all(&mut self, out: &mut Vec<SseEvent>) {
        for i in 0..self.items.len() {
            if let Some(f) = self.close(i) {
                out.push(f);
            }
        }
    }

    fn terminal(&mut self, stop: Option<&StopReason>) -> Vec<SseEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut out = Vec::new();
        self.close_all(&mut out);

        let (status, incomplete) = {
            let mut report = response_report(WireDialect::OpenAiResponses);
            let (status, incomplete) = response_status(stop, &mut report);
            self.losses.extend(report.losses);
            (status, incomplete)
        };
        let output: Vec<Value> = (0..self.items.len())
            .filter(|&i| self.items[i].added)
            .map(|i| self.item_json(i, "completed"))
            .collect();
        let mut response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": output,
            "error": Value::Null,
            "incomplete_details": incomplete,
        });
        if let Some(u) = &self.usage {
            response["usage"] = encode_usage(u);
        }
        let ty = if response["status"] == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        out.push(self.frame(ty, json!({ "response": response })));
        out
    }
}

impl StreamEncoder for ResponsesStreamEncoder {
    /// `response.created` / `response.in_progress` envelopes.
    fn preamble(&mut self) -> Vec<SseEvent> {
        if self.preambled {
            return Vec::new();
        }
        self.preambled = true;
        let response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model,
            "output": [],
            "error": Value::Null,
            "incomplete_details": Value::Null,
        });
        vec![
            self.frame("response.created", json!({ "response": response })),
            self.frame("response.in_progress", json!({ "response": response })),
        ]
    }

    fn encode(&mut self, event: &StreamEvent) -> Vec<SseEvent> {
        let mut out = Vec::new();
        match event {
            StreamEvent::PartStarted { index, kind } => match kind {
                PartKind::Text | PartKind::Reasoning => {
                    let i = self.item(*index, *kind);
                    if let Some(f) = self.open(i) {
                        out.push(f);
                    }
                }
                // Tool identity arrives with `ToolCallStarted`; native
                // item objects arrive with `Native`. Defer opening so the
                // wire item is complete when it is announced.
                PartKind::ToolCall | PartKind::Other => {
                    self.item(*index, *kind);
                }
                PartKind::Media => {
                    self.media_parts.insert(*index);
                    self.losses.push(ConversionLoss {
                        path: format!("parts[{index}] (media part)"),
                        kind: LossKind::Unsupported,
                    });
                }
            },
            StreamEvent::TextDelta { index, text } => {
                if text.is_empty() {
                    return out;
                }
                let i = self.item(*index, PartKind::Text);
                if let Some(f) = self.open(i) {
                    out.push(f);
                }
                self.items[i].text.push_str(text);
                out.push(self.delta(i, "response.output_text.delta", text, true));
            }
            StreamEvent::ReasoningDelta { index, text } => {
                if text.is_empty() {
                    return out;
                }
                let i = self.item(*index, PartKind::Reasoning);
                if let Some(f) = self.open(i) {
                    out.push(f);
                }
                self.items[i].reasoning.push_str(text);
                out.push(self.delta(i, "response.reasoning_summary_text.delta", text, false));
            }
            StreamEvent::ToolCallStarted { index, id, name } => {
                let i = self.item(*index, PartKind::ToolCall);
                self.items[i].tool_id = id.clone();
                self.items[i].tool_name = name.clone();
                if let Some(f) = self.open(i) {
                    out.push(f);
                }
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                if arguments.is_empty() {
                    return out;
                }
                let i = self.item(*index, PartKind::ToolCall);
                if let Some(f) = self.open(i) {
                    out.push(f);
                }
                self.items[i].arguments.push_str(arguments);
                out.push(self.delta(i, "response.function_call_arguments.delta", arguments, true));
            }
            StreamEvent::MediaDelta { index, .. } => {
                if self.media_parts.insert(*index) {
                    self.losses.push(ConversionLoss {
                        path: format!("parts[{index}] (media deltas)"),
                        kind: LossKind::Unsupported,
                    });
                }
            }
            StreamEvent::PartFinished { index } => {
                if let Some(i) = self.indices.get(index).copied() {
                    if self.items[i].kind != PartKind::Other {
                        if let Some(f) = self.open(i) {
                            out.push(f);
                        }
                        if let Some(f) = self.close(i) {
                            out.push(f);
                        }
                    }
                }
            }
            StreamEvent::Native { part_index, value } => {
                out.extend(self.encode_native(*part_index, value));
            }
            StreamEvent::Usage(u) => {
                if self.finished {
                    self.losses.push(ConversionLoss {
                        path: "usage (after terminal event)".into(),
                        kind: LossKind::Lossy,
                    });
                } else {
                    self.usage = Some(u.clone());
                }
            }
            StreamEvent::Finished { stop } => out.extend(self.terminal(stop.as_ref())),
        }
        out
    }

    /// Close every open item and terminate the response envelope if the
    /// backend ended without a `Finished` event.
    fn finish(&mut self) -> Vec<SseEvent> {
        self.terminal(None)
    }

    fn drain_losses(&mut self) -> Vec<ConversionLoss> {
        std::mem::take(&mut self.losses)
    }
}

impl ResponsesStreamEncoder {
    /// Provider-native values: event-shaped objects relay under their own
    /// event name with a remapped `output_index`; item-shaped objects open
    /// and close their part as a verbatim item.
    fn encode_native(&mut self, part_index: Option<u32>, value: &Value) -> Vec<SseEvent> {
        let mut out = Vec::new();
        let Some(canonical) = part_index else {
            if let Some(ty) = value.get("type").and_then(Value::as_str) {
                out.push(self.frame(ty, value.clone()));
            } else {
                self.losses.push(ConversionLoss {
                    path: "stream (provider-native value)".into(),
                    kind: LossKind::Unsupported,
                });
            }
            return out;
        };
        if value.get("output_index").is_some() {
            // Event-shaped: preserve the event, but its wire output index is
            // ours, not the upstream encoder's.
            let Some(ty) = value.get("type").and_then(Value::as_str) else {
                self.losses.push(ConversionLoss {
                    path: format!("parts[{canonical}] (provider-native value)"),
                    kind: LossKind::Unsupported,
                });
                return out;
            };
            let i = self.item(canonical, PartKind::Other);
            let mut v = value.clone();
            v["output_index"] = json!(self.items[i].index);
            out.push(self.frame(ty, v));
            return out;
        }
        // Item-shaped: keep the item as this part's native payload.
        let Some(ty) = value.get("type").and_then(Value::as_str) else {
            self.losses.push(ConversionLoss {
                path: format!("parts[{canonical}] (provider-native value)"),
                kind: LossKind::Unsupported,
            });
            return out;
        };
        let i = self.item(canonical, native_part_kind(ty));
        self.items[i].native = Some(value.clone());
        if let Some(f) = self.open(i) {
            out.push(f);
        }
        if let Some(f) = self.close(i) {
            out.push(f);
        }
        out
    }
}

/// Item type → canonical part kind, for opening a native item.
fn native_part_kind(ty: &str) -> PartKind {
    match ty {
        "message" => PartKind::Text,
        "reasoning" => PartKind::Reasoning,
        "function_call" => PartKind::ToolCall,
        _ => PartKind::Other,
    }
}
