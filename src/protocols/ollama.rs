//! Ollama wire codec: canonical request → `/api/chat` and `/api/embed`
//! JSON, buffered responses and `/api/tags` listings → canonical output
//! and `ModelInfo`, newline-delimited stream objects → canonical events.
//!
//! Pure functions plus a stateful stream decoder; no transport here.
//! Ollama streams are newline-delimited JSON (one object per line), not
//! SSE, so this file carries a small incremental line decoder and event
//! stream that mirror the transport's SSE machinery for line framing.

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde_json::{Map, Value, json};

use crate::api::*;
use crate::protocols::base64_encode;

/// Build a `POST /api/chat` request body. `extensions` merge last and may
/// add or explicitly override fields (`keep_alive`, `format`, `think`,
/// nested `options`, …) — that is their purpose.
///
/// `stream: false` is explicit because the wire defaults to streaming.
/// `tool_choice` has no field on this wire: `Auto` is the server default
/// and emits nothing; every other choice is refused, never dropped.
pub fn build_request(req: &ChatRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    body.insert("messages".into(), encode_messages(&req.messages)?);
    body.insert("stream".into(), false.into());
    if !req.tools.is_empty() {
        body.insert("tools".into(), encode_tools(&req.tools));
    }
    if let Some(choice) = &req.tool_choice {
        if *choice != ToolChoice::Auto {
            return Err(Error::Unsupported(
                "tool_choice is not representable on the Ollama chat API".into(),
            ));
        }
    }
    let s = &req.sampling;
    let mut options = Map::new();
    if let Some(t) = s.temperature {
        options.insert("temperature".into(), t.into());
    }
    if let Some(p) = s.top_p {
        options.insert("top_p".into(), p.into());
    }
    if let Some(m) = s.max_output_tokens {
        options.insert("num_predict".into(), m.into());
    }
    if !s.stop.is_empty() {
        options.insert("stop".into(), s.stop.clone().into());
    }
    if !options.is_empty() {
        body.insert("options".into(), options.into());
    }
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_messages(messages: &[Message]) -> Result<Value, Error> {
    let mut out = Vec::new();
    for m in messages {
        // Tool results are their own `tool` messages on this wire.
        let mut content_parts = Vec::new();
        for part in &m.content {
            if let ContentPart::ToolResult(tr) = part {
                out.push(json!({
                    "role": "tool",
                    "content": tool_result_text(tr),
                }));
            } else {
                content_parts.push(part);
            }
        }
        if content_parts.is_empty() {
            continue;
        }
        let role = match &m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Other(r) => r.as_str(),
            other => {
                return Err(Error::Unsupported(format!(
                    "role {other:?} is not representable on the Ollama chat API"
                )));
            }
        };
        out.push(encode_message(role, &content_parts)?);
    }
    Ok(out.into())
}

/// One message: text folds into `content`, images into the flat base64
/// `images` array, assistant tool calls into `tool_calls`.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_message(role: &str, parts: &[&ContentPart]) -> Result<Value, Error> {
    let mut text = String::new();
    let mut images = Vec::new();
    let mut tool_calls = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text: t } => text.push_str(t),
            ContentPart::Media { kind, source, detail } => {
                if *kind != MediaKind::Image {
                    return Err(Error::Unsupported(
                        "only image media parts are representable on the Ollama chat API".into(),
                    ));
                }
                images.push(encode_image(source, detail)?);
            }
            ContentPart::ToolCall(tc) => {
                if role != "assistant" {
                    return Err(Error::Unsupported(
                        "tool calls belong to assistant messages only".into(),
                    ));
                }
                // Ollama carries arguments as a JSON object, not a string.
                tool_calls.push(json!({
                    "function": { "name": tc.name, "arguments": tc.arguments },
                }));
            }
            // The wire has no reasoning replay field; `think` outputs are
            // regenerated server-side on every request.
            ContentPart::Reasoning(_) => {
                return Err(Error::Unsupported(format!(
                    "reasoning parts are not representable for role {role}"
                )));
            }
            ContentPart::ToolResult(_) => unreachable!("tool results are split earlier"),
            other => {
                return Err(Error::Unsupported(format!(
                    "content part {other:?} is not representable for role {role}"
                )));
            }
        }
    }
    let mut msg = Map::new();
    msg.insert("role".into(), role.into());
    msg.insert("content".into(), text.into());
    if !images.is_empty() {
        msg.insert("images".into(), images.into());
    }
    if !tool_calls.is_empty() {
        msg.insert("tool_calls".into(), tool_calls.into());
    }
    Ok(msg.into())
}

/// Ollama `images` entries are raw base64 strings: inline data only.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_image(source: &MediaSource, detail: &Option<String>) -> Result<Value, Error> {
    if detail.is_some() {
        return Err(Error::Unsupported(
            "media detail hints are not representable on the Ollama chat API".into(),
        ));
    }
    match source {
        MediaSource::Inline { data, .. } => Ok(base64_encode(data).into()),
        MediaSource::Url(_) => Err(Error::Unsupported(
            "ollama chat accepts inline image data only, not URLs".into(),
        )),
        MediaSource::Resource(id) => Err(Error::Unsupported(format!(
            "unresolved resource {id:?}; the adapter resolves resources first"
        ))),
        other => Err(Error::Unsupported(format!(
            "media source {other:?} is not representable on the Ollama chat API"
        ))),
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

/// Function tools share the familiar `{type:"function",function:{…}}`
/// envelope; `strict` has no Ollama counterpart and stays off the wire.
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
                },
            })
        })
        .collect()
}

/// Decode a buffered `/api/chat` response (`"done": true`).
pub fn decode_response(body: &[u8]) -> Result<Output, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("ollama chat response: {e}")))?;
    if v["error"].is_string() {
        return Err(provider_error(None, &v));
    }
    let message = &v["message"];
    let mut parts = Vec::new();
    if let Some(text) = message["content"].as_str() {
        if !text.is_empty() {
            parts.push(ContentPart::Text {
                text: text.to_owned(),
            });
        }
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for tc in calls {
            parts.push(ContentPart::ToolCall(decode_tool_call(tc)));
        }
    }
    let claimed = [
        "model",
        "created_at",
        "message",
        "done",
        "done_reason",
        "prompt_eval_count",
        "eval_count",
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    ];
    let mut extensions = Map::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                extensions.insert(k.clone(), val.clone());
            }
        }
    }
    let usage = usage_from(&v);
    Ok(Output {
        parts,
        stop: v["done_reason"]
            .as_str()
            .map(stop_reason)
            .or_else(|| v["done"].as_bool().filter(|d| *d).map(|_| StopReason::Complete)),
        usage: (usage.input_tokens.is_some()
            || usage.output_tokens.is_some()
            || !usage.extensions.is_empty())
        .then_some(usage),
        extensions,
    })
}

/// Ollama tool calls carry no id; matching is positional.
fn decode_tool_call(v: &Value) -> ToolCall {
    let function = &v["function"];
    let arguments = match &function["arguments"] {
        // Defensive: some servers emit the string form.
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| function["arguments"].clone()),
        other => other.clone(),
    };
    ToolCall {
        id: v["id"].as_str().unwrap_or_default().to_owned(),
        name: function["name"].as_str().unwrap_or_default().to_owned(),
        arguments,
    }
}

fn stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Complete,
        "length" => StopReason::Length,
        other => StopReason::Other(other.to_owned()),
    }
}

/// `/api/*` accounting rides top-level `*_eval_count` and `*_duration`
/// fields rather than a `usage` object.
fn usage_from(v: &Value) -> Usage {
    let mut extensions = Map::new();
    for k in [
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    ] {
        if let Some(d) = v.get(k) {
            extensions.insert(k.to_owned(), d.clone());
        }
    }
    Usage {
        input_tokens: v["prompt_eval_count"].as_u64(),
        output_tokens: v["eval_count"].as_u64(),
        extensions,
    }
}

/// Build a `POST /api/embed` request body. The wire takes `input` as a
/// string or an array; the array form is always valid. There is no
/// output-dimension parameter on this surface.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
pub fn build_embed_request(req: &EmbedRequest) -> Result<Value, Error> {
    if req.dimensions.is_some() {
        return Err(Error::Unsupported(
            "ollama embed has no output-dimension parameter".into(),
        ));
    }
    let mut input: Vec<Value> = Vec::new();
    for i in &req.inputs {
        match i {
            EmbedInput::Text(t) => input.push(t.clone().into()),
            EmbedInput::Media(_) => {
                return Err(Error::Unsupported(
                    "ollama embed accepts text inputs only".into(),
                ));
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "embed input {other:?} is not representable on the Ollama embed API"
                )));
            }
        }
    }
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    body.insert("input".into(), input.into());
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// Decode a `/api/embed` response.
pub fn decode_embed(body: &[u8]) -> Result<EmbedOutput, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("ollama embed response: {e}")))?;
    if v["error"].is_string() {
        return Err(provider_error(None, &v));
    }
    let rows = v["embeddings"]
        .as_array()
        .ok_or_else(|| Error::Protocol("embed response has no embeddings".into()))?;
    let mut embeddings = Vec::with_capacity(rows.len());
    for row in rows {
        let row = row
            .as_array()
            .ok_or_else(|| Error::Protocol("embedding row is not an array".into()))?;
        embeddings.push(row.iter().map(|n| n.as_f64().unwrap_or(0.0) as f32).collect());
    }
    let usage = usage_from(&v);
    let claimed = [
        "model",
        "embeddings",
        "prompt_eval_count",
        "eval_count",
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    ];
    let mut extensions = Map::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                extensions.insert(k.clone(), val.clone());
            }
        }
    }
    Ok(EmbedOutput {
        embeddings,
        usage: (usage.input_tokens.is_some()
            || usage.output_tokens.is_some()
            || !usage.extensions.is_empty())
        .then_some(usage),
        extensions,
    })
}

/// Decode a `GET /api/tags` listing into canonical `ModelInfo` entries.
/// The `name` field is the callable identifier; everything else
/// (size, digest, families, …) rides `extensions` verbatim.
pub fn decode_models(body: &[u8]) -> Result<Vec<ModelInfo>, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("ollama tags response is not JSON: {e}")))?;
    let models = v
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Protocol("tags response missing models[]".into()))?;
    models
        .iter()
        .map(|m| {
            let id = m
                .get("name")
                .or_else(|| m.get("model"))
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Protocol("model entry missing name".into()))?
                .to_owned();
            let mut extensions = m.as_object().cloned().unwrap_or_default();
            extensions.remove("name");
            extensions.remove("owned_by");
            Ok(ModelInfo {
                id,
                owned_by: None,
                extensions,
            })
        })
        .collect()
}

/// Map a provider error body into `Error::Provider`.
/// `status` is the HTTP status when known (stream errors may lack one).
/// Ollama's envelope is a flat `{"error": "..."}` string.
pub fn provider_error(status: Option<u16>, body: &Value) -> Error {
    Error::Provider {
        status,
        code: body["code"].as_str().map(str::to_owned),
        message: body["error"]
            .as_str()
            .or_else(|| body["message"].as_str())
            .unwrap_or("provider error")
            .to_owned(),
        retry: match status {
            // Rejected before accepting work: 4xx (including unknown
            // model names) never bills.
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

/// Incremental newline-delimited-JSON framing. Lines may split across
/// arbitrary chunk boundaries and may end in `\n` or `\r\n`; `feed`
/// yields only complete non-empty lines.
#[derive(Default)]
pub struct NdjsonDecoder {
    buf: Vec<u8>,
}

impl NdjsonDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; returns every complete non-empty line now decodable.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..pos).collect();
            self.buf.drain(..1);
            if let Some(text) = decode_line(&line) {
                out.push(text);
            }
        }
        out
    }

    /// End of stream: flush a trailing line that had no terminator.
    pub fn finish(&mut self) -> Vec<String> {
        let line = std::mem::take(&mut self.buf);
        decode_line(&line).into_iter().collect()
    }
}

fn decode_line(line: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(line);
    let text = text.strip_suffix('\r').unwrap_or(&text);
    if text.trim().is_empty() {
        return None;
    }
    Some(text.to_owned())
}

/// A byte stream framed into NDJSON lines.
pub struct NdjsonStream {
    inner: DynStream<'static, bytes::Bytes>,
    decoder: NdjsonDecoder,
    pending: VecDeque<String>,
    finished: bool,
}

impl NdjsonStream {
    pub fn new(inner: DynStream<'static, bytes::Bytes>) -> Self {
        Self {
            inner,
            decoder: NdjsonDecoder::new(),
            pending: VecDeque::new(),
            finished: false,
        }
    }
}

impl Stream for NdjsonStream {
    type Item = Result<String, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(line) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(line)));
            }
            if self.finished {
                return Poll::Ready(None);
            }
            match self.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    let lines = self.decoder.feed(&bytes);
                    self.pending.extend(lines);
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    self.finished = true;
                    let lines = self.decoder.finish();
                    self.pending.extend(lines);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Maps NDJSON lines into canonical stream events; the line-framing
/// counterpart of `SseEventDecoder`.
pub trait NdjsonEventDecoder: Send {
    /// Decode one line into zero or more canonical events.
    fn decode(&mut self, line: &str) -> Result<Vec<StreamEvent>, Error>;
}

/// An NDJSON byte stream decoded straight into canonical `StreamEvent`s.
pub struct NdjsonEventStream {
    inner: NdjsonStream,
    decoder: Box<dyn NdjsonEventDecoder>,
    pending: VecDeque<StreamEvent>,
}

impl NdjsonEventStream {
    pub fn new(inner: DynStream<'static, bytes::Bytes>, decoder: Box<dyn NdjsonEventDecoder>) -> Self {
        Self {
            inner: NdjsonStream::new(inner),
            decoder,
            pending: VecDeque::new(),
        }
    }
}

impl Stream for NdjsonEventStream {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(ev)));
            }
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(line))) => match self.decoder.decode(&line) {
                    Ok(events) => self.pending.extend(events),
                    Err(e) => return Poll::Ready(Some(Err(e))),
                },
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Stateful decoder: `/api/chat` NDJSON lines → canonical events.
///
/// Every line is a complete partial message; content deltas append to a
/// lazily opened text part and `tool_calls` arrays arrive complete in a
/// single line — their array position maps onto canonical part indices.
#[derive(Default)]
pub struct ChatStreamDecoder {
    next_part: u32,
    text_part: Option<u32>,
    tool_parts: HashMap<u64, u32>,
}

impl ChatStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl NdjsonEventDecoder for ChatStreamDecoder {
    fn decode(&mut self, line: &str) -> Result<Vec<StreamEvent>, Error> {
        let v: Value = serde_json::from_str(line)
            .map_err(|e| Error::Protocol(format!("ollama stream line: {e}")))?;
        if v["error"].is_string() {
            return Err(provider_error(None, &v));
        }
        let mut out = Vec::new();
        let message = &v["message"];
        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                let index = match self.text_part {
                    Some(i) => i,
                    None => {
                        let i = self.next_part;
                        self.next_part += 1;
                        self.text_part = Some(i);
                        out.push(StreamEvent::PartStarted {
                            index: i,
                            kind: PartKind::Text,
                        });
                        i
                    }
                };
                out.push(StreamEvent::TextDelta {
                    index,
                    text: text.to_owned(),
                });
            }
        }
        if let Some(calls) = message["tool_calls"].as_array() {
            for (i, call) in calls.iter().enumerate() {
                let wire = i as u64;
                let index = match self.tool_parts.get(&wire) {
                    Some(i) => *i,
                    None => {
                        let i = self.next_part;
                        self.next_part += 1;
                        self.tool_parts.insert(wire, i);
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
                if let Some(arguments) = call["function"].get("arguments") {
                    // The wire carries an arguments object; the canonical
                    // delta is its JSON text.
                    let text = match arguments {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    if !text.is_empty() && text != "{}" {
                        out.push(StreamEvent::ToolCallDelta {
                            index,
                            arguments: text,
                        });
                    }
                }
            }
        }
        if v["done"].as_bool() == Some(true) {
            let usage = usage_from(&v);
            if usage.input_tokens.is_some()
                || usage.output_tokens.is_some()
                || !usage.extensions.is_empty()
            {
                out.push(StreamEvent::Usage(usage));
            }
            out.push(StreamEvent::Finished {
                stop: v["done_reason"]
                    .as_str()
                    .map(stop_reason)
                    .or(Some(StopReason::Complete)),
            });
        }
        Ok(out)
    }
}
