//! Cohere wire codec: canonical request → `v2` chat/embed/rerank JSON,
//! `v2` SSE frames → canonical output and stream events. The `v1`
//! classify surface shares the same error envelope and lives here too.
//!
//! Pure functions and a stateful stream decoder; no transport here.
//! Cohere-compatible endpoints share this codec; profile-owned
//! differences (input types, truncation, presets) ride `extensions`.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::api::*;
use crate::protocols::base64_encode;
use crate::transport::SseEvent;

/// The protocol family this codec implements.
fn protocol() -> Protocol {
    Protocol::Other("cohere".into())
}

/// Build a `POST /v2/chat` request body. `extensions` merge last and may
/// add or explicitly override fields — that is their purpose.
pub fn build_request(req: &ChatRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    body.insert("messages".into(), encode_messages(&req.messages)?);
    if !req.tools.is_empty() {
        body.insert("tools".into(), encode_tools(&req.tools));
    }
    if let Some(choice) = &req.tool_choice {
        if let Some(v) = encode_tool_choice(choice)? {
            body.insert("tool_choice".into(), v);
        }
    }
    let s = &req.sampling;
    if let Some(t) = s.temperature {
        body.insert("temperature".into(), t.into());
    }
    if let Some(p) = s.top_p {
        body.insert("p".into(), p.into());
    }
    if let Some(m) = s.max_output_tokens {
        body.insert("max_tokens".into(), m.into());
    }
    if !s.stop.is_empty() {
        body.insert("stop_sequences".into(), s.stop.clone().into());
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
                    other => {
                        return Err(Error::Unsupported(format!(
                            "role {other:?} is not representable in the Cohere chat API"
                        )));
                    }
                };
                out.push(json!({ "role": role, "content": encode_content(&content_parts, role)? }));
            }
        }
    }
    Ok(out.into())
}

#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_assistant(parts: &[&ContentPart]) -> Result<Value, Error> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut msg = Map::new();
    msg.insert("role".into(), "assistant".into());
    for part in parts {
        match part {
            ContentPart::Text { text: t } => text.push_str(t),
            // v2 tool calls carry arguments as a JSON string, not an object.
            ContentPart::ToolCall(tc) => tool_calls.push(json!({
                "id": tc.id,
                "type": "function",
                "function": { "name": tc.name, "arguments": tc.arguments.to_string() },
            })),
            ContentPart::Reasoning(rb) => {
                // Replay only what this protocol issued (e.g. `tool_plan`);
                // opaque fields merge back into the assistant message.
                match &rb.opaque {
                    Value::Object(fields) if rb.protocol == protocol() => {
                        for (k, v) in fields {
                            msg.insert(k.clone(), v.clone());
                        }
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "reasoning block cannot be replayed to the Cohere chat API".into(),
                        ));
                    }
                }
            }
            ContentPart::Media { .. } => {
                return Err(Error::Unsupported(
                    "assistant media parts are not representable in the Cohere chat API".into(),
                ));
            }
            ContentPart::ToolResult(_) => unreachable!("tool results are split earlier"),
            other => {
                return Err(Error::Unsupported(format!(
                    "content part {other:?} is not representable in the Cohere chat API"
                )));
            }
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

/// `content` accepts a plain string or a block array. The string form
/// covers the all-text case; v2 blocks are text-only today.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_content(parts: &[&ContentPart], role: &str) -> Result<Value, Error> {
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
        match part {
            ContentPart::Text { text } => out.push(json!({ "type": "text", "text": text })),
            ContentPart::Media { .. } => {
                return Err(Error::Unsupported(
                    "media parts are not representable in the Cohere v2 chat API".into(),
                ));
            }
            ContentPart::Reasoning(_) => {
                return Err(Error::Unsupported(format!(
                    "reasoning parts are not representable for role {role}"
                )));
            }
            ContentPart::ToolCall(_) => {
                return Err(Error::Unsupported(
                    "tool calls belong to assistant messages only".into(),
                ));
            }
            ContentPart::ToolResult(_) => unreachable!("tool results are split earlier"),
            other => {
                return Err(Error::Unsupported(format!(
                    "content part {other:?} is not representable for role {role}"
                )));
            }
        }
    }
    Ok(out.into())
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

/// v2 tool definitions; `strict` has no Cohere counterpart and does not
/// appear on the wire.
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

/// v2 accepts `REQUIRED`, `NONE` or a function-object tool choice;
/// `Auto` is the server default so it emits no field.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_tool_choice(choice: &ToolChoice) -> Result<Option<Value>, Error> {
    Ok(match choice {
        ToolChoice::Auto => None,
        ToolChoice::None => Some("NONE".into()),
        ToolChoice::Required => Some("REQUIRED".into()),
        ToolChoice::Named(name) => Some(json!({
            "type": "function",
            "function": { "name": name },
        })),
        other => {
            return Err(Error::Unsupported(format!(
                "tool choice {other:?} is not representable in the Cohere chat API"
            )));
        }
    })
}

/// Decode a buffered `/v2/chat` response.
pub fn decode_response(body: &[u8]) -> Result<Output, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("cohere chat response: {e}")))?;
    if let Some(err) = v.get("error").filter(|e| e.is_object()) {
        return Err(provider_error(None, err));
    }
    let message = &v["message"];
    let mut parts = Vec::new();
    match &message["content"] {
        Value::Array(blocks) => {
            for block in blocks {
                parts.push(decode_content_block(block));
            }
        }
        Value::String(text) => {
            if !text.is_empty() {
                parts.push(ContentPart::Text {
                    text: text.clone(),
                });
            }
        }
        _ => {}
    }
    // A tool plan is provider-native planning state: it keeps its wire
    // field as opaque replay state and stays ahead of the tool calls.
    if let Some(plan) = message["tool_plan"].as_str() {
        if !plan.is_empty() {
            parts.push(ContentPart::Reasoning(ReasoningBlock {
                protocol: protocol(),
                opaque: json!({ "tool_plan": plan }),
                summary: Some(plan.to_owned()),
            }));
        }
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for tc in calls {
            parts.push(ContentPart::ToolCall(decode_tool_call(tc)?));
        }
    }
    let claimed = ["id", "message", "finish_reason", "usage"];
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
        stop: v["finish_reason"].as_str().map(stop_reason),
        usage: v.get("usage").filter(|u| !u.is_null()).map(decode_usage),
        extensions,
    })
}

/// A v2 content block → a canonical part. Unknown block types stay
/// replayable as opaque provider state.
fn decode_content_block(block: &Value) -> ContentPart {
    match block["type"].as_str() {
        Some("text") => ContentPart::Text {
            text: block["text"].as_str().unwrap_or_default().to_owned(),
        },
        _ => ContentPart::Reasoning(ReasoningBlock {
            protocol: protocol(),
            opaque: block.clone(),
            summary: None,
        }),
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

fn stop_reason(reason: &str) -> StopReason {
    match reason {
        "COMPLETE" => StopReason::Complete,
        "MAX_TOKENS" => StopReason::Length,
        "TOOL_CALLS" => StopReason::ToolUse,
        other => StopReason::Other(other.to_owned()),
    }
}

/// v2 usage prefers the observed `tokens` block and falls back to
/// `billed_units`; both stay visible through extensions.
fn decode_usage(u: &Value) -> Usage {
    let input = u["tokens"]["input_tokens"]
        .as_u64()
        .or_else(|| u["billed_units"]["input_tokens"].as_u64());
    let output = u["tokens"]["output_tokens"]
        .as_u64()
        .or_else(|| u["billed_units"]["output_tokens"].as_u64());
    let mut extensions = Map::new();
    if let Some(obj) = u.as_object() {
        for (k, v) in obj {
            if k != "tokens" {
                extensions.insert(k.clone(), v.clone());
            }
        }
    }
    if let Some(tokens) = u["tokens"].as_object() {
        let extra: Extensions = tokens
            .iter()
            .filter(|(k, _)| !["input_tokens", "output_tokens"].contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !extra.is_empty() {
            extensions.insert("tokens".into(), extra.into());
        }
    }
    Usage {
        input_tokens: input,
        output_tokens: output,
        extensions,
    }
}

/// Build a `POST /v2/embed` request body.
///
/// The wire takes `texts` or `images`, never both; `input_type` defaults
/// to the matching observed value and yields to `extensions`. Canonical
/// `dimensions` map to v2 `output_dimension`.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
pub fn build_embed_request(req: &EmbedRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    let mut texts: Vec<Value> = Vec::new();
    let mut images = Vec::new();
    for input in &req.inputs {
        match input {
            EmbedInput::Text(t) => texts.push(t.clone().into()),
            EmbedInput::Media(source) => images.push(embed_image(source)?),
            other => {
                return Err(Error::Unsupported(format!(
                    "embed input {other:?} is not representable in the Cohere embed API"
                )));
            }
        }
    }
    if !texts.is_empty() && !images.is_empty() {
        return Err(Error::Unsupported(
            "cohere embed takes texts or images, not both".into(),
        ));
    }
    if !texts.is_empty() {
        body.insert("texts".into(), texts.into());
        body.insert("input_type".into(), "search_document".into());
    }
    if !images.is_empty() {
        body.insert("images".into(), images.into());
        body.insert("input_type".into(), "image".into());
    }
    body.insert("embedding_types".into(), json!(["float"]));
    if let Some(d) = req.dimensions {
        body.insert("output_dimension".into(), d.into());
    }
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// v2 image inputs are URL or base64 data-URI strings.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn embed_image(source: &MediaSource) -> Result<Value, Error> {
    match source {
        MediaSource::Url(u) => Ok(u.clone().into()),
        MediaSource::Inline { media_type, data } => {
            Ok(format!("data:{media_type};base64,{}", base64_encode(data)).into())
        }
        MediaSource::Resource(id) => Err(Error::Unsupported(format!(
            "unresolved resource {id:?}; the adapter resolves resources first"
        ))),
        other => Err(Error::Unsupported(format!(
            "media source {other:?} is not representable in the Cohere embed API"
        ))),
    }
}

/// Decode a `/v2/embed` response.
pub fn decode_embed(body: &[u8]) -> Result<EmbedOutput, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("cohere embed response: {e}")))?;
    let floats = v["embeddings"]["float"]
        .as_array()
        .ok_or_else(|| Error::Protocol("embed response has no float embeddings".into()))?;
    let mut embeddings = Vec::with_capacity(floats.len());
    for row in floats {
        let row = row
            .as_array()
            .ok_or_else(|| Error::Protocol("embedding row is not an array".into()))?;
        embeddings.push(row.iter().map(|n| n.as_f64().unwrap_or(0.0) as f32).collect());
    }
    let usage = v
        .get("meta")
        .filter(|m| m.is_object())
        .map(decode_meta_usage);
    let claimed = ["id", "response_type", "embeddings", "meta"];
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
        usage,
        extensions,
    })
}

/// `meta.billed_units` is the accounting block shared by embed, rerank
/// and classify responses.
fn decode_meta_usage(meta: &Value) -> Usage {
    let billed = &meta["billed_units"];
    let mut extensions = Map::new();
    if let Some(obj) = billed.as_object() {
        let extra: Extensions = obj
            .iter()
            .filter(|(k, _)| !["input_tokens", "output_tokens"].contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !extra.is_empty() {
            extensions.insert("billed_units".into(), extra.into());
        }
    }
    if let Some(obj) = meta.as_object() {
        for (k, v) in obj {
            if k != "billed_units" {
                extensions.insert(k.clone(), v.clone());
            }
        }
    }
    Usage {
        input_tokens: billed["input_tokens"].as_u64(),
        output_tokens: billed["output_tokens"].as_u64(),
        extensions,
    }
}

/// Build a `POST /v2/rerank` request body.
pub fn build_rerank_request(req: &RerankRequest) -> Result<Value, Error> {
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    body.insert("query".into(), req.query.clone().into());
    body.insert("documents".into(), req.documents.clone().into());
    if let Some(top_n) = req.top_n {
        body.insert("top_n".into(), top_n.into());
    }
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// Decode a `/v2/rerank` response. Per-hit fields beyond index/score
/// (document echoes, chunk offsets) have no canonical slot, so the raw
/// result list is preserved in extensions whenever they appear.
pub fn decode_rerank(body: &[u8]) -> Result<RerankOutput, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("cohere rerank response: {e}")))?;
    let results = v["results"]
        .as_array()
        .ok_or_else(|| Error::Protocol("rerank response has no results".into()))?;
    let mut hits = Vec::with_capacity(results.len());
    let mut has_extras = false;
    for r in results {
        has_extras = has_extras
            || r.as_object()
                .map(|o| o.keys().any(|k| !["index", "relevance_score"].contains(&k.as_str())))
                .unwrap_or(false);
        hits.push(RerankHit {
            index: r["index"].as_u64().unwrap_or(0) as usize,
            score: r["relevance_score"].as_f64().unwrap_or(0.0),
        });
    }
    let usage = v
        .get("meta")
        .filter(|m| m.is_object())
        .map(decode_meta_usage);
    let claimed = ["id", "results", "meta"];
    let mut extensions = Map::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                extensions.insert(k.clone(), val.clone());
            }
        }
    }
    if has_extras {
        extensions.insert("results".into(), v["results"].clone());
    }
    Ok(RerankOutput {
        results: hits,
        usage,
        extensions,
    })
}

/// Build a `POST /v1/classify` request body.
///
/// v1 classify scores inputs against caller-supplied labelled examples;
/// it has no closed-label parameter, so a canonical `labels` list has no
/// honest mapping — examples ride `extensions` instead.
pub fn build_classify_request(req: &ClassifyRequest) -> Result<Value, Error> {
    if !req.labels.is_empty() {
        return Err(Error::Unsupported(
            "cohere classify scores labelled examples, not a closed label set; \
             pass `examples` via extensions"
                .into(),
        ));
    }
    let mut body = Map::new();
    body.insert("model".into(), req.model.clone().into());
    body.insert("inputs".into(), req.inputs.clone().into());
    for (k, v) in &req.extensions {
        body.insert(k.clone(), v.clone());
    }
    Ok(body.into())
}

/// Decode a `/v1/classify` response. Each classification echoes its
/// input text; the canonical `input` index is positional, matching the
/// request's `inputs` order. The raw list stays in extensions because
/// per-label confidence maps have no canonical slot.
pub fn decode_classify(body: &[u8]) -> Result<ClassifyOutput, Error> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| Error::Protocol(format!("cohere classify response: {e}")))?;
    let classifications = v["classifications"]
        .as_array()
        .ok_or_else(|| Error::Protocol("classify response has no classifications".into()))?;
    let mut predictions = Vec::with_capacity(classifications.len());
    for (i, c) in classifications.iter().enumerate() {
        predictions.push(Classification {
            input: i,
            label: c["prediction"].as_str().unwrap_or_default().to_owned(),
            confidence: c["confidence"].as_f64(),
        });
    }
    let claimed = ["id", "classifications"];
    let mut extensions = Map::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                extensions.insert(k.clone(), val.clone());
            }
        }
    }
    extensions.insert("classifications".into(), v["classifications"].clone());
    Ok(ClassifyOutput {
        predictions,
        usage: None,
        extensions,
    })
}

/// Map a provider error body into `Error::Provider`.
/// `status` is the HTTP status when known (stream errors may lack one).
/// Cohere's envelope is a flat `{"message": "..."}`.
pub fn provider_error(status: Option<u16>, body: &Value) -> Error {
    let err = body
        .get("error")
        .filter(|e| e.is_object())
        .unwrap_or(body);
    Error::Provider {
        status,
        code: err["code"].as_str().map(str::to_owned),
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

/// Stateful decoder: v2 chat SSE frames → canonical events.
///
/// The wire `index` on content/tool events maps to canonical part
/// indices, so text, tool plans and tool calls keep their identities
/// even when the provider interleaves them.
#[derive(Default)]
pub struct ChatStreamDecoder {
    parts: HashMap<u64, u32>,
    next_part: u32,
}

impl ChatStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Canonical index for a wire index; lazily assigns and reports a
    /// `PartStarted` event only for newly seen indices.
    fn open(&mut self, wire: u64, kind: PartKind) -> (u32, Option<StreamEvent>) {
        if let Some(i) = self.parts.get(&wire) {
            return (*i, None);
        }
        let i = self.next_part;
        self.next_part += 1;
        self.parts.insert(wire, i);
        (i, Some(StreamEvent::PartStarted { index: i, kind }))
    }

    /// Canonical index without a `PartStarted`: tool calls announce
    /// themselves through `ToolCallStarted`.
    fn part(&mut self, wire: u64) -> u32 {
        if let Some(i) = self.parts.get(&wire) {
            return *i;
        }
        let i = self.next_part;
        self.next_part += 1;
        self.parts.insert(wire, i);
        i
    }
}

impl crate::transport::SseEventDecoder for ChatStreamDecoder {
    fn decode(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, Error> {
        if event.data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let v: Value = serde_json::from_str(&event.data)
            .map_err(|e| Error::Protocol(format!("cohere stream frame: {e}")))?;
        if v.get("error").is_some() {
            return Err(provider_error(None, &v));
        }
        let ty = v["type"]
            .as_str()
            .or_else(|| event.event.as_deref())
            .unwrap_or_default();
        let wire = v["index"].as_u64().unwrap_or(u64::MAX);
        let delta = &v["delta"]["message"];
        let mut out = Vec::new();
        match ty {
            // Envelope only: the message id is not a stream event.
            "message-start" => {}
            "content-start" => {
                let content = &delta["content"];
                match content["type"].as_str().unwrap_or("text") {
                    "text" => {
                        if let (_, Some(ev)) = self.open(wire, PartKind::Text) {
                            out.push(ev);
                        }
                    }
                    _ => {
                        let (index, ev) = self.open(wire, PartKind::Other);
                        if let Some(ev) = ev {
                            out.push(ev);
                        }
                        out.push(StreamEvent::Native {
                            part_index: Some(index),
                            value: content.clone(),
                        });
                    }
                }
            }
            "content-delta" => {
                if let Some(text) = delta["content"]["text"].as_str() {
                    if !text.is_empty() {
                        let (index, ev) = self.open(wire, PartKind::Text);
                        if let Some(ev) = ev {
                            out.push(ev);
                        }
                        out.push(StreamEvent::TextDelta {
                            index,
                            text: text.to_owned(),
                        });
                    }
                }
            }
            "content-end" => {
                if let Some(index) = self.parts.get(&wire) {
                    out.push(StreamEvent::PartFinished { index: *index });
                }
            }
            // The plan streams as reasoning text before any tool call.
            "tool-plan-delta" => {
                if let Some(text) = delta["tool_plan"].as_str() {
                    if !text.is_empty() {
                        let (index, ev) = self.open(wire, PartKind::Reasoning);
                        if let Some(ev) = ev {
                            out.push(ev);
                        }
                        out.push(StreamEvent::ReasoningDelta {
                            index,
                            text: text.to_owned(),
                        });
                    }
                }
            }
            "tool-call-start" => {
                let index = self.part(wire);
                let tc = &delta["tool_calls"];
                let function = &tc["function"];
                out.push(StreamEvent::ToolCallStarted {
                    index,
                    id: tc["id"].as_str().unwrap_or_default().to_owned(),
                    name: function["name"].as_str().unwrap_or_default().to_owned(),
                });
                if let Some(args) = function["arguments"].as_str() {
                    if !args.is_empty() {
                        out.push(StreamEvent::ToolCallDelta {
                            index,
                            arguments: args.to_owned(),
                        });
                    }
                }
            }
            "tool-call-delta" => {
                let index = self.part(wire);
                if let Some(args) = delta["tool_calls"]["function"]["arguments"].as_str() {
                    if !args.is_empty() {
                        out.push(StreamEvent::ToolCallDelta {
                            index,
                            arguments: args.to_owned(),
                        });
                    }
                }
            }
            "tool-call-end" => {
                if let Some(index) = self.parts.get(&wire) {
                    out.push(StreamEvent::PartFinished { index: *index });
                }
            }
            "message-end" => {
                if let Some(u) = v["delta"].get("usage").filter(|u| u.is_object()) {
                    out.push(StreamEvent::Usage(decode_usage(u)));
                }
                if let Some(reason) = v["delta"]["finish_reason"].as_str() {
                    out.push(StreamEvent::Finished {
                        stop: Some(stop_reason(reason)),
                    });
                }
            }
            // Citations, debug frames and unknown events stay verbatim.
            _ => {
                out.push(StreamEvent::Native {
                    part_index: None,
                    value: v.clone(),
                });
            }
        }
        Ok(out)
    }
}
