//! The external-client access lane: reach provider models through a
//! host-spawned client process.
//!
//! Some subscriptions are reachable only through a provider's own CLI or
//! agent binary — editor/agent CLIs, ACP-style servers, plugin shims.
//! This module is the adapter for that lane: `AuthKind::External`
//! declares the dependency, [`EffectHost::spawn`] runs the process under
//! host control, and a [`ClientProtocol`] describes the client's IPC.
//!
//! The custody boundary is deliberate: the client binary carries its own
//! login. The library never extracts, refreshes or relays that credential
//! — it encodes a request, feeds it to the process and decodes the
//! output. `Credential::None` is the only material the resolver may
//! return for these targets; anything else is refused because there is
//! no honest channel to attach it to a spawned process.
//!
//! Layer split: the foundation (this module) owns byte framing, part
//! lifecycle and process lifecycle; each client protocol owns only its
//! IPC semantics. [`JsonLinesProtocol`] covers the dominant shape —
//! newline-delimited JSON requests in, newline-delimited JSON events out
//! (the ACP/Copilot-CLI family). One process serves one operation.

use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use futures_core::Stream;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::*;
use crate::auth::{AuthKind, Credential, CredentialResolver, CredentialScope};
use crate::effects::{EffectHost, Process, ProcessSpec, ResourceRef};
use crate::transport::Transport;

/// Bytes of diagnostic process output the session retains for error
/// reporting (the merged stdout/stderr tail — `Process` cannot separate
/// the two pipes, so the tail is the raw output tail).
const DIAGNOSTIC_TAIL: usize = 4096;
/// How much of the tail surfaces in a `Provider` error message.
const TAIL_SNIPPET: usize = 512;

/// How the foundation splits process output before handing frames to
/// [`ClientProtocol::decode`]. The mechanism lives here so a protocol
/// implementation only maps complete frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClientFraming {
    /// Newline-delimited frames (JSONL/NDJSON — the ACP/Copilot-CLI
    /// shape). Lines may split across arbitrary chunk boundaries;
    /// `\r\n` is accepted; empty lines are skipped.
    Lines,
    /// The entire process output is one frame, delivered once at
    /// end-of-stream — for one-shot clients that take the request via
    /// argv or a single stdin write and print one result.
    Whole,
}

/// The per-client IPC contract for an external client process.
///
/// Implementations describe how the client is launched, how a
/// `ChatRequest` becomes stdin bytes, and how a complete output frame
/// becomes canonical events. `decode` and `finish` take `&self` on
/// purpose: all incremental state (partial frames, part indices, output
/// tail) is the foundation's per-process session state, so one protocol
/// object is safe under concurrent requests.
pub trait ClientProtocol: Send + Sync {
    /// The declared protocol-family name. `ServiceTarget::protocol` must
    /// be `Protocol::Other(name)` with this name for the adapter to
    /// bind — the gate tying the target's wire family to this client.
    fn name(&self) -> &str;

    /// How to launch the client for `model`: requirement string the
    /// host checks (e.g. `"cursor-agent >= 1.2"`), program and argv.
    fn spec(&self, model: &str) -> ProcessSpec;

    /// The stdin frames to write after spawn. Empty for clients that
    /// take the request via argv or environment only. Frames are sent
    /// in order before output is drained.
    fn encode(&self, req: &ChatRequest) -> Result<Vec<bytes::Bytes>, Error>;

    /// Map one complete output frame (per `framing()`) into canonical
    /// events appended to `out`. Implementations tolerate client
    /// diagnostics on the merged stream by skipping or surfacing them
    /// as `StreamEvent::Native`; they must not require mutable session
    /// state.
    fn decode(&self, chunk: &[u8], out: &mut Vec<StreamEvent>) -> Result<(), Error>;

    /// End-of-process handling. A nonzero exit is a `Error::Provider`;
    /// the adapter enriches it with the captured output tail.
    fn finish(&self, exit_code: i32, out: &mut Vec<StreamEvent>) -> Result<(), Error>;

    /// The framing the foundation applies before `decode`.
    fn framing(&self) -> ClientFraming {
        ClientFraming::Lines
    }
}

// ---------------------------------------------------------------------------
// JsonLinesProtocol
// ---------------------------------------------------------------------------

/// The dominant external-client shape: one JSON-RPC-style request line
/// in, newline-delimited JSON event objects out.
///
/// Outbound, `encode` writes a single line
/// `{"jsonrpc":"2.0","id":1,"method":<method>,"params":{model, messages,
/// tools, sampling, …}}` where `params` carries canonical field names
/// verbatim plus request `extensions` merged last.
///
/// Inbound, each line is classified generically:
///
/// * JSON-RPC error responses (`{"id", "error":{code,message}}`) and
///   `{"type":"error"}` events → `Error::Provider`;
/// * JSON-RPC results (`{"id", "result":{…}}`) and terminal events
///   (`type` ∈ `result|done|finished|complete(d)|end`, or `done:true`) →
///   optional trailing text delta, `Usage` (camelCase or snake_case
///   token fields), then `Finished{stop}` from
///   `stopReason|stop_reason|finishReason|finish_reason|stop|reason`;
/// * tool events (`type` contains `tool`) → `ToolCallStarted` from
///   `id|toolCallId|tool_call_id|callId` + `name|toolName|tool_name|title`,
///   then `ToolCallDelta` carrying `arguments|args|input` as JSON text;
/// * reasoning (`type` contains `thought|reasoning|thinking`) →
///   `ReasoningDelta`;
/// * `usage`/`tokens` events → `Usage`;
/// * everything else → a text delta when a text payload is found at
///   `text|delta`, `content` (string, `.text`, or text-block array),
///   `data.content|data.text`, `message` (string or `.content` blocks),
///   else the event passes through verbatim as `StreamEvent::Native`.
///
/// Notifications `{"method", "params":{…}}` decode their `params.update`
/// (or `params`) as the event object, so ACP `session/update` shapes map
/// through the same layer. Non-JSON lines are tolerated as diagnostics —
/// the session's output tail preserves them for error reporting.
/// Part indices come from `index|partIndex|part|part_index`, default 0.
pub struct JsonLinesProtocol {
    /// Protocol-family name for the `Protocol::Other` gate.
    name: String,
    /// Requirement string forwarded to `ProcessSpec::requirement`.
    requirement: String,
    program: String,
    /// Argv template; `{model}` is substituted per request.
    args: Vec<String>,
    env: Vec<(String, String)>,
    cwd: Option<String>,
    /// JSON-RPC method name for the request line.
    method: String,
    /// Client-specific params merged under request fields and
    /// extensions (lowest precedence).
    params: Extensions,
}

impl JsonLinesProtocol {
    /// A JSONL client: declared protocol name, runtime requirement and
    /// program. Defaults: no args, method `"session/prompt"`.
    pub fn new(name: &str, requirement: &str, program: &str) -> Self {
        Self {
            name: name.to_owned(),
            requirement: requirement.to_owned(),
            program: program.to_owned(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: None,
            method: "session/prompt".to_owned(),
            params: Extensions::new(),
        }
    }

    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    pub fn with_env(mut self, env: Vec<(String, String)>) -> Self {
        self.env = env;
        self
    }

    pub fn with_cwd(mut self, cwd: String) -> Self {
        self.cwd = Some(cwd);
        self
    }

    /// Override the request method (e.g. `"chat"`, `"session/prompt"`).
    pub fn with_method(mut self, method: &str) -> Self {
        self.method = method.to_owned();
        self
    }

    /// Extra params merged into the request object below request fields.
    pub fn with_params(mut self, params: Extensions) -> Self {
        self.params = params;
        self
    }

    fn terminal(&self, ev: &Value, out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        // A bare string payload is the whole answer text.
        if let Value::String(t) = ev {
            if !t.is_empty() {
                out.push(StreamEvent::TextDelta {
                    index: 0,
                    text: t.clone(),
                });
            }
        }
        if let Some(t) = first_str(ev, &["result", "text", "content", "delta"]) {
            if !t.is_empty() {
                out.push(StreamEvent::TextDelta {
                    index: 0,
                    text: t.to_owned(),
                });
            }
        }
        if let Some(u) = ev
            .get("usage")
            .or_else(|| ev.get("meta").and_then(|m| m.get("usage")))
            .and_then(usage_from)
        {
            out.push(StreamEvent::Usage(u));
        }
        let stop = first_str(
            ev,
            &[
                "stopReason",
                "stop_reason",
                "finishReason",
                "finish_reason",
                "stop",
                "reason",
            ],
        )
        .map(stop_reason);
        // Unclaimed fields stay visible verbatim.
        let rest = unclaimed(
            ev,
            &[
                "result",
                "text",
                "content",
                "delta",
                "usage",
                "meta",
                "stopReason",
                "stop_reason",
                "finishReason",
                "finish_reason",
                "stop",
                "reason",
                "id",
                "jsonrpc",
            ],
        );
        if !rest.is_empty() {
            out.push(StreamEvent::Native {
                part_index: None,
                value: Value::Object(rest),
            });
        }
        out.push(StreamEvent::Finished { stop });
        Ok(())
    }

    fn map_event(&self, ev: &Value, out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        if !ev.is_object() {
            out.push(StreamEvent::Native {
                part_index: None,
                value: ev.clone(),
            });
            return Ok(());
        }
        let kind = first_str(ev, &["type", "sessionUpdate", "event", "kind"])
            .map(|s| s.to_ascii_lowercase());
        let index = first_u64(ev, &["index", "partIndex", "part", "part_index"])
            .map(|i| i as u32)
            .unwrap_or(0);
        let kind_str = kind.as_deref().unwrap_or("");
        // Error events, in either envelope.
        if kind_str.contains("error")
            || kind_str == "failed"
            || ev.get("error").is_some_and(|e| e.is_object() || e.is_string())
        {
            return Err(event_error(ev.get("error").unwrap_or(ev)));
        }
        // Terminal events.
        if ev.get("done").and_then(Value::as_bool) == Some(true) || is_terminal(kind_str) {
            return self.terminal(ev, out);
        }
        // Usage events.
        if kind_str == "usage" || kind_str == "tokens" || kind_str.contains("usage") {
            match usage_from(ev.get("usage").unwrap_or(ev)) {
                Some(u) => out.push(StreamEvent::Usage(u)),
                None => out.push(StreamEvent::Native {
                    part_index: None,
                    value: ev.clone(),
                }),
            }
            return Ok(());
        }
        // Tool-call events.
        if kind_str.contains("tool") {
            let id = first_str(
                ev,
                &["id", "toolCallId", "tool_call_id", "callId", "call_id"],
            )
            .unwrap_or_default();
            let name = first_str(ev, &["name", "toolName", "tool_name", "title"])
                .unwrap_or_default();
            out.push(StreamEvent::ToolCallStarted {
                index,
                id: id.to_owned(),
                name: name.to_owned(),
            });
            if let Some(args) = ev
                .get("arguments")
                .or_else(|| ev.get("args"))
                .or_else(|| ev.get("input"))
            {
                let text = match args {
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
            return Ok(());
        }
        // Reasoning events.
        if kind_str.contains("thought")
            || kind_str.contains("reasoning")
            || kind_str.contains("thinking")
        {
            match first_str(ev, &["text", "delta", "thought", "reasoning", "thinking"])
                .map(str::to_owned)
                .or_else(|| text_probe(ev))
            {
                Some(t) if !t.is_empty() => {
                    out.push(StreamEvent::ReasoningDelta { index, text: t });
                }
                _ => out.push(StreamEvent::Native {
                    part_index: None,
                    value: ev.clone(),
                }),
            }
            return Ok(());
        }
        // Text-bearing events of any kind, then lossless passthrough.
        if let Some(t) = text_probe(ev) {
            if !t.is_empty() {
                out.push(StreamEvent::TextDelta { index, text: t });
                return Ok(());
            }
        }
        out.push(StreamEvent::Native {
            part_index: None,
            value: ev.clone(),
        });
        Ok(())
    }
}

impl ClientProtocol for JsonLinesProtocol {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec(&self, model: &str) -> ProcessSpec {
        ProcessSpec {
            requirement: self.requirement.clone(),
            program: self.program.clone(),
            args: self
                .args
                .iter()
                .map(|a| a.replace("{model}", model))
                .collect(),
            env: self.env.clone(),
            cwd: self.cwd.clone(),
        }
    }

    /// One request line: `{"jsonrpc":"2.0","id":1,"method":…,"params":…}`.
    fn encode(&self, req: &ChatRequest) -> Result<Vec<bytes::Bytes>, Error> {
        let mut params = self.params.clone();
        params.insert("model".into(), req.model.clone().into());
        params.insert("messages".into(), encode_messages(&req.messages)?);
        if !req.tools.is_empty() {
            params.insert("tools".into(), encode_tools(&req.tools));
        }
        if let Some(choice) = &req.tool_choice {
            params.insert("tool_choice".into(), encode_tool_choice(choice));
        }
        let s = &req.sampling;
        if let Some(t) = s.temperature {
            params.insert("temperature".into(), t.into());
        }
        if let Some(p) = s.top_p {
            params.insert("top_p".into(), p.into());
        }
        if let Some(m) = s.max_output_tokens {
            params.insert("max_output_tokens".into(), m.into());
        }
        if !s.stop.is_empty() {
            params.insert("stop".into(), s.stop.clone().into());
        }
        for (k, v) in &req.extensions {
            params.insert(k.clone(), v.clone());
        }
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": self.method,
            "params": params,
        });
        let mut line = serde_json::to_vec(&body)
            .map_err(|e| Error::Protocol(format!("request encode: {e}")))?;
        line.push(b'\n');
        Ok(vec![line.into()])
    }

    /// One complete line → canonical events. Non-JSON lines are skipped
    /// as client diagnostics (retained in the session's output tail);
    /// a line that starts as JSON but fails to parse is a violation.
    fn decode(&self, chunk: &[u8], out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        let trimmed: &[u8] = {
            let mut t = chunk;
            while let Some((&b, rest)) = t.split_first() {
                if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                    t = rest;
                } else {
                    break;
                }
            }
            t
        };
        if trimmed.is_empty() {
            return Ok(());
        }
        let v: Value = match serde_json::from_slice(trimmed) {
            Ok(v) => v,
            Err(e) => {
                if trimmed[0] == b'{' || trimmed[0] == b'[' {
                    return Err(Error::UpstreamProtocol(format!("malformed JSON line: {e}")));
                }
                return Ok(());
            }
        };
        // A JSON-RPC response: id (or the version marker), a
        // result/error member and no event discriminator.
        let is_response = (v.get("id").is_some() || v.get("jsonrpc").is_some())
            && v.get("method").is_none()
            && v.get("type").is_none()
            && v.get("sessionUpdate").is_none()
            && (v.get("result").is_some() || v.get("error").is_some());
        if is_response {
            if let Some(err) = v.get("error") {
                return Err(event_error(err));
            }
            return self.terminal(v.get("result").unwrap_or(&Value::Null), out);
        }
        // A notification or bare event: `params.update` (ACP
        // `session/update`) else `params`, else the object itself.
        let ev = if v.get("method").is_some() {
            let params = &v["params"];
            params.get("update").unwrap_or(params)
        } else {
            &v
        };
        self.map_event(ev, out)
    }

    fn finish(&self, exit_code: i32, _out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        if exit_code == 0 {
            return Ok(());
        }
        Err(Error::Provider {
            status: None,
            code: Some(format!("exit:{exit_code}")),
            message: format!(
                "external client {:?} exited with status {exit_code}",
                self.program
            ),
            // Acceptance is uncertain once a process ran partway.
            retry: RetrySafety::Unsafe,
            raw: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Request encoding helpers (canonical → neutral JSON params)
// ---------------------------------------------------------------------------

#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_messages(messages: &[Message]) -> Result<Value, Error> {
    let mut out = Vec::new();
    for m in messages {
        let role = match &m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Other(r) => r.as_str(),
            other => {
                return Err(Error::Unsupported(format!(
                    "role {other:?} is not representable for an external client"
                )));
            }
        };
        let structured = m
            .content
            .iter()
            .any(|p| !matches!(p, ContentPart::Text { .. }));
        let content = if structured {
            let blocks: Result<Vec<Value>, Error> =
                m.content.iter().map(encode_part).collect();
            Value::Array(blocks?)
        } else {
            let mut text = String::new();
            for p in &m.content {
                if let ContentPart::Text { text: t } = p {
                    text.push_str(t);
                }
            }
            text.into()
        };
        out.push(json!({"role": role, "content": content}));
    }
    Ok(out.into())
}

/// One canonical part → a neutral `{"type":…}` block. `Resource` media
/// must already be resolved by the adapter's effect port.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_part(part: &ContentPart) -> Result<Value, Error> {
    match part {
        ContentPart::Text { text } => Ok(json!({"type": "text", "text": text})),
        ContentPart::Media {
            kind,
            source,
            detail,
        } => {
            let mut block = Extensions::new();
            let t = match kind {
                MediaKind::Image => "image",
                MediaKind::Audio => "audio",
                MediaKind::Video => "video",
                MediaKind::File => "file",
                _ => "file",
            };
            block.insert("type".into(), t.into());
            match source {
                MediaSource::Inline { media_type, data } => {
                    block.insert("mimeType".into(), media_type.clone().into());
                    block.insert("data".into(), BASE64.encode(data).into());
                }
                MediaSource::Url(u) => {
                    block.insert("url".into(), u.clone().into());
                }
                MediaSource::Resource(id) => {
                    return Err(Error::Unsupported(format!(
                        "unresolved resource {id:?}; the adapter resolves resources first"
                    )));
                }
                other => {
                    return Err(Error::Unsupported(format!(
                        "media source {other:?} is not representable for an external client"
                    )));
                }
            }
            if let Some(d) = detail {
                block.insert("detail".into(), d.clone().into());
            }
            Ok(block.into())
        }
        ContentPart::ToolCall(tc) => Ok(json!({
            "type": "toolCall",
            "id": tc.id,
            "name": tc.name,
            "arguments": tc.arguments,
        })),
        ContentPart::ToolResult(tr) => {
            let content: Result<Vec<Value>, Error> =
                tr.content.iter().map(encode_part).collect();
            Ok(json!({
                "type": "toolResult",
                "callId": tr.call_id,
                "content": content?,
                "isError": tr.is_error,
            }))
        }
        // Reasoning state replays verbatim; the client decides what to
        // do with a foreign opaque payload.
        ContentPart::Reasoning(rb) => {
            let mut block = Extensions::new();
            block.insert("type".into(), "reasoning".into());
            block.insert("data".into(), rb.opaque.clone());
            if let Some(s) = &rb.summary {
                block.insert("summary".into(), s.clone().into());
            }
            Ok(block.into())
        }
        other => Err(Error::Unsupported(format!(
            "content part {other:?} is not representable for an external client"
        ))),
    }
}

fn encode_tools(tools: &[ToolDef]) -> Value {
    tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": t.strict,
            })
        })
        .collect()
}

#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => "auto".into(),
        ToolChoice::None => "none".into(),
        ToolChoice::Required => "required".into(),
        ToolChoice::Named(n) => json!({"type": "named", "name": n}),
        other => return json!({ "unsupported": format!("{other:?}") }),
    }
}

// ---------------------------------------------------------------------------
// Event-decoding helpers (JSON object → canonical events)
// ---------------------------------------------------------------------------

fn first_str<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| v.get(*k).and_then(Value::as_str))
}

fn first_u64(v: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|k| v.get(*k).and_then(|x| x.as_u64().or_else(|| x.as_i64().map(|i| i as u64))))
}

fn is_terminal(kind: &str) -> bool {
    matches!(
        kind,
        "result" | "done" | "finished" | "finish" | "end" | "complete" | "completed" | "end_turn"
    )
}

fn stop_reason(s: &str) -> StopReason {
    match s {
        "stop" | "end_turn" | "complete" | "completed" | "finished" | "done" => StopReason::Complete,
        "length" | "max_tokens" | "max_output_tokens" | "truncated" => StopReason::Length,
        "tool_use" | "tool_calls" | "function_call" => StopReason::ToolUse,
        "filtered" | "content_filter" | "refusal" => StopReason::Filtered,
        "pause" | "paused" => StopReason::Paused,
        other => StopReason::Other(other.to_owned()),
    }
}

/// Token counters under the common camelCase and snake_case spellings;
/// unclaimed fields ride `extensions` verbatim.
fn usage_from(u: &Value) -> Option<Usage> {
    let input = first_u64(
        u,
        &[
            "inputTokens",
            "input_tokens",
            "promptTokens",
            "prompt_tokens",
        ],
    );
    let output = first_u64(
        u,
        &[
            "outputTokens",
            "output_tokens",
            "completionTokens",
            "completion_tokens",
        ],
    );
    let extensions = unclaimed(
        u,
        &[
            "inputTokens",
            "input_tokens",
            "promptTokens",
            "prompt_tokens",
            "outputTokens",
            "output_tokens",
            "completionTokens",
            "completion_tokens",
        ],
    );
    if input.is_none() && output.is_none() && extensions.is_empty() {
        return None;
    }
    Some(Usage {
        input_tokens: input,
        output_tokens: output,
        extensions,
    })
}

/// The object minus the keys a mapping already consumed.
fn unclaimed(v: &Value, claimed: &[&str]) -> Extensions {
    let mut rest = Extensions::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !claimed.contains(&k.as_str()) {
                rest.insert(k.clone(), val.clone());
            }
        }
    }
    rest
}

/// A client error object (`{"code","message"}`) or bare string →
/// `Error::Provider`. JSON-RPC reserved codes (parse, invalid request,
/// method/params, internal) mean the request was rejected before work —
/// `Safe`; anything else is `Unsafe`.
fn event_error(src: &Value) -> Error {
    let code = match src.get("code") {
        Some(Value::Number(n)) => Some(n.to_string()),
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };
    let message = src
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| src.as_str())
        .unwrap_or("external client error")
        .to_owned();
    // JSON-RPC reserved codes for pre-work rejections are Safe; -32603
    // (internal error) and -32000.. server errors already ran work.
    let retry = match code.as_deref().and_then(|c| c.parse::<i64>().ok()) {
        Some(-32700 | -32600 | -32601 | -32602) => RetrySafety::Safe,
        _ => RetrySafety::Unsafe,
    };
    Error::Provider {
        status: None,
        code,
        message,
        retry,
        raw: Some(src.clone()),
    }
}

/// Probe the common text-payload shapes: `text`/`delta` strings, a
/// `content` string / `{text}` object / text-block array, `data.text` /
/// `data.content`, and `message` string or `.content` blocks.
fn text_probe(ev: &Value) -> Option<String> {
    for k in ["text", "delta"] {
        if let Some(t) = ev.get(k).and_then(Value::as_str) {
            return Some(t.to_owned());
        }
    }
    if let Some(t) = ev["content"].as_str() {
        return Some(t.to_owned());
    }
    if let Some(t) = ev["content"].get("text").and_then(Value::as_str) {
        return Some(t.to_owned());
    }
    if let Some(arr) = ev["content"].as_array() {
        if let Some(t) = join_text_blocks(arr) {
            return Some(t);
        }
    }
    if let Some(t) = ev["data"].get("text").and_then(Value::as_str) {
        return Some(t.to_owned());
    }
    if let Some(t) = ev["data"].get("content").and_then(Value::as_str) {
        return Some(t.to_owned());
    }
    if let Some(t) = ev["message"].as_str() {
        return Some(t.to_owned());
    }
    if let Some(arr) = ev["message"].get("content").and_then(Value::as_array) {
        if let Some(t) = join_text_blocks(arr) {
            return Some(t);
        }
    }
    None
}

fn join_text_blocks(arr: &[Value]) -> Option<String> {
    let mut s = String::new();
    for b in arr {
        if let Some(t) = b.get("text").and_then(Value::as_str) {
            s.push_str(t);
        } else if let Some(t) = b.as_str() {
            s.push_str(t);
        }
    }
    if s.is_empty() { None } else { Some(s) }
}

// ---------------------------------------------------------------------------
// ExternalAdapter
// ---------------------------------------------------------------------------

/// Adapter for the external-client lane: spawns the client process per
/// operation through the host effect port and drives a `ClientProtocol`.
///
/// `transport` is retained for signature parity with the other adapters
/// (external clients make no HTTP calls through this crate) and for
/// future multiplexed transports. Every operation family except
/// `chat`/`chat_stream` reports `Unsupported` — external CLIs generally
/// expose no listing, embedding or job surfaces.
pub struct ExternalAdapter {
    target: ServiceTarget,
    /// Declared client identity from `AuthKind::External` (diagnostics).
    client: String,
    #[allow(dead_code)] // parity field; external lane never sends HTTP
    transport: Arc<dyn Transport>,
    resolver: Arc<dyn CredentialResolver>,
    effects: Arc<dyn EffectHost>,
    protocol: Arc<dyn ClientProtocol>,
}

impl ExternalAdapter {
    /// Bind an external client. Requires `auth` to be
    /// `AuthKind::External` and `target.protocol` to be
    /// `Protocol::Other(protocol.name())` — the client declares its wire
    /// family; the target must speak it.
    pub fn new(
        target: ServiceTarget,
        auth: AuthKind,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
        effects: Arc<dyn EffectHost>,
        protocol: Arc<dyn ClientProtocol>,
    ) -> Result<Self, Error> {
        let client = match &auth {
            AuthKind::External { client } => client.clone(),
            other => {
                return Err(Error::Unsupported(format!(
                    "ExternalAdapter requires AuthKind::External, got {other:?}"
                )));
            }
        };
        match &target.protocol {
            Protocol::Other(name) if name == protocol.name() => {}
            other => {
                return Err(Error::Unsupported(format!(
                    "ExternalAdapter cannot serve protocol {other:?} (client declares {:?})",
                    protocol.name()
                )));
            }
        }
        Ok(Self {
            target,
            client,
            transport,
            resolver,
            effects,
            protocol,
        })
    }

    /// The resolver must yield `Credential::None`: the client binary
    /// carries its own login and there is no channel to attach material
    /// to a spawned process without leaking it.
    #[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
    async fn check_credential(&self) -> Result<(), Error> {
        let scope = CredentialScope {
            target: &self.target,
            auth: &AuthKind::External {
                client: self.client.clone(),
            },
        };
        match self.resolver.resolve(scope).await? {
            Credential::None => Ok(()),
            Credential::ApiKey(_) => Err(Error::Credential(
                "external client processes carry their own auth; API-key material cannot attach".into(),
            )),
            Credential::Bearer { .. } => Err(Error::Credential(
                "external client processes carry their own auth; bearer material cannot attach".into(),
            )),
            Credential::Cloud { .. } => Err(Error::Credential(
                "external client processes carry their own auth; cloud identity cannot attach".into(),
            )),
            other => Err(Error::Credential(format!(
                "credential kind {other:?} cannot attach to an external client process"
            ))),
        }
    }

    /// Resolve `MediaSource::Resource` parts into inline data through
    /// the host effect port before the request enters `encode`.
    async fn resolve_media(&self, req: &mut ChatRequest, ctrl: &RequestControl) -> Result<(), Error> {
        for m in &mut req.messages {
            for part in &mut m.content {
                if let ContentPart::Media { kind: _, source, .. } = part {
                    if let MediaSource::Resource(id) = source {
                        let mut rr = ResourceRef {
                            id: id.clone(),
                            media_type: None,
                        };
                        let data = self.effects.read_resource(&mut rr, ctrl).await?;
                        *source = MediaSource::Inline {
                            media_type: rr
                                .media_type
                                .unwrap_or_else(|| "application/octet-stream".into()),
                            data,
                        };
                    }
                }
            }
        }
        Ok(())
    }

    /// Credential check → spec → spawn → write the encoded stdin frames.
    /// Any failure after spawn terminates and reaps the process.
    async fn open_session(
        &self,
        req: &ChatRequest,
        ctrl: &RequestControl,
    ) -> Result<ProcessSession, Error> {
        if ctrl.cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        self.check_credential().await?;
        let spec = self.protocol.spec(&req.model);
        let frames = self.protocol.encode(req)?;
        let process = self.effects.spawn(&spec, ctrl).await?;
        let mut session =
            ProcessSession::new(process, self.protocol.clone(), ctrl.cancel.clone());
        for frame in frames {
            if let Err(e) = session.process.send_input(frame) {
                session.abort();
                return Err(e);
            }
        }
        Ok(session)
    }
}

#[async_trait::async_trait]
impl Adapter for ExternalAdapter {
    fn target(&self) -> &ServiceTarget {
        &self.target
    }

    /// Spawn → write frames → drain → assemble the canonical `Output`.
    async fn chat(&self, mut req: ChatRequest, ctrl: &RequestControl) -> Result<Output, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let mut session = self.open_session(&req, ctrl).await?;
        let mut events = Vec::new();
        while let Some(item) =
            std::future::poll_fn(|cx| Pin::new(&mut session).poll_next(cx)).await
        {
            events.push(item?);
        }
        Ok(assemble_output(&events, self.protocol.name()))
    }

    /// The returned stream owns the process: drop or termination always
    /// runs `terminate`/`wait`, and cancellation surfaces as a terminal
    /// `Error::Cancelled` item after any already-decoded events.
    async fn chat_stream(
        &self,
        mut req: ChatRequest,
        ctrl: &RequestControl,
    ) -> Result<DynStream<'static, StreamEvent>, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let session = self.open_session(&req, ctrl).await?;
        Ok(Box::pin(session))
    }
}

// ---------------------------------------------------------------------------
// Process session: framing + part lifecycle + process lifecycle
// ---------------------------------------------------------------------------

/// Incremental newline framing owned by the foundation. Lines may split
/// across arbitrary chunk boundaries; `\r\n` is accepted.
struct LineFramer {
    buf: Vec<u8>,
}

impl LineFramer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Feed bytes; returns every complete non-empty line now decodable.
    fn feed(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..pos).collect();
            self.buf.drain(..1);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if !line.is_empty() {
                out.push(line);
            }
        }
        out
    }

    /// End of stream: a trailing unterminated line still counts.
    fn finish(&mut self) -> Option<Vec<u8>> {
        let mut line = std::mem::take(&mut self.buf);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        if line.is_empty() { None } else { Some(line) }
    }
}

/// One running client process decoded into canonical events.
///
/// Owns the process for the session's life: every exit path —
/// end-of-stream, decode failure, cancellation or `Drop` — runs
/// `terminate` then `wait` exactly once. `output()` is re-polled each
/// cycle, so host implementations must return a stream that resumes the
/// process's merged output rather than a competing reader.
/// What an incoming event's `index` means on the wire. Client protocols
/// emit wire-scoped indices that may collide across part kinds (a
/// wire with no index field defaults everything to 0); the session owns
/// canonical index allocation so distinct logical parts never merge.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PartKey {
    /// A bare delta stream keyed by (kind, wire index).
    Delta(u8, u32),
    /// A tool call identified by its call id.
    ToolId(String),
    /// A tool call identified only by its wire index.
    ToolWire(u32),
}

#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn kind_code(kind: PartKind) -> u8 {
    match kind {
        PartKind::Text => 0,
        PartKind::Reasoning => 1,
        PartKind::Media => 2,
        PartKind::ToolCall => 3,
        PartKind::Other => 4,
        _ => 5,
    }
}

struct ProcessSession {
    process: Box<dyn Process>,
    protocol: Arc<dyn ClientProtocol>,
    framing: ClientFraming,
    cancel: CancellationToken,
    framer: LineFramer,
    /// Accumulated output under `ClientFraming::Whole`.
    whole: Vec<u8>,
    pending: VecDeque<StreamEvent>,
    /// Semantic key → canonical index for currently open parts.
    slots: BTreeMap<PartKey, u32>,
    /// Wire index → canonical index for currently open parts.
    by_wire: BTreeMap<u32, u32>,
    /// Canonical index → kind for currently open parts.
    open: BTreeMap<u32, PartKind>,
    next_index: u32,
    /// Bounded tail of raw output for error diagnostics.
    tail: Vec<u8>,
    eof: bool,
    waited: bool,
    done: bool,
    /// Terminal error to yield once pending events drain.
    terminal: Option<Error>,
}

impl ProcessSession {
    fn new(
        process: Box<dyn Process>,
        protocol: Arc<dyn ClientProtocol>,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            framing: protocol.framing(),
            process,
            protocol,
            cancel,
            framer: LineFramer::new(),
            whole: Vec::new(),
            pending: VecDeque::new(),
            slots: BTreeMap::new(),
            by_wire: BTreeMap::new(),
            open: BTreeMap::new(),
            next_index: 0,
            tail: Vec::new(),
            eof: false,
            waited: false,
            done: false,
            terminal: None,
        }
    }

    /// Terminate and reap the process, tolerating repeated calls.
    fn abort(&mut self) {
        let _ = self.process.terminate();
        if !self.waited {
            self.waited = true;
            let _ = self.process.wait();
        }
    }

    fn push_tail(&mut self, chunk: &[u8]) {
        self.tail.extend_from_slice(chunk);
        if self.tail.len() > DIAGNOSTIC_TAIL {
            let excess = self.tail.len() - DIAGNOSTIC_TAIL;
            self.tail.drain(..excess);
        }
    }

    /// Append the diagnostic tail to a provider error's message.
    fn with_tail(&self, e: Error) -> Error {
        if self.tail.is_empty() {
            return e;
        }
        let text = String::from_utf8_lossy(&self.tail);
        let tail: String = text
            .trim()
            .chars()
            .rev()
            .take(TAIL_SNIPPET)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        if tail.is_empty() {
            return e;
        }
        match e {
            Error::Provider {
                status,
                code,
                message,
                retry,
                raw,
            } => Error::Provider {
                status,
                code,
                message: format!("{message}\nprocess output tail: {tail}"),
                retry,
                raw,
            },
            other => other,
        }
    }

    /// Split a chunk into frames per the protocol's framing and decode
    /// each through the client protocol.
    #[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
    fn ingest(&mut self, chunk: &[u8]) {
        self.push_tail(chunk);
        let frames = match self.framing {
            ClientFraming::Lines => self.framer.feed(chunk),
            ClientFraming::Whole => {
                self.whole.extend_from_slice(chunk);
                Vec::new()
            }
            _ => Vec::new(),
        };
        for frame in frames {
            self.decode_frame(&frame);
            if self.terminal.is_some() {
                return;
            }
        }
    }

    fn decode_frame(&mut self, frame: &[u8]) {
        let mut raw = Vec::new();
        match self.protocol.decode(frame, &mut raw) {
            Ok(()) => {
                for ev in raw {
                    self.track(ev);
                }
            }
            Err(e) => {
                // Partial progress survives: decoded events stay queued;
                // the error ends the stream after them.
                self.abort();
                self.eof = true;
                if self.terminal.is_none() {
                    self.terminal = Some(e);
                }
            }
        }
    }

    /// Part lifecycle owned by the foundation. Canonical indices are
    /// allocated here — one open part per semantic key — so client
    /// protocols may emit bare deltas with wire indices that collide
    /// across kinds (everything defaults to 0) and still produce a
    /// well-formed, non-merging canonical event order. A repeated
    /// `ToolCallStarted` for a call already open is an update: the start
    /// collapses and the wire index is re-pointed so the same line's
    /// deltas reach the existing part.
    #[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
    fn track(&mut self, ev: StreamEvent) {
        match ev {
            StreamEvent::PartStarted { index, kind } => {
                // The protocol's explicit open maps onto canonical
                // allocation; `open_part` emits `PartStarted` itself.
                self.delta_index(index, kind);
            }
            StreamEvent::ToolCallStarted { index, id, name } => {
                let key = if id.is_empty() {
                    PartKey::ToolWire(index)
                } else {
                    PartKey::ToolId(id.clone())
                };
                if let Some(&canon) = self
                    .slots
                    .get(&key)
                    .filter(|c| self.open.contains_key(*c))
                {
                    // Update event for a call still open: no second
                    // start; re-point the wire index so this line's
                    // deltas reach the existing part.
                    self.by_wire.insert(index, canon);
                    return;
                }
                let canon = self.alloc(key, index, PartKind::ToolCall);
                self.pending.push_back(StreamEvent::ToolCallStarted {
                    index: canon,
                    id,
                    name,
                });
            }
            StreamEvent::PartFinished { index } => {
                match self.by_wire.remove(&index) {
                    Some(canon) => {
                        self.open.remove(&canon);
                        self.slots.retain(|_, c| *c != canon);
                        self.pending
                            .push_back(StreamEvent::PartFinished { index: canon });
                    }
                    None => self
                        .pending
                        .push_back(StreamEvent::PartFinished { index }),
                }
            }
            StreamEvent::TextDelta { index, text } => {
                let canon = self.delta_index(index, PartKind::Text);
                self.pending.push_back(StreamEvent::TextDelta {
                    index: canon,
                    text,
                });
            }
            StreamEvent::ReasoningDelta { index, text } => {
                let canon = self.delta_index(index, PartKind::Reasoning);
                self.pending.push_back(StreamEvent::ReasoningDelta {
                    index: canon,
                    text,
                });
            }
            StreamEvent::MediaDelta {
                index,
                data,
                media_type,
            } => {
                let canon = self.delta_index(index, PartKind::Media);
                self.pending.push_back(StreamEvent::MediaDelta {
                    index: canon,
                    data,
                    media_type,
                });
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                let key = PartKey::ToolWire(index);
                let canon = match self
                    .by_wire
                    .get(&index)
                    .filter(|c| self.open.get(*c) == Some(&PartKind::ToolCall))
                {
                    Some(&c) => c,
                    None => match self.slots.get(&key) {
                        Some(&c) if self.open.contains_key(&c) => c,
                        _ => self.alloc(key, index, PartKind::ToolCall),
                    },
                };
                self.pending.push_back(StreamEvent::ToolCallDelta {
                    index: canon,
                    arguments,
                });
            }
            other => self.pending.push_back(other),
        }
    }

    /// Open a canonical part for `key` if none is open; emits
    /// `PartStarted` itself and returns the canonical index.
    fn open_part(&mut self, key: PartKey, wire: u32, kind: PartKind) -> u32 {
        match self.slots.get(&key) {
            Some(&c) if self.open.contains_key(&c) => c,
            _ => {
                let canon = self.alloc(key, wire, kind);
                self.pending.push_back(StreamEvent::PartStarted {
                    index: canon,
                    kind,
                });
                canon
            }
        }
    }

    fn delta_index(&mut self, wire: u32, kind: PartKind) -> u32 {
        self.open_part(PartKey::Delta(kind_code(kind), wire), wire, kind)
    }

    /// Allocate a fresh canonical index for a new part.
    fn alloc(&mut self, key: PartKey, wire: u32, kind: PartKind) -> u32 {
        let canon = self.next_index;
        self.next_index += 1;
        self.slots.insert(key, canon);
        self.by_wire.insert(wire, canon);
        self.open.insert(canon, kind);
        canon
    }

    /// Output ended: flush the trailing frame, reap the process, run the
    /// protocol's `finish`, then close any still-open parts.
    #[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
    fn on_eof(&mut self) {
        self.eof = true;
        match self.framing {
            ClientFraming::Lines => {
                if let Some(frame) = self.framer.finish() {
                    self.decode_frame(&frame);
                }
            }
            ClientFraming::Whole => {
                let whole = std::mem::take(&mut self.whole);
                if !whole.is_empty() {
                    self.decode_frame(&whole);
                }
            }
            _ => {}
        }
        if !self.waited {
            self.waited = true;
            match self.process.wait() {
                Ok(code) => {
                    let mut raw = Vec::new();
                    match self.protocol.finish(code, &mut raw) {
                        Ok(()) => {
                            for ev in raw {
                                self.track(ev);
                            }
                        }
                        Err(e) => {
                            if self.terminal.is_none() {
                                self.terminal = Some(self.with_tail(e));
                            }
                        }
                    }
                }
                Err(e) => {
                    if self.terminal.is_none() {
                        self.terminal = Some(e);
                    }
                }
            }
        }
        for (index, _) in std::mem::take(&mut self.open) {
            self.pending.push_back(StreamEvent::PartFinished { index });
        }
        self.by_wire.clear();
        self.slots.clear();
    }

    /// Cancel path: terminate + reap, then surface `Error::Cancelled`
    /// after the already-decoded events.
    fn cancelled(&mut self) {
        self.abort();
        self.eof = true;
        if self.terminal.is_none() {
            self.terminal = Some(Error::Cancelled);
        }
    }
}

impl Stream for ProcessSession {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(ev)));
            }
            if let Some(e) = self.terminal.take() {
                self.done = true;
                return Poll::Ready(Some(Err(e)));
            }
            if self.done || self.eof {
                self.done = true;
                return Poll::Ready(None);
            }
            if self.cancel.is_cancelled() {
                self.cancelled();
                continue;
            }
            let polled = {
                // The borrow ends before `wait`/`terminate` need the
                // process; `output()` must resume the merged stream.
                let mut out = self.process.output();
                out.as_mut().poll_next(cx)
            };
            match polled {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.ingest(&chunk);
                }
                Poll::Ready(Some(Err(e))) => {
                    self.abort();
                    self.eof = true;
                    if self.terminal.is_none() {
                        self.terminal = Some(e);
                    }
                }
                Poll::Ready(None) => {
                    self.on_eof();
                }
                Poll::Pending => {
                    // Register the cancel token's waker so cancellation
                    // wakes the drain while the process output pends.
                    let mut cancelled = std::pin::pin!(self.cancel.cancelled());
                    if cancelled.as_mut().poll(cx).is_ready() {
                        continue;
                    }
                    return Poll::Pending;
                }
            }
        }
    }
}

impl Drop for ProcessSession {
    /// Dropping a live stream always terminates the client process.
    fn drop(&mut self) {
        let _ = self.process.terminate();
        if !self.waited {
            self.waited = true;
            let _ = self.process.wait();
        }
    }
}

/// Assemble a canonical `Output` from a completed event sequence.
/// Parts emit in index order; usage and stop take the last reported
/// values; `Native` payloads collect under `native_events`.
#[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
fn assemble_output(events: &[StreamEvent], protocol_name: &str) -> Output {
    #[derive(Default)]
    struct Slot {
        tool: Option<(String, String, String)>, // id, name, arguments
        text: String,
        reasoning: String,
        media: Vec<u8>,
        media_type: Option<String>,
    }
    let mut slots: BTreeMap<u32, Slot> = BTreeMap::new();
    let mut usage = None;
    let mut stop = None;
    let mut native = Vec::new();
    for ev in events {
        match ev {
            StreamEvent::TextDelta { index, text } => {
                slots.entry(*index).or_default().text.push_str(text);
            }
            StreamEvent::ReasoningDelta { index, text } => {
                slots.entry(*index).or_default().reasoning.push_str(text);
            }
            StreamEvent::ToolCallStarted { index, id, name } => {
                slots.entry(*index).or_default().tool =
                    Some((id.clone(), name.clone(), String::new()));
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                let slot = slots.entry(*index).or_default();
                match &mut slot.tool {
                    Some((_, _, args)) => args.push_str(arguments),
                    None => slot.tool = Some((String::new(), String::new(), arguments.clone())),
                }
            }
            StreamEvent::MediaDelta {
                index,
                data,
                media_type,
            } => {
                let slot = slots.entry(*index).or_default();
                slot.media.extend_from_slice(data);
                if slot.media_type.is_none() {
                    slot.media_type = media_type.clone();
                }
            }
            StreamEvent::Usage(u) => usage = Some(u.clone()),
            StreamEvent::Finished { stop: s } => stop = s.clone(),
            StreamEvent::Native { value, .. } => native.push(value.clone()),
            _ => {}
        }
    }
    let mut parts = Vec::new();
    for (_, slot) in slots {
        if let Some((id, name, args)) = slot.tool {
            let arguments = serde_json::from_str(&args).unwrap_or(Value::String(args));
            parts.push(ContentPart::ToolCall(ToolCall {
                id,
                name,
                arguments,
            }));
        } else if !slot.reasoning.is_empty() {
            parts.push(ContentPart::Reasoning(ReasoningBlock {
                protocol: Protocol::Other(protocol_name.to_owned()),
                opaque: json!({"text": slot.reasoning}),
                summary: Some(slot.reasoning),
            }));
        } else if !slot.media.is_empty() {
            parts.push(ContentPart::Media {
                kind: MediaKind::File,
                source: MediaSource::Inline {
                    media_type: slot
                        .media_type
                        .unwrap_or_else(|| "application/octet-stream".into()),
                    data: slot.media.into(),
                },
                detail: None,
            });
        } else {
            parts.push(ContentPart::Text { text: slot.text });
        }
    }
    let mut extensions = Extensions::new();
    if !native.is_empty() {
        extensions.insert("native_events".into(), native.into());
    }
    Output {
        parts,
        stop,
        usage,
        extensions,
    }
}
