//! Inbound wire dialects for gateway-style consumers.
//!
//! `protocols` adapts canonical requests to provider wire formats
//! (outbound). This module runs the opposite direction: it accepts a
//! client request already shaped like a provider dialect, decodes it
//! into the canonical model so the host may run it through
//! `Adapter::chat` against *any* backend, then encodes the canonical
//! output and stream events back into the dialect the client spoke.
//!
//! Lossy edges are explicit: representable-but-imperfect mappings are
//! recorded in `ConversionReport` (or accumulated on a stream encoder)
//! rather than failing; only malformed input or unrepresentable
//! semantics produce `Error`.

mod anthropic_in;
mod openai_in;

use serde_json::{Map, Value};

use crate::api::*;
use crate::transport::SseEvent;

pub use anthropic_in::AnthropicStreamEncoder;
pub use openai_in::OpenAiStreamEncoder;

/// A client-facing wire dialect: the request/response shape a gateway
/// caller used, independent of the backend protocol that serves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WireDialect {
    /// OpenAI Chat Completions (`POST /chat/completions`).
    OpenAiChatCompletions,
    /// Anthropic Messages (`POST /v1/messages`).
    AnthropicMessages,
}

impl WireDialect {
    /// The native protocol family this dialect's wire shape belongs to.
    pub fn protocol(self) -> Protocol {
        match self {
            WireDialect::OpenAiChatCompletions => Protocol::OpenAiChatCompletions,
            WireDialect::AnthropicMessages => Protocol::AnthropicMessages,
        }
    }
}

/// Decode a dialect request body into a canonical `ChatRequest`.
///
/// Malformed input fails with `Error::Protocol`. Fields with no
/// canonical home are preserved verbatim in `ChatRequest::extensions`
/// and reported in the returned `ConversionReport`, so a same-dialect
/// backend still receives them while the report keeps the relay honest.
pub fn decode_chat_request(
    dialect: WireDialect,
    body: &[u8],
) -> Result<(ChatRequest, ConversionReport), Error> {
    match dialect {
        WireDialect::OpenAiChatCompletions => openai_in::decode_request(body),
        WireDialect::AnthropicMessages => anthropic_in::decode_request(body),
    }
}

/// Encode a canonical `Output` into the dialect's response body.
///
/// `model` is the identifier echoed back to the client — normally the
/// model name from its own request. Parts the dialect cannot express
/// are reported in the `ConversionReport`, never dropped silently.
pub fn encode_chat_output(
    dialect: WireDialect,
    output: &Output,
    model: &str,
) -> (Value, ConversionReport) {
    match dialect {
        WireDialect::OpenAiChatCompletions => openai_in::encode_output(output, model),
        WireDialect::AnthropicMessages => anthropic_in::encode_output(output, model),
    }
}

/// Stateful encoder: canonical `StreamEvent`s → dialect SSE frames.
///
/// Encoders own the small amount of state a faithful stream needs —
/// dense tool-call indices for Chat Completions, content-block indices
/// for Messages. Frames are emitted in the dialect's own event shape,
/// ready to serialize as `event:`/`data:` lines.
pub trait StreamEncoder: Send {
    /// The frames that open the stream before the first event —
    /// Messages emits `message_start`; Chat Completions opens nothing
    /// (the first data chunk carries `delta.role` instead). Idempotent.
    fn preamble(&mut self) -> Vec<SseEvent>;
    /// Encode one canonical event into zero or more SSE frames.
    fn encode(&mut self, event: &StreamEvent) -> Vec<SseEvent>;
    /// Terminal frames after the last event (`data: [DONE]`,
    /// `message_stop`, …). Idempotent once the terminator is sent.
    fn finish(&mut self) -> Vec<SseEvent>;
    /// Conversion losses observed while encoding since the last call.
    fn drain_losses(&mut self) -> Vec<ConversionLoss>;
}

/// Open a stream encoder for `dialect`. `model` is echoed into every
/// chunk or envelope. Emit `encode_stream_start` frames first, feed
/// events through `encode`, then call `finish` for the terminator.
pub fn stream_encoder(dialect: WireDialect, model: &str) -> Box<dyn StreamEncoder> {
    match dialect {
        WireDialect::OpenAiChatCompletions => Box::new(OpenAiStreamEncoder::new(model)),
        WireDialect::AnthropicMessages => Box::new(AnthropicStreamEncoder::new(model)),
    }
}

/// The frames that open a dialect stream before the first event.
/// Equivalent to a fresh encoder's `preamble()` — emit one or the
/// other, not both.
pub fn encode_stream_start(dialect: WireDialect, model: &str) -> Vec<SseEvent> {
    stream_encoder(dialect, model).preamble()
}

// ---------------------------------------------------------------------------
// Bridge: dialect request → adapter → dialect response, in one call
// ---------------------------------------------------------------------------

use std::sync::Arc;

/// A joined relay for gateway-style consumers: holds an adapter plus the
/// client-facing dialect and performs the full decode → adapter call →
/// encode trip in one method. Conversion losses are returned alongside
/// results; the host decides whether to log, reject, or forward them.
pub struct WireBridge {
    adapter: Arc<dyn Adapter>,
    dialect: WireDialect,
}

impl WireBridge {
    pub fn new(adapter: Arc<dyn Adapter>, dialect: WireDialect) -> Self {
        Self { adapter, dialect }
    }

    /// Non-streaming chat: decode `body` in this bridge's dialect, run it
    /// through the adapter, encode the output back. The returned model
    /// name echoed to the client comes from the client's own request.
    pub async fn chat(
        &self,
        body: &[u8],
        ctrl: &RequestControl,
    ) -> Result<(Value, ConversionReport), Error> {
        let (request, mut report) = decode_chat_request(self.dialect, body)?;
        let model = request.model.clone();
        let output = self.adapter.chat(request, ctrl).await?;
        let (value, out_report) = encode_chat_output(self.dialect, &output, &model);
        report.losses.extend(out_report.losses);
        Ok((value, report))
    }

    /// Streaming chat: returns a stream of dialect-shaped SSE frames.
    /// Preamble frames emit first; canonical events pass through the
    /// dialect encoder; `finish` frames terminate the stream.
    pub async fn chat_stream(
        &self,
        body: &[u8],
        ctrl: &RequestControl,
    ) -> Result<DynStream<'static, SseEvent>, Error> {
        let (request, _report) = decode_chat_request(self.dialect, body)?;
        let model = request.model.clone();
        let inner = self.adapter.chat_stream(request, ctrl).await?;
        Ok(Box::pin(BridgeStream {
            pending: stream_encoder(self.dialect, &model).preamble().into(),
            encoder: stream_encoder(self.dialect, &model),
            inner,
            finished: false,
        }))
    }
}

/// The stream behind [`WireBridge::chat_stream`]: drains pending frames
/// first, then translates canonical events through the dialect encoder,
/// and emits terminator frames once the inner stream ends.
struct BridgeStream {
    pending: std::collections::VecDeque<SseEvent>,
    encoder: Box<dyn StreamEncoder>,
    inner: DynStream<'static, StreamEvent>,
    finished: bool,
}

impl futures_core::Stream for BridgeStream {
    type Item = Result<SseEvent, Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            if let Some(frame) = self.pending.pop_front() {
                return std::task::Poll::Ready(Some(Ok(frame)));
            }
            if self.finished {
                return std::task::Poll::Ready(None);
            }
            match std::pin::Pin::new(&mut self.inner).poll_next(cx) {
                std::task::Poll::Pending => return std::task::Poll::Pending,
                std::task::Poll::Ready(None) => {
                    self.finished = true;
                    let frames = self.encoder.finish();
                    self.pending.extend(frames);
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return std::task::Poll::Ready(Some(Err(e)));
                }
                std::task::Poll::Ready(Some(Ok(ev))) => {
                    let frames = self.encoder.encode(&ev);
                    self.pending.extend(frames);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared internals
// ---------------------------------------------------------------------------

/// Marker protocol for the canonical side of a conversion report.
pub(crate) fn canonical_protocol() -> Protocol {
    Protocol::Other("a4mp-canonical".into())
}

/// Report for a dialect request → canonical decode.
pub(crate) fn request_report(dialect: WireDialect) -> ConversionReport {
    ConversionReport {
        source: dialect.protocol(),
        target: canonical_protocol(),
        losses: Vec::new(),
    }
}

/// Report for a canonical output → dialect encode.
pub(crate) fn response_report(dialect: WireDialect) -> ConversionReport {
    ConversionReport {
        source: canonical_protocol(),
        target: dialect.protocol(),
        losses: Vec::new(),
    }
}

/// Record one conversion loss.
pub(crate) fn loss(report: &mut ConversionReport, kind: LossKind, path: impl Into<String>) {
    report.losses.push(ConversionLoss {
        path: path.into(),
        kind,
    });
}

/// Preserve an unclaimed request field in `extensions` and flag it.
/// Forwarding keeps the field meaningful for a same-dialect backend;
/// the report marks its semantics as dialect-bound.
pub(crate) fn preserve_extension(
    extensions: &mut Extensions,
    report: &mut ConversionReport,
    key: &str,
    value: Value,
) {
    extensions.insert(key.to_owned(), value);
    loss(report, LossKind::Lossy, format!("request.{key}"));
}

/// Keys of `obj` outside `claimed`, in encounter order.
pub(crate) fn extra_keys<'a>(obj: &'a Map<String, Value>, claimed: &[&str]) -> Vec<&'a str> {
    obj.keys()
        .map(String::as_str)
        .filter(|k| !claimed.contains(k))
        .collect()
}

/// A `stop`/`stop_sequences` field: a single string or a string array.
/// `None` means the value had another shape entirely.
pub(crate) fn string_list(v: &Value) -> Option<Vec<String>> {
    match v {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                match item.as_str() {
                    Some(s) => out.push(s.to_owned()),
                    None => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Decode base64 wire data; malformed payloads are protocol violations.
pub(crate) fn base64_decode(data: &str, path: &str) -> Result<bytes::Bytes, Error> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map(bytes::Bytes::from)
        .map_err(|e| Error::Protocol(format!("{path}: invalid base64: {e}")))
}

/// A dialect-prefixed random identifier (`chatcmpl-…`, `msg_…`).
pub(crate) fn wire_id(prefix: &str) -> String {
    format!("{prefix}{:016x}", rand::random::<u64>())
}

/// Seconds since the Unix epoch for `created` fields.
pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One SSE frame carrying a JSON payload.
pub(crate) fn sse(event: Option<&str>, data: Value) -> SseEvent {
    SseEvent {
        event: event.map(str::to_owned),
        data: data.to_string(),
        id: None,
    }
}
