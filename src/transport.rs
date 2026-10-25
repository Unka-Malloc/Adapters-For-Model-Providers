//! Transport port: minimal HTTP request/response types plus the injected
//! transport trait. Adapters never create connections directly; the host
//! owns pools, DNS, TLS policy and proxies. Body streams are pull-based,
//! so backpressure is structural, not a separate buffer.

mod sse;
pub use sse::*;

#[cfg(feature = "reqwest-transport")]
mod reqwest_transport;
#[cfg(feature = "reqwest-transport")]
pub use reqwest_transport::ReqwestTransport;

#[cfg(feature = "realtime-ws")]
mod ws_transport;
#[cfg(feature = "realtime-ws")]
pub use ws_transport::WsTransport;

use std::fmt;

use crate::api::{DynStream, Error, RequestControl, RetrySafety};

/// Header values safe to render in diagnostics. Every other header may
/// carry a credential and is redacted by the manual `Debug` impls.
const DEBUG_SAFE_HEADERS: &[&str] = &["accept", "content-length", "content-type", "user-agent"];

struct RedactedHeaders<'a>(&'a http::HeaderMap);

impl fmt::Debug for RedactedHeaders<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (name, value) in self.0 {
            if DEBUG_SAFE_HEADERS.contains(&name.as_str()) {
                map.entry(&name.as_str(), &value.to_str().unwrap_or("<non-text>"));
            } else {
                map.entry(&name.as_str(), &"<redacted>");
            }
        }
        map.finish()
    }
}

/// Render a URL without userinfo, query values or fragments. Query strings
/// can carry provider keys; paths and hosts remain useful to diagnostics.
fn redacted_url(url: &url::Url) -> String {
    let mut out = format!("{}://", url.scheme());
    if let Some(host) = url.host_str() {
        out.push_str(host);
    }
    if let Some(port) = url.port() {
        out.push(':');
        out.push_str(&port.to_string());
    }
    out.push_str(url.path());
    if url.query().is_some() {
        out.push_str("?<redacted>");
    }
    out
}

fn redacted_body(len: usize) -> String {
    format!("<redacted {len} bytes>")
}

/// An HTTP request issued through the transport port.
///
/// `Debug` shows the method, a credential-stripped URL, header names and
/// body length; header values outside a small safe set and the body are
/// redacted so transport logging cannot disclose API keys or prompts.
#[derive(Clone)]
pub struct HttpRequest {
    pub method: http::Method,
    pub url: url::Url,
    pub headers: http::HeaderMap,
    pub body: bytes::Bytes,
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("url", &redacted_url(&self.url))
            .field("headers", &RedactedHeaders(&self.headers))
            .field("body", &redacted_body(self.body.len()))
            .finish()
    }
}

impl HttpRequest {
    pub fn get(url: url::Url) -> Self {
        Self {
            method: http::Method::GET,
            url,
            headers: http::HeaderMap::new(),
            body: bytes::Bytes::new(),
        }
    }

    pub fn post_json(url: url::Url, body: serde_json::Value) -> Self {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        Self {
            method: http::Method::POST,
            url,
            headers,
            body: bytes::Bytes::from(serde_json::to_vec(&body).unwrap_or_default()),
        }
    }
}

/// A buffered HTTP response.
///
/// `Debug` shows the status, safe header values and body length; other
/// header values and the body are redacted because token-endpoint and
/// provider error bodies can carry credentials.
#[derive(Clone)]
pub struct HttpResponse {
    pub status: http::StatusCode,
    pub headers: http::HeaderMap,
    pub body: bytes::Bytes,
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &RedactedHeaders(&self.headers))
            .field("body", &redacted_body(self.body.len()))
            .finish()
    }
}

/// A streaming HTTP response: status and headers first, then a pull-based
/// byte body. The adapter decodes incrementally; nothing accumulates the
/// whole body implicitly.
pub struct StreamingResponse {
    pub status: http::StatusCode,
    pub headers: http::HeaderMap,
    /// Byte stream; item errors are transport failures mid-body.
    pub body: DynStream<'static, bytes::Bytes>,
}

/// Transport-layer failure: before the request, or mid-body.
/// Provider-level rejections are `Error::Provider`, not transport errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TransportError {
    /// Connect/DNS/TLS refused before the provider saw the request.
    #[error("connect: {0}")]
    Connect(String),
    /// The peer closed the connection mid-request or mid-body.
    #[error("disconnected: {0}")]
    Disconnected(String),
    /// Body decode or I/O failure.
    #[error("io: {0}")]
    Io(String),
    /// Caller cancellation surfaced through the transport. The request may
    /// already have been dispatched when cancellation raced it.
    #[error("cancelled")]
    Cancelled,
}

impl TransportError {
    /// Whether repeating the request could duplicate billed work.
    ///
    /// `Connect` never reached the provider: safe. Everything else is
    /// `Unsafe` — cancellation may race a dispatched request, and
    /// mid-body failure may follow upstream acceptance, so automatic
    /// retries are forbidden; the caller decides explicitly.
    pub fn retry_safety(&self) -> RetrySafety {
        match self {
            TransportError::Connect(_) => RetrySafety::Safe,
            TransportError::Cancelled
            | TransportError::Disconnected(_)
            | TransportError::Io(_) => RetrySafety::Unsafe,
        }
    }
}

/// A bidirectional frame channel (WebSocket or equivalent) for realtime
/// sessions. Pull-based inbound preserves backpressure; `None` from
/// `recv` means the peer closed.
#[async_trait::async_trait]
pub trait DuplexChannel: Send {
    /// Pull the next inbound frame.
    async fn recv(&mut self) -> Result<Option<bytes::Bytes>, Error>;
    /// Send one outbound frame (text or binary per the session's
    /// negotiation; the adapter picks the wire shape).
    async fn send(&self, data: bytes::Bytes) -> Result<(), Error>;
    /// Close normally and release the connection.
    async fn close(self: Box<Self>) -> Result<(), Error>;
}

/// The injected transport port. One shared implementation backs every
/// adapter in a client, giving one connection pool and one place for
/// host policy (TLS, proxy, DNS, metrics).
#[async_trait::async_trait]
pub trait Transport: Send + Sync {
    /// Buffered request/response.
    async fn send(&self, req: HttpRequest, ctrl: &RequestControl)
    -> Result<HttpResponse, TransportError>;

    /// Streaming request/response for SSE and chunked bodies.
    async fn send_stream(
        &self,
        req: HttpRequest,
        ctrl: &RequestControl,
    ) -> Result<StreamingResponse, TransportError>;

    /// Bidirectional channel for realtime session protocols.
    /// Default reports unsupported; realtime-capable transports
    /// (WebSocket implementations, host-provided channels) override.
    async fn open_channel(
        &self,
        _req: HttpRequest,
        _ctrl: &RequestControl,
    ) -> Result<Box<dyn DuplexChannel>, TransportError> {
        Err(TransportError::Io(
            "duplex channels not supported by this transport".into(),
        ))
    }
}
