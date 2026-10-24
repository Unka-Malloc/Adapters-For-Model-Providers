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

use crate::api::{DynStream, Error, RequestControl, RetrySafety};

/// An HTTP request issued through the transport port.
#[derive(Clone, Debug)]
pub struct HttpRequest {
    pub method: http::Method,
    pub url: url::Url,
    pub headers: http::HeaderMap,
    pub body: bytes::Bytes,
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
#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub status: http::StatusCode,
    pub headers: http::HeaderMap,
    pub body: bytes::Bytes,
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
    /// Caller cancellation surfaced through the transport.
    #[error("cancelled")]
    Cancelled,
}

impl TransportError {
    /// Whether repeating the request could duplicate billed work.
    ///
    /// `Connect` never reached the provider: safe. Everything else is
    /// `Unsafe` — mid-body failure may follow upstream acceptance, so
    /// automatic retries are forbidden; the caller decides explicitly.
    pub fn retry_safety(&self) -> RetrySafety {
        match self {
            TransportError::Connect(_) | TransportError::Cancelled => RetrySafety::Safe,
            TransportError::Disconnected(_) | TransportError::Io(_) => RetrySafety::Unsafe,
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
