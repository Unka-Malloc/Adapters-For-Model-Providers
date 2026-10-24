//! WebSocket-backed duplex channels for realtime sessions
//! (feature `realtime-ws`). The handshake is a real RFC 6455 client
//! upgrade: `ws(s)` follows the request's `http(s)` scheme, `wss` uses
//! rustls with webpki roots (matching the reqwest transport), and every
//! request header except the reserved handshake set passes through.
//!
//! HTTP verbs stay with an optional delegated transport
//! (`WsTransport::with_http`); without one this transport is channels
//! only.

use std::sync::Arc;

use futures_util::future::Either;
use futures_util::lock::Mutex;
use futures_util::{Sink, SinkExt, StreamExt, pin_mut};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};

use crate::api::{Error, RequestControl};
use crate::transport::{
    DuplexChannel, HttpRequest, HttpResponse, StreamingResponse, Transport, TransportError,
};

/// Headers the handshake must generate itself; a duplicate would
/// corrupt the upgrade request, so caller copies are skipped.
const HANDSHAKE_HEADERS: [&str; 4] = [
    "connection",
    "upgrade",
    "sec-websocket-key",
    "sec-websocket-version",
];

/// A `Transport` whose `open_channel` dials real WebSocket connections.
/// `send`/`send_stream` delegate to an optional HTTP transport so a
/// single object can back every adapter operation.
#[derive(Clone, Default)]
pub struct WsTransport {
    http: Option<Arc<dyn Transport>>,
}

impl WsTransport {
    /// Channel-only transport; HTTP methods report unsupported.
    pub fn new() -> Self {
        Self::default()
    }

    /// Delegate `send`/`send_stream` to `http` while this transport
    /// owns duplex channels.
    pub fn with_http(http: Arc<dyn Transport>) -> Self {
        Self { http: Some(http) }
    }
}

/// Map a mid-stream socket failure to the transport taxonomy.
fn map_ws(e: WsError) -> TransportError {
    match e {
        WsError::ConnectionClosed | WsError::AlreadyClosed => {
            TransportError::Disconnected(e.to_string())
        }
        _ => TransportError::Io(e.to_string()),
    }
}

/// A failed handshake never established a session: always `Connect`
/// (retry-safe), with the HTTP rejection status kept when present.
fn map_connect(e: WsError) -> TransportError {
    match e {
        WsError::Http(res) => TransportError::Connect(format!(
            "websocket upgrade rejected: {}",
            res.status()
        )),
        other => TransportError::Connect(other.to_string()),
    }
}

/// One live WebSocket framed as a `DuplexChannel`. The socket is split
/// so `send(&self)` and `recv(&mut self)` proceed independently; the
/// write half sits behind an async mutex only for the duration of each
/// write.
struct WsChannel<R, W> {
    read: R,
    write: Mutex<W>,
}

#[async_trait::async_trait]
impl<R, W> DuplexChannel for WsChannel<R, W>
where
    // `send(&self)` shares `&WsChannel`, so the whole channel must be
    // `Sync`; `R` needs it, `Mutex<W>` gets it from `W: Send`.
    R: futures_core::Stream<Item = Result<Message, WsError>> + Unpin + Send + Sync,
    W: Sink<Message, Error = WsError> + Unpin + Send,
{
    async fn recv(&mut self) -> Result<Option<bytes::Bytes>, Error> {
        loop {
            match self.read.next().await {
                Some(Ok(msg)) => match msg {
                    Message::Text(_) | Message::Binary(_) => {
                        return Ok(Some(msg.into_data().into()));
                    }
                    Message::Close(_) => {
                        // Answer the peer's close handshake, then report EOF.
                        let _ = self.write.lock().await.close().await;
                        return Ok(None);
                    }
                    // Ping/Pong are answered inside tungstenite; raw
                    // frames carry no payload for the session.
                    _ => {}
                },
                Some(Err(e)) => return Err(Error::Transport(map_ws(e))),
                None => return Ok(None),
            }
        }
    }

    async fn send(&self, data: bytes::Bytes) -> Result<(), Error> {
        // Sessions speak JSON text; non-UTF-8 payloads degrade to binary.
        let msg = match String::from_utf8(data.to_vec()) {
            Ok(text) => Message::text(text),
            Err(_) => Message::binary(data),
        };
        self.write
            .lock()
            .await
            .send(msg)
            .await
            .map_err(|e| Error::Transport(map_ws(e)))
    }

    async fn close(self: Box<Self>) -> Result<(), Error> {
        let _ = self
            .write
            .lock()
            .await
            .close()
            .await
            .map_err(|e| Error::Transport(map_ws(e)))?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl Transport for WsTransport {
    async fn send(
        &self,
        req: HttpRequest,
        ctrl: &RequestControl,
    ) -> Result<HttpResponse, TransportError> {
        match &self.http {
            Some(http) => http.send(req, ctrl).await,
            None => Err(TransportError::Io(
                "WsTransport implements only duplex channels".into(),
            )),
        }
    }

    async fn send_stream(
        &self,
        req: HttpRequest,
        ctrl: &RequestControl,
    ) -> Result<StreamingResponse, TransportError> {
        match &self.http {
            Some(http) => http.send_stream(req, ctrl).await,
            None => Err(TransportError::Io(
                "WsTransport implements only duplex channels".into(),
            )),
        }
    }

    async fn open_channel(
        &self,
        req: HttpRequest,
        ctrl: &RequestControl,
    ) -> Result<Box<dyn DuplexChannel>, TransportError> {
        let uri: http::Uri = req
            .url
            .as_str()
            .parse()
            .map_err(|e| TransportError::Connect(format!("invalid websocket url: {e}")))?;
        // `Uri::into_client_request` builds the upgrade request (Host,
        // Upgrade, Sec-WebSocket-*); the `Request` impl would pass the
        // request through without them, so it is not used here.
        let mut request = uri
            .into_client_request()
            .map_err(|e| TransportError::Connect(e.to_string()))?;
        for (name, value) in req.headers.iter() {
            if HANDSHAKE_HEADERS.contains(&name.as_str()) {
                continue;
            }
            request.headers_mut().insert(name.clone(), value.clone());
        }
        let connect = connect_async(request);
        let cancelled = ctrl.cancel.cancelled();
        pin_mut!(connect, cancelled);
        let ws = match futures_util::future::select(connect, cancelled).await {
            Either::Left((Ok((ws, _response)), _)) => ws,
            Either::Left((Err(e), _)) => return Err(map_connect(e)),
            Either::Right(((), _)) => return Err(TransportError::Cancelled),
        };
        let (write, read) = ws.split();
        Ok(Box::new(WsChannel {
            read,
            write: Mutex::new(write),
        }))
    }
}
