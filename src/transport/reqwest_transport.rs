//! Default transport backed by `reqwest` (feature `reqwest-transport`,
//! on by default). Hosts that need their own pool, TLS or proxy policy
//! inject `Transport` directly; this is the batteries-included default
//! so the SDK is usable without extra wiring.

use futures_util::{StreamExt, future::Either};

use crate::api::RequestControl;
use crate::transport::{
    HttpRequest, HttpResponse, StreamingResponse, Transport, TransportError,
};

/// `reqwest`-backed `Transport`. Clones share the connection pool.
#[derive(Clone)]
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Wrap a caller-configured `reqwest::Client` (custom TLS, proxy,
    /// pool limits) without giving up the `Transport` port.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    fn request(&self, req: &HttpRequest) -> reqwest::RequestBuilder {
        self.client
            .request(req.method.clone(), req.url.clone())
            .headers(req.headers.clone())
            .body(req.body.clone())
    }

    /// Race the request against caller cancellation; a cancelled caller
    /// surfaces as `TransportError::Cancelled`, never a phantom failure.
    async fn send_inner(
        &self,
        req: &HttpRequest,
        ctrl: &RequestControl,
    ) -> Result<reqwest::Response, TransportError> {
        let send = self.request(req).send();
        let cancelled = ctrl.cancel.cancelled();
        futures_util::pin_mut!(send, cancelled);
        match futures_util::future::select(send, cancelled).await {
            Either::Left((Ok(res), _)) => Ok(res),
            Either::Left((Err(e), _)) => Err(map_reqwest(e)),
            Either::Right(((), _)) => Err(TransportError::Cancelled),
        }
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self::new()
    }
}

fn map_reqwest(e: reqwest::Error) -> TransportError {
    if e.is_connect() {
        TransportError::Connect(e.to_string())
    } else {
        TransportError::Io(e.to_string())
    }
}

#[async_trait::async_trait]
impl Transport for ReqwestTransport {
    async fn send(
        &self,
        req: HttpRequest,
        ctrl: &RequestControl,
    ) -> Result<HttpResponse, TransportError> {
        let res = self.send_inner(&req, ctrl).await?;
        let status = res.status();
        let headers = res.headers().clone();
        let body = res.bytes().await.map_err(map_reqwest)?;
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn send_stream(
        &self,
        req: HttpRequest,
        ctrl: &RequestControl,
    ) -> Result<StreamingResponse, TransportError> {
        let res = self.send_inner(&req, ctrl).await?;
        let status = res.status();
        let headers = res.headers().clone();
        let body = res
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| crate::api::Error::Transport(map_reqwest(e))));
        Ok(StreamingResponse {
            status,
            headers,
            body: Box::pin(body),
        })
    }
}
