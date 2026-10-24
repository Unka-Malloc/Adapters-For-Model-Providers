//! Shared test fixtures: an in-memory transport that replays recorded
//! responses and captures requests for assertions.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use a4mp::*;

/// Recorded request for assertion.
#[derive(Clone)]
pub struct RecordedRequest {
    pub method: http::Method,
    pub url: url::Url,
    pub headers: http::HeaderMap,
    pub body: Vec<u8>,
}

/// One queued response (buffered or streaming).
pub enum Response {
    Json(u16, serde_json::Value),
    Text(u16, &'static str),
    /// Raw SSE body emitted as one or more byte chunks.
    Sse(Vec<&'static str>),
    Fail(TransportError),
}

/// A scripted transport: every `send`/`send_stream` pops the next queued
/// response and records the request. Deterministic, no sockets.
#[derive(Clone, Default)]
pub struct MockTransport {
    pub requests: Arc<Mutex<Vec<RecordedRequest>>>,
    pub responses: Arc<Mutex<VecDeque<Response>>>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, response: Response) -> &Self {
        self.responses.lock().unwrap().push_back(response);
        self
    }

    pub fn recorded(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn record(&self, req: &HttpRequest) {
        self.requests.lock().unwrap().push(RecordedRequest {
            method: req.method.clone(),
            url: req.url.clone(),
            headers: req.headers.clone(),
            body: req.body.to_vec(),
        });
    }

    fn next(&self) -> Response {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Response::Fail(TransportError::Io("no scripted response".into())))
    }
}

#[async_trait::async_trait]
impl Transport for MockTransport {
    async fn send(
        &self,
        req: HttpRequest,
        _ctrl: &RequestControl,
    ) -> Result<HttpResponse, TransportError> {
        self.record(&req);
        match self.next() {
            Response::Json(status, v) => Ok(HttpResponse {
                status: http::StatusCode::from_u16(status).unwrap(),
                headers: http::HeaderMap::new(),
                body: serde_json::to_vec(&v).unwrap().into(),
            }),
            Response::Text(status, t) => Ok(HttpResponse {
                status: http::StatusCode::from_u16(status).unwrap(),
                headers: http::HeaderMap::new(),
                body: t.as_bytes().to_vec().into(),
            }),
            Response::Sse(_) => Err(TransportError::Io("stream response for send".into())),
            Response::Fail(e) => Err(e),
        }
    }

    async fn send_stream(
        &self,
        req: HttpRequest,
        _ctrl: &RequestControl,
    ) -> Result<StreamingResponse, TransportError> {
        self.record(&req);
        match self.next() {
            Response::Sse(chunks) => {
                let body: DynStream<'static, bytes::Bytes> =
                    Box::pin(futures_util::stream::iter(
                        chunks.into_iter().map(|c| {
                            Ok::<_, Error>(bytes::Bytes::from_static(c.as_bytes()))
                        }),
                    ));
                Ok(StreamingResponse {
                    status: http::StatusCode::OK,
                    headers: http::HeaderMap::new(),
                    body,
                })
            }
            Response::Json(status, v) => {
                let body: DynStream<'static, bytes::Bytes> =
                    Box::pin(futures_util::stream::iter([Ok::<_, Error>(
                        serde_json::to_vec(&v).unwrap().into(),
                    )]));
                Ok(StreamingResponse {
                    status: http::StatusCode::from_u16(status).unwrap(),
                    headers: http::HeaderMap::new(),
                    body,
                })
            }
            Response::Text(status, t) => {
                let body: DynStream<'static, bytes::Bytes> =
                    Box::pin(futures_util::stream::iter([Ok::<_, Error>(
                        bytes::Bytes::from_static(t.as_bytes()),
                    )]));
                Ok(StreamingResponse {
                    status: http::StatusCode::from_u16(status).unwrap(),
                    headers: http::HeaderMap::new(),
                    body,
                })
            }
            Response::Fail(e) => Err(e),
        }
    }
}

/// Collect a `DynStream` into a Vec.
pub async fn collect<T>(mut stream: DynStream<'_, T>) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    while let Some(item) = std::future::poll_fn(|cx| stream.as_mut().poll_next(cx)).await {
        out.push(item?);
    }
    Ok(out)
}

/// A request body parsed back to JSON for assertions.
pub fn body_json(req: &RecordedRequest) -> serde_json::Value {
    serde_json::from_slice(&req.body).unwrap()
}
