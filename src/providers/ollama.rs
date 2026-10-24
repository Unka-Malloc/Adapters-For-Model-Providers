//! The Ollama adapter: binds a `ServiceTarget` speaking the Ollama
//! protocol (`Protocol::Other("ollama")`) to a transport and credential
//! port.
//!
//! Local Ollama servers need no credential (`Credential::None`);
//! Ollama Cloud (`https://ollama.com`) accepts a bearer token — both
//! resolve through the same credential port. Rerank and classify have
//! no Ollama surface and stay `Unsupported`.

use std::sync::Arc;

use serde_json::json;

use crate::api::*;
use crate::auth::{AuthKind, Credential, CredentialResolver, CredentialScope};
use crate::effects::{EffectHost, ResourceRef};
use crate::protocols::ollama;
use crate::transport::{HttpRequest, Transport};

/// Adapter for the Ollama protocol family.
pub struct OllamaAdapter {
    target: ServiceTarget,
    auth: AuthKind,
    transport: Arc<dyn Transport>,
    resolver: Arc<dyn CredentialResolver>,
    effects: Option<Arc<dyn EffectHost>>,
}

impl OllamaAdapter {
    pub fn new(
        target: ServiceTarget,
        auth: AuthKind,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
    ) -> Result<Self, Error> {
        match &target.protocol {
            Protocol::Other(name) if name == "ollama" => {}
            other => {
                return Err(Error::Unsupported(format!(
                    "OllamaAdapter cannot serve protocol {other:?}"
                )));
            }
        }
        Ok(Self {
            target,
            auth,
            transport,
            resolver,
            effects: None,
        })
    }

    /// Attach the host effect port used to resolve media resources.
    pub fn with_effects(mut self, effects: Arc<dyn EffectHost>) -> Self {
        self.effects = Some(effects);
        self
    }

    /// No credential → no header (local servers); API key or bearer
    /// material → `Authorization: Bearer` (Ollama Cloud).
    #[allow(unreachable_patterns)] // reachable only to downstream crates (non_exhaustive enums)
    async fn authenticate(&self) -> Result<http::HeaderMap, Error> {
        let scope = CredentialScope {
            target: &self.target,
            auth: &self.auth,
        };
        let credential = self.resolver.resolve(scope).await?;
        let mut headers = self.target.headers.clone();
        match credential {
            Credential::None => {}
            Credential::ApiKey(secret) | Credential::Bearer { token: secret, .. } => {
                headers.insert(
                    http::header::AUTHORIZATION,
                    http::HeaderValue::from_str(&format!("Bearer {}", secret.expose()))
                        .map_err(|_| Error::Credential("credential is not a valid header value".into()))?,
                );
            }
            Credential::Cloud { .. } => {
                return Err(Error::Unsupported(
                    "cloud identity is not yet implemented for this adapter".into(),
                ));
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "credential kind {other:?} is not yet implemented for this adapter"
                )));
            }
        }
        Ok(headers)
    }

    /// Resolve `MediaSource::Resource` parts into inline data through the
    /// host effect port before the request enters the codec; the codec
    /// then base64-encodes the bytes into `images`.
    async fn resolve_media(&self, req: &mut ChatRequest, ctrl: &RequestControl) -> Result<(), Error> {
        let needs_resolve = req.messages.iter().any(|m| {
            m.content.iter().any(|p| {
                matches!(
                    p,
                    ContentPart::Media {
                        source: MediaSource::Resource(_),
                        ..
                    }
                )
            })
        });
        if !needs_resolve {
            return Ok(());
        }
        let effects = self
            .effects
            .as_ref()
            .ok_or_else(|| Error::Unsupported("media resources need a host effect port".into()))?;
        for m in &mut req.messages {
            for part in &mut m.content {
                if let ContentPart::Media { kind: _, source, .. } = part {
                    if let MediaSource::Resource(id) = source {
                        let rr = ResourceRef {
                            id: id.clone(),
                            media_type: None,
                        };
                        let data = effects.read_resource(&rr, ctrl).await?;
                        *source = MediaSource::Inline {
                            media_type: "application/octet-stream".into(),
                            data,
                        };
                    }
                }
            }
        }
        Ok(())
    }

    async fn send_json(&self, req: HttpRequest, ctrl: &RequestControl) -> Result<Vec<u8>, Error> {
        let res = self.transport.send(req, ctrl).await?;
        if !res.status.is_success() {
            return Err(ollama::http_error(res.status.as_u16(), &res.body));
        }
        Ok(res.body.to_vec())
    }

    async fn open_stream(
        &self,
        req: HttpRequest,
        ctrl: &RequestControl,
    ) -> Result<crate::transport::StreamingResponse, Error> {
        let res = self.transport.send_stream(req, ctrl).await?;
        if !res.status.is_success() {
            // Drain the error body before building the error.
            let mut body = Vec::new();
            let mut stream = res.body;
            while let Some(chunk) = std::future::poll_fn(|cx| stream.as_mut().poll_next(cx)).await
            {
                match chunk {
                    Ok(b) => body.extend_from_slice(&b),
                    Err(e) => return Err(e),
                }
            }
            return Err(ollama::http_error(res.status.as_u16(), &body));
        }
        Ok(res)
    }
}

#[async_trait::async_trait]
impl Adapter for OllamaAdapter {
    fn target(&self) -> &ServiceTarget {
        &self.target
    }

    /// `GET /api/tags` — the local model listing.
    async fn models(&self, ctrl: &RequestControl) -> Result<Vec<ModelInfo>, Error> {
        let headers = self.authenticate().await?;
        let mut http = HttpRequest::get(self.target.endpoint("api/tags"));
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        ollama::decode_models(&body)
    }

    /// `POST /api/chat` with `"stream": false` — the wire defaults to
    /// streaming, so non-streaming is explicit.
    async fn chat(&self, mut req: ChatRequest, ctrl: &RequestControl) -> Result<Output, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        let body = ollama::build_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("api/chat"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        ollama::decode_response(&body)
    }

    /// `POST /api/chat` with `"stream": true` — NDJSON-framed generation.
    async fn chat_stream(
        &self,
        mut req: ChatRequest,
        ctrl: &RequestControl,
    ) -> Result<DynStream<'static, StreamEvent>, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        let mut body = ollama::build_request(&req)?;
        body["stream"] = json!(true);
        let mut http = HttpRequest::post_json(self.target.endpoint("api/chat"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let res = self.open_stream(http, ctrl).await?;
        Ok(Box::pin(ollama::NdjsonEventStream::new(
            res.body,
            Box::new(ollama::ChatStreamDecoder::new()),
        )))
    }

    /// `POST /api/embed` — text embeddings.
    async fn embed(&self, req: EmbedRequest, ctrl: &RequestControl) -> Result<EmbedOutput, Error> {
        let headers = self.authenticate().await?;
        let body = ollama::build_embed_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("api/embed"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        ollama::decode_embed(&body)
    }
}
