//! The Cohere adapter: binds a `ServiceTarget` speaking the Cohere
//! protocol (`Protocol::Other("cohere")`) to a transport and credential
//! port.
//!
//! Covers the v2 chat/embed/rerank surfaces and the v1 classify surface;
//! per-endpoint differences arrive through `ServiceTarget` and request
//! `extensions`. There is no model listing on these surfaces, so
//! `models` stays `Unsupported`.

use std::sync::Arc;

use serde_json::json;

use crate::api::*;
use crate::auth::{AuthKind, Credential, CredentialResolver, CredentialScope};
use crate::effects::{EffectHost, ResourceRef};
use crate::protocols::cohere;
use crate::transport::{EventStream, HttpRequest, Transport};

/// Adapter for the Cohere protocol family.
pub struct CohereAdapter {
    target: ServiceTarget,
    auth: AuthKind,
    transport: Arc<dyn Transport>,
    resolver: Arc<dyn CredentialResolver>,
    effects: Option<Arc<dyn EffectHost>>,
}

impl CohereAdapter {
    pub fn new(
        target: ServiceTarget,
        auth: AuthKind,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
    ) -> Result<Self, Error> {
        match &target.protocol {
            Protocol::Other(name) if name == "cohere" => {}
            other => {
                return Err(Error::Unsupported(format!(
                    "CohereAdapter cannot serve protocol {other:?}"
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

    /// Both API-key and bearer credentials travel as `Authorization:
    /// Bearer`; the resolver owns custody and rotation.
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
    /// host effect port before the request enters the codec.
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

    /// Resolve `EmbedInput::Media` resources the same way chat media is
    /// resolved; the codec then encodes the inline data.
    async fn resolve_embed_media(
        &self,
        req: &mut EmbedRequest,
        ctrl: &RequestControl,
    ) -> Result<(), Error> {
        let needs_resolve = req
            .inputs
            .iter()
            .any(|i| matches!(i, EmbedInput::Media(MediaSource::Resource(_))));
        if !needs_resolve {
            return Ok(());
        }
        let effects = self
            .effects
            .as_ref()
            .ok_or_else(|| Error::Unsupported("media resources need a host effect port".into()))?;
        for input in &mut req.inputs {
            if let EmbedInput::Media(MediaSource::Resource(id)) = input {
                let rr = ResourceRef {
                    id: id.clone(),
                    media_type: None,
                };
                let data = effects.read_resource(&rr, ctrl).await?;
                *input = EmbedInput::Media(MediaSource::Inline {
                    media_type: "application/octet-stream".into(),
                    data,
                });
            }
        }
        Ok(())
    }

    async fn send_json(&self, req: HttpRequest, ctrl: &RequestControl) -> Result<Vec<u8>, Error> {
        let res = self.transport.send(req, ctrl).await?;
        if !res.status.is_success() {
            return Err(cohere::http_error(res.status.as_u16(), &res.body));
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
            return Err(cohere::http_error(res.status.as_u16(), &body));
        }
        Ok(res)
    }
}

#[async_trait::async_trait]
impl Adapter for CohereAdapter {
    fn target(&self) -> &ServiceTarget {
        &self.target
    }

    /// `POST /v2/chat` — non-streaming generation.
    async fn chat(&self, mut req: ChatRequest, ctrl: &RequestControl) -> Result<Output, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        let body = cohere::build_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("v2/chat"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        cohere::decode_response(&body)
    }

    /// `POST /v2/chat` with `"stream": true` — SSE-framed generation.
    async fn chat_stream(
        &self,
        mut req: ChatRequest,
        ctrl: &RequestControl,
    ) -> Result<DynStream<'static, StreamEvent>, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        let mut body = cohere::build_request(&req)?;
        body["stream"] = json!(true);
        let mut http = HttpRequest::post_json(self.target.endpoint("v2/chat"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let res = self.open_stream(http, ctrl).await?;
        Ok(Box::pin(EventStream::new(
            res.body,
            Box::new(cohere::ChatStreamDecoder::new()),
        )))
    }

    /// `POST /v2/embed` — text or image embeddings.
    async fn embed(&self, mut req: EmbedRequest, ctrl: &RequestControl) -> Result<EmbedOutput, Error> {
        self.resolve_embed_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        let body = cohere::build_embed_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("v2/embed"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        cohere::decode_embed(&body)
    }

    /// `POST /v2/rerank` — document reranking.
    async fn rerank(&self, req: RerankRequest, ctrl: &RequestControl) -> Result<RerankOutput, Error> {
        let headers = self.authenticate().await?;
        let body = cohere::build_rerank_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("v2/rerank"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        cohere::decode_rerank(&body)
    }

    /// `POST /v1/classify` — the legacy classify surface; it shares the
    /// service's error envelope but not the v2 message shapes.
    async fn classify(
        &self,
        req: ClassifyRequest,
        ctrl: &RequestControl,
    ) -> Result<ClassifyOutput, Error> {
        let headers = self.authenticate().await?;
        let body = cohere::build_classify_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("v1/classify"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        cohere::decode_classify(&body)
    }
}
