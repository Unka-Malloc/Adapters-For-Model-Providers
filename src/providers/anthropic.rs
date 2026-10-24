//! The Anthropic Messages adapter: binds a `ServiceTarget` speaking the
//! Messages protocol to a transport and credential port.
//!
//! Anthropic-compatible endpoints use this adapter; per-endpoint
//! differences arrive through `ServiceTarget` (base URL, extra headers
//! such as `anthropic-beta`) and request `extensions` (thinking
//! configuration, metadata, beta body fields).

use std::sync::Arc;

use serde_json::json;

use crate::api::*;
use crate::auth::{AuthKind, Credential, CredentialResolver, CredentialScope};
use crate::effects::{EffectHost, ResourceRef};
use crate::protocols::anthropic_messages;
use crate::transport::{EventStream, HttpRequest, Transport};

/// Adapter for the Anthropic Messages protocol family.
pub struct AnthropicAdapter {
    target: ServiceTarget,
    auth: AuthKind,
    transport: Arc<dyn Transport>,
    resolver: Arc<dyn CredentialResolver>,
    effects: Option<Arc<dyn EffectHost>>,
}

impl AnthropicAdapter {
    pub fn new(
        target: ServiceTarget,
        auth: AuthKind,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
    ) -> Result<Self, Error> {
        match target.protocol {
            Protocol::AnthropicMessages => {}
            ref other => {
                return Err(Error::Unsupported(format!(
                    "AnthropicAdapter cannot serve protocol {other:?}"
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

    /// Resolved credential + protocol headers. The adapter always owns
    /// `anthropic-version`; beta features ride caller-declared target
    /// headers, never generated here.
    async fn authenticate(&self) -> Result<http::HeaderMap, Error> {
        let scope = CredentialScope {
            target: &self.target,
            auth: &self.auth,
        };
        let credential = self.resolver.resolve(scope).await?;
        let mut headers = self.target.headers.clone();
        headers.insert(
            http::header::HeaderName::from_static("anthropic-version"),
            http::HeaderValue::from_static("2023-06-01"),
        );
        match credential {
            Credential::None => {}
            Credential::ApiKey(secret) => {
                headers.insert(
                    http::header::HeaderName::from_static("x-api-key"),
                    http::HeaderValue::from_str(secret.expose()).map_err(|_| {
                        Error::Credential("credential is not a valid header value".into())
                    })?,
                );
            }
            Credential::Bearer { token, .. } => {
                headers.insert(
                    http::header::AUTHORIZATION,
                    http::HeaderValue::from_str(&format!("Bearer {}", token.expose())).map_err(
                        |_| Error::Credential("credential is not a valid header value".into()),
                    )?,
                );
            }
            Credential::Cloud { .. } => {
                return Err(Error::Unsupported(
                    "cloud identity is not yet implemented for this adapter".into(),
                ));
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
                        let mut rr = ResourceRef {
                            id: id.clone(),
                            media_type: None,
                        };
                        let data = effects.read_resource(&mut rr, ctrl).await?;
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

    async fn send_json(&self, req: HttpRequest, ctrl: &RequestControl) -> Result<Vec<u8>, Error> {
        let res = self.transport.send(req, ctrl).await?;
        if !res.status.is_success() {
            return Err(anthropic_messages::http_error(res.status.as_u16(), &res.body));
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
            return Err(anthropic_messages::http_error(res.status.as_u16(), &body));
        }
        Ok(res)
    }
}

#[async_trait::async_trait]
impl Adapter for AnthropicAdapter {
    fn target(&self) -> &ServiceTarget {
        &self.target
    }

    async fn chat(&self, mut req: ChatRequest, ctrl: &RequestControl) -> Result<Output, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        let body = anthropic_messages::build_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("v1/messages"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        anthropic_messages::decode_response(&body)
    }

    async fn chat_stream(
        &self,
        mut req: ChatRequest,
        ctrl: &RequestControl,
    ) -> Result<DynStream<'static, StreamEvent>, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        // Streaming shares the endpoint and body; only the flag differs.
        let mut body = anthropic_messages::build_request(&req)?;
        body["stream"] = json!(true);
        let mut http = HttpRequest::post_json(self.target.endpoint("v1/messages"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let res = self.open_stream(http, ctrl).await?;
        Ok(Box::pin(EventStream::new(
            res.body,
            Box::new(anthropic_messages::MessagesStreamDecoder::new()),
        )))
    }
}
