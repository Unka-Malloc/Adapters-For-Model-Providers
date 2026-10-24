//! The Amazon Bedrock Converse adapter: binds a `ServiceTarget` speaking
//! the Converse protocol to a transport and credential port.
//!
//! This lane covers bearer-token access: Bedrock API keys and compatible
//! proxies resolve to `Authorization: Bearer …`. `Credential::Cloud`
//! (SigV4 signing material) is the cloud-identity lane's concern and is
//! rejected explicitly here so it can be wired without changing this
//! adapter's behavior for bearer deployments.

use std::sync::Arc;

use crate::api::*;
use crate::auth::{AuthKind, Credential, CredentialResolver, CredentialScope};
use crate::effects::{EffectHost, ResourceRef};
use crate::protocols::bedrock_converse;
use crate::transport::{HttpRequest, Transport};

/// Adapter for the Amazon Bedrock Converse protocol family.
pub struct BedrockAdapter {
    target: ServiceTarget,
    auth: AuthKind,
    transport: Arc<dyn Transport>,
    resolver: Arc<dyn CredentialResolver>,
    effects: Option<Arc<dyn EffectHost>>,
}

impl BedrockAdapter {
    pub fn new(
        target: ServiceTarget,
        auth: AuthKind,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
    ) -> Result<Self, Error> {
        match target.protocol {
            Protocol::BedrockConverse => {}
            ref other => {
                return Err(Error::Unsupported(format!(
                    "BedrockAdapter cannot serve protocol {other:?}"
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

    /// Resolved credential → headers for API-key and bearer material.
    /// Cloud (SigV4) material yields no headers here — the finished
    /// request is signed by `sign_request` after every header is applied.
    async fn authenticate(&self) -> Result<(http::HeaderMap, Credential), Error> {
        let scope = CredentialScope {
            target: &self.target,
            auth: &self.auth,
        };
        let credential = self.resolver.resolve(scope).await?;
        let mut headers = self.target.headers.clone();
        match &credential {
            Credential::None | Credential::Cloud { .. } => {}
            Credential::ApiKey(secret) | Credential::Bearer { token: secret, .. } => {
                headers.insert(
                    http::header::AUTHORIZATION,
                    http::HeaderValue::from_str(&format!("Bearer {}", secret.expose()))
                        .map_err(|_| {
                            Error::Credential("credential is not a valid header value".into())
                        })?,
                );
            }
        }
        Ok((headers, credential))
    }

    /// Cloud-identity credentials sign the finished request (SigV4
    /// covers path, headers and body hash); bearer material needs no
    /// further step.
    fn sign_request(&self, http: &mut HttpRequest, credential: &Credential) -> Result<(), Error> {
        if let Credential::Cloud { .. } = credential {
            crate::sigv4::SigV4::new("bedrock").sign_now(http, credential, &self.target)?;
        }
        Ok(())
    }

    /// Resolve `MediaSource::Resource` parts into inline data through the
    /// host effect port before the request enters the codec. The codec
    /// infers the image format from the resolved bytes.
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

    /// `model/{id}/converse` or `model/{id}/converse-stream` with the
    /// model id percent-encoded into a single path segment.
    fn endpoint(&self, model: &str, operation: &str) -> url::Url {
        self.target.endpoint(&format!(
            "model/{}/{operation}",
            bedrock_converse::encode_path_segment(model)
        ))
    }

    async fn send_json(&self, req: HttpRequest, ctrl: &RequestControl) -> Result<Vec<u8>, Error> {
        let res = self.transport.send(req, ctrl).await?;
        if !res.status.is_success() {
            return Err(bedrock_converse::http_error(res.status.as_u16(), &res.body));
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
            return Err(bedrock_converse::http_error(res.status.as_u16(), &body));
        }
        Ok(res)
    }
}

#[async_trait::async_trait]
impl Adapter for BedrockAdapter {
    fn target(&self) -> &ServiceTarget {
        &self.target
    }

    async fn chat(&self, mut req: ChatRequest, ctrl: &RequestControl) -> Result<Output, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let (headers, credential) = self.authenticate().await?;
        let body = bedrock_converse::build_request(&req)?;
        let mut http = HttpRequest::post_json(self.endpoint(&req.model, "converse"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        self.sign_request(&mut http, &credential)?;
        let body = self.send_json(http, ctrl).await?;
        bedrock_converse::decode_response(&body)
    }

    async fn chat_stream(
        &self,
        mut req: ChatRequest,
        ctrl: &RequestControl,
    ) -> Result<DynStream<'static, StreamEvent>, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let (headers, credential) = self.authenticate().await?;
        // Same body, different endpoint: `converse-stream` answers with
        // AWS event-stream binary frames, not SSE.
        let body = bedrock_converse::build_request(&req)?;
        let mut http = HttpRequest::post_json(self.endpoint(&req.model, "converse-stream"), body);
        http.headers.insert(
            http::header::ACCEPT,
            http::HeaderValue::from_static("application/vnd.amazon.eventstream"),
        );
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        self.sign_request(&mut http, &credential)?;
        let res = self.open_stream(http, ctrl).await?;
        Ok(Box::pin(bedrock_converse::ConverseEventStream::new(
            res.body,
        )))
    }
}
