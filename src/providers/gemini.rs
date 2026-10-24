//! The Gemini adapter: binds a `ServiceTarget` speaking the
//! generateContent protocol family to a transport and credential port.
//!
//! Non-streaming calls POST `v1beta/models/{model}:generateContent`;
//! streaming calls POST `v1beta/models/{model}:streamGenerateContent`
//! with `alt=sse`. API-key credentials travel in `x-goog-api-key`;
//! OAuth bearer and cloud-identity tokens use `Authorization: Bearer`.

use std::sync::Arc;

use crate::api::*;
use crate::auth::{AuthKind, Credential, CredentialResolver, CredentialScope};
use crate::effects::{EffectHost, ResourceRef};
use crate::protocols::{gemini_generate, gemini_media};
use crate::transport::{EventStream, HttpRequest, Transport};

/// Adapter for the Gemini generateContent protocol family.
pub struct GeminiAdapter {
    target: ServiceTarget,
    auth: AuthKind,
    transport: Arc<dyn Transport>,
    resolver: Arc<dyn CredentialResolver>,
    effects: Option<Arc<dyn EffectHost>>,
}

impl GeminiAdapter {
    pub fn new(
        target: ServiceTarget,
        auth: AuthKind,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
    ) -> Result<Self, Error> {
        match target.protocol {
            Protocol::GeminiGenerateContent => {}
            ref other => {
                return Err(Error::Unsupported(format!(
                    "GeminiAdapter cannot serve protocol {other:?}"
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
            Credential::ApiKey(secret) => {
                headers.insert(
                    "x-goog-api-key",
                    http::HeaderValue::from_str(secret.expose()).map_err(|_| {
                        Error::Credential("credential is not a valid header value".into())
                    })?,
                );
            }
            Credential::Bearer { token, .. } | Credential::Cloud { token, .. } => {
                headers.insert(
                    http::header::AUTHORIZATION,
                    http::HeaderValue::from_str(&format!("Bearer {}", token.expose())).map_err(
                        |_| Error::Credential("credential is not a valid header value".into()),
                    )?,
                );
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
    /// host effect port before the request enters the codec. A `detail`
    /// declared on the part supplies the resource's media type.
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
                if let ContentPart::Media { source, detail, .. } = part {
                    if let MediaSource::Resource(id) = source {
                        let mut rr = ResourceRef {
                            id: id.clone(),
                            media_type: detail.clone(),
                        };
                        let data = effects.read_resource(&mut rr, ctrl).await?;
                        *source = MediaSource::Inline {
                            media_type: rr
                                .media_type
                                .or_else(|| detail.clone())
                                .unwrap_or_else(|| "application/octet-stream".into()),
                            data,
                        };
                    }
                }
            }
        }
        Ok(())
    }

    fn generate_url(&self, model: &str) -> url::Url {
        self.target
            .endpoint(&format!("v1beta/models/{model}:generateContent"))
    }

    fn stream_url(&self, model: &str) -> url::Url {
        let mut url = self
            .target
            .endpoint(&format!("v1beta/models/{model}:streamGenerateContent"));
        url.query_pairs_mut().append_pair("alt", "sse");
        url
    }

    async fn send_json(&self, req: HttpRequest, ctrl: &RequestControl) -> Result<Vec<u8>, Error> {
        let res = self.transport.send(req, ctrl).await?;
        if !res.status.is_success() {
            return Err(gemini_generate::http_error(res.status.as_u16(), &res.body));
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
            return Err(gemini_generate::http_error(res.status.as_u16(), &body));
        }
        Ok(res)
    }

    /// `v1beta/{operation}[suffix]` — each `/`-separated segment of the
    /// provider's opaque operation name is percent-encoded as a path
    /// segment; `suffix` (e.g. `:cancel`) is appended after the last one.
    fn operation_url(&self, name: &str, suffix: &str) -> url::Url {
        let mut url = self.target.endpoint("v1beta");
        let Ok(mut segments) = url.path_segments_mut() else {
            return self.target.endpoint(&format!("v1beta/{name}{suffix}"));
        };
        for segment in name.trim_matches('/').split('/') {
            if !segment.is_empty() {
                segments.push(segment);
            }
        }
        drop(segments);
        if !suffix.is_empty() {
            url.set_path(&format!("{}{suffix}", url.path()));
        }
        url
    }
}

#[async_trait::async_trait]
impl Adapter for GeminiAdapter {
    fn target(&self) -> &ServiceTarget {
        &self.target
    }

    async fn chat(&self, mut req: ChatRequest, ctrl: &RequestControl) -> Result<Output, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        let body = gemini_generate::build_request(&req)?;
        let mut http = HttpRequest::post_json(self.generate_url(&req.model), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        gemini_generate::decode_response(&body)
    }

    async fn chat_stream(
        &self,
        mut req: ChatRequest,
        ctrl: &RequestControl,
    ) -> Result<DynStream<'static, StreamEvent>, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        let body = gemini_generate::build_request(&req)?;
        let mut http = HttpRequest::post_json(self.stream_url(&req.model), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let res = self.open_stream(http, ctrl).await?;
        Ok(Box::pin(EventStream::new(
            res.body,
            Box::new(gemini_generate::GenerateStreamDecoder::new()),
        )))
    }

    /// `POST v1beta/models/{model}:predict` — the Imagen-style image
    /// surface: `predictions[].bytesBase64Encoded` decode into inline
    /// media.
    async fn image(&self, req: ImageRequest, ctrl: &RequestControl) -> Result<MediaOutput, Error> {
        let headers = self.authenticate().await?;
        let body = gemini_media::build_image_request(&req)?;
        let mut http = HttpRequest::post_json(
            self.target
                .endpoint(&format!("v1beta/models/{}:predict", req.model)),
            body,
        );
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        gemini_media::decode_image_output(&body)
    }

    /// Veo video generation is a long-running operation, not a
    /// synchronous call: submit through `submit_job`, poll `job_state`.
    async fn video(&self, _req: VideoRequest, _ctrl: &RequestControl) -> Result<MediaOutput, Error> {
        Err(self.unsupported(
            "video (Veo runs on predictLongRunning; use submit_job + job_state)",
        ))
    }

    /// `POST v1beta/models/{model}:predictLongRunning` — only
    /// `JobOperation::Video` maps onto this wire; chat, embed, image and
    /// audio are synchronous surfaces on Gemini and answer `Unsupported`.
    async fn submit_job(&self, op: JobOperation, ctrl: &RequestControl) -> Result<JobHandle, Error> {
        let JobOperation::Video(req) = &op else {
            return Err(self.unsupported(
                "submit_job for this operation (only video is a long-running job on Gemini)",
            ));
        };
        let headers = self.authenticate().await?;
        let body = gemini_media::build_video_request(req)?;
        let mut http = HttpRequest::post_json(
            self.target
                .endpoint(&format!("v1beta/models/{}:predictLongRunning", req.model)),
            body,
        );
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        gemini_media::decode_operation_handle(&self.target, &body)
    }

    /// `GET v1beta/{operation}` — poll the long-running operation.
    async fn job_state(&self, handle: &JobHandle, ctrl: &RequestControl) -> Result<JobState, Error> {
        let headers = self.authenticate().await?;
        let mut http = HttpRequest::get(self.operation_url(&handle.id, ""));
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        gemini_media::decode_operation_state(&body)
    }

    /// `POST v1beta/{operation}:cancel` — the Veo LRO surface supports
    /// the cancel verb.
    async fn cancel_job(&self, handle: &JobHandle, ctrl: &RequestControl) -> Result<(), Error> {
        let headers = self.authenticate().await?;
        let mut http = HttpRequest::post_json(
            self.operation_url(&handle.id, ":cancel"),
            serde_json::json!({}),
        );
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        self.send_json(http, ctrl).await?;
        Ok(())
    }
}
