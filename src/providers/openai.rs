//! The OpenAI-protocol adapter: binds a `ServiceTarget` speaking Chat
//! Completions or Responses to a transport and credential port.
//!
//! Every OpenAI-compatible endpoint (OpenAI itself, DeepSeek, Moonshot,
//! xAI, routers, local servers) uses this adapter; per-endpoint
//! differences arrive through `ServiceTarget` and request `extensions`.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde_json::{Value, json};

use crate::api::*;
use crate::auth::{AuthKind, Credential, CredentialResolver, CredentialScope};
use crate::effects::{EffectHost, ResourceRef};
use crate::protocols::{
    openai_chat, openai_data, openai_media, openai_realtime, openai_responses,
};
use crate::transport::{DuplexChannel, EventStream, HttpRequest, Transport};

/// Adapter for the OpenAI protocol family (Chat Completions + Responses).
pub struct OpenAiAdapter {
    target: ServiceTarget,
    auth: AuthKind,
    transport: Arc<dyn Transport>,
    resolver: Arc<dyn CredentialResolver>,
    effects: Option<Arc<dyn EffectHost>>,
}

impl OpenAiAdapter {
    pub fn new(
        target: ServiceTarget,
        auth: AuthKind,
        transport: Arc<dyn Transport>,
        resolver: Arc<dyn CredentialResolver>,
    ) -> Result<Self, Error> {
        match target.protocol {
            Protocol::OpenAiChatCompletions | Protocol::OpenAiResponses => {}
            ref other => {
                return Err(Error::Unsupported(format!(
                    "OpenAiAdapter cannot serve protocol {other:?}"
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

    /// Resolve one `MediaSource::Resource` into inline data through the
    /// host effect port (the single-source counterpart of
    /// `resolve_media` for non-chat payloads).
    async fn resolve_source(
        &self,
        source: &mut MediaSource,
        ctrl: &RequestControl,
    ) -> Result<(), Error> {
        if let MediaSource::Resource(id) = source {
            let effects = self
                .effects
                .as_ref()
                .ok_or_else(|| Error::Unsupported("media resources need a host effect port".into()))?;
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
        Ok(())
    }

    async fn send_json(&self, req: HttpRequest, ctrl: &RequestControl) -> Result<Vec<u8>, Error> {
        let res = self.transport.send(req, ctrl).await?;
        if !res.status.is_success() {
            return Err(openai_chat::http_error(res.status.as_u16(), &res.body));
        }
        Ok(res.body.to_vec())
    }

    /// POST a multipart/form-data body (file uploads). Returns the raw
    /// response body; status errors map through `openai_chat::http_error`.
    async fn send_multipart(
        &self,
        path: &str,
        fields: Vec<(String, String)>,
        file: openai_media::MultipartFile<'_>,
        ctrl: &RequestControl,
    ) -> Result<Vec<u8>, Error> {
        let headers = self.authenticate().await?;
        let (content_type, body) = openai_media::encode_multipart(&fields, &file);
        let mut http = HttpRequest {
            method: http::Method::POST,
            url: self.target.endpoint(path),
            headers: http::HeaderMap::new(),
            body: body.into(),
        };
        http.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_str(&content_type)
                .map_err(|_| Error::Protocol("invalid multipart content type".into()))?,
        );
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        self.send_json(http, ctrl).await
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
            return Err(openai_chat::http_error(res.status.as_u16(), &body));
        }
        Ok(res)
    }
}

#[async_trait::async_trait]
impl Adapter for OpenAiAdapter {
    fn target(&self) -> &ServiceTarget {
        &self.target
    }

    /// `GET /models` — the OpenAI-compatible discovery surface.
    async fn models(&self, ctrl: &RequestControl) -> Result<Vec<ModelInfo>, Error> {
        let headers = self.authenticate().await?;
        let mut http = HttpRequest::get(self.target.endpoint("models"));
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        openai_chat::decode_models(&body)
    }

    async fn chat(&self, mut req: ChatRequest, ctrl: &RequestControl) -> Result<Output, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        match self.target.protocol {
            Protocol::OpenAiChatCompletions => {
                let body = openai_chat::build_request(&req)?;
                let mut http = HttpRequest::post_json(self.target.endpoint("chat/completions"), body);
                for (k, v) in headers.iter() {
                    http.headers.append(k, v.clone());
                }
                let body = self.send_json(http, ctrl).await?;
                openai_chat::decode_response(&body)
            }
            Protocol::OpenAiResponses => {
                let body = openai_responses::build_request(&req)?;
                let mut http = HttpRequest::post_json(self.target.endpoint("responses"), body);
                for (k, v) in headers.iter() {
                    http.headers.append(k, v.clone());
                }
                let body = self.send_json(http, ctrl).await?;
                openai_responses::decode_response(&body)
            }
            ref other => Err(Error::Unsupported(format!(
                "OpenAiAdapter cannot serve protocol {other:?}"
            ))),
        }
    }

    async fn chat_stream(
        &self,
        mut req: ChatRequest,
        ctrl: &RequestControl,
    ) -> Result<DynStream<'static, StreamEvent>, Error> {
        self.resolve_media(&mut req, ctrl).await?;
        let headers = self.authenticate().await?;
        match self.target.protocol {
            Protocol::OpenAiChatCompletions => {
                let mut body = openai_chat::build_request(&req)?;
                body["stream"] = json!(true);
                body.as_object_mut().unwrap().entry("stream_options".to_owned()).or_insert_with(|| {
                    json!({"include_usage": true})
                });
                let mut http = HttpRequest::post_json(self.target.endpoint("chat/completions"), body);
                for (k, v) in headers.iter() {
                    http.headers.append(k, v.clone());
                }
                let res = self.open_stream(http, ctrl).await?;
                Ok(Box::pin(EventStream::new(
                    res.body,
                    Box::new(openai_chat::ChatStreamDecoder::new()),
                )))
            }
            Protocol::OpenAiResponses => {
                let mut body = openai_responses::build_request(&req)?;
                body["stream"] = json!(true);
                let mut http = HttpRequest::post_json(self.target.endpoint("responses"), body);
                for (k, v) in headers.iter() {
                    http.headers.append(k, v.clone());
                }
                let res = self.open_stream(http, ctrl).await?;
                Ok(Box::pin(EventStream::new(
                    res.body,
                    Box::new(openai_responses::ResponsesStreamDecoder::new()),
                )))
            }
            ref other => Err(Error::Unsupported(format!(
                "OpenAiAdapter cannot serve protocol {other:?}"
            ))),
        }
    }

    /// `POST /embeddings` — the OpenAI-compatible embedding surface.
    async fn embed(&self, req: EmbedRequest, ctrl: &RequestControl) -> Result<EmbedOutput, Error> {
        let headers = self.authenticate().await?;
        let body = openai_data::build_embed_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("embeddings"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        openai_data::decode_embed_output(&body)
    }

    /// `POST /moderations` — the OpenAI-compatible moderation surface.
    async fn moderate(
        &self,
        req: ModerateRequest,
        ctrl: &RequestControl,
    ) -> Result<ModerateOutput, Error> {
        let headers = self.authenticate().await?;
        let body = openai_data::build_moderate_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("moderations"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        openai_data::decode_moderate_output(&body)
    }

    /// `POST /images/generations` — the OpenAI-compatible image surface.
    async fn image(&self, req: ImageRequest, ctrl: &RequestControl) -> Result<MediaOutput, Error> {
        let headers = self.authenticate().await?;
        let body = openai_media::build_image_request(&req)?;
        let mut http = HttpRequest::post_json(self.target.endpoint("images/generations"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        openai_media::decode_image_output(&body)
    }

    /// `POST /audio/speech` + `POST /audio/transcriptions`.
    async fn audio(&self, req: AudioRequest, ctrl: &RequestControl) -> Result<MediaOutput, Error> {
        match req {
            AudioRequest::Synthesize {
                model,
                input,
                voice,
                format,
                extensions,
            } => {
                let headers = self.authenticate().await?;
                let media_type = openai_media::speech_media_type(format.as_deref());
                let body = openai_media::build_speech_request(
                    &model,
                    &input,
                    voice.as_deref(),
                    format.as_deref(),
                    &extensions,
                )?;
                let mut http =
                    HttpRequest::post_json(self.target.endpoint("audio/speech"), body);
                for (k, v) in headers.iter() {
                    http.headers.append(k, v.clone());
                }
                // The answer is raw audio bytes, not JSON: send_json only
                // performs the status check and hands back the body.
                let bytes = self.send_json(http, ctrl).await?;
                Ok(MediaOutput {
                    media: vec![MediaSource::Inline {
                        media_type,
                        data: bytes.into(),
                    }],
                    ..Default::default()
                })
            }
            AudioRequest::Transcribe {
                model,
                audio,
                language,
                extensions,
            } => {
                let mut source = audio;
                self.resolve_source(&mut source, ctrl).await?;
                let (data, media_type) = match source {
                    MediaSource::Inline { media_type, data } => (data, media_type),
                    MediaSource::Url(_) => {
                        return Err(Error::Unsupported(
                            "OpenAI transcription uploads audio bytes; fetch remote audio first".into(),
                        ));
                    }
                    MediaSource::Resource(_) => {
                        return Err(Error::Unsupported(
                            "media resources need a host effect port".into(),
                        ));
                    }
                };
                let fields =
                    openai_media::transcription_fields(&model, language.as_deref(), &extensions);
                let file = openai_media::MultipartFile {
                    field: "file",
                    filename: openai_media::audio_filename(&media_type),
                    content_type: media_type,
                    data: &data,
                };
                let body = self
                    .send_multipart("audio/transcriptions", fields, file, ctrl)
                    .await?;
                openai_media::decode_transcription(&body)
            }
        }
    }

    /// OpenAI Batch: upload a JSONL request file (`POST /files`,
    /// `purpose=batch`) then create the batch (`POST /batches`). Chat
    /// and Embed operations map onto their `/v1/` endpoints; `Custom`
    /// carries an explicit `/v1/…` endpoint path plus body.
    async fn submit_job(&self, op: JobOperation, ctrl: &RequestControl) -> Result<JobHandle, Error> {
        let (endpoint, body) = match &op {
            JobOperation::Chat(req) => (
                "/v1/chat/completions".to_owned(),
                openai_chat::build_request(req)?,
            ),
            JobOperation::Embed(req) => {
                ("/v1/embeddings".to_owned(), openai_data::build_embed_request(req)?)
            }
            JobOperation::Custom { kind, payload } if kind.starts_with("/v1/") => {
                (kind.clone(), payload.clone())
            }
            _ => {
                return Err(self.unsupported(
                    "submit_job for this operation (batches accept /v1/ endpoint payloads)",
                ));
            }
        };
        // Upload the one-line JSONL request file.
        let jsonl = openai_data::batch_line("a4mp-0", &endpoint, body)?;
        let file = openai_media::MultipartFile {
            field: "file",
            filename: "input.jsonl".to_owned(),
            content_type: "application/jsonl".to_owned(),
            data: &jsonl,
        };
        let body = self
            .send_multipart(
                "files",
                vec![("purpose".to_owned(), "batch".to_owned())],
                file,
                ctrl,
            )
            .await?;
        let input_file_id = openai_data::decode_file_id(&body)?;

        // Create the batch.
        let headers = self.authenticate().await?;
        let body = json!({
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": "24h",
        });
        let mut http = HttpRequest::post_json(self.target.endpoint("batches"), body);
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        let batch = openai_data::decode_batch(&body)?;
        Ok(JobHandle {
            provider: self.target.provider.clone(),
            protocol: self.target.protocol.clone(),
            id: batch.id,
            state: json!({
                "endpoint": endpoint,
                "input_file_id": input_file_id,
            }),
        })
    }

    /// `GET /batches/{id}` → canonical job state. A completed batch's
    /// output file (`GET /files/{id}/content`) parses as JSONL rows.
    async fn job_state(&self, handle: &JobHandle, ctrl: &RequestControl) -> Result<JobState, Error> {
        let headers = self.authenticate().await?;
        let mut http = HttpRequest::get(self.target.endpoint(&format!("batches/{}", handle.id)));
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        let body = self.send_json(http, ctrl).await?;
        let batch = openai_data::decode_batch(&body)?;
        match batch.status.as_str() {
            "validating" => Ok(JobState::Submitted),
            "in_progress" | "finalizing" | "cancelling" => Ok(JobState::Running),
            "completed" => {
                let mut result = JobResult::default();
                if let Some(file_id) = &batch.output_file_id {
                    let mut http = HttpRequest::get(
                        self.target.endpoint(&format!("files/{file_id}/content")),
                    );
                    for (k, v) in headers.iter() {
                        http.headers.append(k, v.clone());
                    }
                    let body = self.send_json(http, ctrl).await?;
                    result.output = Value::Array(openai_data::decode_jsonl(&body)?);
                    result.files.push(RemoteFile {
                        id: file_id.clone(),
                        purpose: Some("batch_output".into()),
                        url: None,
                    });
                }
                Ok(JobState::Succeeded(result))
            }
            "failed" | "expired" => {
                let (code, message) = batch
                    .errors
                    .first()
                    .map(|(c, m)| (c.clone().or_else(|| Some(batch.status.clone())), m.clone()))
                    .unwrap_or_else(|| {
                        (Some(batch.status.clone()), format!("batch {}", batch.status))
                    });
                Ok(JobState::Failed { code, message })
            }
            "cancelled" => Ok(JobState::Cancelled),
            other => Err(Error::UpstreamProtocol(format!("unknown batch status {other:?}"))),
        }
    }

    /// `POST /batches/{id}/cancel`.
    async fn cancel_job(&self, handle: &JobHandle, ctrl: &RequestControl) -> Result<(), Error> {
        let headers = self.authenticate().await?;
        let mut http = HttpRequest::post_json(
            self.target.endpoint(&format!("batches/{}/cancel", handle.id)),
            json!({}),
        );
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        self.send_json(http, ctrl).await?;
        Ok(())
    }

    /// `wss://{base}/realtime?model=…` with `OpenAI-Beta: realtime=v1`.
    /// Requires a transport implementing `open_channel` (the `realtime-ws`
    /// feature's WebSocket transport, or a host-provided channel).
    async fn open_realtime(
        &self,
        config: RealtimeConfig,
        ctrl: &RequestControl,
    ) -> Result<Box<dyn RealtimeSession>, Error> {
        let headers = self.authenticate().await?;
        let mut http =
            HttpRequest::get(openai_realtime::session_url(&self.target, &config.model));
        for (k, v) in headers.iter() {
            http.headers.append(k, v.clone());
        }
        if !http.headers.contains_key("openai-beta") {
            http.headers.insert(
                http::header::HeaderName::from_static("openai-beta"),
                http::HeaderValue::from_static("realtime=v1"),
            );
        }
        let channel = self.transport.open_channel(http, ctrl).await?;
        let mut session = OpenAiRealtimeSession {
            channel,
            encoder: openai_realtime::RealtimeEncoder::new(),
            decoder: openai_realtime::RealtimeDecoder::new(&config),
            pending: VecDeque::new(),
            remote_closed: false,
        };
        if let Some(update) = openai_realtime::encode_session_update(&config) {
            session.send_frame(update).await?;
        }
        Ok(Box::new(session))
    }
}

/// A live OpenAI Realtime session over a `DuplexChannel`. `send` encodes
/// canonical inputs through `RealtimeEncoder`; `events()` drives the
/// inbound decode loop. Each `poll` creates a fresh `recv` future, so
/// `DuplexChannel::recv` implementations must be cancellation-safe (a
/// dropped poll must not lose frames).
pub struct OpenAiRealtimeSession {
    channel: Box<dyn DuplexChannel>,
    encoder: openai_realtime::RealtimeEncoder,
    decoder: openai_realtime::RealtimeDecoder,
    pending: VecDeque<RealtimeEvent>,
    /// Peer close (or terminal channel error) observed; survives across
    /// `events()` calls so a fresh stream cannot resurrect the channel.
    remote_closed: bool,
}

impl OpenAiRealtimeSession {
    // `&mut self` keeps the async future `Send`: `&OpenAiRealtimeSession`
    // could not be `Send` because `dyn DuplexChannel` is not `Sync`.
    async fn send_frame(&mut self, frame: Value) -> Result<(), Error> {
        let body = serde_json::to_vec(&frame)
            .map_err(|e| Error::Protocol(format!("realtime client event: {e}")))?;
        self.channel.send(body.into()).await
    }
}

#[async_trait::async_trait]
impl RealtimeSession for OpenAiRealtimeSession {
    async fn send(&mut self, event: RealtimeInput) -> Result<(), Error> {
        for frame in self.encoder.encode(&event)? {
            self.send_frame(frame).await?;
        }
        Ok(())
    }

    fn events(&mut self) -> DynStream<'_, RealtimeEvent> {
        Box::pin(SessionEvents { session: self })
    }

    async fn close(self: Box<Self>) -> Result<(), Error> {
        self.channel.close().await
    }
}

/// The session's inbound stream: pulls one frame, decodes it, drains the
/// resulting events, then pulls again. Decode failures surface as `Err`
/// items without ending the stream (provider `error` events are
/// non-fatal); a channel error or peer close ends it for good.
struct SessionEvents<'a> {
    session: &'a mut OpenAiRealtimeSession,
}

impl Stream for SessionEvents<'_> {
    type Item = Result<RealtimeEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(ev) = self.session.pending.pop_front() {
                return Poll::Ready(Some(Ok(ev)));
            }
            if self.session.remote_closed {
                return Poll::Ready(None);
            }
            // The recv future borrows the channel; it must be dropped
            // before touching the session again, so poll it inside a
            // block and carry only the result out.
            let polled = {
                let mut recv = Box::pin(self.session.channel.recv());
                match recv.as_mut().poll(cx) {
                    Poll::Pending => None,
                    Poll::Ready(result) => Some(result),
                }
            };
            match polled {
                Some(Ok(Some(frame))) => match self.session.decoder.decode(&frame) {
                    Ok(events) => self.session.pending.extend(events),
                    Err(e) => return Poll::Ready(Some(Err(e))),
                },
                Some(Ok(None)) => {
                    self.session.remote_closed = true;
                    return Poll::Ready(Some(Ok(RealtimeEvent::Closed { reason: None })));
                }
                Some(Err(e)) => {
                    self.session.remote_closed = true;
                    return Poll::Ready(Some(Err(e)));
                }
                None => return Poll::Pending,
            }
        }
    }
}
