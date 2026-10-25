//! Original fixtures for the OpenAI Realtime session family. A scripted
//! `DuplexChannel` replays inbound frames and captures outbound JSON;
//! a `ChannelTransport` hands it to `open_channel`. No sockets.

mod common;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use a4mp::protocols::openai_realtime;
use a4mp::*;
use serde_json::json;

// ---------------------------------------------------------------------------
// Scripted channel + channel transport
// ---------------------------------------------------------------------------

/// Replays queued inbound frames (end of script = peer closed) and
/// records every outbound frame parsed back to JSON.
struct ScriptedChannel {
    inbound: VecDeque<Result<Option<bytes::Bytes>, Error>>,
    sent: Arc<Mutex<Vec<serde_json::Value>>>,
    closed: Arc<Mutex<bool>>,
}

impl ScriptedChannel {
    fn new(frames: impl IntoIterator<Item = serde_json::Value>) -> Self {
        Self {
            inbound: frames
                .into_iter()
                .map(|f| Ok(Some(serde_json::to_vec(&f).unwrap().into())))
                .collect(),
            sent: Arc::new(Mutex::new(Vec::new())),
            closed: Arc::new(Mutex::new(false)),
        }
    }

    fn sent(&self) -> Arc<Mutex<Vec<serde_json::Value>>> {
        self.sent.clone()
    }

    fn closed(&self) -> Arc<Mutex<bool>> {
        self.closed.clone()
    }
}

#[async_trait::async_trait]
impl DuplexChannel for ScriptedChannel {
    async fn recv(&mut self) -> Result<Option<bytes::Bytes>, Error> {
        self.inbound.pop_front().unwrap_or(Ok(None))
    }

    async fn send(&self, data: bytes::Bytes) -> Result<(), Error> {
        let frame: serde_json::Value = serde_json::from_slice(&data).unwrap();
        self.sent.lock().unwrap().push(frame);
        Ok(())
    }

    async fn close(self: Box<Self>) -> Result<(), Error> {
        *self.closed.lock().unwrap() = true;
        Ok(())
    }
}

/// A transport that only implements `open_channel`: it records the
/// handshake request and returns the scripted channel.
struct ChannelTransport {
    channel: Mutex<Option<Box<dyn DuplexChannel>>>,
    requests: Arc<Mutex<Vec<common::RecordedRequest>>>,
}

impl ChannelTransport {
    fn new(channel: Box<dyn DuplexChannel>) -> Self {
        Self {
            channel: Mutex::new(Some(channel)),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn requests(&self) -> Arc<Mutex<Vec<common::RecordedRequest>>> {
        self.requests.clone()
    }
}

#[async_trait::async_trait]
impl Transport for ChannelTransport {
    async fn send(
        &self,
        _req: HttpRequest,
        _ctrl: &RequestControl,
    ) -> Result<HttpResponse, TransportError> {
        unreachable!("realtime sessions do not issue buffered requests")
    }

    async fn send_stream(
        &self,
        _req: HttpRequest,
        _ctrl: &RequestControl,
    ) -> Result<StreamingResponse, TransportError> {
        unreachable!("realtime sessions do not issue streamed responses")
    }

    async fn open_channel(
        &self,
        req: HttpRequest,
        _ctrl: &RequestControl,
    ) -> Result<Box<dyn DuplexChannel>, TransportError> {
        self.requests
            .lock()
            .unwrap()
            .push(common::RecordedRequest {
                method: req.method.clone(),
                url: req.url.clone(),
                headers: req.headers.clone(),
                body: req.body.to_vec(),
            });
        self.channel
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| TransportError::Io("channel already taken".into()))
    }
}

fn adapter(transport: ChannelTransport) -> OpenAiAdapter {
    let target = ServiceTarget::new(
        "test-vendor",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://api.vendor.test/v1").unwrap(),
    );
    OpenAiAdapter::new(
        target,
        AuthKind::Bearer,
        Arc::new(transport),
        Arc::new(StaticCredential(Credential::Bearer {
            token: Secret::new("sk-test"),
            expires_at: None,
        })),
    )
    .unwrap()
}

/// Pull the next event out of a session's `events()` stream.
async fn next_event(
    session: &mut Box<dyn RealtimeSession>,
) -> Option<Result<RealtimeEvent, Error>> {
    let mut stream = session.events();
    std::future::poll_fn(|cx| stream.as_mut().poll_next(cx)).await
}

// ---------------------------------------------------------------------------
// Handshake + session establishment
// ---------------------------------------------------------------------------

#[tokio::test]
async fn open_realtime_handshakes_ws_url_beta_header_and_session_update() {
    let channel = ScriptedChannel::new([json!({
        "type": "session.created",
        "session": { "id": "sess_1", "model": "gpt-realtime-test" },
    })]);
    let sent = channel.sent();
    let transport = ChannelTransport::new(Box::new(channel));
    let requests = transport.requests();
    let a = adapter(transport);

    let mut config = RealtimeConfig {
        model: "gpt-realtime-test".into(),
        modalities: vec!["text".into(), "audio".into()],
        ..Default::default()
    };
    config.extensions.insert("voice".into(), json!("alloy"));
    config
        .extensions
        .insert("turn_detection".into(), json!({"type": "server_vad"}));

    let mut session = a
        .open_realtime(config, &RequestControl::default())
        .await
        .unwrap();

    // Handshake request: ws scheme, /v1/realtime path, model in query,
    // Authorization + OpenAI-Beta headers.
    let req = requests.lock().unwrap()[0].clone();
    assert_eq!(req.method, http::Method::GET);
    assert_eq!(req.url.scheme(), "wss");
    assert_eq!(req.url.host_str(), Some("api.vendor.test"));
    assert_eq!(req.url.path(), "/v1/realtime");
    let model = req
        .url
        .query_pairs()
        .find(|(k, _)| k == "model")
        .map(|(_, v)| v.into_owned());
    assert_eq!(model.as_deref(), Some("gpt-realtime-test"));
    assert_eq!(req.headers.get("authorization").unwrap(), "Bearer sk-test");
    assert_eq!(req.headers.get("openai-beta").unwrap(), "realtime=v1");

    // session.update carries modalities plus extension fields verbatim.
    let sent = sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0],
        json!({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "turn_detection": {"type": "server_vad"},
            }
        })
    );
    drop(sent);

    // The server's session.created surfaces as a verbatim Native event.
    match next_event(&mut session).await.unwrap().unwrap() {
        RealtimeEvent::Native(v) => assert_eq!(v["type"], "session.created"),
        other => panic!("expected session.created native event, got {other:?}"),
    }
}

#[tokio::test]
async fn open_realtime_with_empty_config_sends_no_session_update() {
    let channel = ScriptedChannel::new([]);
    let sent = channel.sent();
    let transport = ChannelTransport::new(Box::new(channel));
    let a = adapter(transport);

    let _session = a
        .open_realtime(
            RealtimeConfig {
                model: "m".into(),
                ..Default::default()
            },
            &RequestControl::default(),
        )
        .await
        .unwrap();
    assert!(sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn open_realtime_surfaces_unsupported_transport() {
    // MockTransport does not override open_channel; the default reports
    // unsupported, which the adapter propagates as a transport error.
    let target = ServiceTarget::new(
        "test-vendor",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://api.vendor.test/v1").unwrap(),
    );
    let a = OpenAiAdapter::new(
        target,
        AuthKind::None,
        Arc::new(common::MockTransport::new()),
        Arc::new(NoCredentials),
    )
    .unwrap();
    let err = match a
        .open_realtime(RealtimeConfig::default(), &RequestControl::default())
        .await
    {
        Err(e) => e,
        Ok(_) => panic!("expected unsupported-transport error"),
    };
    assert!(matches!(err, Error::Transport(TransportError::Io(_))));
}

// ---------------------------------------------------------------------------
// Outbound encoding
// ---------------------------------------------------------------------------

#[tokio::test]
async fn send_encodes_canonical_inputs_as_client_events() {
    let channel = ScriptedChannel::new([]);
    let sent = channel.sent();
    let transport = ChannelTransport::new(Box::new(channel));
    let a = adapter(transport);
    let mut session = a
        .open_realtime(RealtimeConfig::default(), &RequestControl::default())
        .await
        .unwrap();

    session
        .send(RealtimeInput::Text("hello".into()))
        .await
        .unwrap();
    session
        .send(RealtimeInput::Audio {
            data: bytes::Bytes::from_static(&[1, 2, 3]),
            media_type: "audio/pcm16".into(),
        })
        .await
        .unwrap();
    session
        .send(RealtimeInput::ToolResult(ToolResult {
            call_id: "call_1".into(),
            content: vec![ContentPart::Text { text: "42".into() }],
            is_error: false,
        }))
        .await
        .unwrap();
    // Commit with audio pending: buffer commit, then response.create.
    session.send(RealtimeInput::Commit).await.unwrap();
    // Commit without pending audio: only response.create.
    session.send(RealtimeInput::Commit).await.unwrap();
    session.send(RealtimeInput::Cancel).await.unwrap();
    session
        .send(RealtimeInput::Native(json!({
            "type": "session.update",
            "session": {"turn_detection": null}
        })))
        .await
        .unwrap();

    let sent = sent.lock().unwrap();
    assert_eq!(
        sent.as_slice(),
        &[
            json!({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "hello"}]}
            }),
            json!({"type": "input_audio_buffer.append", "audio": "AQID"}),
            json!({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output",
                         "call_id": "call_1", "output": "42"}
            }),
            json!({"type": "input_audio_buffer.commit"}),
            json!({"type": "response.create"}),
            json!({"type": "response.create"}),
            json!({"type": "response.cancel"}),
            json!({"type": "session.update", "session": {"turn_detection": null}}),
        ]
    );
}

#[tokio::test]
async fn send_rejects_malformed_native_frame() {
    let channel = ScriptedChannel::new([]);
    let transport = ChannelTransport::new(Box::new(channel));
    let a = adapter(transport);
    let mut session = a
        .open_realtime(RealtimeConfig::default(), &RequestControl::default())
        .await
        .unwrap();
    for bad in [json!({"no_type": true}), json!([1, 2])] {
        let err = session.send(RealtimeInput::Native(bad)).await.unwrap_err();
        assert!(matches!(err, Error::Protocol(_)));
    }
}

// ---------------------------------------------------------------------------
// Inbound decoding
// ---------------------------------------------------------------------------

#[tokio::test]
async fn events_decode_deltas_toolcalls_usage_error_and_close() {
    let frames = [
        json!({"type": "session.created", "session": {"id": "s1"}}),
        json!({"type": "response.created", "response": {"id": "r1"}}),
        json!({"type": "response.output_item.added", "item": {"type": "message"}}),
        json!({"type": "response.text.delta", "delta": "Hel"}),
        json!({"type": "response.text.delta", "delta": "lo"}),
        // base64 of [0xDE, 0xAD, 0xBE, 0xEF]
        json!({"type": "response.audio.delta", "delta": "3q2+7w=="}),
        json!({"type": "response.audio_transcript.delta", "delta": "wor"}),
        json!({"type": "response.audio_transcript.done", "transcript": "word"}),
        json!({"type": "conversation.item.input_audio_transcription.completed",
               "transcript": "user said hi"}),
        json!({"type": "response.function_call_arguments.delta",
               "call_id": "c1", "delta": "{\"a\""}),
        json!({"type": "response.function_call_arguments.done",
               "call_id": "c1", "name": "lookup", "arguments": "{\"a\":1}"}),
        json!({"type": "input_audio_buffer.speech_started"}),
        json!({"type": "rate_limits.updated", "rate_limits": []}),
        json!({"type": "response.done", "response": {
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                      "input_token_details": {"audio_tokens": 3}}
        }}),
        json!({"type": "error",
               "error": {"type": "server_error", "code": "internal", "message": "boom"}}),
    ];
    let channel = ScriptedChannel::new(frames);
    let transport = ChannelTransport::new(Box::new(channel));
    let a = adapter(transport);
    let mut session = a
        .open_realtime(RealtimeConfig::default(), &RequestControl::default())
        .await
        .unwrap();

    let mut items = Vec::new();
    while let Some(item) = next_event(&mut session).await {
        items.push(item);
    }

    let mut text = String::new();
    let mut audio = Vec::new();
    let mut transcripts = Vec::new();
    let mut natives = Vec::new();
    let mut calls = Vec::new();
    let mut usage = None;
    let mut provider_err = None;
    let mut closed = false;
    for item in items {
        match item {
            Ok(RealtimeEvent::Text(t)) => text.push_str(&t),
            Ok(RealtimeEvent::Audio { data, media_type }) => {
                assert_eq!(media_type, "audio/pcm16");
                audio.extend_from_slice(&data);
            }
            Ok(RealtimeEvent::Transcript { text, final_ }) => {
                transcripts.push((text, final_))
            }
            Ok(RealtimeEvent::ToolCall(c)) => calls.push(c),
            Ok(RealtimeEvent::Usage(u)) => usage = Some(u),
            Ok(RealtimeEvent::Native(v)) => natives.push(v),
            Ok(RealtimeEvent::Closed { reason }) => {
                assert_eq!(reason, None);
                closed = true;
            }
            Err(e) => provider_err = Some(e),
            Ok(_) => {}
        }
    }

    assert_eq!(text, "Hello");
    assert_eq!(audio, [0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(
        transcripts,
        [
            ("wor".to_owned(), false),
            ("word".to_owned(), true),
            ("user said hi".to_owned(), true),
        ]
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "c1");
    assert_eq!(calls[0].name, "lookup");
    assert_eq!(calls[0].arguments, json!({"a": 1}));
    let usage = usage.unwrap();
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(4));
    assert_eq!(
        usage.extensions.get("input_token_details").unwrap()["audio_tokens"],
        3
    );
    // Lifecycle/bookkeeping frames arrive verbatim, including the
    // arg-delta that has no canonical shape and response.done itself.
    let types: Vec<&str> = natives
        .iter()
        .map(|v| v["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        types,
        [
            "session.created",
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "input_audio_buffer.speech_started",
            "rate_limits.updated",
            "response.done",
        ]
    );
    match provider_err.unwrap() {
        Error::Provider { code, message, .. } => {
            // `provider_error` names the error by the wire `type` field.
            assert_eq!(code.as_deref(), Some("server_error"));
            assert_eq!(message, "boom");
        }
        other => panic!("expected provider error, got {other:?}"),
    }
    assert!(closed);
}

#[tokio::test]
async fn send_and_recv_interleave() {
    let channel = ScriptedChannel::new([
        json!({"type": "response.text.delta", "delta": "a"}),
        json!({"type": "response.text.delta", "delta": "b"}),
    ]);
    let sent = channel.sent();
    let transport = ChannelTransport::new(Box::new(channel));
    let a = adapter(transport);
    let mut session = a
        .open_realtime(RealtimeConfig::default(), &RequestControl::default())
        .await
        .unwrap();

    session
        .send(RealtimeInput::Text("first".into()))
        .await
        .unwrap();
    match next_event(&mut session).await.unwrap().unwrap() {
        RealtimeEvent::Text(t) => assert_eq!(t, "a"),
        other => panic!("expected text delta, got {other:?}"),
    }
    session
        .send(RealtimeInput::Audio {
            data: bytes::Bytes::from_static(&[9]),
            media_type: "audio/pcm16".into(),
        })
        .await
        .unwrap();
    match next_event(&mut session).await.unwrap().unwrap() {
        RealtimeEvent::Text(t) => assert_eq!(t, "b"),
        other => panic!("expected text delta, got {other:?}"),
    }
    // Script exhausted: the peer-close event arrives, then the stream ends.
    assert!(matches!(
        next_event(&mut session).await.unwrap().unwrap(),
        RealtimeEvent::Closed { .. }
    ));
    assert!(next_event(&mut session).await.is_none());

    assert_eq!(
        sent.lock().unwrap().as_slice(),
        &[
            json!({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "first"}]}
            }),
            json!({"type": "input_audio_buffer.append", "audio": "CQ=="}),
        ]
    );
}

#[tokio::test]
async fn close_reports_to_channel() {
    let channel = ScriptedChannel::new([]);
    let closed = channel.closed();
    let transport = ChannelTransport::new(Box::new(channel));
    let a = adapter(transport);
    let session = a
        .open_realtime(RealtimeConfig::default(), &RequestControl::default())
        .await
        .unwrap();
    session.close().await.unwrap();
    assert!(*closed.lock().unwrap());
}

// ---------------------------------------------------------------------------
// Codec-level checks (transport-free)
// ---------------------------------------------------------------------------

#[test]
fn session_url_maps_scheme_and_model() {
    let target = ServiceTarget::new(
        "t",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://api.vendor.test/v1/").unwrap(),
    );
    let url = openai_realtime::session_url(&target, "m-1");
    assert_eq!(url.as_str(), "wss://api.vendor.test/v1/realtime?model=m-1");

    let plain = ServiceTarget::new(
        "t",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("http://localhost:8080").unwrap(),
    );
    let url = openai_realtime::session_url(&plain, "m 1");
    assert_eq!(url.as_str(), "ws://localhost:8080/realtime?model=m+1");
}

#[test]
fn session_update_is_none_when_nothing_to_set() {
    assert!(openai_realtime::encode_session_update(&RealtimeConfig::default()).is_none());
    let mut config = RealtimeConfig::default();
    config
        .extensions
        .insert("output_audio_format".into(), json!("g711_ulaw"));
    let update = openai_realtime::encode_session_update(&config).unwrap();
    assert_eq!(update["session"]["output_audio_format"], "g711_ulaw");
}

#[test]
fn decoder_uses_configured_output_format_for_media_type() {
    let mut config = RealtimeConfig::default();
    config
        .extensions
        .insert("output_audio_format".into(), json!("g711_alaw"));
    let mut dec = openai_realtime::RealtimeDecoder::new(&config);
    let events = dec
        .decode(br#"{"type":"response.audio.delta","delta":"AA=="}"#)
        .unwrap();
    match &events[0] {
        RealtimeEvent::Audio { media_type, .. } => {
            assert_eq!(media_type, "audio/g711_alaw")
        }
        other => panic!("expected audio event, got {other:?}"),
    }
}

#[test]
fn decode_rejects_non_json_and_preserves_unknown_events() {
    let mut dec = openai_realtime::RealtimeDecoder::new(&RealtimeConfig::default());
    assert!(matches!(
        dec.decode(b"not json"),
        Err(Error::UpstreamProtocol(_))
    ));
    let events = dec
        .decode(br#"{"type":"some.future.event","payload":1}"#)
        .unwrap();
    match &events[0] {
        RealtimeEvent::Native(v) => assert_eq!(v["payload"], 1),
        other => panic!("expected native event, got {other:?}"),
    }
}
