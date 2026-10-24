//! Original fixtures for the external-client lane: a scripted `Process`
//! stub behind a stub `EffectHost`, a `JsonLinesProtocol` client config,
//! request→stdin bytes, event decode (incl. mid-line reassembly), exit
//! codes, lifecycle termination and cancellation. Payloads are
//! hand-authored from the IPC contract, not copied from real CLIs.
//!
//! The provider module is included through `#[path]` so this file works
//! before `src/providers/mod.rs` registers it: `crate::` paths inside the
//! included file resolve through the shim modules below.

// Shared fixtures: only `MockTransport` is used (the external lane makes
// no HTTP calls; the transport slot is parity); other helpers stay
// available to sibling suites.
#[allow(dead_code)]
mod common;

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use a4mp::*;
use common::{MockTransport, collect};
use futures_core::Stream;
use serde_json::{Value, json};

use a4mp::providers::{ClientFraming, ClientProtocol, ExternalAdapter, JsonLinesProtocol};

// ---------------------------------------------------------------------------
// ScriptProcess: scripted output chunks, recorded stdin, scripted exit
// ---------------------------------------------------------------------------

/// Shared observation handle: what the process saw and how it ended.
#[derive(Default)]
struct ProcHandle {
    stdin: Mutex<Vec<bytes::Bytes>>,
    terminates: Mutex<u32>,
    waits: Mutex<u32>,
}

/// A stub `Process`: scripted output chunks (a chunk may split a JSON
/// line mid-line), optional stall (output pends forever once drained),
/// a scripted exit code, recorded stdin frames and lifecycle counts.
struct ScriptProcess {
    handle: Arc<ProcHandle>,
    chunks: VecDeque<Result<bytes::Bytes, Error>>,
    /// When drained, yield `Pending` forever instead of `None`.
    stall: bool,
    exit: i32,
}

impl ScriptProcess {
    fn new(chunks: Vec<&str>, exit: i32) -> (Self, Arc<ProcHandle>) {
        let h = Arc::new(ProcHandle::default());
        (
            Self {
                handle: h.clone(),
                chunks: chunks
                    .into_iter()
                    .map(|c| Ok(bytes::Bytes::from(c.to_owned())))
                    .collect(),
                stall: false,
                exit,
            },
            h,
        )
    }

    fn stalling(mut self) -> Self {
        self.stall = true;
        self
    }
}

/// The resumable per-poll output view: each `output()` call returns a
/// stream that continues where the previous one left off, as the
/// session's poll model requires of host implementations.
struct ProcOut<'a> {
    p: &'a mut ScriptProcess,
}

impl Stream for ProcOut<'_> {
    type Item = Result<bytes::Bytes, Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.p.chunks.pop_front() {
            Some(item) => Poll::Ready(Some(item)),
            None if self.p.stall => Poll::Pending,
            None => Poll::Ready(None),
        }
    }
}

impl Process for ScriptProcess {
    fn output(&mut self) -> DynStream<'_, bytes::Bytes> {
        Box::pin(ProcOut { p: self })
    }

    fn send_input(&mut self, data: bytes::Bytes) -> Result<(), Error> {
        self.handle.stdin.lock().unwrap().push(data);
        Ok(())
    }

    fn terminate(&mut self) -> Result<(), Error> {
        *self.handle.terminates.lock().unwrap() += 1;
        Ok(())
    }

    fn wait(&mut self) -> Result<i32, Error> {
        *self.handle.waits.lock().unwrap() += 1;
        Ok(self.exit)
    }
}

/// Stub effect host: records every `ProcessSpec` and hands out the one
/// scripted process. A second spawn without a script reports refusal.
#[derive(Default)]
struct StubEffects {
    next: Mutex<Option<ScriptProcess>>,
    specs: Mutex<Vec<ProcessSpec>>,
}

impl StubEffects {
    fn with(proc: ScriptProcess) -> Self {
        Self {
            next: Mutex::new(Some(proc)),
            specs: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait::async_trait]
impl EffectHost for StubEffects {
    async fn spawn(
        &self,
        spec: &ProcessSpec,
        _ctrl: &RequestControl,
    ) -> Result<Box<dyn Process>, Error> {
        self.specs.lock().unwrap().push(spec.clone());
        self.next
            .lock()
            .unwrap()
            .take()
            .map(|p| Box::new(p) as Box<dyn Process>)
            .ok_or_else(|| Error::Unsupported("no scripted process".into()))
    }
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

fn jsonl_protocol() -> JsonLinesProtocol {
    JsonLinesProtocol::new("acp-test", "cursor-agent >= 1.2", "cursor-agent")
        .with_args(vec![
            "-p".into(),
            "--output-format".into(),
            "stream-json".into(),
            "--model".into(),
            "{model}".into(),
        ])
        .with_method("session/prompt")
}

fn bind(
    proc: ScriptProcess,
    protocol: Arc<dyn ClientProtocol>,
    protocol_name: &str,
) -> (ExternalAdapter, Arc<StubEffects>) {
    let effects = Arc::new(StubEffects::with(proc));
    let target = ServiceTarget::new(
        "cursor",
        Protocol::Other(protocol_name.into()),
        url::Url::parse("https://agent.invalid/").unwrap(),
    );
    let adapter = ExternalAdapter::new(
        target,
        AuthKind::External {
            client: "cursor-agent".into(),
        },
        Arc::new(MockTransport::new()),
        Arc::new(NoCredentials),
        effects.clone(),
        protocol,
    )
    .unwrap();
    (adapter, effects)
}

fn jsonl_adapter(proc: ScriptProcess) -> (ExternalAdapter, Arc<StubEffects>) {
    bind(proc, Arc::new(jsonl_protocol()), "acp-test")
}

fn stdin_json(h: &ProcHandle) -> Value {
    let frames = h.stdin.lock().unwrap();
    assert_eq!(frames.len(), 1, "expected exactly one stdin frame");
    let frame = frames[0].clone();
    assert!(frame.ends_with(b"\n"), "frame must be newline-terminated");
    serde_json::from_slice(&frame).unwrap()
}

// ---------------------------------------------------------------------------
// Request → stdin bytes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_writes_one_jsonrpc_request_line() {
    let (proc, h) = ScriptProcess::new(
        vec!["{\"id\":1,\"result\":{\"stopReason\":\"end_turn\",\"usage\":{\"inputTokens\":5,\"outputTokens\":2}}}\n"],
        0,
    );
    let (a, fx) = jsonl_adapter(proc);

    let mut req = ChatRequest {
        model: "gpt-5".into(),
        ..Default::default()
    };
    req.messages
        .push(Message::text(Role::User, "say hi"));
    req.tools.push(ToolDef {
        name: "lookup".into(),
        description: "find things".into(),
        parameters: json!({"type": "object"}),
        strict: true,
    });
    req.sampling.temperature = Some(0.5);
    req.sampling.max_output_tokens = Some(64);
    req.extensions.insert("cwd".into(), json!("/repo"));

    let out = a.chat(req, &RequestControl::default()).await.unwrap();

    // Launch spec: requirement string and argv template substitution.
    let spec = &fx.specs.lock().unwrap()[0];
    assert_eq!(spec.requirement, "cursor-agent >= 1.2");
    assert_eq!(spec.program, "cursor-agent");
    assert_eq!(
        spec.args,
        vec!["-p", "--output-format", "stream-json", "--model", "gpt-5"]
    );

    // The single stdin frame is one JSON-RPC request line.
    let body = stdin_json(&h);
    assert_eq!(body["jsonrpc"], "2.0");
    assert_eq!(body["id"], 1);
    assert_eq!(body["method"], "session/prompt");
    assert_eq!(body["params"]["model"], "gpt-5");
    assert_eq!(
        body["params"]["messages"],
        json!([{"role": "user", "content": "say hi"}])
    );
    assert_eq!(body["params"]["tools"][0]["name"], "lookup");
    assert_eq!(body["params"]["tools"][0]["strict"], true);
    assert_eq!(body["params"]["temperature"], 0.5);
    assert_eq!(body["params"]["max_output_tokens"], 64);
    assert_eq!(body["params"]["cwd"], "/repo");

    assert!(matches!(out.stop, Some(StopReason::Complete)));
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(5));
    assert_eq!(usage.output_tokens, Some(2));
    // Exit 0 ran finish → wait observed; drop terminated the process.
    assert!(*h.waits.lock().unwrap() >= 1);
    assert!(*h.terminates.lock().unwrap() >= 1);
}

// ---------------------------------------------------------------------------
// JSONL event decode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_decodes_delta_usage_and_finished() {
    let (proc, _h) = ScriptProcess::new(
        vec![
            "{\"type\":\"delta\",\"text\":\"Hel\"}\n",
            "{\"type\":\"delta\",\"text\":\"lo\"}\n",
            "{\"type\":\"usage\",\"input_tokens\":4,\"output_tokens\":1}\n",
            "{\"type\":\"result\",\"stop_reason\":\"end_turn\"}\n",
        ],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let stream = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    let events = collect(stream).await.unwrap();

    let mut text = String::new();
    let mut started = Vec::new();
    let mut finished_parts = Vec::new();
    let mut usage_in = None;
    let mut stop = None;
    for e in &events {
        match e {
            StreamEvent::PartStarted { index, kind } => started.push((*index, *kind)),
            StreamEvent::TextDelta { text: t, .. } => text.push_str(t),
            StreamEvent::PartFinished { index } => finished_parts.push(*index),
            StreamEvent::Usage(u) => usage_in = u.input_tokens,
            StreamEvent::Finished { stop: s } => stop = s.clone(),
            _ => {}
        }
    }
    // The foundation synthesizes PartStarted/PartFinished around bare
    // deltas: the client protocol only mapped frames.
    assert_eq!(started, [(0, PartKind::Text)]);
    assert_eq!(finished_parts, [0]);
    assert_eq!(text, "Hello");
    assert_eq!(usage_in, Some(4));
    assert!(matches!(stop, Some(StopReason::Complete)));
}

#[tokio::test]
async fn jsonl_line_split_mid_line_is_reassembled() {
    // One JSON line split inside a string token across two chunks.
    let (proc, _h) = ScriptProcess::new(
        vec![
            "{\"type\":\"delta\",\"text\":\"sp",
            "lit\"}\n{\"type\":\"done\"}\n",
        ],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let events = collect(
        a.chat_stream(ChatRequest::default(), &RequestControl::default())
            .await
            .unwrap(),
    )
    .await
    .unwrap();
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "split");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::Finished { .. }))
    );
}

#[tokio::test]
async fn acp_notification_envelope_decodes_updates() {
    // ACP-shaped notifications: params.update carries the event object.
    let (proc, _h) = ScriptProcess::new(
        vec![
            "{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"update\":{\"sessionUpdate\":\"agent_thought_chunk\",\"content\":{\"type\":\"text\",\"text\":\"thinking\"}}}}\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"update\":{\"sessionUpdate\":\"agent_message_chunk\",\"content\":{\"type\":\"text\",\"text\":\"answer\"}}}}\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"update\":{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"tc1\",\"title\":\"search\",\"status\":\"in_progress\"}}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"stopReason\":\"end_turn\",\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3,\"total_tokens\":12}}}\n",
        ],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let events = collect(
        a.chat_stream(ChatRequest::default(), &RequestControl::default())
            .await
            .unwrap(),
    )
    .await
    .unwrap();

    let reasoning: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ReasoningDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "thinking");
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "answer");
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::ToolCallStarted { id, name, .. } if id == "tc1" && name == "search"
    )));
    let usage = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::Usage(u) => Some(u),
            _ => None,
        })
        .last()
        .unwrap();
    assert_eq!(usage.input_tokens, Some(9));
    assert_eq!(usage.output_tokens, Some(3));
    // Unclaimed usage fields stay verbatim.
    assert_eq!(usage.extensions["total_tokens"], 12);
}

#[tokio::test]
async fn non_json_diagnostics_are_tolerated_and_captured() {
    let (proc, _h) = ScriptProcess::new(
        vec![
            "progress: warming up\n{\"type\":\"delta\",\"text\":\"ok\"}\n",
        ],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    match &out.parts[0] {
        ContentPart::Text { text } => assert_eq!(text, "ok"),
        other => panic!("expected text part, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Exit codes and errors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn nonzero_exit_is_provider_error_with_output_tail() {
    let (proc, _h) = ScriptProcess::new(
        vec![
            "{\"type\":\"delta\",\"text\":\"partial\"}\n",
            "fatal: quota exhausted\n",
        ],
        3,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let err = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap_err();
    match err {
        Error::Provider {
            code, message, retry, ..
        } => {
            assert_eq!(code.as_deref(), Some("exit:3"));
            assert!(message.contains("quota exhausted"), "{message}");
            assert!(matches!(retry, RetrySafety::Unsafe));
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn jsonrpc_error_response_maps_to_provider_error() {
    let (proc, _h) = ScriptProcess::new(
        vec!["{\"id\":1,\"error\":{\"code\":-32601,\"message\":\"method not found\"}}\n"],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let err = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap_err();
    match err {
        Error::Provider {
            code,
            message,
            retry,
            ..
        } => {
            assert_eq!(code.as_deref(), Some("-32601"));
            assert!(message.contains("method not found"), "{message}");
            // A JSON-RPC reserved code means rejected before work.
            assert!(matches!(retry, RetrySafety::Safe));
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_error_events_terminate_process() {
    let (proc, h) = ScriptProcess::new(
        vec![
            "{\"type\":\"delta\",\"text\":\"half\"}\n",
            "{\"type\":\"error\",\"message\":\"model blew up\"}\n",
            "{\"type\":\"delta\",\"text\":\"never\"}\n",
        ],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let events = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    // Already-decoded events survive; the error then ends the stream.
    match collect(events).await {
        Err(Error::Provider { message, .. }) => assert_eq!(message, "model blew up"),
        other => panic!("expected provider error, got {other:?}"),
    }
    assert!(*h.terminates.lock().unwrap() >= 1);
}

// ---------------------------------------------------------------------------
// Binding gates
// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_external_auth_is_rejected() {
    let (proc, _h) = ScriptProcess::new(vec![], 0);
    let target = ServiceTarget::new(
        "cursor",
        Protocol::Other("acp-test".into()),
        url::Url::parse("https://agent.invalid/").unwrap(),
    );
    match ExternalAdapter::new(
        target,
        AuthKind::Bearer,
        Arc::new(MockTransport::new()),
        Arc::new(NoCredentials),
        Arc::new(StubEffects::with(proc)),
        Arc::new(jsonl_protocol()),
    ) {
        Err(Error::Unsupported(_)) => {}
        Err(other) => panic!("expected unsupported, got {other:?}"),
        Ok(_) => panic!("AuthKind::Bearer must not bind an external lane"),
    }
}

#[tokio::test]
async fn protocol_gate_requires_matching_other_name() {
    let (proc, _h) = ScriptProcess::new(vec![], 0);
    // Wrong family name.
    let target = ServiceTarget::new(
        "cursor",
        Protocol::Other("not-acp".into()),
        url::Url::parse("https://agent.invalid/").unwrap(),
    );
    match ExternalAdapter::new(
        target,
        AuthKind::External {
            client: "cursor-agent".into(),
        },
        Arc::new(MockTransport::new()),
        Arc::new(NoCredentials),
        Arc::new(StubEffects::with(proc)),
        Arc::new(jsonl_protocol()),
    ) {
        Err(Error::Unsupported(m)) => assert!(m.contains("acp-test"), "{m}"),
        Err(other) => panic!("expected unsupported, got {other:?}"),
        Ok(_) => panic!("a mismatched protocol family must not bind"),
    }

    // A named non-Other protocol is rejected too.
    let (proc2, _h2) = ScriptProcess::new(vec![], 0);
    let target = ServiceTarget::new(
        "cursor",
        Protocol::OpenAiChatCompletions,
        url::Url::parse("https://agent.invalid/").unwrap(),
    );
    match ExternalAdapter::new(
        target,
        AuthKind::External {
            client: "cursor-agent".into(),
        },
        Arc::new(MockTransport::new()),
        Arc::new(NoCredentials),
        Arc::new(StubEffects::with(proc2)),
        Arc::new(jsonl_protocol()),
    ) {
        Err(Error::Unsupported(_)) => {}
        Err(other) => panic!("expected unsupported, got {other:?}"),
        Ok(_) => panic!("a named non-Other protocol must not bind"),
    }
}

#[tokio::test]
async fn non_none_credential_is_rejected_before_spawn() {
    let (proc, _h) = ScriptProcess::new(vec![], 0);
    let effects = Arc::new(StubEffects::with(proc));
    let target = ServiceTarget::new(
        "cursor",
        Protocol::Other("acp-test".into()),
        url::Url::parse("https://agent.invalid/").unwrap(),
    );
    let a = ExternalAdapter::new(
        target,
        AuthKind::External {
            client: "cursor-agent".into(),
        },
        Arc::new(MockTransport::new()),
        Arc::new(StaticCredential(Credential::ApiKey(Secret::new("leak-me-not")))),
        effects.clone(),
        Arc::new(jsonl_protocol()),
    )
    .unwrap();
    let err = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Credential(_)));
    // Refused before spawn: the client process never ran with foreign
    // material.
    assert!(effects.specs.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Lifecycle: cancellation, end-of-stream, drop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_during_stall_terminates_the_process() {
    let (proc, h) = ScriptProcess::new(vec!["{\"type\":\"delta\",\"text\":\"a\"}\n"], 0);
    let (a, _fx) = jsonl_adapter(proc.stalling());
    let ctrl = RequestControl::default();
    let cancel = ctrl.cancel.clone();
    let stream = a
        .chat_stream(ChatRequest::default(), &ctrl)
        .await
        .unwrap();
    cancel.cancel();
    match collect(stream).await {
        Err(Error::Cancelled) => {}
        other => panic!("expected cancellation, got {other:?}"),
    }
    assert!(*h.terminates.lock().unwrap() >= 1);
    assert!(*h.waits.lock().unwrap() >= 1);
}

#[tokio::test]
async fn dropping_the_stream_terminates_the_process() {
    let (proc, h) = ScriptProcess::new(
        vec![
            "{\"type\":\"delta\",\"text\":\"x\"}\n",
            "{\"type\":\"delta\",\"text\":\"y\"}\n",
        ],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc.stalling());
    let mut stream = a
        .chat_stream(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();
    // Pull one event, then drop with output still pending.
    let first = std::future::poll_fn(|cx| stream.as_mut().poll_next(cx)).await;
    assert!(matches!(first, Some(Ok(StreamEvent::PartStarted { .. }))));
    drop(stream);
    assert!(*h.terminates.lock().unwrap() >= 1);
    assert!(*h.waits.lock().unwrap() >= 1);
}

#[tokio::test]
async fn end_of_stream_waits_and_closes_parts() {
    let (proc, h) = ScriptProcess::new(
        vec!["{\"type\":\"delta\",\"text\":\"x\"}\n"],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let events = collect(
        a.chat_stream(ChatRequest::default(), &RequestControl::default())
            .await
            .unwrap(),
    )
    .await
    .unwrap();
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::PartFinished { index: 0 }
    )));
    assert!(*h.waits.lock().unwrap() >= 1);
    assert!(*h.terminates.lock().unwrap() >= 1);
}

// ---------------------------------------------------------------------------
// Output assembly
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_assembles_output_from_the_event_stream() {
    let (proc, _h) = ScriptProcess::new(
        vec![
            "{\"type\":\"thought\",\"text\":\"plan\"}\n",
            "{\"type\":\"delta\",\"text\":\"Hello \"}\n",
            "{\"type\":\"delta\",\"text\":\"world\"}\n",
            "{\"type\":\"tool_call\",\"id\":\"t9\",\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}\n",
            "{\"id\":1,\"result\":{\"stop_reason\":\"tool_use\",\"usage\":{\"input_tokens\":7,\"output_tokens\":5}}}\n",
        ],
        0,
    );
    let (a, _fx) = jsonl_adapter(proc);
    let out = a
        .chat(ChatRequest::default(), &RequestControl::default())
        .await
        .unwrap();

    assert_eq!(out.parts.len(), 3);
    match &out.parts[0] {
        ContentPart::Reasoning(rb) => {
            assert_eq!(rb.protocol, Protocol::Other("acp-test".into()));
            assert_eq!(rb.summary.as_deref(), Some("plan"));
        }
        other => panic!("expected reasoning part, got {other:?}"),
    }
    match &out.parts[1] {
        ContentPart::Text { text } => assert_eq!(text, "Hello world"),
        other => panic!("expected text part, got {other:?}"),
    }
    match &out.parts[2] {
        ContentPart::ToolCall(tc) => {
            assert_eq!(tc.id, "t9");
            assert_eq!(tc.name, "lookup");
            assert_eq!(tc.arguments, json!({"q": "x"}));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    assert!(matches!(out.stop, Some(StopReason::ToolUse)));
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, Some(5));
}

// ---------------------------------------------------------------------------
// ClientFraming::Whole — the argv/stdin-once client shape
// ---------------------------------------------------------------------------

/// A one-shot test client: takes the request via argv (`encode` is
/// empty), prints a single JSON object, exits. Covers the non-line
/// framing arm of the foundation and proves `ClientProtocol` is the
/// extension point (not a JSONL-only hook).
struct WholeShot;

impl ClientProtocol for WholeShot {
    fn name(&self) -> &str {
        "whole-shot"
    }

    fn spec(&self, model: &str) -> ProcessSpec {
        ProcessSpec {
            requirement: "oneshot-cli >= 0.1".into(),
            program: "oneshot".into(),
            args: vec!["--model".into(), model.into()],
            env: vec![],
            cwd: None,
        }
    }

    fn encode(&self, _req: &ChatRequest) -> Result<Vec<bytes::Bytes>, Error> {
        Ok(vec![])
    }

    fn framing(&self) -> ClientFraming {
        ClientFraming::Whole
    }

    fn decode(&self, chunk: &[u8], out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        let v: Value = serde_json::from_slice(chunk)
            .map_err(|e| Error::Protocol(format!("oneshot output: {e}")))?;
        if let Some(t) = v["text"].as_str() {
            out.push(StreamEvent::TextDelta {
                index: 0,
                text: t.to_owned(),
            });
        }
        out.push(StreamEvent::Finished {
            stop: Some(StopReason::Complete),
        });
        Ok(())
    }

    fn finish(&self, exit_code: i32, _out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        if exit_code == 0 {
            Ok(())
        } else {
            Err(Error::Provider {
                status: None,
                code: Some(format!("exit:{exit_code}")),
                message: "oneshot failed".into(),
                retry: RetrySafety::Unsafe,
                raw: None,
            })
        }
    }
}

#[tokio::test]
async fn whole_framing_delivers_one_frame_at_eof() {
    let (proc, h) = ScriptProcess::new(
        // Output split across chunks; Whole buffers it all.
        vec!["{\"text\":\"one", " shot\"}"],
        0,
    );
    let (a, fx) = bind(proc, Arc::new(WholeShot), "whole-shot");
    let out = a
        .chat(
            ChatRequest {
                model: "m9".into(),
                ..Default::default()
            },
            &RequestControl::default(),
        )
        .await
        .unwrap();
    match &out.parts[0] {
        ContentPart::Text { text } => assert_eq!(text, "one shot"),
        other => panic!("expected text, got {other:?}"),
    }
    // argv carried the model; no stdin frames were written.
    let spec = &fx.specs.lock().unwrap()[0];
    assert_eq!(spec.args, vec!["--model", "m9"]);
    assert!(h.stdin.lock().unwrap().is_empty());
    assert!(matches!(out.stop, Some(StopReason::Complete)));
}


