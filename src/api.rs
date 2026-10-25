//! Public SDK contracts: provider identity, model operations, results,
//! invocation lifecycles, request control and errors.

use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::transport::TransportError;

/// A pull-based event or byte stream. Backpressure flows to the caller;
/// adapters must not buffer whole responses or unbounded pending frames.
pub type DynStream<'a, T> = Pin<Box<dyn Stream<Item = Result<T, Error>> + Send + 'a>>;

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Canonical identity of a provider service operator (not the model author).
/// Open-ended by design: a provider id is a string, never a closed enum.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderId(String);

impl ProviderId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProviderId({:?})", self.0)
    }
}
impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl From<&str> for ProviderId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}
impl From<String> for ProviderId {
    fn from(s: String) -> Self {
        Self(s)
    }
}
impl AsRef<str> for ProviderId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Native wire protocol family a service target speaks.
///
/// `Other` keeps the set open without weakening the named families.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Protocol {
    OpenAiChatCompletions,
    OpenAiResponses,
    AnthropicMessages,
    GeminiGenerateContent,
    BedrockConverse,
    /// A provider-native protocol not yet named by this crate.
    Other(String),
}

/// One reachable provider service: endpoint, protocol and deployment detail.
/// Region/deployment and access variant belong here, not in `ProviderId`.
#[derive(Clone, Debug)]
pub struct ServiceTarget {
    pub provider: ProviderId,
    pub protocol: Protocol,
    pub base_url: url::Url,
    pub region: Option<String>,
    pub headers: http::HeaderMap,
}

impl ServiceTarget {
    pub fn new(provider: impl Into<ProviderId>, protocol: Protocol, base_url: url::Url) -> Self {
        Self {
            provider: provider.into(),
            protocol,
            base_url,
            region: None,
            headers: http::HeaderMap::new(),
        }
    }

    /// Join an API path onto the base URL, normalizing the trailing slash
    /// so `v1` + `chat/completions` yields `v1/chat/completions`.
    pub fn endpoint(&self, path: &str) -> url::Url {
        let mut base = self.base_url.clone();
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path()));
        }
        base.join(path).unwrap_or_else(|_| self.base_url.clone())
    }
}

/// A model addressed through a service target. `model` is the upstream
/// identifier; there is no closed vendor or model enum.
#[derive(Clone, Debug)]
pub struct ModelRef {
    pub target: ServiceTarget,
    pub model: String,
}

/// A model reported by live endpoint discovery. Reported presence is an
/// observed claim for that account/endpoint — never a verified-capability
/// or entitlement statement.
#[derive(Clone, Debug, Default)]
pub struct ModelInfo {
    pub id: String,
    pub owned_by: Option<String>,
    /// Provider-native model fields not represented above.
    pub extensions: Extensions,
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// Model operation family. Each family keeps its own request/result
/// contract; nothing is emulated through chat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OperationKind {
    Chat,
    Embed,
    Rerank,
    Classify,
    Moderate,
    Image,
    Audio,
    Video,
    Batch,
    Realtime,
}

/// Whether a failed operation may be repeated without replaying billed work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetrySafety {
    /// Safe: the provider never accepted the work (connect refused,
    /// local pre-send validation/conversion, explicit rejection, idempotent
    /// read).
    Safe,
    /// Upstream acceptance is uncertain or confirmed: an automatic retry
    /// could duplicate billed work. Only an explicit caller decision may
    /// repeat it.
    Unsafe,
}

// ---------------------------------------------------------------------------
// Content model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Role {
    System,
    User,
    Assistant,
    /// Provider-specific role kept verbatim.
    Other(String),
}

/// Media payload: remote URL, inline bytes, or a host-resolved resource.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum MediaSource {
    Url(String),
    Inline { media_type: String, data: bytes::Bytes },
    /// Opaque handle the host resolves through its effect port.
    Resource(String),
}

/// The kind of media a part carries — required to pick the correct
/// native encoding (image vs audio vs file differ on the wire).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaKind {
    Image,
    Audio,
    Video,
    File,
}

/// One ordered content part inside a message or output.
///
/// Order is preserved across protocols that support ordered blocks.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ContentPart {
    Text {
        text: String,
    },
    Media {
        kind: MediaKind,
        source: MediaSource,
        /// Optional provider detail (e.g. image detail level).
        detail: Option<String>,
    },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    /// Provider-native opaque reasoning state (signature, encrypted thought,
    /// interleaved thinking). Never flattened into visible text.
    Reasoning(ReasoningBlock),
}

/// A complete tool invocation requested by the model.
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A tool execution result returned to the model.
#[derive(Clone, Debug)]
pub struct ToolResult {
    pub call_id: String,
    pub content: Vec<ContentPart>,
    pub is_error: bool,
}

/// Opaque, origin-bound reasoning state. Replayable only to the protocol
/// family that issued it; never converted to user-visible text.
#[derive(Clone, Debug)]
pub struct ReasoningBlock {
    /// The protocol family that issued this state.
    pub protocol: Protocol,
    /// Provider-native payload (signature, encrypted blob, summary refs).
    pub opaque: serde_json::Value,
    /// Optional provider-supplied summary; never chain-of-thought.
    pub summary: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl Message {
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text { text: text.into() }],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool parameters.
    pub parameters: serde_json::Value,
    pub strict: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

/// Sampling parameters shared across protocols. Fields a target cannot
/// express surface as conversion losses, never silent drops.
#[derive(Clone, Debug, Default)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub stop: Vec<String>,
}

/// Provider-native request fields merged into the target protocol's body.
/// Keys the protocol cannot accept are rejected, never silently dropped.
pub type Extensions = serde_json::Map<String, serde_json::Value>;

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub tool_choice: Option<ToolChoice>,
    /// Whether the provider may fan the request out into parallel tool
    /// calls. Maps to Chat/Responses `parallel_tool_calls` and Anthropic
    /// `tool_choice.disable_parallel_tool_use`.
    pub parallel_tool_calls: Option<bool>,
    pub sampling: Sampling,
    pub extensions: Extensions,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum EmbedInput {
    Text(String),
    Media(MediaSource),
}

#[derive(Clone, Debug, Default)]
pub struct EmbedRequest {
    pub model: String,
    pub inputs: Vec<EmbedInput>,
    pub dimensions: Option<u32>,
    pub extensions: Extensions,
}

#[derive(Clone, Debug)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
    pub extensions: Extensions,
}

#[derive(Clone, Debug)]
pub struct ClassifyRequest {
    pub model: String,
    pub inputs: Vec<String>,
    /// Allowed labels when the provider requires a closed set.
    pub labels: Vec<String>,
    pub extensions: Extensions,
}

#[derive(Clone, Debug)]
pub struct ModerateRequest {
    pub model: String,
    pub inputs: Vec<String>,
    pub extensions: Extensions,
}

#[derive(Clone, Debug)]
pub struct ImageRequest {
    pub model: String,
    pub prompt: String,
    pub count: Option<u32>,
    /// Provider-specific size/aspect string kept verbatim.
    pub size: Option<String>,
    pub extensions: Extensions,
}

/// Audio operations share a family but keep distinct directions.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum AudioRequest {
    Synthesize {
        model: String,
        input: String,
        voice: Option<String>,
        format: Option<String>,
        extensions: Extensions,
    },
    Transcribe {
        model: String,
        audio: MediaSource,
        language: Option<String>,
        extensions: Extensions,
    },
}

#[derive(Clone, Debug)]
pub struct VideoRequest {
    pub model: String,
    pub prompt: String,
    /// Requested duration hint; providers may ignore or approximate it.
    pub duration_secs: Option<f32>,
    pub extensions: Extensions,
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Output {
    /// Ordered output parts (text, tool calls, opaque reasoning, media).
    pub parts: Vec<ContentPart>,
    pub stop: Option<StopReason>,
    pub usage: Option<Usage>,
    /// Provider-native response fields not represented above.
    pub extensions: Extensions,
}

#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Provider-native usage fields (cache hits, reasoning tokens, …).
    pub extensions: Extensions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StopReason {
    Complete,
    Length,
    ToolUse,
    Filtered,
    Paused,
    Other(String),
}

#[derive(Clone, Debug, Default)]
pub struct EmbedOutput {
    pub embeddings: Vec<Vec<f32>>,
    pub usage: Option<Usage>,
    pub extensions: Extensions,
}

#[derive(Clone, Debug)]
pub struct RerankHit {
    pub index: usize,
    pub score: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RerankOutput {
    pub results: Vec<RerankHit>,
    pub usage: Option<Usage>,
    pub extensions: Extensions,
}

#[derive(Clone, Debug)]
pub struct Classification {
    /// Index into the request's `inputs`.
    pub input: usize,
    pub label: String,
    pub confidence: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct ClassifyOutput {
    pub predictions: Vec<Classification>,
    pub usage: Option<Usage>,
    pub extensions: Extensions,
}

#[derive(Clone, Debug, Default)]
pub struct ModerationResult {
    pub flagged: bool,
    /// Category → flagged/score map, provider-native shape preserved.
    pub categories: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Default)]
pub struct ModerateOutput {
    pub results: Vec<ModerationResult>,
    pub usage: Option<Usage>,
    pub extensions: Extensions,
}

/// Generated or transformed media (image, audio, video).
#[derive(Clone, Debug, Default)]
pub struct MediaOutput {
    pub media: Vec<MediaSource>,
    /// Textual output some providers return alongside media.
    pub text: Vec<String>,
    pub usage: Option<Usage>,
    pub extensions: Extensions,
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Kind of an output part currently streaming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartKind {
    Text,
    ToolCall,
    Reasoning,
    Media,
    /// Provider-native part kind kept verbatim.
    Other,
}

/// Incremental events of a streaming operation. A complete buffered
/// response is never required; partial progress survives failures.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum StreamEvent {
    /// A new output part begins.
    PartStarted { index: u32, kind: PartKind },
    /// Text appended to an open part.
    TextDelta { index: u32, text: String },
    /// A tool call begins; carries the full call identity up front.
    /// Partial arguments follow as `ToolCallDelta`.
    ToolCallStarted { index: u32, id: String, name: String },
    /// Partial tool-call arguments appended to an open part.
    ToolCallDelta { index: u32, arguments: String },
    /// Reasoning/thinking text appended to an open reasoning part.
    ReasoningDelta { index: u32, text: String },
    /// Media bytes appended to an open part.
    MediaDelta { index: u32, data: bytes::Bytes, media_type: Option<String> },
    /// An open part is complete.
    PartFinished { index: u32 },
    /// A complete, provider-native object with no canonical form
    /// (opaque reasoning signature, server tool block, …).
    Native { part_index: Option<u32>, value: serde_json::Value },
    /// Cumulative or incremental usage reported mid- or post-stream.
    Usage(Usage),
    /// Terminal event with the final stop reason.
    Finished { stop: Option<StopReason> },
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// Credential-free, resumable handle for an accepted asynchronous job.
/// Carrying it across processes or restarts resubmits nothing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobHandle {
    pub provider: ProviderId,
    pub protocol: Protocol,
    /// Provider job identifier.
    pub id: String,
    /// Opaque provider resume token; never contains credentials.
    #[serde(default)]
    pub state: serde_json::Value,
}

/// Operation payload submitted as an asynchronous job.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum JobOperation {
    Chat(ChatRequest),
    Embed(EmbedRequest),
    Image(ImageRequest),
    Audio(AudioRequest),
    Video(VideoRequest),
    /// Operation-specific payload outside the named set.
    Custom {
        kind: String,
        payload: serde_json::Value,
    },
}

#[derive(Clone, Debug)]
pub struct RemoteFile {
    pub id: String,
    pub purpose: Option<String>,
    /// Downloadable location when the provider exposes one directly.
    pub url: Option<String>,
}

/// Provider-produced job output plus any associated remote resources.
#[derive(Clone, Debug, Default)]
pub struct JobResult {
    /// Operation-specific result payload (native shape).
    pub output: serde_json::Value,
    pub files: Vec<RemoteFile>,
}

/// Lifecycle state of an accepted job. Distinct from transport failure:
/// a failed fetch retries the same job; it never resubmits it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum JobState {
    Submitted,
    Running,
    Succeeded(JobResult),
    Failed { code: Option<String>, message: String },
    Cancelled,
}

// ---------------------------------------------------------------------------
// Realtime
// ---------------------------------------------------------------------------

/// Configuration for opening a bidirectional realtime session.
#[derive(Clone, Debug, Default)]
pub struct RealtimeConfig {
    pub model: String,
    /// Session modalities, provider-native naming preserved.
    pub modalities: Vec<String>,
    pub extensions: Extensions,
}

/// One event into an open realtime session.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RealtimeInput {
    Text(String),
    Audio { data: bytes::Bytes, media_type: String },
    ToolResult(ToolResult),
    /// Provider-native client event verbatim.
    Native(serde_json::Value),
    Commit,
    Cancel,
}

/// One event emitted by a realtime session.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RealtimeEvent {
    Text(String),
    Audio { data: bytes::Bytes, media_type: String },
    Transcript { text: String, final_: bool },
    ToolCall(ToolCall),
    Usage(Usage),
    /// Provider-native server event verbatim.
    Native(serde_json::Value),
    Closed { reason: Option<String> },
}

/// An open bidirectional session. Input and output are independent;
/// it is not an SSE response.
#[async_trait::async_trait]
pub trait RealtimeSession: Send {
    async fn send(&mut self, event: RealtimeInput) -> Result<(), Error>;
    /// Events from the server, independent of `send` ordering.
    fn events(&mut self) -> DynStream<'_, RealtimeEvent>;
    /// Graceful close; distinct from remote failure.
    async fn close(self: Box<Self>) -> Result<(), Error>;
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

/// Per-request control: caller cancellation plus tracing identity.
///
/// There are deliberately no deadlines, silence timeouts or attempt
/// limits: a viable request runs to completion; the caller cancels.
#[derive(Clone, Debug)]
pub struct RequestControl {
    pub cancel: tokio_util::sync::CancellationToken,
    pub request_id: Option<String>,
}

impl Default for RequestControl {
    fn default() -> Self {
        Self {
            cancel: tokio_util::sync::CancellationToken::new(),
            request_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// One semantic loss detected in a cross-protocol conversion.
#[derive(Clone, Debug)]
pub struct ConversionLoss {
    /// Request/response path where the semantic lives.
    pub path: String,
    /// What could not be represented.
    pub kind: LossKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossKind {
    /// The target protocol cannot express this semantic at all.
    Unsupported,
    /// Representable only with changed meaning.
    Lossy,
}

/// Conversion report attached to rejected transformations.
#[derive(Clone, Debug)]
pub struct ConversionReport {
    pub source: Protocol,
    pub target: Protocol,
    pub losses: Vec<ConversionLoss>,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Transport-layer failure before/during the byte stream.
    #[error("transport: {0}")]
    Transport(#[from] TransportError),

    /// Provider rejected or failed the request with its own semantics.
    #[error("provider error{status} ({code}): {message}", status = status.map(|s| format!(" {s}")).unwrap_or_default(), code = code.as_deref().unwrap_or("-"))]
    Provider {
        status: Option<u16>,
        code: Option<String>,
        message: String,
        /// Whether an automatic retry could repeat billed work.
        retry: RetrySafety,
        /// Provider-native error payload preserved verbatim.
        raw: Option<serde_json::Value>,
    },

    /// The provider profile does not implement this operation/capability.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A cross-protocol conversion could not preserve semantics.
    #[error("conversion {report:?}")]
    Conversion { report: Box<ConversionReport> },

    /// Credential generation or resolution failed.
    #[error("credentials: {0}")]
    Credential(String),

    /// The upstream payload violated the protocol contract. Violations are
    /// observed after the request was dispatched, so upstream work may
    /// already have been accepted.
    #[error("protocol violation: {0}")]
    Protocol(String),

    /// Caller cancellation via `RequestControl::cancel`. Cancellation can
    /// race a request that has already reached the provider, so the
    /// upstream outcome is unknown.
    #[error("cancelled")]
    Cancelled,
}

impl Error {
    /// Whether an automatic retry is safe for this failure.
    ///
    /// Only failures that provably occurred before dispatch are `Safe`:
    /// connect refusal, local conversion/validation, unsupported
    /// operations and credential generation. Cancellation, protocol
    /// violations and mid-flight transport failures may follow upstream
    /// acceptance, so repeating them could duplicate billed work.
    pub fn retry_safety(&self) -> RetrySafety {
        match self {
            Error::Transport(e) => e.retry_safety(),
            Error::Provider { retry, .. } => *retry,
            Error::Unsupported(_) => RetrySafety::Safe,
            // Local, pre-send failures.
            Error::Conversion { .. } | Error::Credential(_) => RetrySafety::Safe,
            // Post-dispatch failures: upstream acceptance cannot be ruled out.
            Error::Cancelled | Error::Protocol(_) => RetrySafety::Unsafe,
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// A bound provider adapter: one `ServiceTarget` plus resolved
/// authentication mechanics. Every model operation family is a first-class
/// method; unimplemented families return `Error::Unsupported` explicitly
/// rather than being emulated through another family.
#[async_trait::async_trait]
pub trait Adapter: Send + Sync {
    fn target(&self) -> &ServiceTarget;

    fn unsupported(&self, op: &'static str) -> Error {
        Error::Unsupported(format!(
            "{} does not implement {op}",
            self.target().provider
        ))
    }

    /// Live model discovery against the bound endpoint. A missing or
    /// failed listing means "unknown", never "no models exist"; an
    /// adapter without a discovery surface reports `Unsupported`.
    async fn models(&self, _ctrl: &RequestControl) -> Result<Vec<ModelInfo>, Error> {
        Err(self.unsupported("models"))
    }

    async fn chat(&self, _req: ChatRequest, _ctrl: &RequestControl) -> Result<Output, Error> {
        Err(self.unsupported("chat"))
    }

    /// The returned stream owns its data; it does not borrow `ctrl`,
    /// so it may outlive the call and move across tasks.
    async fn chat_stream(
        &self,
        _req: ChatRequest,
        _ctrl: &RequestControl,
    ) -> Result<DynStream<'static, StreamEvent>, Error> {
        Err(self.unsupported("chat_stream"))
    }

    async fn embed(&self, _req: EmbedRequest, _ctrl: &RequestControl) -> Result<EmbedOutput, Error> {
        Err(self.unsupported("embed"))
    }

    async fn rerank(&self, _req: RerankRequest, _ctrl: &RequestControl) -> Result<RerankOutput, Error> {
        Err(self.unsupported("rerank"))
    }

    async fn classify(
        &self,
        _req: ClassifyRequest,
        _ctrl: &RequestControl,
    ) -> Result<ClassifyOutput, Error> {
        Err(self.unsupported("classify"))
    }

    async fn moderate(
        &self,
        _req: ModerateRequest,
        _ctrl: &RequestControl,
    ) -> Result<ModerateOutput, Error> {
        Err(self.unsupported("moderate"))
    }

    async fn image(&self, _req: ImageRequest, _ctrl: &RequestControl) -> Result<MediaOutput, Error> {
        Err(self.unsupported("image"))
    }

    async fn audio(&self, _req: AudioRequest, _ctrl: &RequestControl) -> Result<MediaOutput, Error> {
        Err(self.unsupported("audio"))
    }

    async fn video(&self, _req: VideoRequest, _ctrl: &RequestControl) -> Result<MediaOutput, Error> {
        Err(self.unsupported("video"))
    }

    /// Submit an asynchronous job; returns a resumable, credential-free handle.
    async fn submit_job(&self, _op: JobOperation, _ctrl: &RequestControl) -> Result<JobHandle, Error> {
        Err(self.unsupported("submit_job"))
    }

    /// Poll an accepted job. A transport failure retries the same job;
    /// it must never resubmit work.
    async fn job_state(&self, _handle: &JobHandle, _ctrl: &RequestControl) -> Result<JobState, Error> {
        Err(self.unsupported("job_state"))
    }

    /// Cancel an accepted job.
    async fn cancel_job(&self, _handle: &JobHandle, _ctrl: &RequestControl) -> Result<(), Error> {
        Err(self.unsupported("cancel_job"))
    }

    /// Open a bidirectional realtime session.
    async fn open_realtime(
        &self,
        _config: RealtimeConfig,
        _ctrl: &RequestControl,
    ) -> Result<Box<dyn RealtimeSession>, Error> {
        Err(self.unsupported("open_realtime"))
    }
}
