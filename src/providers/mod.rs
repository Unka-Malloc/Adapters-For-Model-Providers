//! Provider adapters: a `ServiceTarget` + authentication mechanics +
//! injected transport/effect ports bound into an `Adapter`.
//!
//! Profiles (endpoint, region, auth kind, parameter differences) are
//! declared per provider in later milestones; adapters here are the
//! protocol-family implementations they bind.

pub mod anthropic;
pub mod bedrock;
pub mod cohere;
pub mod gemini;
pub mod ollama;
pub mod openai;

pub use anthropic::AnthropicAdapter;
pub use bedrock::BedrockAdapter;
pub use cohere::CohereAdapter;
pub use gemini::GeminiAdapter;
pub use ollama::OllamaAdapter;
pub use openai::OpenAiAdapter;
