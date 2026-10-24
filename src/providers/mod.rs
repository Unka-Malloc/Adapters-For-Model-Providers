//! Provider adapters: a `ServiceTarget` + authentication mechanics +
//! injected transport/effect ports bound into an `Adapter`.
//!
//! Profiles (endpoint, region, auth kind, parameter differences) are
//! declared per provider in later milestones; adapters here are the
//! protocol-family implementations they bind.

pub mod anthropic;
pub mod gemini;
pub mod openai;

pub use anthropic::AnthropicAdapter;
pub use gemini::GeminiAdapter;
pub use openai::OpenAiAdapter;
