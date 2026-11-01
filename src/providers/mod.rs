//! Provider adapters: a `ServiceTarget` + authentication mechanics +
//! injected transport/effect ports bound into an `Adapter`.
//!
//! Profiles (endpoints, regions, accepted auth kinds, per-resource URL
//! templates) are declared as data in `crate::catalog`; the adapters
//! here are the protocol-family implementations a `Client` binds them
//! to through `Catalog::bind`.

pub mod anthropic;
pub mod bedrock;
pub mod cohere;
pub mod external;
pub mod gemini;
pub mod ollama;
pub mod openai;

pub use anthropic::AnthropicAdapter;
pub use bedrock::BedrockAdapter;
pub use cohere::CohereAdapter;
pub use external::{ClientFraming, ClientProtocol, ExternalAdapter, JsonLinesProtocol};
pub use gemini::GeminiAdapter;
pub use ollama::OllamaAdapter;
pub use openai::OpenAiAdapter;
