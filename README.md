# A4MP — Adapters For Model Providers

One Rust foundation for talking to model providers — all of them, through
every access mode they actually offer, without re-deriving each provider's
wire quirks inside every application that wants to use them.

Provider compatibility is a one-time cost that keeps being paid again and
again: each application reverse-engineers endpoints, auth flows, stream
framing, tool-call shapes and error semantics on its own. A4MP collects
that work into a single, independently versioned library. Applications
depend on a released version; routine application changes stop being a
risk to provider integrations.

```toml
[dependencies]
a4mp = { version = "0.3", features = ["reqwest-transport"] }
```

## What it is

- **A native Rust SDK.** Canonical request/output types and a first-class
  `Adapter` trait per operation family — chat, streaming chat, model
  discovery, embeddings, rerank, classification, moderation, images,
  audio, video, asynchronous jobs and realtime sessions.
- **An explicit-unsupported policy.** When a provider has no surface for
  an operation, the adapter returns `Error::Unsupported`. Nothing is
  silently emulated through chat, and representable-but-lossy mappings
  are surfaced through `ConversionReport` rather than swallowed.
- **Every access mode.** API keys, bearer tokens, OAuth (PKCE
  authorization-code and device flows), cloud identity (AWS SigV4 today),
  and external client/plugin-backed subscriptions that run through
  host-spawned processes — all behind the same canonical API.
- **A provider catalog.** ~240 declarative profiles with official
  endpoints, aliases, auth modes and provenance — including regional
  variants and per-resource templated endpoints (Azure resource names,
  Cloudflare account IDs, OCI regions). Provider identity is open-ended;
  any `Protocol`-registered family and any `bind_target` endpoint works.
- **Optional inbound wire conversion.** Gateway consumers can accept
  client requests already shaped like OpenAI Chat Completions, OpenAI
  Responses or Anthropic Messages, run them through *any* backend
  adapter, and answer in the dialect the client spoke — via
  `WireBridge`.

## What it is not

- Not a gateway server. It listens on no port, manages no local service,
  and picks no provider for you.
- Not a credential vault. `CredentialResolver` is a host port; the
  library never stores, logs, or silently reads another application's
  login state.
- Not a port of any existing project. The implementation is written
  against provider documentation and protocol behavior; reference
  projects informed the coverage map only.

## Architecture

```text
your application ── canonical API ──> Adapter (per provider protocol)
                                        │  codecs: openai_chat, openai_responses,
                                        │  anthropic_messages, gemini_generate,
                                        │  bedrock_converse, cohere, ollama,
                                        │  openai_data, openai_media, openai_realtime
                                        │
                                        ▼
                            host ports: Transport (buffered + streaming),
                            DuplexChannel (realtime), CredentialResolver,
                            EffectHost (resources, processes, OAuth, timers)
```

Everything side-effecting is a host port. The default `reqwest-transport`
feature provides HTTP; `realtime-ws` adds a tokio-tungstenite duplex
channel. Swap in your own to control pooling, proxies, or runtimes.

## Quickstart

```rust,no_run
use std::sync::Arc;
use a4mp::{api::*, auth::*, catalog::*, effects::EffectHost, transport::ReqwestTransport};

# async fn example() -> Result<(), a4mp::api::Error> {
# let resolver = Arc::new(NoCredentials);
# struct HostEffects;
# impl EffectHost for HostEffects {}
# let effects = Arc::new(HostEffects);
let client = Client::new(
    Catalog::with_builtin_adapters(),
    Arc::new(ReqwestTransport::new()),
    resolver,          // your CredentialResolver impl
)
.with_effects(effects); // your EffectHost impl

// Any catalog provider, alias, or explicit target.
let adapter = client.bind_provider(
    "anthropic",                      // or "deepseek", "kimi", "kilo", …
    None,                             // profile's preferred protocol
    None,                             // region
    AuthKind::ApiKey,
)?;

let output = adapter.chat(
    ChatRequest {
        model: "claude-sonnet-4-5".into(),
        messages: vec![Message::text(Role::User, "Hello")],
        ..Default::default()
    },
    &RequestControl::default(),
).await?;
# Ok(())
# }
```

Streaming, media, jobs and realtime follow the same pattern — see
`src/api.rs` for the canonical types and each adapter's documented
coverage.

## Coverage status

Provider profiles are declarative data with documented endpoints; binding
one asserts a supported access surface, not live availability. Operation
coverage varies by provider surface — every adapter documents what it
implements and returns `Error::Unsupported` for the rest. Per-provider
details live in `src/catalog/profiles/`; per-adapter decisions are in the
module docs of each provider file.

The test suite verifies codecs, framing, signing and adapter behavior
against deterministic fixtures — it does not spend money calling live
services. Live verification is a separate, explicitly authorized step.

## License

Apache-2.0. See [LICENSE](LICENSE).
