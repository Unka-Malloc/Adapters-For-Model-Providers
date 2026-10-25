# Integration modes

One way to consume this crate: **SDK — embed it**. `a4mp` is a codec and
transport library — canonical request/response types, per-provider wire
decoders/encoders, transports and credential resolution. It has no daemon
surface, so there is no sidecar mode.

## SDK — embed the crate

```toml
[dependencies]
a4mp = "0.1"               # default-features = false keeps it runtime-free
```

```rust
// Canonical middle: decode the client's dialect, build the upstream's.
let request = a4mp::wire::decode_chat_request(dialect, &body)?;
let upstream_body = a4mp::protocols::anthropic::build_request(&request)?;
let output = a4mp::protocols::anthropic::decode_response(&resp_body)?;
let client_body = a4mp::wire::encode_chat_output(dialect, &output)?;

// Streaming: upstream decoder → canonical StreamEvent → client encoder.
let event = decoder.decode(&sse_event)?;        // → StreamEvent
let out = encoder.encode(&event)?;              // → client SSE frame
```

Layers, top to bottom (depend only on what you use):

- `wire` — client-facing dialect codecs (OpenAI-Chat, OpenAI Responses,
  Anthropic); `default-features = false` ships codec-only, no HTTP runtime.
- `protocols` — upstream provider request builders + response/stream
  decoders for Chat, Responses and Anthropic.
- `transport` + `credentials` — optional async adapter layer (`Adapter`,
  `Transport`, `CredentialResolver`) for hosts that don't bring their own.

## Skills

- `skills/sdk/` — embedding guidance (canonical types, codec layering,
  stream relay, feature flags).
