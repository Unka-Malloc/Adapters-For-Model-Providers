# Integration modes

One way to consume this crate: **SDK — embed it**. `a4mp` is a codec and
transport library — canonical request/response types, per-provider wire
decoders/encoders, transports and credential resolution. It has no daemon
surface, so there is no sidecar mode.

## SDK — embed the crate

```toml
[dependencies]
a4mp = "0.2"               # default-features = false drops the reqwest transport
```

```rust
// Canonical middle: decode the client's dialect, build the upstream's.
let (request, decode_report) = a4mp::wire::decode_chat_request(dialect, &body)?;
let upstream_body = a4mp::protocols::anthropic_messages::build_request(&request)?;
let output = a4mp::protocols::anthropic_messages::decode_response(&resp_body)?;
let (client_body, encode_report) =
    a4mp::wire::encode_chat_output(dialect, &output, &request.model);

// Streaming: upstream decoder → canonical StreamEvent → client encoder.
let events: Vec<a4mp::StreamEvent> = upstream_decoder.decode(&sse_event)?;
let mut client_encoder = a4mp::wire::stream_encoder(dialect, &request.model);
for event in &events {
    let frames = client_encoder.encode(event);   // → client SSE frames
}
```

Layers, top to bottom (depend only on what you use):

- `wire` — client-facing dialect codecs (OpenAI-Chat, OpenAI Responses,
  Anthropic); `default-features = false` drops the HTTP client, while
  `RequestControl` still links tokio-util's cancellation token.
- `protocols` — upstream provider request builders + response/stream
  decoders for Chat, Responses and Anthropic.
- `transport` + `credentials` — optional async adapter layer (`Adapter`,
  `Transport`, `CredentialResolver`) for hosts that don't bring their own.

## Skills

- `skills/sdk/` — embedding guidance (canonical types, codec layering,
  stream relay, feature flags).
