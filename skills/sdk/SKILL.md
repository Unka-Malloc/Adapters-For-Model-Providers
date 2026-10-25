---
name: a4mp-sdk
description: Embed a4mp in a Rust host — canonical request/response types, wire dialects, provider protocols, optional transport layer.
---

# a4mp SDK mode

Embed `a4mp` — a pure codec/transport library with no daemon surface.

## When to use

- The host is a gateway, proxy or direct client that needs
  provider-dialect translation through one canonical middle type.
- Codec-only use drops the HTTP transport: `default-features = false`
  removes reqwest (the HTTP client); `RequestControl` still links
  tokio-util's cancellation token. Bring your own transport (ureq, hyper, …).

## Canonical calls

```rust
// Buffered: client dialect → canonical → upstream dialect.
let (req, decode_report) = a4mp::wire::decode_chat_request(in_dialect, &body)?;
let up = a4mp::protocols::openai_chat::build_request(&req)?;
let out = a4mp::protocols::openai_chat::decode_response(&upstream_response)?;
let (body, encode_report) =
    a4mp::wire::encode_chat_output(in_dialect, &out, &req.model);

// Streaming: SseEventDecoder → StreamEvent → StreamEncoder.
let mut encoder = a4mp::wire::stream_encoder(in_dialect, &req.model);
for event in upstream_decoder.decode(&sse_event)? {
    for frame in encoder.encode(&event) {
        // serialize frame as event:/data: lines
    }
}
```

## Rules that keep the contract

- **Canonical is the middle**: translate client→canonical→upstream, never
  client→upstream directly — every dialect pair then stays N×1, not N×N.
- A field the canonical type cannot express is reported as loss, not
  silently dropped — check decode diagnostics before promising fidelity.
- The `stream` flag is hop-local: the host injects it into the outgoing
  body for the upstream it's speaking to; it is not a canonical field.
- Codec layer needs no HTTP client or executor; only the optional
  `Adapter`/`Transport` layer assumes an async runtime.
