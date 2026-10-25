---
name: a4mp-sdk
description: Embed a4mp in a Rust host — canonical request/response types, wire dialects, provider protocols, optional transport layer.
---

# a4mp SDK mode

Embed `a4mp` — a pure codec/transport library with no daemon surface.

## When to use

- The host is a gateway, proxy or direct client that needs
  provider-dialect translation through one canonical middle type.
- Codec-only use pulls no HTTP runtime: `default-features = false` keeps
  reqwest/tokio out; bring your own transport (ureq, hyper, …).

## Canonical calls

```rust
// Buffered: client dialect → canonical → upstream dialect.
let req = a4mp::wire::decode_chat_request(in_dialect, &body)?;
let up = a4mp::protocols::<provider>::build_request(&req)?;
let out = a4mp::protocols::<provider>::decode_response(&resp)?;
let body = a4mp::wire::encode_chat_output(in_dialect, &out)?;

// Streaming: SseEventDecoder → StreamEvent → StreamEncoder.
```

## Rules that keep the contract

- **Canonical is the middle**: translate client→canonical→upstream, never
  client→upstream directly — every dialect pair then stays N×1, not N×N.
- A field the canonical type cannot express is reported as loss, not
  silently dropped — check decode diagnostics before promising fidelity.
- The `stream` flag is hop-local: the host injects it into the outgoing
  body for the upstream it's speaking to; it is not a canonical field.
- Codec layer needs no runtime; only the optional `Adapter`/`Transport`
  layer assumes tokio.
