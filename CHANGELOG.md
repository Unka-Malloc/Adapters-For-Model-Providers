# Changelog

All notable changes to `a4mp` are recorded here. Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Behavior change:** the builtin local-server profiles — `omlx`,
  `mlx-serve`, `atomic-chat`, `ollama` (native and OpenAI-compatible
  targets) and `lmstudio` — plus privatemode's `local-proxy` region no
  longer carry compiled-in addresses. Their base URLs are
  `{endpoint}`/`{endpoint}/v1` templates, so they bind through
  `bind_provider_with` with an `endpoint` variable holding the runtime
  URL; `bind_provider` on them now fails with `Error::Protocol` naming
  `{endpoint}`.

### Fixed

- One endpoint policy now covers every materialization path — literal
  base URLs, `{name}` atoms and leading `{endpoint}` URLs alike:
  `https` binds anywhere, `http` binds only toward private or loopback
  authorities, and cleartext to a public host is refused with a
  distinct message naming the accepted classes. A leading `{name}`
  variable's value is validated by `check_endpoint_var` rather than
  re-parsed inline, so the two value classes share one check.
- Caller-supplied `http` endpoints are confined to private and loopback
  hosts — loopback, RFC1918, link-local, CGNAT `100.64/10`,
  `localhost`, `.localhost` and `.internal` — so credential-bearing
  traffic cannot go cleartext to a public address; named private hosts
  are trusted to the operator's own resolver, while spoofable classes
  (mDNS `.local`, search-domain single labels, arbitrary `.lan` zones)
  are not accepted. `https` endpoints bind anywhere and embedded
  credentials remain refused. Variables a target's template does not
  name stay inert, so one shared vars map can feed several targets, and
  trailing slashes on an `endpoint` value no longer yield `//v1` paths.

- `WireBridge::chat_stream` now creates one stream encoder and uses it for
  both the preamble and the rest of the stream. Responses streams therefore
  announce the same response id in `response.created`, `response.in_progress`
  and `response.completed`.
- `AnthropicStreamEncoder` closes every still-open content block and defers
  `message_stop` to `finish()`, so a canonical stream that ends at `Finished`
  (or without one) produces balanced block frames and any post-`Finished`
  usage is emitted before the terminator.
- `DeviceGrant`, `HttpRequest`, `HttpResponse` and `ProcessSpec` now have
  manual `Debug` implementations. Device/user codes, the raw
  device-authorization response, credential-bearing headers, request and
  response bodies, process argument values and environment values are
  redacted; non-secret metadata stays visible for diagnostics.
- The `Credential::Cloud` redaction test uses a non-PEM private-key marker as
  its synthetic secret, so the deterministic privacy scan no longer mistakes
  the test fixture for a real private key.

### Tests

- Bridge-level Responses coverage asserts one response id across the preamble
  and terminal frame.
- Anthropic stream coverage asserts balanced `content_block_start` /
  `content_block_stop` frames when a part is open at `Finished` or at stream
  end, and asserts `message_stop` stays last when usage follows `Finished`.
  A bridge-level test relays an OpenAI Chat-shaped canonical stream (no
  `PartFinished`) to an Anthropic client and checks block balancing.
- Endpoint-template coverage parametrizes a `bind_provider_with` pass
  over every converted local profile — `/v1` paths on chat surfaces,
  the bare endpoint on Ollama's native dialect, privatemode's
  region-selected `local-proxy` — plus trailing-slash normalization,
  public-IP `http` refusal (TEST-NET-3), private-host `http`
  acceptance, and shared vars maps carrying keys for sibling targets.

## [0.3.2] - 2026-10-25

### Fixed

- Gemini provider errors decoded from a 2xx response body or stream frame now
  report `RetrySafety::Unsafe`, including 429-shaped error envelopes and
  blocked prompts. Retrying such a response can resend the prompt and re-bill
  work the provider already accepted. Non-2xx `http_error` responses keep their
  status-based classification because the HTTP status itself proves rejection
  before acceptance.
- `Credential` now has a manual `Debug` implementation. Cloud-identity context
  fields whose names denote secrets (`session_token`, `secret_access_key`,
  `secretAccessKey`, `private_key`, …) are redacted recursively, so a host that
  logs a resolved or rejected credential cannot leak an AWS STS session token.
  Non-secret mechanism metadata such as `region` and `service` stays visible.

### Tests

- Post-2xx Gemini error envelopes are covered on chat, streaming, image and
  long-running-operation submit paths, while real non-2xx rejections remain
  `Safe`.
- `Credential::Cloud` debug output is tested for redaction at nested depth,
  including camelCase and snake_case secret keys.

## [0.3.1] - 2026-10-25

### Fixed

- `Error::Cancelled`, the new `Error::UpstreamProtocol`, and
  `TransportError::Cancelled` now report `RetrySafety::Unsafe`. These
  failures can surface after a request reached the provider, so an automatic
  retry could repeat billed input and output work.
- Post-dispatch response, stream, OAuth-token, broker and batch-status
  protocol violations now use `Error::UpstreamProtocol`. Local pre-dispatch
  validation (`Error::Protocol`, conversion, credentials, unsupported
  operations) and explicit provider rejections remain `Safe`.
- `RetrySafety` and error documentation now describe the boundary in billing
  terms: `Safe` means the provider did not accept or bill the work.

### Tests

- Contract coverage for cancelled transport errors, `Error::Cancelled`,
  local `Error::Protocol` (including a real wire-decoder failure) and
  `Error::UpstreamProtocol`, plus a real-socket transport test that cancels
  after the request has been dispatched.
- The README quickstart now uses the current `Client`, `Catalog`,
  `ChatRequest` and `RequestControl` APIs, and is compiled as a doctest so
  it cannot drift again.

## [0.3.0] - 2026-10-25

### Changed

- **Breaking:** `oauth::pkce()` now returns
  `Result<(Secret, String), Error>` instead of a bare tuple. Callers must
  handle entropy failure:

  ```rust
  let (verifier, challenge) = a4mp::oauth::pkce()?;
  ```

  `oauth::code_flow` keeps its signature and propagates the failure itself.

### Fixed

- OAuth PKCE and `state` generation no longer panic when the OS entropy
  source fails. `oauth::code_flow` returns `Error::Credential` before
  presenting an authorization URL, so a failed entropy read cannot yield a
  weak verifier or state value.

## [0.2.2] - 2026-10-25

### Fixed

- Responses stream decoding backfills text, reasoning, refusal and tool
  arguments from terminal `.done` or item payloads when no non-empty delta
  was observed; empty leading deltas no longer suppress the backfill.

### Documentation

- Stream examples emit the preamble and finish frames, and name the
  transport + auth layer correctly.

## [0.2.1] - 2026-10-25

### Fixed

- Responses stream decoding backfills text, reasoning and tool arguments
  when a provider sends only `.done` payloads, and synthesizes tool
  identity from the terminal `function_call` item.
- The Responses `stop` field reports one conversion loss instead of two.
- Shared one data-URI media parser between the Chat and Responses wire
  dialects.

### Documentation

- Corrected SDK and integration samples (tuple decode, model argument,
  event vector, 0.2 pins).

### Tests

- Added done-only stream coverage, a duplicate-loss regression and a real
  trailing-SSE-frame assertion in place of a tautological one.

## [0.2.0] - 2026-10-25

### Added

- Canonical `ChatRequest::parallel_tool_calls`, emitted by the Chat,
  Responses and Anthropic request builders and decoded by the Chat and
  Anthropic inbound codecs.
- OpenAI Responses inbound wire dialect: request items, response objects
  and `response.*` stream frames through canonical types.

### Changed

- Upgraded `rand` from 0.8 to 0.9, using `TryRngCore` for `OsRng` fills.

## [0.1.0] - 2026-10-25

### Added

- First public release: typed provider adapters, the provider catalog,
  OAuth/PKCE/device and SigV4 authentication, and optional wire codecs.
