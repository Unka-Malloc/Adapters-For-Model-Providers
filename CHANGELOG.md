# Changelog

All notable changes to `a4mp` are recorded here. Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- The README quickstart now compiles against the current `Client` and
  `RequestControl` APIs.

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
