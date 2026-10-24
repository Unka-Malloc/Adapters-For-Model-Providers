//! Host effect port: local resources and external client processes.
//!
//! Adapters declare needs; the host decides fulfilment. The library never
//! opens user files, spawns runtimes or reads another application's
//! login state directly — every effect crosses this port explicitly.

use std::time::Duration;

use crate::api::{DynStream, Error, RequestControl};

/// A named host-resolved resource (file attachment, model asset, …).
#[derive(Clone, Debug)]
pub struct ResourceRef {
    /// Opaque handle meaningful only to the host.
    pub id: String,
    pub media_type: Option<String>,
}

/// How a required external client process should run.
#[derive(Clone, Debug)]
pub struct ProcessSpec {
    /// Declared runtime requirement (e.g. "cursor-agent >= 1.2",
    /// "python3 plugin entrypoint"). The host checks availability.
    pub requirement: String,
    pub program: String,
    pub args: Vec<String>,
    /// Environment overrides layered onto the real host environment.
    pub env: Vec<(String, String)>,
    /// Working directory when required; otherwise inherit.
    pub cwd: Option<String>,
}

/// A running external process with a bounded, pull-based output stream.
/// Adapters read incrementally; the host enforces lifecycle and cleanup.
pub trait Process: Send {
    /// stdout/stderr merged byte stream (pull-based backpressure).
    ///
    /// Implementations must return a *resumable* stream: consumers may
    /// re-call `output` after a dropped or completed poll and continue
    /// where the previous stream left off. A fresh competing reader per
    /// call would lose bytes the adapter already owns.
    fn output(&mut self) -> DynStream<'_, bytes::Bytes>;
    /// Write bytes to the process stdin. There is no explicit
    /// close-input signal — drop/termination ends input.
    fn send_input(&mut self, data: bytes::Bytes) -> Result<(), Error>;
    /// Ask the process to exit; `wait` reports the final status.
    fn terminate(&mut self) -> Result<(), Error>;
    /// Wait for exit and report the exit status.
    fn wait(&mut self) -> Result<i32, Error>;
}

/// Host-implemented local effects. Implemented by the application
/// (LicoUp gateway, CLI, test harness); the library holds only the port.
#[async_trait::async_trait]
pub trait EffectHost: Send + Sync {
    /// Read a host-resolved resource by id. Implementations may fill
    /// `resource.media_type` so callers can inline the bytes with a real
    /// MIME type instead of guessing.
    async fn read_resource(
        &self,
        resource: &mut ResourceRef,
        ctrl: &RequestControl,
    ) -> Result<bytes::Bytes, Error> {
        let _ = ctrl;
        Err(Error::Unsupported(format!(
            "host cannot read resource {:?}",
            resource.id
        )))
    }

    /// Spawn a required external client process. Only adapters that
    /// declare `AuthKind::External` call this; the host may refuse.
    async fn spawn(
        &self,
        spec: &ProcessSpec,
        ctrl: &RequestControl,
    ) -> Result<Box<dyn Process>, Error> {
        let _ = ctrl;
        Err(Error::Unsupported(format!(
            "host cannot spawn {:?}",
            spec.requirement
        )))
    }

    /// Redirect landing point for interactive authorization flows.
    /// `None` means the host cannot capture callbacks (device flow or
    /// paste-back flows still work).
    fn redirect_uri(&self) -> Option<url::Url> {
        None
    }

    /// Present an authorization URL to the user: open a browser, show
    /// a QR code or display text — the host chooses.
    async fn present_url(&self, url: &url::Url, ctrl: &RequestControl) -> Result<(), Error> {
        let _ = ctrl;
        Err(Error::Unsupported(format!(
            "host cannot present {url}"
        )))
    }

    /// Await the provider's redirect and return the full callback URL.
    /// Implementations listen on a loopback port, capture a custom
    /// scheme or accept a paste-back.
    async fn await_callback(&self, ctrl: &RequestControl) -> Result<String, Error> {
        let _ = ctrl;
        Err(Error::Unsupported("host cannot capture auth callbacks".into()))
    }

    /// Wait `duration` before continuing a flow (device-code polling,
    /// refresh scheduling). The library carries no async runtime, so
    /// sleeping itself is a host effect; cancellation must interrupt it.
    async fn sleep(&self, duration: Duration, ctrl: &RequestControl) -> Result<(), Error> {
        let _ = (duration, ctrl);
        Err(Error::Unsupported("host supplies no timer".into()))
    }
}
