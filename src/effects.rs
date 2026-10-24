//! Host effect port: local resources and external client processes.
//!
//! Adapters declare needs; the host decides fulfilment. The library never
//! opens user files, spawns runtimes or reads another application's
//! login state directly — every effect crosses this port explicitly.

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
    fn output(&mut self) -> DynStream<'_, bytes::Bytes>;
    /// Write bytes to the process stdin.
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
    /// Read a host-resolved resource by id.
    async fn read_resource(
        &self,
        resource: &ResourceRef,
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
}
