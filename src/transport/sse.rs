//! Incremental Server-Sent Events decoding and typed event streams.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::api::{DynStream, Error, StreamEvent};

/// One decoded SSE frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, if present.
    pub event: Option<String>,
    /// Joined `data:` lines.
    pub data: String,
    /// The `id:` field, if present.
    pub id: Option<String>,
}

/// A single SSE frame is capped at 8 MiB: frames carry one JSON event
/// apiece and are kilobytes in practice, so an unterminated line or
/// separator-free blob beyond this is a peer pathological enough to cut
/// off rather than keep buffering.
const MAX_FRAME: usize = 8 * 1024 * 1024;

/// Incremental byte→frame decoder. Frames may split across arbitrary
/// chunk boundaries; `feed` yields only complete frames.
#[derive(Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
    /// Bytes already proven not to start a frame separator — scanning
    /// for `\n\n` resumes here instead of rescanning every chunk from
    /// zero (a terminator is always `\n`-led, so the cursor may sit at a
    /// `\n` whose lookahead was incomplete).
    scan: usize,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; returns every complete frame now decodable.
    /// `Error::Protocol` when the buffered frame outgrows the cap —
    /// an unbounded line or separator-less blob is a broken peer.
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, Error> {
        self.buf.extend_from_slice(bytes);
        if self.buf.len() > MAX_FRAME {
            return Err(Error::Protocol(format!(
                "SSE frame exceeds {} MiB cap",
                MAX_FRAME / (1024 * 1024)
            )));
        }
        let mut out = Vec::new();
        loop {
            match find_frame_end(&self.buf, self.scan) {
                (Some((frame_len, total)), _) => {
                    let frame: Vec<u8> = self.buf.drain(..frame_len).collect();
                    self.buf.drain(..total - frame_len);
                    self.scan = 0;
                    if let Some(ev) = parse_frame(&frame) {
                        out.push(ev);
                    }
                }
                (None, resume) => {
                    self.scan = resume;
                    break;
                }
            }
        }
        Ok(out)
    }

    /// End of stream: flush a trailing frame that had no terminator.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        self.scan = 0;
        let frame = std::mem::take(&mut self.buf);
        if frame.is_empty() {
            return Vec::new();
        }
        parse_frame(&frame).into_iter().collect()
    }
}

/// Locate the first complete frame at or after `from`. Returns
/// `(Some((frame_len, total_len_including_separator)), _)` on a hit, or
/// `(None, resume)` where `resume` is the position scanning may safely
/// restart from next feed — the earliest `\n` whose lookahead could not
/// complete a separator, or `buf.len()` when none was seen.
fn find_frame_end(buf: &[u8], from: usize) -> (Option<(usize, usize)>, usize) {
    let mut i = from;
    while i < buf.len() {
        if buf[i] != b'\n' {
            i += 1;
            continue;
        }
        if i + 1 >= buf.len() {
            // Trailing `\n` — could lead a `\n\n` or `\n\r\n` next feed.
            return (None, i);
        }
        // "\n\n": LF-terminated line followed by a bare LF.
        if buf[i + 1] == b'\n' {
            let end = if i > 0 && buf[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            return (Some((end, i + 2)), 0);
        }
        if buf[i + 1] == b'\r' {
            if i + 2 >= buf.len() {
                // Trailing `\n\r` — could complete to `\n\r\n` next feed.
                return (None, i);
            }
            // "\n\r\n": LF-terminated line followed by a CRLF line.
            if buf[i + 2] == b'\n' {
                let end = if i > 0 && buf[i - 1] == b'\r' {
                    i - 1
                } else {
                    i
                };
                return (Some((end, i + 3)), 0);
            }
        }
        i += 1;
    }
    (None, buf.len())
}

fn parse_frame(frame: &[u8]) -> Option<SseEvent> {
    let text = String::from_utf8_lossy(frame);
    let mut ev = SseEvent::default();
    let mut data = String::new();
    let mut has_data = false;
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() || line.starts_with(':') {
            continue; // blank lines and comments/heartbeats
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "data" => {
                if has_data {
                    data.push('\n');
                }
                data.push_str(value);
                has_data = true;
            }
            "event" => ev.event = Some(value.to_owned()),
            "id" => ev.id = Some(value.to_owned()),
            _ => {} // `retry:` and unknown fields are ignored
        }
    }
    if !has_data && ev.event.is_none() && ev.id.is_none() {
        return None;
    }
    ev.data = data;
    Some(ev)
}

/// A byte stream decoded into SSE frames.
pub struct SseStream {
    inner: DynStream<'static, bytes::Bytes>,
    decoder: SseDecoder,
    pending: VecDeque<SseEvent>,
    finished: bool,
}

impl SseStream {
    pub fn new(inner: DynStream<'static, bytes::Bytes>) -> Self {
        Self {
            inner,
            decoder: SseDecoder::new(),
            pending: VecDeque::new(),
            finished: false,
        }
    }
}

impl Stream for SseStream {
    type Item = Result<SseEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(ev)));
            }
            if self.finished {
                return Poll::Ready(None);
            }
            match self.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => match self.decoder.feed(&bytes) {
                    Ok(events) => self.pending.extend(events),
                    Err(e) => {
                        self.finished = true;
                        return Poll::Ready(Some(Err(e)));
                    }
                },
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    self.finished = true;
                    let events = self.decoder.finish();
                    self.pending.extend(events);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Maps decoded SSE frames into canonical stream events.
/// Each protocol implements this against its own chunk shape.
pub trait SseEventDecoder: Send {
    /// Decode one frame into zero or more canonical events.
    /// `event.data` equal to the protocol's terminator (e.g. `[DONE]`)
    /// produces no events; the stream then ends naturally.
    fn decode(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, Error>;
}

/// An SSE byte stream decoded straight into canonical `StreamEvent`s.
pub struct EventStream {
    inner: SseStream,
    decoder: Box<dyn SseEventDecoder>,
    pending: VecDeque<StreamEvent>,
}

impl EventStream {
    pub fn new(inner: DynStream<'static, bytes::Bytes>, decoder: Box<dyn SseEventDecoder>) -> Self {
        Self {
            inner: SseStream::new(inner),
            decoder,
            pending: VecDeque::new(),
        }
    }
}

impl Stream for EventStream {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(ev)));
            }
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(frame))) => match self.decoder.decode(&frame) {
                    Ok(events) => self.pending.extend(events),
                    Err(e) => return Poll::Ready(Some(Err(e))),
                },
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
