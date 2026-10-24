//! Native protocol codecs: canonical contracts ↔ provider wire formats.
//!
//! Pure conversion only — no transport, no credentials. Each codec is
//! shared by every service target speaking that protocol family; profile
//! differences ride request `extensions` or adapter-level handling.

pub mod anthropic_messages;
pub mod bedrock_converse;
pub mod cohere;
pub mod gemini_generate;
pub mod gemini_media;
pub mod ollama;
pub mod openai_chat;
pub mod openai_data;
pub mod openai_media;
pub mod openai_realtime;
pub mod openai_responses;

/// Base64 encoding for inline media payloads.
pub(crate) fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = b0 << 16 | b1 << 8 | b2;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Standard base64 (with padding) → raw bytes. Whitespace and `=`
/// padding are skipped; any other non-alphabet byte is a protocol error.
pub(crate) fn base64_decode(s: &str) -> Result<bytes::Bytes, crate::api::Error> {
    fn value(b: u8) -> Option<u32> {
        match b {
            b'A'..=b'Z' => Some((b - b'A') as u32),
            b'a'..=b'z' => Some((b - b'a' + 26) as u32),
            b'0'..=b'9' => Some((b - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &b in s.as_bytes() {
        if b == b'=' || b == b'\n' || b == b'\r' {
            continue;
        }
        let v = value(b)
            .ok_or_else(|| crate::api::Error::Protocol("invalid base64 payload".into()))?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out.into())
}
