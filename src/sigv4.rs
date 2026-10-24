//! AWS Signature Version 4 request signing.
//!
//! Implemented from the documented SigV4 algorithm — canonical request,
//! string to sign, HMAC signing-key chain — with no AWS SDK code. AWS
//! hosted providers such as Bedrock authenticate with
//! `AuthKind::CloudIdentity`: the adapter builds its [`HttpRequest`], then
//! calls [`SigV4::sign`] (or [`SigV4::sign_now`]) before handing the
//! request to the transport.
//!
//! # Credential contract
//!
//! The signer consumes [`Credential::Cloud`]; `token` is the AWS SECRET
//! ACCESS KEY — it seeds the HMAC chain and is never written to the
//! request, an error or a log. `context` carries the identity fields:
//!
//! | context key        | required | meaning                                          |
//! |--------------------|----------|--------------------------------------------------|
//! | `"access_key_id"`  | yes      | Access key id; appears in `Authorization`.       |
//! | `"session_token"`  | no       | STS session token → `x-amz-security-token`, signed. |
//! | `"region"`         | no       | Signing region; falls back to `target.region`.   |
//! | `"service"`        | no       | Signing service; falls back to `SigV4::service`. |

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::api::{Error, ServiceTarget};
use crate::auth::Credential;
use crate::transport::HttpRequest;

/// `Credential::Cloud` context key for the AWS access key id (required).
pub const CTX_ACCESS_KEY_ID: &str = "access_key_id";
/// `Credential::Cloud` context key for an STS session token (optional).
pub const CTX_SESSION_TOKEN: &str = "session_token";
/// `Credential::Cloud` context key that overrides `ServiceTarget::region`.
pub const CTX_REGION: &str = "region";
/// `Credential::Cloud` context key that overrides [`SigV4::service`].
pub const CTX_SERVICE: &str = "service";

const ALGORITHM: &str = "AWS4-HMAC-SHA256";
const TERMINATOR: &str = "aws4_request";
/// `x-amz-content-sha256` is always set and signed: bedrock-runtime
/// requires it, and every AWS service accepts it.
const HDR_PAYLOAD_SHA: &str = "x-amz-content-sha256";

type HmacSha256 = Hmac<Sha256>;

/// A stateless SigV4 signer for one AWS service family. `service` is the
/// credential-scope service name used when the credential context does not
/// name one (`"bedrock"` covers the Bedrock control and runtime lanes).
#[derive(Clone, Debug)]
pub struct SigV4 {
    /// Default signing service name (e.g. `"bedrock"`).
    pub service: String,
}

impl SigV4 {
    pub fn new(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
        }
    }

    /// Sign `req` in place as of `now`. Injects `host`, `x-amz-date`,
    /// `x-amz-content-sha256` and — for session credentials —
    /// `x-amz-security-token` into `req.headers`, signs every header then
    /// present, and writes the `Authorization` header. Deterministic:
    /// identical inputs produce an identical signature, and re-signing an
    /// already-signed request replaces the stale `Authorization` cleanly.
    pub fn sign(
        &self,
        req: &mut HttpRequest,
        cred: &Credential,
        target: &ServiceTarget,
        now: SystemTime,
    ) -> Result<(), Error> {
        let (token, context) = match cred {
            Credential::Cloud {
                token, context, ..
            } => (token, context),
            _ => {
                return Err(Error::Credential(
                    "sigv4: expected Credential::Cloud material".into(),
                ))
            }
        };
        let access_key_id = ctx_str(context, CTX_ACCESS_KEY_ID).ok_or_else(|| {
            Error::Credential(format!(
                "sigv4: credential context missing \"{CTX_ACCESS_KEY_ID}\""
            ))
        })?;
        let region = ctx_str(context, CTX_REGION)
            .map(str::to_owned)
            .or_else(|| target.region.clone())
            .ok_or_else(|| {
                Error::Credential(format!(
                    "sigv4: no signing region — credential context \"{CTX_REGION}\" \
                     is unset and ServiceTarget::region is None"
                ))
            })?;
        let service = ctx_str(context, CTX_SERVICE).unwrap_or(&self.service);
        let session_token = ctx_str(context, CTX_SESSION_TOKEN);

        let time = amz_time(now);
        let payload_hash = sha256_hex(&req.body);
        let host = host_header(&req.url)?;

        // Replace anything from a previous signing pass so the canonical
        // header set is exactly what this request carries on the wire.
        req.headers.remove(http::header::AUTHORIZATION);
        req.headers.remove(http::header::HOST);
        req.headers.remove("x-amz-date");
        req.headers.remove(HDR_PAYLOAD_SHA);
        req.headers.remove("x-amz-security-token");
        req.headers.insert(http::header::HOST, header_value(&host)?);
        req.headers.insert("x-amz-date", header_value(&time.amz_date)?);
        req.headers.insert(HDR_PAYLOAD_SHA, header_value(&payload_hash)?);
        if let Some(session_token) = session_token {
            req.headers.insert(
                "x-amz-security-token",
                header_value(session_token).map_err(|_| {
                    Error::Credential(
                        "sigv4: session_token is not a valid header value".into(),
                    )
                })?,
            );
        }

        let (canonical_headers, signed_headers) = canonical_headers(&req.headers);
        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            req.method.as_str(),
            canonical_uri(&req.url),
            canonical_query(&req.url),
            canonical_headers,
            signed_headers,
            payload_hash,
        );

        let scope = format!("{}/{}/{}/{}", time.date_stamp, region, service, TERMINATOR);
        let string_to_sign = format!(
            "{ALGORITHM}\n{}\n{}\n{}",
            time.amz_date,
            scope,
            sha256_hex(canonical_request.as_bytes()),
        );

        // Derived signing key: HMAC("AWS4" + secret, date) → region →
        // service → terminator, then one final HMAC over the string.
        let k_date = hmac(
            format!("AWS4{}", token.expose()).as_bytes(),
            time.date_stamp.as_bytes(),
        );
        let k_region = hmac(&k_date, region.as_bytes());
        let k_service = hmac(&k_region, service.as_bytes());
        let k_signing = hmac(&k_service, TERMINATOR.as_bytes());
        let signature = hex(&hmac(&k_signing, string_to_sign.as_bytes()));

        let authorization = format!(
            "{ALGORITHM} Credential={access_key_id}/{scope}, \
             SignedHeaders={signed_headers}, Signature={signature}"
        );
        req.headers.insert(
            http::header::AUTHORIZATION,
            header_value(&authorization).map_err(|_| {
                Error::Credential("sigv4: access_key_id is not a valid header value".into())
            })?,
        );
        Ok(())
    }

    /// [`SigV4::sign`] with the current wall clock.
    pub fn sign_now(
        &self,
        req: &mut HttpRequest,
        cred: &Credential,
        target: &ServiceTarget,
    ) -> Result<(), Error> {
        self.sign(req, cred, target, SystemTime::now())
    }
}

/// Whether the credential carries cloud-identity material SigV4 can use.
pub fn is_cloud(cred: &Credential) -> bool {
    matches!(cred, Credential::Cloud { .. })
}

/// Read a non-empty string field out of a `Credential::Cloud` context map.
fn ctx_str<'a>(
    context: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a str> {
    context
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
}

/// Build a header value; the input is never echoed into the error because
/// callers pass material that may be sensitive (session tokens, authz).
fn header_value(s: &str) -> Result<http::HeaderValue, Error> {
    http::HeaderValue::from_str(s).map_err(|_| {
        Error::Protocol("sigv4: signing field is not a valid header value".into())
    })
}

fn hmac(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC accepts keys of any size");
    mac.update(data);
    mac.finalize().into_bytes().into()
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

// ---------------------------------------------------------------------------
// Canonical request pieces
// ---------------------------------------------------------------------------

/// `Host` header value for the request URL, keeping a non-default port.
fn host_header(url: &url::Url) -> Result<String, Error> {
    let host = url
        .host_str()
        .ok_or_else(|| Error::Protocol(format!("sigv4: URL has no host: {url}")))?;
    Ok(match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_owned(),
    })
}

/// `YYYYMMDD'T'HHMMSS'Z'` and `YYYYMMDD` for one instant in UTC.
struct AmzTime {
    amz_date: String,
    date_stamp: String,
}

fn amz_time(now: SystemTime) -> AmzTime {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day) = civil_from_days((secs / 86_400) as i64);
    let clock = secs % 86_400;
    let (h, m, s) = (clock / 3_600, (clock % 3_600) / 60, clock % 60);
    let date_stamp = format!("{year:04}{month:02}{day:02}");
    AmzTime {
        amz_date: format!("{date_stamp}T{h:02}{m:02}{s:02}Z"),
        date_stamp,
    }
}

/// Days since the Unix epoch → proleptic-Gregorian `(year, month, day)`.
/// The civil-from-days algorithm (Howard Hinnant, public domain), so no
/// calendar dependency is needed for `x-amz-date`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // day of era: [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// AWS URI-encode: every byte outside `A-Z a-z 0-9 - _ . ~` becomes
/// `%XX` (uppercase hex). Applied per path segment / per query part, so
/// '/' is never inside the input and stays a separator outside it.
fn uri_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

/// Decode `%XX` triplets (any case); anything else passes through as-is.
/// '+' stays literal — query params are not form-decoded.
fn percent_decode(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push(hi << 4 | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Canonical URI: the path split on '/', each segment percent-decoded
/// then URI-encoded per AWS rules (normalizing hex case and reserved
/// characters), rejoined with '/'. An empty path becomes "/".
fn canonical_uri(url: &url::Url) -> String {
    let path = url.path();
    let canonical = path
        .split('/')
        .map(|segment| uri_encode(&percent_decode(segment)))
        .collect::<Vec<_>>()
        .join("/");
    if canonical.is_empty() {
        "/".to_owned()
    } else {
        canonical
    }
}

/// Canonical query string: `name=value` pairs percent-decoded, URI-encoded
/// and sorted by encoded name then value, joined with '&'.
fn canonical_query(url: &url::Url) -> String {
    let mut pairs: Vec<(String, String)> = Vec::new();
    if let Some(query) = url.query() {
        for part in query.split('&') {
            if part.is_empty() {
                continue;
            }
            let (name, value) = part.split_once('=').unwrap_or((part, ""));
            pairs.push((
                uri_encode(&percent_decode(name)),
                uri_encode(&percent_decode(value)),
            ));
        }
    }
    pairs.sort();
    pairs
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// Canonical headers block (each `name:value\n`, sorted by name, values
/// trimmed with sequential whitespace collapsed) plus the `;`-joined
/// signed-headers list.
fn canonical_headers(headers: &http::HeaderMap) -> (String, String) {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in headers.keys() {
        for value in headers.get_all(name) {
            let normalized = String::from_utf8_lossy(value.as_bytes())
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            map.entry(name.as_str().to_owned()).or_default().push(normalized);
        }
    }
    let mut canonical = String::new();
    let mut signed = String::new();
    for (name, values) in &map {
        let _ = writeln!(canonical, "{name}:{}", values.join(","));
        if !signed.is_empty() {
            signed.push(';');
        }
        signed.push_str(name);
    }
    (canonical, signed)
}
