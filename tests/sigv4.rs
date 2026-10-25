//! SigV4 tests: signatures pinned by an independent reference derivation
//! over hand-written canonical requests, canonicalization behavior,
//! credential-shape errors and determinism.
//!
//! Signing material is minted per run: no credential-shaped value is
//! written into this file, and every expected signature is derived from
//! the run's material by the reference HMAC chain below.
//!
//! The module under test is included through `#[path]` so this file works
//! before `pub mod sigv4` is registered in `src/lib.rs`: the `crate::`
//! paths inside the included file resolve through the `a4mp` root imports
//! below. Once lib.rs exports the module, the local `mod sigv4` shadows
//! the name and this test keeps passing unchanged.

use std::time::{Duration, SystemTime};

// Root bindings: `crate::api` / `crate::auth` / `crate::transport` inside
// the `#[path]`-included file resolve to these.
use a4mp::{Credential, Error, HttpRequest, Protocol, Secret, ServiceTarget};
use a4mp::{api, auth, transport};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use sigv4::{SigV4, is_cloud};

#[allow(dead_code)]
#[path = "../src/sigv4.rs"]
mod sigv4;

mod common;

/// One run's signing material: access key id, signing secret and session
/// token, each minted at run time.
struct Keys {
    access_id: String,
    signing: String,
    session: String,
}

impl Keys {
    fn mint() -> Self {
        Self {
            access_id: common::fixture_token("access-id"),
            signing: common::fixture_token("signing"),
            session: common::fixture_token("session"),
        }
    }
}

/// sha256("") — the empty-body payload hash.
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// 2015-08-30T12:36:00Z — the timestamp used by the AWS test suite.
fn now() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_440_938_160)
}

fn target(region: Option<&str>, base: &str) -> ServiceTarget {
    let mut t = ServiceTarget::new(
        "aws-test",
        Protocol::BedrockConverse,
        url::Url::parse(base).unwrap(),
    );
    t.region = region.map(str::to_owned);
    t
}

fn cloud(keys: &Keys, context: &[(&str, &str)]) -> Credential {
    let mut map = serde_json::Map::new();
    for (k, v) in context {
        map.insert((*k).to_owned(), (*v).into());
    }
    Credential::Cloud {
        token: Secret::new(keys.signing.as_str()),
        expires_at: None,
        context: map,
    }
}

fn authz(req: &HttpRequest) -> String {
    req.headers
        .get(http::header::AUTHORIZATION)
        .expect("authorization header")
        .to_str()
        .unwrap()
        .to_owned()
}

// ---------------------------------------------------------------------------
// Independent reference derivation (shares no code with the module under
// test — the canonical request strings are written out by hand from the
// documented layout).
// ---------------------------------------------------------------------------

fn ref_hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("hmac accepts any key");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn ref_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// HMAC chain kDate → kRegion → kService → kSigning, per AWS docs.
fn ref_signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = ref_hmac(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k_region = ref_hmac(&k_date, region.as_bytes());
    let k_service = ref_hmac(&k_region, service.as_bytes());
    ref_hmac(&k_service, b"aws4_request")
}

/// Signature for a hand-written canonical request string.
fn ref_signature(k_signing: &[u8], amz_date: &str, scope: &str, canonical_request: &str) -> String {
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        ref_hex(&Sha256::digest(canonical_request.as_bytes()))
    );
    ref_hex(&ref_hmac(k_signing, string_to_sign.as_bytes()))
}

/// RFC 4231 test case 1 pins the reference HMAC-SHA256 primitive, so the
/// derivation every signature below is checked against is itself anchored
/// to a published vector.
#[test]
fn reference_hmac_reproduces_rfc_4231() {
    assert_eq!(
        ref_hex(&ref_hmac(&[0x0b; 20], b"Hi There")),
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    );
}

// ---------------------------------------------------------------------------
// Known-answer tests
// ---------------------------------------------------------------------------

/// The request of the AWS SigV4 test-suite vector
/// `get-vanilla-query-order-key-case` (GET on example.amazonaws.com,
/// us-east-1, service "service", 20150830T123600Z), signed with the run's
/// material. The module always adds `x-amz-content-sha256` because
/// bedrock-runtime requires it, so the hand-written canonical request
/// carries that header; the reference derivation over it must equal the
/// module's signature.
#[test]
fn vanilla_query_order_key_case_matches_the_reference() {
    let keys = Keys::mint();
    let canonical = concat!(
        "GET\n/\nParam1=value1&Param2=value2\n",
        "host:example.amazonaws.com\n",
        "x-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n",
        "x-amz-date:20150830T123600Z\n\n",
        "host;x-amz-content-sha256;x-amz-date\n",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    let scope = "20150830/us-east-1/service/aws4_request";
    let k_signing = ref_signing_key(&keys.signing, "20150830", "us-east-1", "service");
    let expected = ref_signature(&k_signing, "20150830T123600Z", scope, canonical);

    let fixture = cloud(
        &keys,
        &[
            ("access_key_id", &keys.access_id),
            ("region", "us-east-1"),
            ("service", "service"),
        ],
    );
    let mut req = HttpRequest::get(
        url::Url::parse("https://example.amazonaws.com/?Param2=value2&Param1=value1").unwrap(),
    );
    SigV4::new("service")
        .sign(
            &mut req,
            &fixture,
            &target(None, "https://example.amazonaws.com/"),
            now(),
        )
        .unwrap();

    assert_eq!(req.headers["x-amz-date"], "20150830T123600Z");
    assert_eq!(req.headers["x-amz-content-sha256"], EMPTY_SHA256);
    assert_eq!(req.headers[http::header::HOST], "example.amazonaws.com");
    assert_eq!(
        authz(&req),
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, \
             SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={expected}",
            keys.access_id
        )
    );
}

/// Bedrock-runtime-shaped POST: body hash header set, session token set
/// and covered by the signature, path segment `:` URI-encoded.
#[test]
fn bedrock_post_signs_session_token_and_body() {
    /// sha256 of the request body below.
    const PAYLOAD_SHA: &str = "faf67977867133068b4f8078e1cd5c049141ca4e5c4cbf802077df480d1a466c";
    let keys = Keys::mint();

    // No region/service in the context: they fall back to target.region
    // and SigV4::service respectively.
    let fixture = cloud(
        &keys,
        &[
            ("access_key_id", &keys.access_id),
            ("session_token", &keys.session),
        ],
    );
    let target = target(
        Some("us-east-1"),
        "https://bedrock-runtime.us-east-1.amazonaws.com/",
    );
    let mut req = HttpRequest {
        method: http::Method::POST,
        url: url::Url::parse(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2:0/converse",
        )
        .unwrap(),
        headers: http::HeaderMap::new(),
        body: bytes::Bytes::from_static(
            br#"{"max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    };
    req.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    assert_eq!(ref_hex(&Sha256::digest(&req.body)), PAYLOAD_SHA);

    SigV4::new("bedrock")
        .sign(&mut req, &fixture, &target, now())
        .unwrap();

    assert_eq!(req.headers["x-amz-security-token"], keys.session.as_str());
    assert_eq!(req.headers["x-amz-content-sha256"], PAYLOAD_SHA);
    let authz = authz(&req);
    assert!(authz.contains(&format!(
        "Credential={}/20150830/us-east-1/bedrock/aws4_request",
        keys.access_id
    )));
    assert!(authz.contains(
        "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
    ));

    // The reference derivation over the hand-written canonical request
    // must equal the module's output.
    let canonical = format!(
        "POST\n/model/anthropic.claude-v2%3A0/converse\n\n\
         content-type:application/json\n\
         host:bedrock-runtime.us-east-1.amazonaws.com\n\
         x-amz-content-sha256:{PAYLOAD_SHA}\n\
         x-amz-date:20150830T123600Z\n\
         x-amz-security-token:{}\n\n\
         content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token\n\
         {PAYLOAD_SHA}",
        keys.session
    );
    let k_signing = ref_signing_key(&keys.signing, "20150830", "us-east-1", "bedrock");
    let expected = ref_signature(
        &k_signing,
        "20150830T123600Z",
        "20150830/us-east-1/bedrock/aws4_request",
        &canonical,
    );
    assert!(authz.ends_with(&format!("Signature={expected}")));
}

// ---------------------------------------------------------------------------
// Canonicalization behavior
// ---------------------------------------------------------------------------

/// Equivalent wire forms must produce identical signatures: query order,
/// pre-encoded vs literal reserved/unreserved characters, bare vs empty
/// keys. A literal '+' is not a space.
#[test]
fn query_sorting_and_encoding_are_canonical() {
    let keys = Keys::mint();
    let fixture = cloud(
        &keys,
        &[("access_key_id", &keys.access_id), ("region", "us-east-1")],
    );
    let target = target(Some("us-east-1"), "https://example.amazonaws.com/");
    let signer = SigV4::new("service");
    let sign = |query: &str| {
        let mut req = HttpRequest::get(
            url::Url::parse(&format!("https://example.amazonaws.com/{query}")).unwrap(),
        );
        signer.sign(&mut req, &fixture, &target, now()).unwrap();
        authz(&req)
    };
    assert_eq!(sign("?b=2&a=1"), sign("?a=1&b=2"));
    assert_eq!(sign("?x=a%2Bb"), sign("?x=a+b"));
    assert_eq!(sign("?x=%7E"), sign("?x=~"));
    assert_eq!(sign("?x=%41"), sign("?x=A"));
    assert_eq!(sign("?k="), sign("?k"));
    assert_ne!(sign("?x=a%20b"), sign("?x=a+b"));
}

/// Context `region`/`service` override `ServiceTarget::region` and
/// `SigV4::service`; the target region is the fallback.
#[test]
fn region_and_service_resolution() {
    let keys = Keys::mint();
    let fixture = cloud(
        &keys,
        &[
            ("access_key_id", &keys.access_id),
            ("region", "us-west-2"),
            ("service", "s3"),
        ],
    );
    let mut req = HttpRequest::get(url::Url::parse("https://s3.example/").unwrap());
    SigV4::new("bedrock")
        .sign(
            &mut req,
            &fixture,
            &target(Some("us-east-1"), "https://s3.example/"),
            now(),
        )
        .unwrap();
    assert!(
        authz(&req).contains(&format!(
            "Credential={}/20150830/us-west-2/s3/aws4_request",
            keys.access_id
        )),
        "context region/service must win: {}",
        authz(&req)
    );

    let fixture = cloud(&keys, &[("access_key_id", &keys.access_id)]);
    let mut req = HttpRequest::get(url::Url::parse("https://x.example/").unwrap());
    SigV4::new("bedrock")
        .sign(
            &mut req,
            &fixture,
            &target(Some("eu-west-1"), "https://x.example/"),
            now(),
        )
        .unwrap();
    assert!(authz(&req).contains("/eu-west-1/bedrock/aws4_request"));
}

// ---------------------------------------------------------------------------
// Credential shape and determinism
// ---------------------------------------------------------------------------

#[test]
fn missing_material_names_the_key() {
    let keys = Keys::mint();
    let signer = SigV4::new("bedrock");
    let tgt = target(Some("us-east-1"), "https://example.amazonaws.com/");
    let mut req = HttpRequest::get(url::Url::parse("https://example.amazonaws.com/").unwrap());

    // A non-Cloud credential cannot sign.
    let bearer = Credential::Bearer {
        token: Secret::new(keys.signing.as_str()),
        expires_at: None,
    };
    match signer.sign(&mut req, &bearer, &tgt, now()) {
        Err(Error::Credential(m)) => assert!(m.contains("Cloud"), "{m}"),
        other => panic!("expected credential error, got {other:?}"),
    }

    // access_key_id is required.
    let fixture = cloud(&keys, &[("region", "us-east-1")]);
    match signer.sign(&mut req, &fixture, &tgt, now()) {
        Err(Error::Credential(m)) => assert!(m.contains("access_key_id"), "{m}"),
        other => panic!("expected credential error, got {other:?}"),
    }

    // A non-string context value counts as missing.
    let mut ctx = serde_json::Map::new();
    ctx.insert("access_key_id".into(), serde_json::json!(42));
    let fixture = Credential::Cloud {
        token: Secret::new(keys.signing.as_str()),
        expires_at: None,
        context: ctx,
    };
    match signer.sign(&mut req, &fixture, &tgt, now()) {
        Err(Error::Credential(m)) => assert!(m.contains("access_key_id"), "{m}"),
        other => panic!("expected credential error, got {other:?}"),
    }

    // No region anywhere.
    let fixture = cloud(&keys, &[("access_key_id", &keys.access_id)]);
    let no_region = target(None, "https://example.amazonaws.com/");
    match signer.sign(&mut req, &fixture, &no_region, now()) {
        Err(Error::Credential(m)) => assert!(m.contains("region"), "{m}"),
        other => panic!("expected credential error, got {other:?}"),
    }
}

#[test]
fn signing_is_deterministic_and_idempotent() {
    let keys = Keys::mint();
    let fixture = cloud(
        &keys,
        &[
            ("access_key_id", &keys.access_id),
            ("region", "us-east-1"),
            ("session_token", &keys.session),
        ],
    );
    let target = target(
        Some("us-east-1"),
        "https://bedrock-runtime.us-east-1.amazonaws.com/",
    );
    let build = || {
        let mut r = HttpRequest::post_json(
            url::Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse")
                .unwrap(),
            serde_json::json!({"a": 1}),
        );
        r.headers
            .insert("x-custom", http::HeaderValue::from_static("v"));
        r
    };
    let signer = SigV4::new("bedrock");
    let (mut a, mut b) = (build(), build());
    signer.sign(&mut a, &fixture, &target, now()).unwrap();
    signer.sign(&mut b, &fixture, &target, now()).unwrap();
    assert_eq!(authz(&a), authz(&b));
    // Re-signing an already-signed request replaces stale signing headers.
    signer.sign(&mut a, &fixture, &target, now()).unwrap();
    assert_eq!(authz(&a), authz(&b));
}

#[test]
fn is_cloud_and_sign_now() {
    let keys = Keys::mint();
    let fixture = cloud(
        &keys,
        &[("access_key_id", &keys.access_id), ("region", "us-east-1")],
    );
    assert!(is_cloud(&fixture));
    assert!(!is_cloud(&Credential::None));
    assert!(!is_cloud(&Credential::ApiKey(Secret::new(
        keys.signing.as_str()
    ))));

    let mut req = HttpRequest::get(url::Url::parse("https://example.amazonaws.com/").unwrap());
    SigV4::new("service")
        .sign_now(
            &mut req,
            &fixture,
            &target(Some("us-east-1"), "https://example.amazonaws.com/"),
        )
        .unwrap();
    assert!(req.headers.contains_key(http::header::AUTHORIZATION));
    assert_eq!(req.headers["x-amz-content-sha256"], EMPTY_SHA256);
}
