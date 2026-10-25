//! OAuth mechanics owned by the library: PKCE, authorization-code and
//! device flows, token exchange and refresh.
//!
//! Interaction (presenting the URL, capturing the callback, waiting
//! between polls) crosses `EffectHost`; issued-token custody stays with
//! the host's `CredentialResolver`. The library never opens a browser,
//! binds a listener or stores a token itself.

use std::time::{Duration, SystemTime};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::TryRngCore;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::api::{Error, RequestControl};
use crate::auth::{Credential, Secret};
use crate::effects::EffectHost;
use crate::transport::{HttpRequest, Transport};

/// An OAuth client registration for one provider's endpoints.
#[derive(Clone, Debug)]
pub struct OAuthClient {
    pub client_id: String,
    /// Client secret only when the provider issues one to this client
    /// kind; public clients rely on PKCE and leave this `None`.
    pub client_secret: Option<Secret>,
    /// Authorization endpoint; `None` when the provider has no
    /// interactive code flow.
    pub authorize_url: Option<url::Url>,
    /// Device-authorization endpoint; `None` when unsupported.
    pub device_url: Option<url::Url>,
    pub token_url: url::Url,
    pub scope: Option<String>,
    /// Provider-required extra parameters on the authorize request.
    pub authorize_extra: Vec<(String, String)>,
    /// Provider-required extra parameters on token requests.
    pub token_extra: Vec<(String, String)>,
}

/// An issued OAuth token set; the raw response is preserved for
/// provider-specific fields (account id, tenant, …).
#[derive(Clone)]
pub struct OAuthToken {
    pub access_token: Secret,
    pub refresh_token: Option<Secret>,
    pub token_type: String,
    pub expires_at: Option<SystemTime>,
    pub scope: Option<String>,
    pub raw: Value,
}

impl OAuthToken {
    /// View as a bearer `Credential` for adapter resolvers.
    pub fn credential(&self) -> Credential {
        Credential::Bearer {
            token: self.access_token.clone(),
            expires_at: self.expires_at,
        }
    }
}

/// Device grant issued by `device_authorize`: the pair shown to the
/// user plus the polling contract.
#[derive(Clone, Debug)]
pub struct DeviceGrant {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: url::Url,
    /// URL embedding the user code when the provider supplies one —
    /// present this in preference to the bare URI.
    pub verification_uri_complete: Option<url::Url>,
    pub expires_at: SystemTime,
    /// Provider-declared poll interval; `slow_down` responses extend it.
    pub interval: Duration,
    pub raw: Value,
}

/// Fill `bytes` from `rng` and encode them URL-safe.
///
/// Entropy failure is returned, never panicked on: a caller that cannot
/// obtain random bytes must not fall back to a weaker verifier or state.
fn random_url_safe<R: TryRngCore>(rng: &mut R, bytes: usize) -> Result<String, Error> {
    let mut buf = vec![0u8; bytes];
    rng.try_fill_bytes(&mut buf)
        .map_err(|e| Error::Credential(format!("OS entropy source failed: {e}")))?;
    Ok(URL_SAFE_NO_PAD.encode(buf))
}

/// A PKCE verifier (keep secret) and its S256 challenge.
///
/// Returns [`Error::Credential`] when the OS entropy source cannot supply
/// bytes; no verifier is produced in that case.
pub fn pkce() -> Result<(Secret, String), Error> {
    pkce_with(&mut rand::rngs::OsRng)
}

fn pkce_with<R: TryRngCore>(rng: &mut R) -> Result<(Secret, String), Error> {
    let verifier = random_url_safe(rng, 48)?;
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    Ok((Secret::new(verifier), challenge))
}

/// Build the authorization URL for the code flow.
pub fn authorize_url(
    client: &OAuthClient,
    redirect_uri: &url::Url,
    state: &str,
    challenge: &str,
) -> Result<url::Url, Error> {
    let mut url = client.authorize_url.clone().ok_or_else(|| {
        Error::Unsupported("provider has no authorization-code endpoint".into())
    })?;
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("response_type", "code");
        q.append_pair("client_id", &client.client_id);
        q.append_pair("redirect_uri", redirect_uri.as_str());
        q.append_pair("state", state);
        q.append_pair("code_challenge", challenge);
        q.append_pair("code_challenge_method", "S256");
        if let Some(scope) = &client.scope {
            q.append_pair("scope", scope);
        }
        for (k, v) in &client.authorize_extra {
            q.append_pair(k, v);
        }
    }
    Ok(url)
}

fn form_body(pairs: &[(String, String)]) -> bytes::Bytes {
    let body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .finish();
    bytes::Bytes::from(body)
}

async fn post_form(
    url: &url::Url,
    pairs: &[(String, String)],
    transport: &dyn Transport,
    ctrl: &RequestControl,
) -> Result<Value, Error> {
    let mut req = HttpRequest::post_json(url.clone(), Value::Null);
    req.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/x-www-form-urlencoded"),
    );
    req.body = form_body(pairs);
    let res = transport.send(req, ctrl).await?;
    let body: Value = serde_json::from_slice(&res.body)
        .map_err(|e| Error::UpstreamProtocol(format!("token endpoint returned non-JSON: {e}")))?;
    Ok(body)
}

fn form_pairs(client: &OAuthClient, grant: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(grant.len() + 4);
    for (k, v) in grant {
        pairs.push((k.to_string(), v.to_string()));
    }
    pairs.push(("client_id".into(), client.client_id.clone()));
    if let Some(secret) = &client.client_secret {
        pairs.push(("client_secret".into(), secret.expose().to_owned()));
    }
    pairs.extend(client.token_extra.iter().cloned());
    pairs
}

fn decode_token(body: &Value) -> Result<OAuthToken, Error> {
    let access = body
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::UpstreamProtocol("token response missing access_token".into()))?;
    let expires_at = body
        .get("expires_in")
        .and_then(Value::as_u64)
        .map(|s| SystemTime::now() + Duration::from_secs(s));
    Ok(OAuthToken {
        access_token: Secret::new(access),
        refresh_token: body
            .get("refresh_token")
            .and_then(Value::as_str)
            .map(Secret::new),
        token_type: body
            .get("token_type")
            .and_then(Value::as_str)
            .unwrap_or("Bearer")
            .to_owned(),
        expires_at,
        scope: body
            .get("scope")
            .and_then(Value::as_str)
            .map(str::to_owned),
        raw: body.clone(),
    })
}

/// Map an OAuth error response: pending-ish states return `None` so the
/// caller keeps polling; terminal states surface.
fn oauth_error(body: &Value) -> Option<Error> {
    let code = body.get("error").and_then(Value::as_str)?;
    let desc = body
        .get("error_description")
        .and_then(Value::as_str)
        .unwrap_or(code);
    match code {
        // Non-terminal polling states.
        "authorization_pending" | "slow_down" => None,
        "access_denied" | "expired_token" => Some(Error::Credential(desc.to_owned())),
        _ => Some(Error::Provider {
            status: None,
            code: Some(code.to_owned()),
            message: desc.to_owned(),
            retry: crate::api::RetrySafety::Safe,
            raw: Some(body.clone()),
        }),
    }
}

/// Device-authorization step: obtain the device/user code pair.
pub async fn device_authorize(
    client: &OAuthClient,
    transport: &dyn Transport,
    ctrl: &RequestControl,
) -> Result<DeviceGrant, Error> {
    let url = client.device_url.clone().ok_or_else(|| {
        Error::Unsupported("provider has no device-authorization endpoint".into())
    })?;
    let mut pairs = vec![("client_id".to_owned(), client.client_id.clone())];
    if let Some(scope) = &client.scope {
        pairs.push(("scope".into(), scope.clone()));
    }
    pairs.extend(client.authorize_extra.iter().cloned());
    let body = post_form(&url, &pairs, transport, ctrl).await?;
    if let Some(e) = oauth_error(&body) {
        return Err(e);
    }
    let get_str = |k: &str| -> Result<String, Error> {
        body.get(k)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| Error::UpstreamProtocol(format!("device grant missing {k}")))
    };
    let parse_url = |s: &str| -> Result<url::Url, Error> {
        url::Url::parse(s).map_err(|e| Error::UpstreamProtocol(format!("bad verification URI: {e}")))
    };
    Ok(DeviceGrant {
        device_code: get_str("device_code")?,
        user_code: get_str("user_code")?,
        verification_uri: parse_url(&get_str("verification_uri")?)?,
        verification_uri_complete: body
            .get("verification_uri_complete")
            .and_then(Value::as_str)
            .map(parse_url)
            .transpose()?,
        expires_at: SystemTime::now()
            + Duration::from_secs(
                body.get("expires_in").and_then(Value::as_u64).unwrap_or(600),
            ),
        interval: Duration::from_secs(
            body.get("interval").and_then(Value::as_u64).unwrap_or(5).max(1),
        ),
        raw: body,
    })
}

/// Complete device flow: authorize, present the URL, poll to a token.
/// Polling runs to the provider's expiry — there is no internal
/// attempt cap; slow_down widens the interval per spec.
pub async fn device_flow(
    client: &OAuthClient,
    transport: &dyn Transport,
    effects: &dyn EffectHost,
    ctrl: &RequestControl,
) -> Result<OAuthToken, Error> {
    let grant = device_authorize(client, transport, ctrl).await?;
    effects
        .present_url(
            grant
                .verification_uri_complete
                .as_ref()
                .unwrap_or(&grant.verification_uri),
            ctrl,
        )
        .await?;
    let mut interval = grant.interval;
    loop {
        if ctrl.cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if SystemTime::now() >= grant.expires_at {
            return Err(Error::Credential("device grant expired".into()));
        }
        let pairs = form_pairs(
            client,
            &[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("device_code", &grant.device_code),
            ],
        );
        let body = post_form(&client.token_url, &pairs, transport, ctrl).await?;
        match body.get("error").and_then(Value::as_str) {
            Some("authorization_pending") => {}
            Some("slow_down") => interval += Duration::from_secs(5),
            Some(_) => return Err(oauth_error(&body).unwrap()),
            None => return decode_token(&body),
        }
        effects.sleep(interval, ctrl).await?;
    }
}

/// Complete authorization-code + PKCE flow through the interaction port.
///
/// Entropy failure is returned as [`Error::Credential`] before an
/// authorization URL is presented.
pub async fn code_flow(
    client: &OAuthClient,
    transport: &dyn Transport,
    effects: &dyn EffectHost,
    ctrl: &RequestControl,
) -> Result<OAuthToken, Error> {
    let mut rng = rand::rngs::OsRng;
    code_flow_with_rng(client, transport, effects, ctrl, &mut rng).await
}

async fn code_flow_with_rng<R: TryRngCore>(
    client: &OAuthClient,
    transport: &dyn Transport,
    effects: &dyn EffectHost,
    ctrl: &RequestControl,
    rng: &mut R,
) -> Result<OAuthToken, Error> {
    let redirect = effects.redirect_uri().ok_or_else(|| {
        Error::Unsupported("host cannot capture auth callbacks".into())
    })?;
    let (verifier, challenge) = pkce_with(&mut *rng)?;
    let state = random_url_safe(&mut *rng, 24)?;
    let url = authorize_url(client, &redirect, &state, &challenge)?;
    effects.present_url(&url, ctrl).await?;
    let callback = effects.await_callback(ctrl).await?;
    let parsed = url::Url::parse(&callback)
        .map_err(|e| Error::Protocol(format!("callback is not a URL: {e}")))?;
    let mut code = None;
    let mut got_state = None;
    let mut err = None;
    for (k, v) in parsed.query_pairs() {
        match k.as_ref() {
            "code" => code = Some(v.into_owned()),
            "state" => got_state = Some(v.into_owned()),
            "error" => err = Some(v.into_owned()),
            _ => {}
        }
    }
    if let Some(e) = err {
        return Err(Error::Credential(format!("authorization failed: {e}")));
    }
    if got_state.as_deref() != Some(state.as_str()) {
        return Err(Error::Protocol("callback state mismatch".into()));
    }
    let code = code.ok_or_else(|| Error::Protocol("callback missing code".into()))?;
    let pairs = form_pairs(
        client,
        &[
            ("grant_type", "authorization_code"),
            ("code", &code),
            ("redirect_uri", redirect.as_str()),
            ("code_verifier", verifier.expose()),
        ],
    );
    let body = post_form(&client.token_url, &pairs, transport, ctrl).await?;
    if let Some(e) = oauth_error(&body) {
        return Err(e);
    }
    decode_token(&body)
}

/// Refresh an issued token; the host stores the result.
pub async fn refresh(
    client: &OAuthClient,
    token: &OAuthToken,
    transport: &dyn Transport,
    ctrl: &RequestControl,
) -> Result<OAuthToken, Error> {
    let refresh_token = token
        .refresh_token
        .as_ref()
        .ok_or_else(|| Error::Credential("token has no refresh_token".into()))?;
    let pairs = form_pairs(
        client,
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token.expose()),
        ],
    );
    let body = post_form(&client.token_url, &pairs, transport, ctrl).await?;
    if let Some(e) = oauth_error(&body) {
        return Err(e);
    }
    let mut new_token = decode_token(&body)?;
    // Providers may omit a rotated refresh token; keep the prior one.
    if new_token.refresh_token.is_none() {
        new_token.refresh_token = token.refresh_token.clone();
    }
    Ok(new_token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{HttpResponse, StreamingResponse, TransportError};

    #[derive(Debug)]
    struct EntropyUnavailable;

    impl std::fmt::Display for EntropyUnavailable {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("simulated OS entropy failure")
        }
    }

    impl std::error::Error for EntropyUnavailable {}

    struct FailingRng;

    impl rand::TryRngCore for FailingRng {
        type Error = EntropyUnavailable;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            Err(EntropyUnavailable)
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            Err(EntropyUnavailable)
        }

        fn try_fill_bytes(&mut self, _dst: &mut [u8]) -> Result<(), Self::Error> {
            Err(EntropyUnavailable)
        }
    }

    struct FixedRng;

    impl rand::TryRngCore for FixedRng {
        type Error = std::convert::Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            Ok(0)
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            Ok(0)
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
            dst.fill(b'a');
            Ok(())
        }
    }

    struct NoIo;

    #[async_trait::async_trait]
    impl Transport for NoIo {
        async fn send(
            &self,
            _req: HttpRequest,
            _ctrl: &RequestControl,
        ) -> Result<HttpResponse, TransportError> {
            panic!("transport must not be reached when entropy generation fails")
        }

        async fn send_stream(
            &self,
            _req: HttpRequest,
            _ctrl: &RequestControl,
        ) -> Result<StreamingResponse, TransportError> {
            panic!("transport must not be reached when entropy generation fails")
        }
    }

    struct NoEffects;

    #[async_trait::async_trait]
    impl EffectHost for NoEffects {
        fn redirect_uri(&self) -> Option<url::Url> {
            Some(url::Url::parse("http://127.0.0.1:8765/callback").unwrap())
        }

        async fn present_url(&self, _url: &url::Url, _ctrl: &RequestControl) -> Result<(), Error> {
            panic!("authorization URL must not be presented when entropy generation fails")
        }

        async fn await_callback(&self, _ctrl: &RequestControl) -> Result<String, Error> {
            panic!("callback must not be awaited when entropy generation fails")
        }
    }

    fn client() -> OAuthClient {
        OAuthClient {
            client_id: "a4mp-test".into(),
            client_secret: None,
            authorize_url: Some(url::Url::parse("https://auth.example/authorize").unwrap()),
            device_url: None,
            token_url: url::Url::parse("https://auth.example/token").unwrap(),
            scope: None,
            authorize_extra: vec![],
            token_extra: vec![],
        }
    }

    fn assert_entropy_failure(err: Error) {
        match err {
            Error::Credential(message) => {
                assert!(
                    message.contains("entropy"),
                    "expected an entropy message, got {message}"
                );
            }
            other => panic!("expected Credential entropy error, got {other:?}"),
        }
    }

    #[test]
    fn random_url_safe_reports_entropy_failure() {
        let mut rng = FailingRng;
        match random_url_safe(&mut rng, 32) {
            Ok(value) => panic!("entropy failure must not produce {value:?}"),
            Err(err) => assert_entropy_failure(err),
        }
    }

    #[test]
    fn pkce_reports_entropy_failure() {
        let mut rng = FailingRng;
        match pkce_with(&mut rng) {
            Ok(_) => panic!("entropy failure must not produce a verifier"),
            Err(err) => assert_entropy_failure(err),
        }
    }

    #[test]
    fn pkce_with_derives_s256_challenge() {
        let mut rng = FixedRng;
        let (verifier, challenge) = pkce_with(&mut rng).unwrap();
        let expected_verifier = URL_SAFE_NO_PAD.encode([b'a'; 48]);
        assert_eq!(verifier.expose(), expected_verifier);
        assert_eq!(
            challenge,
            URL_SAFE_NO_PAD.encode(Sha256::digest(expected_verifier.as_bytes()))
        );
    }

    #[tokio::test]
    async fn code_flow_propagates_entropy_failure() {
        let mut rng = FailingRng;
        let result = code_flow_with_rng(
            &client(),
            &NoIo,
            &NoEffects,
            &RequestControl::default(),
            &mut rng,
        )
        .await;
        match result {
            Ok(_) => panic!("entropy failure must not produce a token"),
            Err(err) => assert_entropy_failure(err),
        }
    }
}
