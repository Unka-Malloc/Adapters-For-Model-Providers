//! OAuth mechanics owned by the library: PKCE, authorization-code and
//! device flows, token exchange and refresh.
//!
//! Interaction (presenting the URL, capturing the callback, waiting
//! between polls) crosses `EffectHost`; issued-token custody stays with
//! the host's `CredentialResolver`. The library never opens a browser,
//! binds a listener or stores a token itself.

use std::time::{Duration, SystemTime};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;
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

fn random_url_safe(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    rand::rngs::OsRng.fill_bytes(&mut buf);
    URL_SAFE_NO_PAD.encode(buf)
}

/// A PKCE verifier (keep secret) and its S256 challenge.
pub fn pkce() -> (Secret, String) {
    let verifier = random_url_safe(48);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    (Secret::new(verifier), challenge)
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
        .map_err(|e| Error::Protocol(format!("token endpoint returned non-JSON: {e}")))?;
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
        .ok_or_else(|| Error::Protocol("token response missing access_token".into()))?;
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
            .ok_or_else(|| Error::Protocol(format!("device grant missing {k}")))
    };
    let parse_url = |s: &str| -> Result<url::Url, Error> {
        url::Url::parse(s).map_err(|e| Error::Protocol(format!("bad verification URI: {e}")))
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
pub async fn code_flow(
    client: &OAuthClient,
    transport: &dyn Transport,
    effects: &dyn EffectHost,
    ctrl: &RequestControl,
) -> Result<OAuthToken, Error> {
    let redirect = effects.redirect_uri().ok_or_else(|| {
        Error::Unsupported("host cannot capture auth callbacks".into())
    })?;
    let (verifier, challenge) = pkce();
    let state = random_url_safe(24);
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
