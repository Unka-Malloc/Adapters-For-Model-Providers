//! Provider-specific OAuth token brokers.
//!
//! Subscription-backed providers layer a proprietary session exchange on
//! top of a standard OAuth grant: GitHub Copilot trades a GitHub
//! device-flow token for a short-lived Copilot session token, and the
//! OpenAI Codex client binds an authorization-code token to a ChatGPT
//! account id. The client registrations, request parameters and
//! exchanges implemented here are the publicly documented behavior of
//! the providers' official first-party clients. Issued material stays
//! in host custody, as with every flow in this crate.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::Value;

use crate::api::{Error, RequestControl, RetrySafety};
use crate::auth::{Credential, Secret};
use crate::effects::EffectHost;
use crate::oauth::{self, DeviceGrant, OAuthClient, OAuthToken};
use crate::transport::{HttpRequest, Transport};

fn const_url(text: &str) -> url::Url {
    url::Url::parse(text).expect("broker endpoint constants are valid URLs")
}

/// Read a claim from a JWT payload. The signature is never verified —
/// claims feed routing headers and account identity, never an
/// authentication decision. `key_path` walks nested objects; an absent
/// path yields `Ok(Value::Null)`.
pub fn jwt_claim(token: &Secret, key_path: &[&str]) -> Result<Value, Error> {
    let payload = token
        .expose()
        .split('.')
        .nth(1)
        .ok_or_else(|| Error::Protocol("token is not a JWT".into()))?;
    let bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|e| Error::Protocol(format!("JWT payload is not base64url: {e}")))?;
    let body: Value = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Protocol(format!("JWT payload is not JSON: {e}")))?;
    let mut claim = &body;
    for key in key_path {
        match claim.get(*key) {
            Some(next) => claim = next,
            None => return Ok(Value::Null),
        }
    }
    Ok(claim.clone())
}

// ---------------------------------------------------------------------------
// GitHub Copilot
// ---------------------------------------------------------------------------

/// `api.github.com` refuses requests without a `User-Agent`; the broker
/// identifies as the documented Copilot plugin build.
const COPILOT_USER_AGENT: &str = "copilot/1.250.0";

/// A Copilot session token minted for `api.githubcopilot.com`.
#[derive(Clone)]
pub struct CopilotToken {
    /// Session token sent as `Authorization: Bearer`.
    pub token: Secret,
    /// Absolute expiry reported by GitHub (`expires_at`, unix seconds).
    pub expires_at: SystemTime,
    /// Provider response preserved verbatim (`refresh_in`, annotations).
    pub raw: Value,
}

/// The public GitHub OAuth App registration first-party Copilot clients
/// use for the device grant (`Iv1.*` is GitHub's OAuth App client-id
/// namespace). Device grant only; this client has no redirect surface.
pub fn copilot_oauth_client() -> OAuthClient {
    OAuthClient {
        client_id: "Iv1.b507a08c87ecfe98".into(),
        client_secret: None,
        authorize_url: None,
        device_url: Some(const_url("https://github.com/login/device/code")),
        token_url: const_url("https://github.com/login/oauth/access_token"),
        scope: Some("read:user".into()),
        authorize_extra: vec![],
        token_extra: vec![],
    }
}

/// Form post to a GitHub OAuth endpoint. GitHub negotiates the response
/// format on `Accept` and defaults to `application/x-www-form-urlencoded`,
/// so the grant asks for JSON explicitly; the shared
/// [`crate::oauth`] post helper cannot carry the header, hence this
/// local variant.
async fn github_post_form(
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
    req.headers.insert(
        http::header::ACCEPT,
        http::HeaderValue::from_static("application/json"),
    );
    req.body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .finish()
        .into();
    let res = transport.send(req, ctrl).await?;
    serde_json::from_slice(&res.body)
        .map_err(|e| Error::UpstreamProtocol(format!("token endpoint returned non-JSON: {e}")))
}

/// Map an OAuth error payload: polling states return `None` so the
/// caller keeps going; terminal states surface.
fn github_oauth_error(body: &Value) -> Option<Error> {
    let code = body.get("error").and_then(Value::as_str)?;
    let desc = body
        .get("error_description")
        .and_then(Value::as_str)
        .unwrap_or(code);
    match code {
        "authorization_pending" | "slow_down" => None,
        "access_denied" | "expired_token" => Some(Error::Credential(desc.to_owned())),
        _ => Some(Error::Provider {
            status: None,
            code: Some(code.to_owned()),
            message: desc.to_owned(),
            retry: RetrySafety::Safe,
            raw: Some(body.clone()),
        }),
    }
}

fn github_decode_token(body: &Value) -> Result<OAuthToken, Error> {
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

/// Device-authorization step against GitHub: same contract as
/// [`crate::oauth::device_authorize`] through the JSON-negotiating post.
async fn github_device_authorize(
    client: &OAuthClient,
    transport: &dyn Transport,
    ctrl: &RequestControl,
) -> Result<DeviceGrant, Error> {
    let url = client.device_url.clone().ok_or_else(|| {
        Error::Unsupported("client has no device-authorization endpoint".into())
    })?;
    let mut pairs = vec![("client_id".to_owned(), client.client_id.clone())];
    if let Some(scope) = &client.scope {
        pairs.push(("scope".into(), scope.clone()));
    }
    pairs.extend(client.authorize_extra.iter().cloned());
    let body = github_post_form(&url, &pairs, transport, ctrl).await?;
    if let Some(e) = github_oauth_error(&body) {
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
            + Duration::from_secs(body.get("expires_in").and_then(Value::as_u64).unwrap_or(600)),
        interval: Duration::from_secs(
            body.get("interval").and_then(Value::as_u64).unwrap_or(5).max(1),
        ),
        raw: body,
    })
}

/// Run the GitHub device grant for the Copilot OAuth App: authorize,
/// present the verification URL through `effects`, then poll to a token.
/// Polling honors `authorization_pending`/`slow_down` until the grant
/// expires; the caller cancels through `ctrl`.
pub async fn copilot_device_flow(
    transport: &dyn Transport,
    effects: &dyn EffectHost,
    ctrl: &RequestControl,
) -> Result<OAuthToken, Error> {
    let client = copilot_oauth_client();
    let grant = github_device_authorize(&client, transport, ctrl).await?;
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
        let pairs = vec![
            (
                "grant_type".to_owned(),
                "urn:ietf:params:oauth:grant-type:device_code".to_owned(),
            ),
            ("device_code".to_owned(), grant.device_code.clone()),
            ("client_id".to_owned(), client.client_id.clone()),
        ];
        let body = github_post_form(&client.token_url, &pairs, transport, ctrl).await?;
        match body.get("error").and_then(Value::as_str) {
            Some("authorization_pending") => {}
            Some("slow_down") => interval += Duration::from_secs(5),
            Some(_) => return Err(github_oauth_error(&body).unwrap()),
            None => return github_decode_token(&body),
        }
        effects.sleep(interval, ctrl).await?;
    }
}

/// Exchange a GitHub OAuth token for a Copilot session token. The
/// `copilot_internal` endpoint answers only for accounts holding an
/// active Copilot subscription: a `401` means the GitHub token itself
/// was rejected (`Error::Credential` — re-run the device grant); other
/// rejections surface as `Error::Provider` with GitHub's payload.
pub async fn copilot_session(
    github_token: &Secret,
    transport: &dyn Transport,
    ctrl: &RequestControl,
) -> Result<CopilotToken, Error> {
    let mut req = HttpRequest::get(const_url(
        "https://api.github.com/copilot_internal/v2/token",
    ));
    req.headers.extend(copilot_headers());
    req.headers.insert(
        http::header::USER_AGENT,
        http::HeaderValue::from_static(COPILOT_USER_AGENT),
    );
    req.headers.insert(
        http::header::ACCEPT,
        http::HeaderValue::from_static("application/json"),
    );
    req.headers.insert(
        http::header::AUTHORIZATION,
        http::HeaderValue::from_str(&format!("token {}", github_token.expose()))
            .map_err(|_| Error::Protocol("github token cannot be a header value".into()))?,
    );
    let res = transport.send(req, ctrl).await?;
    if !res.status.is_success() {
        let body: Option<Value> = serde_json::from_slice(&res.body).ok();
        let message = body
            .as_ref()
            .and_then(|b| b.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("copilot session request rejected")
            .to_owned();
        return Err(if res.status == http::StatusCode::UNAUTHORIZED {
            Error::Credential(format!("github token rejected: {message}"))
        } else {
            Error::Provider {
                status: Some(res.status.as_u16()),
                code: body
                    .as_ref()
                    .and_then(|b| b.get("error"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                message,
                retry: RetrySafety::Safe,
                raw: body,
            }
        });
    }
    let body: Value = serde_json::from_slice(&res.body)
        .map_err(|e| Error::UpstreamProtocol(format!("copilot session response is not JSON: {e}")))?;
    let token = body
        .get("token")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::UpstreamProtocol("copilot session response missing token".into()))?;
    let expires_at = body
        .get("expires_at")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::UpstreamProtocol("copilot session response missing expires_at".into()))?;
    Ok(CopilotToken {
        token: Secret::new(token),
        expires_at: UNIX_EPOCH + Duration::from_secs(expires_at),
        raw: body,
    })
}

/// The Copilot session token viewed as an adapter credential.
pub fn copilot_credential(token: &CopilotToken) -> Credential {
    Credential::Bearer {
        token: token.token.clone(),
        expires_at: Some(token.expires_at),
    }
}

/// Headers `api.githubcopilot.com` requires beyond the bearer token:
/// the editor/plugin identification Copilot's API is documented to
/// expect, plus the integration id first-party clients send.
pub fn copilot_headers() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        "editor-version",
        http::HeaderValue::from_static("vscode/1.95.0"),
    );
    headers.insert(
        "editor-plugin-version",
        http::HeaderValue::from_static("copilot/1.250.0"),
    );
    headers.insert(
        "copilot-integration-id",
        http::HeaderValue::from_static("vscode-chat"),
    );
    headers
}

/// Complete Copilot sign-in: GitHub device grant, then the session
/// exchange. Returns both — the OAuth token is the long-lived secret
/// the host keeps for re-minting; the session token is the short-lived
/// API credential.
pub async fn copilot_login(
    transport: &dyn Transport,
    effects: &dyn EffectHost,
    ctrl: &RequestControl,
) -> Result<(OAuthToken, CopilotToken), Error> {
    let github = copilot_device_flow(transport, effects, ctrl).await?;
    let session = copilot_session(&github.access_token, transport, ctrl).await?;
    Ok((github, session))
}

// ---------------------------------------------------------------------------
// OpenAI Codex (ChatGPT account)
// ---------------------------------------------------------------------------

/// Claim paths `chatgpt_account_id` occupies in the issued JWTs —
/// top-level, or inside the namespaced auth claim.
const CHATGPT_ACCOUNT_ID_PATHS: &[&[&str]] = &[
    &["chatgpt_account_id"],
    &["https://api.openai.com/auth", "chatgpt_account_id"],
];

/// The public OAuth client registration the official Codex CLI uses:
/// an authorization-code + PKCE client whose registered redirect is the
/// fixed loopback callback [`codex_redirect_uri`]. The extra authorize
/// parameters are the documented Codex CLI flow markers.
pub fn codex_oauth_client() -> OAuthClient {
    OAuthClient {
        client_id: "app_EMoamEEZ73f0CkXaXp7hrann".into(),
        client_secret: None,
        authorize_url: Some(const_url("https://auth.openai.com/oauth/authorize")),
        device_url: None,
        token_url: const_url("https://auth.openai.com/oauth/token"),
        scope: Some("openid profile email offline_access".into()),
        authorize_extra: vec![
            ("codex-cli-simplified-flow".into(), "true".into()),
            ("originator".into(), "codex_cli_rs".into()),
            ("id_token_add_organizations".into(), "true".into()),
        ],
        token_extra: vec![],
    }
}

/// The loopback callback the Codex client registration declares. The
/// host's [`EffectHost::redirect_uri`] must expose this URI for the
/// authorization server to accept the code flow.
pub fn codex_redirect_uri() -> url::Url {
    const_url("http://localhost:1455/auth/callback")
}

/// Extract the ChatGPT account id from an issued Codex token. Both the
/// access token and the `id_token` are JWTs carrying the claim at one of
/// the documented locations; the access token is checked first. A token
/// carrying no claim is a protocol violation — Codex requests cannot be
/// routed without the account id.
pub fn codex_account_id(token: &OAuthToken) -> Result<String, Error> {
    let mut candidates = vec![token.access_token.clone()];
    if let Some(id_token) = token.raw.get("id_token").and_then(Value::as_str) {
        candidates.push(Secret::new(id_token));
    }
    for secret in &candidates {
        for path in CHATGPT_ACCOUNT_ID_PATHS {
            if let Ok(claim) = jwt_claim(secret, path) {
                if let Some(id) = claim.as_str() {
                    return Ok(id.to_owned());
                }
            }
        }
    }
    Err(Error::UpstreamProtocol(
        "codex token carries no chatgpt_account_id claim".into(),
    ))
}

/// Headers `chatgpt.com/backend-api/codex` requires beyond the bearer
/// token: the account routing header plus the Responses API beta flag.
pub fn codex_headers(account_id: &str) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    // Provider-issued account ids are ASCII claim strings.
    if let Ok(value) = http::HeaderValue::from_str(account_id) {
        headers.insert("chatgpt-account-id", value);
    }
    headers.insert(
        "openai-beta",
        http::HeaderValue::from_static("responses=experimental"),
    );
    headers
}

/// Complete Codex sign-in: authorization-code + PKCE through the host's
/// loopback capture, then account-id extraction. The host must expose
/// [`codex_redirect_uri`] from [`EffectHost::redirect_uri`].
pub async fn codex_login(
    transport: &dyn Transport,
    effects: &dyn EffectHost,
    ctrl: &RequestControl,
) -> Result<(OAuthToken, String), Error> {
    let token = oauth::code_flow(&codex_oauth_client(), transport, effects, ctrl).await?;
    let account_id = codex_account_id(&token)?;
    Ok((token, account_id))
}
