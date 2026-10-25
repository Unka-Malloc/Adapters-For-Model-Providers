//! Broker tests: GitHub device-grant → Copilot session exchange, and
//! the Codex code-flow → ChatGPT account-id extraction — all through
//! scripted transport and a stub effect host. JWT fixtures are
//! fabricated in-test with real payload encodings.

mod common;

use std::sync::Mutex;
use std::time::{Duration, UNIX_EPOCH};

use a4mp::broker::*;
use a4mp::oauth::OAuthToken;
use a4mp::*;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use common::{MockTransport, Response};

fn ctrl() -> RequestControl {
    RequestControl::default()
}

/// A JWT with `alg: none`: the payload is genuinely base64url-encoded
/// so claim decoding runs against a real wire shape; the signature is
/// never verified anywhere in this crate.
fn fake_jwt(payload: serde_json::Value) -> Secret {
    let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"none","typ":"JWT"}"#);
    let body = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap());
    Secret::new(format!("{header}.{body}.c2ln"))
}

fn oauth_token(access: &Secret, raw: serde_json::Value) -> OAuthToken {
    OAuthToken {
        access_token: access.clone(),
        refresh_token: None,
        token_type: "Bearer".into(),
        expires_at: None,
        scope: None,
        raw,
    }
}

/// Scripted effect host: records presented URLs, rebuilds the redirect
/// callback from the `redirect_uri`/`state` params inside the presented
/// authorize URL (the loopback listener's real behavior), and counts
/// poll sleeps.
#[derive(Default)]
struct StubEffects {
    presented: Mutex<Vec<url::Url>>,
    sleeps: Mutex<u32>,
}

#[async_trait::async_trait]
impl EffectHost for StubEffects {
    fn redirect_uri(&self) -> Option<url::Url> {
        Some(codex_redirect_uri())
    }

    async fn present_url(&self, url: &url::Url, _ctrl: &RequestControl) -> Result<(), Error> {
        self.presented.lock().unwrap().push(url.clone());
        Ok(())
    }

    async fn await_callback(&self, _ctrl: &RequestControl) -> Result<String, Error> {
        let presented = self.presented.lock().unwrap();
        let url = presented.first().unwrap().clone();
        let param = |name: &str| {
            url.query_pairs()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.into_owned())
                .unwrap()
        };
        Ok(format!(
            "{}?code=codex-auth-code&state={}",
            param("redirect_uri"),
            param("state")
        ))
    }

    async fn sleep(&self, _d: Duration, _ctrl: &RequestControl) -> Result<(), Error> {
        *self.sleeps.lock().unwrap() += 1;
        Ok(())
    }
}

fn header<'a>(req: &'a common::RecordedRequest, name: &str) -> &'a str {
    req.headers
        .get(name)
        .unwrap_or_else(|| panic!("request to {} missing header {name}", req.url))
        .to_str()
        .unwrap()
}

#[tokio::test]
async fn copilot_login_runs_device_grant_into_session_token() {
    let transport = MockTransport::new();
    transport
        // Device authorization.
        .push(Response::Json(
            200,
            serde_json::json!({
                "device_code": "dev-gh-1", "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "verification_uri_complete": "https://github.com/login/device?user_code=ABCD-EFGH",
                "expires_in": 900, "interval": 0
            }),
        ))
        // One pending poll, then the GitHub access token.
        .push(Response::Json(
            200,
            serde_json::json!({"error": "authorization_pending"}),
        ))
        .push(Response::Json(
            200,
            serde_json::json!({
                "access_token": "gho_copilot", "token_type": "bearer", "scope": "read:user"
            }),
        ))
        // Copilot session exchange.
        .push(Response::Json(
            200,
            serde_json::json!({
                "token": "tid=copilot-session", "expires_at": 1_800_000_000,
                "refresh_in": 1500
            }),
        ));
    let effects = StubEffects::default();

    let (github, copilot) = copilot_login(&transport, &effects, &ctrl()).await.unwrap();

    assert_eq!(github.access_token.expose(), "gho_copilot");
    assert_eq!(copilot.token.expose(), "tid=copilot-session");
    assert_eq!(
        copilot.expires_at,
        UNIX_EPOCH + Duration::from_secs(1_800_000_000)
    );
    assert_eq!(copilot.raw["refresh_in"], 1500);

    // Presented the complete verification URI; slept once through the
    // pending poll.
    assert_eq!(
        effects.presented.lock().unwrap()[0].as_str(),
        "https://github.com/login/device?user_code=ABCD-EFGH"
    );
    assert_eq!(*effects.sleeps.lock().unwrap(), 1);

    let rec = transport.recorded();
    assert_eq!(rec.len(), 4);
    assert_eq!(rec[0].url.as_str(), "https://github.com/login/device/code");
    assert_eq!(rec[1].url.as_str(), "https://github.com/login/oauth/access_token");
    assert_eq!(rec[2].url.as_str(), "https://github.com/login/oauth/access_token");
    assert_eq!(
        rec[3].url.as_str(),
        "https://api.github.com/copilot_internal/v2/token"
    );

    // GitHub's OAuth endpoints negotiate on Accept; every form post
    // must request JSON.
    for req in &rec[..3] {
        assert_eq!(header(req, "accept"), "application/json");
        assert_eq!(header(req, "content-type"), "application/x-www-form-urlencoded");
    }
    let authorize_body = String::from_utf8(rec[0].body.clone()).unwrap();
    assert!(authorize_body.contains("client_id=Iv1.b507a08c87ecfe98"));
    assert!(authorize_body.contains("scope=read%3Auser"));
    let poll_body = String::from_utf8(rec[1].body.clone()).unwrap();
    assert!(poll_body.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"));
    assert!(poll_body.contains("device_code=dev-gh-1"));

    // The session exchange authenticates with `token <github_token>`
    // and the Copilot editor identification.
    let session = &rec[3];
    assert_eq!(session.method, http::Method::GET);
    assert_eq!(header(session, "authorization"), "token gho_copilot");
    assert_eq!(header(session, "accept"), "application/json");
    assert_eq!(header(session, "editor-version"), "vscode/1.95.0");
    assert_eq!(header(session, "editor-plugin-version"), "copilot/1.250.0");
}

#[tokio::test]
async fn copilot_session_exchange_sends_github_token_and_editor_headers() {
    let transport = MockTransport::new();
    transport.push(Response::Json(
        200,
        serde_json::json!({"token": "tid-x", "expires_at": 100, "refresh_in": 60}),
    ));

    let session = copilot_session(&Secret::new("gho_direct"), &transport, &ctrl())
        .await
        .unwrap();
    assert_eq!(session.token.expose(), "tid-x");
    assert_eq!(session.expires_at, UNIX_EPOCH + Duration::from_secs(100));

    let req = &transport.recorded()[0];
    assert_eq!(header(req, "authorization"), "token gho_direct");
    assert_eq!(header(req, "copilot-integration-id"), "vscode-chat");
    assert!(header(req, "user-agent").starts_with("copilot/"));
}

#[tokio::test]
async fn copilot_session_maps_rejections() {
    // 401 means the GitHub token itself was refused → re-authorize.
    let transport = MockTransport::new();
    transport.push(Response::Json(
        401,
        serde_json::json!({"message": "bad credentials"}),
    ));
    match copilot_session(&Secret::new("gho_bad"), &transport, &ctrl()).await {
        Err(Error::Credential(m)) => assert!(m.contains("bad credentials")),
        Err(other) => panic!("expected Credential error, got {other:?}"),
        Ok(_) => panic!("expected Credential error, got session token"),
    }

    // An account without Copilot gets a provider rejection.
    let transport = MockTransport::new();
    transport.push(Response::Json(
        403,
        serde_json::json!({"message": "no Copilot subscription"}),
    ));
    match copilot_session(&Secret::new("gho_free"), &transport, &ctrl()).await {
        Err(Error::Provider {
            status: Some(403),
            message,
            ..
        }) => assert!(message.contains("Copilot")),
        Err(other) => panic!("expected Provider error, got {other:?}"),
        Ok(_) => panic!("expected Provider error, got session token"),
    }
}

#[test]
fn copilot_headers_and_credential_exports() {
    let headers = copilot_headers();
    assert_eq!(headers["editor-version"], "vscode/1.95.0");
    assert_eq!(headers["editor-plugin-version"], "copilot/1.250.0");
    assert_eq!(headers["copilot-integration-id"], "vscode-chat");

    let session = CopilotToken {
        token: Secret::new("tid"),
        expires_at: UNIX_EPOCH + Duration::from_secs(42),
        raw: serde_json::json!({}),
    };
    match copilot_credential(&session) {
        Credential::Bearer { token, expires_at } => {
            assert_eq!(token.expose(), "tid");
            assert_eq!(expires_at, Some(UNIX_EPOCH + Duration::from_secs(42)));
        }
        other => panic!("expected Bearer credential, got {other:?}"),
    }
}

#[tokio::test]
async fn codex_login_runs_code_flow_and_extracts_account_id() {
    let access = fake_jwt(serde_json::json!({
        "sub": "user-1",
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-nested-1"}
    }));
    let transport = MockTransport::new();
    transport.push(Response::Json(
        200,
        serde_json::json!({
            "access_token": access.expose(), "token_type": "Bearer",
            "expires_in": 3600, "refresh_token": "rt-codex",
            "id_token": access.expose()
        }),
    ));
    let effects = StubEffects::default();

    let (token, account_id) = codex_login(&transport, &effects, &ctrl()).await.unwrap();
    assert_eq!(account_id, "acct-nested-1");
    assert_eq!(token.access_token.expose(), access.expose());
    assert_eq!(token.refresh_token.as_ref().unwrap().expose(), "rt-codex");

    // The authorize URL carries the documented Codex CLI flow params.
    let presented = &effects.presented.lock().unwrap()[0];
    assert_eq!(
        presented.as_str().split('?').next().unwrap(),
        "https://auth.openai.com/oauth/authorize"
    );
    let params: Vec<_> = presented.query_pairs().collect();
    let get = |k: &str| params.iter().find(|(n, _)| n == k).map(|(_, v)| v.to_string());
    assert_eq!(get("client_id").as_deref(), Some("app_EMoamEEZ73f0CkXaXp7hrann"));
    assert_eq!(
        get("redirect_uri").as_deref(),
        Some("http://localhost:1455/auth/callback")
    );
    assert_eq!(
        get("scope").as_deref(),
        Some("openid profile email offline_access")
    );
    assert_eq!(get("codex-cli-simplified-flow").as_deref(), Some("true"));
    assert_eq!(get("originator").as_deref(), Some("codex_cli_rs"));
    assert_eq!(get("code_challenge_method").as_deref(), Some("S256"));

    // The code exchange posted PKCE material to the Codex token endpoint.
    let rec = transport.recorded();
    assert_eq!(rec[0].url.as_str(), "https://auth.openai.com/oauth/token");
    let body = String::from_utf8(rec[0].body.clone()).unwrap();
    assert!(body.contains("grant_type=authorization_code"));
    assert!(body.contains("code=codex-auth-code"));
    assert!(body.contains("code_verifier="));
    assert!(body.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
}

#[test]
fn codex_account_id_reads_top_level_claim() {
    let access = fake_jwt(serde_json::json!({"chatgpt_account_id": "acct-top"}));
    assert_eq!(
        codex_account_id(&oauth_token(&access, serde_json::json!({}))).unwrap(),
        "acct-top"
    );
}

#[test]
fn codex_account_id_reads_nested_auth_claim() {
    let access = fake_jwt(serde_json::json!({
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-ns"}
    }));
    assert_eq!(
        codex_account_id(&oauth_token(&access, serde_json::json!({}))).unwrap(),
        "acct-ns"
    );
}

#[test]
fn codex_account_id_falls_back_to_id_token() {
    // An opaque access token yields nothing; the claim rides the id_token.
    let id = fake_jwt(serde_json::json!({"chatgpt_account_id": "acct-idtok"}));
    let token = oauth_token(
        &Secret::new("opaque-access"),
        serde_json::json!({"id_token": id.expose()}),
    );
    assert_eq!(codex_account_id(&token).unwrap(), "acct-idtok");
}

#[test]
fn codex_account_id_missing_claim_is_protocol_error() {
    let access = fake_jwt(serde_json::json!({"sub": "user-1"}));
    match codex_account_id(&oauth_token(&access, serde_json::json!({}))) {
        Err(Error::UpstreamProtocol(_)) => {}
        other => panic!("expected Protocol error, got {other:?}"),
    }
}

#[test]
fn codex_headers_set_account_and_beta_flag() {
    let headers = codex_headers("acct-9");
    assert_eq!(headers["chatgpt-account-id"], "acct-9");
    assert_eq!(headers["openai-beta"], "responses=experimental");
}

#[test]
fn jwt_claim_walks_paths_and_reports_missing() {
    let jwt = fake_jwt(serde_json::json!({
        "top": "v", "nested": {"deeper": {"leaf": 42}}
    }));
    assert_eq!(jwt_claim(&jwt, &["top"]).unwrap(), "v");
    assert_eq!(jwt_claim(&jwt, &["nested", "deeper", "leaf"]).unwrap(), 42);
    assert_eq!(
        jwt_claim(&jwt, &["nested", "absent"]).unwrap(),
        serde_json::Value::Null
    );

    // A non-JWT value is a protocol violation, not a silent null.
    match jwt_claim(&Secret::new("opaque"), &["top"]) {
        Err(Error::Protocol(_)) => {}
        other => panic!("expected Protocol error, got {other:?}"),
    }
}
