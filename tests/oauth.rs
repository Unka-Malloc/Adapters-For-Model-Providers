//! OAuth flow tests: device polling, PKCE code exchange, refresh —
//! all through mock transport and a scripted effect port.

mod common;

use std::sync::Mutex;
use std::time::Duration;

use a4mp::oauth::{self, OAuthClient};
use a4mp::*;
use common::{MockTransport, Response};

fn client() -> OAuthClient {
    OAuthClient {
        client_id: "a4mp-test".into(),
        client_secret: None,
        authorize_url: Some(url::Url::parse("https://auth.example/authorize").unwrap()),
        device_url: Some(url::Url::parse("https://auth.example/device").unwrap()),
        token_url: url::Url::parse("https://auth.example/token").unwrap(),
        scope: Some("models".into()),
        authorize_extra: vec![],
        token_extra: vec![],
    }
}

fn ctrl() -> RequestControl {
    RequestControl::default()
}

/// Scripted effect host: records presented URLs, returns a fabricated
/// callback embedding the state it captured from the authorize URL.
#[derive(Default)]
struct StubEffects {
    presented: Mutex<Vec<url::Url>>,
    sleeps: Mutex<u32>,
}

#[async_trait::async_trait]
impl EffectHost for StubEffects {
    fn redirect_uri(&self) -> Option<url::Url> {
        Some(url::Url::parse("http://127.0.0.1:8765/callback").unwrap())
    }

    async fn present_url(&self, url: &url::Url, _ctrl: &RequestControl) -> Result<(), Error> {
        self.presented.lock().unwrap().push(url.clone());
        Ok(())
    }

    async fn await_callback(&self, _ctrl: &RequestControl) -> Result<String, Error> {
        let state = self
            .presented
            .lock()
            .unwrap()
            .first()
            .and_then(|u| {
                u.query_pairs()
                    .find(|(k, _)| k == "state")
                    .map(|(_, v)| v.into_owned())
            })
            .unwrap();
        Ok(format!(
            "http://127.0.0.1:8765/callback?code=abc123&state={state}"
        ))
    }

    async fn sleep(&self, _d: Duration, _ctrl: &RequestControl) -> Result<(), Error> {
        *self.sleeps.lock().unwrap() += 1;
        Ok(())
    }
}

#[tokio::test]
async fn device_flow_polls_through_pending_to_token() {
    let transport = MockTransport::new();
    transport
        .push(Response::Json(
            200,
            serde_json::json!({
                "device_code": "dev-1", "user_code": "ABCD-EFGH",
                "verification_uri": "https://auth.example/activate",
                "verification_uri_complete": "https://auth.example/activate?code=ABCD-EFGH",
                "expires_in": 900, "interval": 0
            }),
        ))
        .push(Response::Json(
            200,
            serde_json::json!({"error": "authorization_pending"}),
        ))
        .push(Response::Json(
            200,
            serde_json::json!({
                "access_token": "tok-live", "token_type": "Bearer",
                "expires_in": 3600, "refresh_token": "ref-1"
            }),
        ));
    let effects = StubEffects::default();

    let token = oauth::device_flow(&client(), &transport, &effects, &ctrl())
        .await
        .unwrap();
    assert_eq!(token.access_token.expose(), "tok-live");
    assert_eq!(token.refresh_token.as_ref().unwrap().expose(), "ref-1");
    assert!(token.expires_at.is_some());

    // Presented the complete URI; polled once through a pending response.
    assert_eq!(
        effects.presented.lock().unwrap()[0].as_str(),
        "https://auth.example/activate?code=ABCD-EFGH"
    );
    assert_eq!(*effects.sleeps.lock().unwrap(), 1);

    let recorded = transport.recorded();
    assert_eq!(recorded.len(), 3);
    assert_eq!(recorded[0].url.path(), "/device");
    let poll_body = String::from_utf8(recorded[1].body.clone()).unwrap();
    assert!(poll_body.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"));
    assert!(poll_body.contains("device_code=dev-1"));
    assert!(poll_body.contains("client_id=a4mp-test"));
}

#[tokio::test]
async fn code_flow_exchanges_with_pkce_verifier() {
    let transport = MockTransport::new();
    transport.push(Response::Json(
        200,
        serde_json::json!({"access_token": "tok-code", "token_type": "Bearer"}),
    ));
    let effects = StubEffects::default();

    let token = oauth::code_flow(&client(), &transport, &effects, &ctrl())
        .await
        .unwrap();
    assert_eq!(token.access_token.expose(), "tok-code");

    let presented = &effects.presented.lock().unwrap()[0];
    let params: Vec<_> = presented.query_pairs().collect();
    let get = |k: &str| params.iter().find(|(n, _)| n == k).map(|(_, v)| v.to_string());
    assert_eq!(get("code_challenge_method").as_deref(), Some("S256"));
    assert!(get("code_challenge").is_some());
    assert!(get("state").is_some());
    assert_eq!(get("response_type").as_deref(), Some("code"));

    let body = String::from_utf8(transport.recorded()[0].body.clone()).unwrap();
    assert!(body.contains("grant_type=authorization_code"));
    assert!(body.contains("code=abc123"));
    assert!(body.contains("code_verifier="));
}

#[tokio::test]
async fn device_access_denied_is_credential_error() {
    let transport = MockTransport::new();
    transport
        .push(Response::Json(
            200,
            serde_json::json!({
                "device_code": "d", "user_code": "u",
                "verification_uri": "https://a.example/v", "expires_in": 60, "interval": 0
            }),
        ))
        .push(Response::Json(
            200,
            serde_json::json!({"error": "access_denied"}),
        ));
    let effects = StubEffects::default();
    match oauth::device_flow(&client(), &transport, &effects, &ctrl()).await {
        Err(Error::Credential(_)) => {}
        Err(other) => panic!("expected Credential error, got {other:?}"),
        Ok(_) => panic!("expected Credential error, got token"),
    }
}

#[tokio::test]
async fn refresh_keeps_prior_refresh_token_when_omitted() {
    let transport = MockTransport::new();
    transport.push(Response::Json(
        200,
        serde_json::json!({"access_token": "tok-new", "token_type": "Bearer", "expires_in": 60}),
    ));
    let old = oauth::OAuthToken {
        access_token: Secret::new("tok-old"),
        refresh_token: Some(Secret::new("ref-keep")),
        token_type: "Bearer".into(),
        expires_at: None,
        scope: None,
        raw: serde_json::json!({}),
    };
    let new = oauth::refresh(&client(), &old, &transport, &ctrl())
        .await
        .unwrap();
    assert_eq!(new.access_token.expose(), "tok-new");
    assert_eq!(new.refresh_token.as_ref().unwrap().expose(), "ref-keep");
    let body = String::from_utf8(transport.recorded()[0].body.clone()).unwrap();
    assert!(body.contains("grant_type=refresh_token"));
    assert!(body.contains("refresh_token=ref-keep"));
}
