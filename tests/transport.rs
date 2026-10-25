//! Transport contract tests that exercise a real socket: cancellation after
//! the request has been dispatched must not be classified as safe to retry.

#![cfg(feature = "reqwest-transport")]

use std::time::Duration;

use a4mp::transport::{HttpRequest, ReqwestTransport, Transport, TransportError};
use a4mp::{RequestControl, RetrySafety};

#[tokio::test]
async fn cancellation_after_dispatch_is_unsafe_to_retry() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (accepted, accepted_rx) = tokio::sync::oneshot::channel();

    tokio::spawn(async move {
        let (_conn, _) = listener.accept().await.unwrap();
        let _ = accepted.send(());
        // Hold the connection open without responding: the request has been
        // dispatched and the transport future remains pending.
        tokio::time::sleep(Duration::from_secs(30)).await;
    });

    let transport =
        ReqwestTransport::with_client(reqwest::Client::builder().no_proxy().build().unwrap());
    let ctrl = RequestControl::default();
    let cancel = ctrl.cancel.clone();
    let req = HttpRequest::get(url::Url::parse(&format!("http://{addr}/")).unwrap());
    let send = tokio::spawn(async move { transport.send(req, &ctrl).await });

    // `accept` only completes after the send future was polled and the
    // request reached the server, so cancellation below races a dispatched
    // request.
    tokio::time::timeout(Duration::from_secs(5), accepted_rx)
        .await
        .expect("request dispatch timed out")
        .expect("accept task ended before dispatch");
    cancel.cancel();

    let err = tokio::time::timeout(Duration::from_secs(5), send)
        .await
        .expect("cancelled send did not settle")
        .expect("send task panicked")
        .expect_err("a cancelled request must return an error");
    assert!(matches!(err, TransportError::Cancelled), "{err:?}");
    assert_eq!(err.retry_safety(), RetrySafety::Unsafe);
}
