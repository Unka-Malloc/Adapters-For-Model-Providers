//! Live verification against the local Ollama surface (gemma4:e4b).
//!
//! Pull the model into a running `ollama serve`, then:
//!
//!   A4MP_LIVE_ENDPOINT=<base URL of the ollama server> \
//!   cargo run --release --example live_local
//!
//! Covers the two surfaces a stock `ollama serve` exposes: the Ollama
//! native protocol and the OpenAI-compatible surface under `/v1`, both
//! reached through the runtime-supplied `A4MP_LIVE_ENDPOINT`. No
//! credentials, no proxy, no external cost.

use std::sync::Arc;

use a4mp::api::*;
use a4mp::auth::{AuthKind, NoCredentials};
use a4mp::providers::{OllamaAdapter, OpenAiAdapter};
use a4mp::transport::ReqwestTransport;
use futures_util::StreamExt;

fn target(protocol: Protocol, base: String) -> ServiceTarget {
    ServiceTarget::new(
        ProviderId::new("live-local"),
        protocol,
        base.parse().unwrap(),
    )
}

fn brief(out: &Output) -> String {
    let mut s = String::new();
    for p in &out.parts {
        match p {
            ContentPart::Text { text } => s.push_str(&format!("text({}B) ", text.len())),
            ContentPart::Reasoning(r) => {
                let n =
                    r.summary.as_deref().map(str::len).unwrap_or(0) + r.opaque.to_string().len();
                s.push_str(&format!("reasoning(~{n}B) "))
            }
            ContentPart::ToolCall(t) => s.push_str(&format!("tool:{} ", t.name)),
            _ => s.push_str("other "),
        }
    }
    format!("stop={:?} parts=[{}] usage={:?}", out.stop, s, out.usage)
}

fn req(model: &str, text: &str) -> ChatRequest {
    ChatRequest {
        model: model.into(),
        messages: vec![Message::text(Role::User, text)],
        tools: vec![],
        tool_choice: None,
        parallel_tool_calls: None,
        sampling: Sampling::default(),
        extensions: Extensions::default(),
    }
}

async fn run_chat(adapter: &dyn Adapter, label: &str, model: &str) {
    match adapter
        .chat(
            req(model, "Reply with exactly: pong"),
            &RequestControl::default(),
        )
        .await
    {
        Ok(out) => println!("[{label}] chat OK      {}", brief(&out)),
        Err(e) => println!("[{label}] chat FAIL    {e}"),
    }
}

async fn run_stream(adapter: &dyn Adapter, label: &str, model: &str) {
    match adapter
        .chat_stream(req(model, "count 1 to 3"), &RequestControl::default())
        .await
    {
        Ok(mut st) => {
            let mut events = 0usize;
            let mut text = String::new();
            let mut saw_finished = false;
            while let Some(ev) = st.next().await {
                match ev {
                    Ok(StreamEvent::TextDelta { text: t, .. }) => {
                        events += 1;
                        text.push_str(&t);
                    }
                    Ok(StreamEvent::Finished { .. }) => {
                        events += 1;
                        saw_finished = true;
                    }
                    Ok(_) => events += 1,
                    Err(e) => {
                        println!("[{label}] stream FAIL  mid-stream: {e}");
                        return;
                    }
                }
            }
            println!(
                "[{label}] stream OK    events={events} finished={saw_finished} text={:?}",
                &text[..text.len().min(60)]
            );
        }
        Err(e) => println!("[{label}] stream FAIL  {e}"),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let endpoint = std::env::var("A4MP_LIVE_ENDPOINT").unwrap_or_else(|_| {
        eprintln!("A4MP_LIVE_ENDPOINT is required (the ollama server's base URL)");
        std::process::exit(2);
    });
    let transport = Arc::new(ReqwestTransport::new());
    let resolver = Arc::new(NoCredentials);
    let model = "gemma4:e4b";

    // 1. Ollama native protocol on the supplied endpoint
    let ol = OllamaAdapter::new(
        target(Protocol::Other("ollama".into()), endpoint.clone()),
        AuthKind::None,
        transport.clone(),
        resolver.clone(),
    )
    .unwrap();
    run_chat(&ol, "ollama", model).await;
    run_stream(&ol, "ollama", model).await;
    match ol.models(&RequestControl::default()).await {
        Ok(m) => println!("[ollama] models OK     {} entries", m.len()),
        Err(e) => println!("[ollama] models FAIL   {e}"),
    }

    // 2. OpenAI-compatible surface on the same endpoint's /v1
    let oa = OpenAiAdapter::new(
        target(Protocol::OpenAiChatCompletions, format!("{endpoint}/v1")),
        AuthKind::None,
        transport.clone(),
        resolver.clone(),
    )
    .unwrap();
    run_chat(&oa, "openai", model).await;
    run_stream(&oa, "openai", model).await;
    match oa.models(&RequestControl::default()).await {
        Ok(m) => println!("[openai] models OK     {} entries", m.len()),
        Err(e) => println!("[openai] models FAIL   {e}"),
    }
}
