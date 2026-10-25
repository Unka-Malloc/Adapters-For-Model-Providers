//! Live verification against the Tailscale Windows-laptop model.
//!
//! Live verification against a self-hosted model rig.
//!
//! Point it at a host running:
//!   - llama-server on :8080 (native OpenAI + vision)
//!   - dialect gateway on :8090 (Anthropic + Gemini surfaces)
//!   - ollama serve on :11434 (native Ollama + /api/embed)
//!
//!   A4MP_LIVE_HOST=host.tailnet cargo run --example live_tailnet
//!
//! Set NO_PROXY for the host if a local HTTP proxy is configured.

use std::sync::Arc;

use a4mp::api::*;
use a4mp::auth::{AuthKind, NoCredentials};
use a4mp::providers::{AnthropicAdapter, GeminiAdapter, OllamaAdapter, OpenAiAdapter};
use a4mp::transport::ReqwestTransport;
use futures_util::StreamExt;

fn target(protocol: Protocol, base: String) -> ServiceTarget {
    ServiceTarget::new(ProviderId::new("live-local"), protocol, base.parse().unwrap())
}

fn brief(out: &Output) -> String {
    let mut s = String::new();
    for p in &out.parts {
        match p {
            ContentPart::Text { text } => s.push_str(&format!("text({}B) ", text.len())),
            ContentPart::Reasoning(r) => {
                let n = r.summary.as_deref().map(str::len).unwrap_or(0)
                    + r.opaque.to_string().len();
                s.push_str(&format!("reasoning(~{n}B) "))
            }
            ContentPart::ToolCall(t) => s.push_str(&format!("tool:{} ", t.name)),
            ContentPart::Media { kind, .. } => s.push_str(&format!("media:{kind:?} ")),
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
        sampling: Sampling::default(),
        extensions: Extensions::default(),
    }
}

async fn run_chat(adapter: &dyn Adapter, label: &str, model: &str) {
    match adapter.chat(req(model, "Reply with exactly: pong"), &RequestControl::default()).await {
        Ok(out) => println!("[{label}] chat OK      {}", brief(&out)),
        Err(e) => println!("[{label}] chat FAIL    {e}"),
    }
}

async fn run_stream(adapter: &dyn Adapter, label: &str, model: &str) {
    match adapter.chat_stream(req(model, "count 1 to 3"), &RequestControl::default()).await {
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
    let host = std::env::var("A4MP_LIVE_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let transport = Arc::new(ReqwestTransport::new());
    let resolver = Arc::new(NoCredentials);
    let model = "qwen3.6-35b-a3b";

    // 1. Native OpenAI surface (llama-server :8080)
    let oa = OpenAiAdapter::new(
        target(Protocol::OpenAiChatCompletions, format!("http://{host}:8080/v1")),
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

    // 2. Anthropic Messages via gateway :8090
    let an = AnthropicAdapter::new(
        target(Protocol::AnthropicMessages, format!("http://{host}:8090")),
        AuthKind::ApiKey,
        transport.clone(),
        resolver.clone(),
    )
    .unwrap();
    run_chat(&an, "anthropic", model).await;
    run_stream(&an, "anthropic", model).await;

    // 3. Gemini generateContent via gateway :8090
    let ge = GeminiAdapter::new(
        target(Protocol::GeminiGenerateContent, format!("http://{host}:8090")),
        AuthKind::ApiKey,
        transport.clone(),
        resolver.clone(),
    )
    .unwrap();
    run_chat(&ge, "gemini", model).await;
    run_stream(&ge, "gemini", model).await;

    // 4. Ollama NATIVE on :11434 (real ollama serve, qwen36 registered
    //    from the same GGUF; nomic-embed for vectors)
    let ol = OllamaAdapter::new(
        target(Protocol::Other("ollama".into()), format!("http://{host}:11434")),
        AuthKind::None,
        transport.clone(),
        resolver.clone(),
    )
    .unwrap();
    run_chat(&ol, "ollama", "qwen36").await;
    run_stream(&ol, "ollama", "qwen36").await;
    match ol.models(&RequestControl::default()).await {
        Ok(m) => println!("[ollama] models OK     {} entries", m.len()),
        Err(e) => println!("[ollama] models FAIL   {e}"),
    }
    match ol
        .embed(
            EmbedRequest {
                model: "nomic-embed".into(),
                inputs: vec![EmbedInput::Text("hello world".into())],
                dimensions: None,
                extensions: Extensions::default(),
            },
            &RequestControl::default(),
        )
        .await
    {
        Ok(o) => println!(
            "[ollama] embed OK      {} vectors x {} dims",
            o.embeddings.len(),
            o.embeddings.first().map(|e| e.len()).unwrap_or(0)
        ),
        Err(e) => println!("[ollama] embed FAIL    {e}"),
    }

    // 5. Multimodal image input on the OpenAI surface (mmproj active)
    const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeUlEQVR4nO3PQQkAMAzAwCqpf1ETMxF7HINABFzm7H7dcEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFj13PLIEAOXyUUwAAAABJRU5ErkJggg==";
    use base64::Engine;
    let png = base64::engine::general_purpose::STANDARD
        .decode(RED_PNG_B64)
        .unwrap();
    let img_req = ChatRequest {
        model: model.into(),
        messages: vec![Message {
            role: Role::User,
            content: vec![
                ContentPart::Text {
                    text: "What single color fills this image? One word.".into(),
                },
                ContentPart::Media {
                    kind: MediaKind::Image,
                    source: MediaSource::Inline {
                        media_type: "image/png".into(),
                        data: png.into(),
                    },
                    detail: None,
                },
            ],
        }],
        tools: vec![],
        tool_choice: None,
        sampling: Sampling::default(),
        extensions: Extensions::default(),
    };
    match oa.chat(img_req, &RequestControl::default()).await {
        Ok(out) => println!("[openai] image OK      {}", brief(&out)),
        Err(e) => println!("[openai] image FAIL    {e}"),
    }
}
