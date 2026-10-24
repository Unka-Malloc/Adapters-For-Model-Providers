//! Built-in provider profiles: the declarative data layer.
//!
//! One entry per provider service surface — id, aliases, accepted
//! authentication mechanisms, endpoint×protocol targets with regional
//! variants, and catalogue provenance. Presence asserts a supported
//! access surface, never live availability. The full inventory
//! reconciliation accrues here; entries cite only documented endpoints.

use super::{AuthSpec, ProviderProfile, ServiceDef};
use crate::api::Protocol;

const CHAT: Protocol = Protocol::OpenAiChatCompletions;
const RESP: Protocol = Protocol::OpenAiResponses;
const ANTH: Protocol = Protocol::AnthropicMessages;
const GEM: Protocol = Protocol::GeminiGenerateContent;

fn native(name: &'static str) -> Protocol {
    Protocol::Other(name.to_string())
}

fn target(protocol: Protocol, base_url: &'static str) -> ServiceDef {
    ServiceDef {
        protocol,
        base_url,
        region: None,
        headers: &[],
    }
}

fn region(protocol: Protocol, base_url: &'static str, region: &'static str) -> ServiceDef {
    ServiceDef {
        protocol,
        base_url,
        region: Some(region),
        headers: &[],
    }
}

const KEY: &[AuthSpec] = &[AuthSpec::ApiKey];
const BEARER: &[AuthSpec] = &[AuthSpec::Bearer];

/// The reconciled profile set. Ordered for readability; lookup is by id.
pub fn builtin_profiles() -> Vec<ProviderProfile> {
    vec![
        ProviderProfile {
            id: "openai".into(),
            name: "OpenAI",
            aliases: &["openai-codex-api"],
            auth: KEY.to_vec(),
            targets: vec![
                target(RESP, "https://api.openai.com/v1"),
                target(CHAT, "https://api.openai.com/v1"),
            ],
            catalogue: Some("openai"),
        },
        ProviderProfile {
            id: "anthropic".into(),
            name: "Anthropic",
            aliases: &["claude"],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![ServiceDef {
                protocol: ANTH,
                base_url: "https://api.anthropic.com",
                region: None,
                headers: &[("anthropic-version", "2023-06-01")],
            }],
            catalogue: Some("anthropic"),
        },
        ProviderProfile {
            id: "google".into(),
            name: "Google Gemini",
            aliases: &["gemini", "google-gemini"],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![target(GEM, "https://generativelanguage.googleapis.com")],
            catalogue: Some("google"),
        },
        ProviderProfile {
            id: "deepseek".into(),
            name: "DeepSeek",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.deepseek.com")],
            catalogue: Some("deepseek"),
        },
        ProviderProfile {
            id: "moonshotai".into(),
            name: "Moonshot AI (Kimi)",
            aliases: &["moonshot", "kimi"],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://api.moonshot.ai/v1", "intl"),
                region(CHAT, "https://api.moonshot.cn/v1", "cn"),
            ],
            catalogue: Some("moonshotai"),
        },
        ProviderProfile {
            id: "kimi-code".into(),
            name: "Kimi For Coding",
            aliases: &["kimi-coding"],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://api.kimi.com/coding/v1", "global"),
                region(CHAT, "https://api.kimi.com/coding/v1", "cn"),
            ],
            catalogue: Some("kimi-code-plan-global"),
        },
        ProviderProfile {
            id: "xai".into(),
            name: "xAI",
            aliases: &["grok-api"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.x.ai/v1")],
            catalogue: Some("xai"),
        },
        ProviderProfile {
            id: "mistral".into(),
            name: "Mistral",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                target(CHAT, "https://api.mistral.ai/v1"),
                target(native("mistral"), "https://api.mistral.ai"),
            ],
            catalogue: Some("mistral"),
        },
        ProviderProfile {
            id: "groq".into(),
            name: "Groq",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.groq.com/openai/v1")],
            catalogue: Some("groq"),
        },
        ProviderProfile {
            id: "together".into(),
            name: "Together AI",
            aliases: &["togetherai"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.together.xyz/v1")],
            catalogue: Some("togetherai"),
        },
        ProviderProfile {
            id: "fireworks".into(),
            name: "Fireworks AI",
            aliases: &["fireworks-ai"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.fireworks.ai/inference/v1")],
            catalogue: Some("fireworks-ai"),
        },
        ProviderProfile {
            id: "openrouter".into(),
            name: "OpenRouter",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://openrouter.ai/api/v1")],
            catalogue: Some("openrouter"),
        },
        ProviderProfile {
            id: "cohere".into(),
            name: "Cohere",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                target(native("cohere"), "https://api.cohere.com"),
                target(CHAT, "https://api.cohere.com/compatibility/v1"),
            ],
            catalogue: Some("cohere"),
        },
        ProviderProfile {
            id: "zhipu".into(),
            name: "Zhipu GLM",
            aliases: &["zhipuai", "bigmodel", "glm"],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://open.bigmodel.cn/api/paas/v4", "api"),
                region(CHAT, "https://open.bigmodel.cn/api/coding/paas/v4", "coding-plan"),
            ],
            catalogue: Some("zhipuai"),
        },
        ProviderProfile {
            id: "zai".into(),
            name: "Z.ai",
            aliases: &["z-ai"],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://api.z.ai/api/paas/v4", "api"),
                region(CHAT, "https://api.z.ai/api/coding/paas/v4", "coding-plan"),
            ],
            catalogue: Some("zhipuai"),
        },
        ProviderProfile {
            id: "minimax".into(),
            name: "MiniMax",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://api.minimax.io/v1", "intl"),
                region(CHAT, "https://api.minimaxi.com/v1", "cn"),
            ],
            catalogue: Some("minimax"),
        },
        ProviderProfile {
            id: "stepfun".into(),
            name: "StepFun",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://api.stepfun.ai/v1", "global"),
                region(CHAT, "https://api.stepfun.com/v1", "cn"),
            ],
            catalogue: Some("stepfun"),
        },
        ProviderProfile {
            id: "xiaomi".into(),
            name: "Xiaomi MiMo",
            aliases: &["mimo"],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://api.xiaomimimo.com/v1", "api"),
                region(CHAT, "https://token-plan-cn.xiaomimimo.com/v1", "plan-cn"),
                region(CHAT, "https://token-plan-sgp.xiaomimimo.com/v1", "plan-sgp"),
                region(CHAT, "https://token-plan-ams.xiaomimimo.com/v1", "plan-ams"),
            ],
            catalogue: Some("xiaomi"),
        },
        ProviderProfile {
            id: "alibaba".into(),
            name: "Alibaba (Qwen / Bailian)",
            aliases: &["qwen", "dashscope", "bailian"],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://dashscope.aliyuncs.com/compatible-mode/v1", "cn"),
                region(CHAT, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "intl"),
            ],
            catalogue: Some("alibaba"),
        },
        ProviderProfile {
            id: "baidu-qianfan".into(),
            name: "Baidu Qianfan",
            aliases: &["qianfan"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://qianfan.baidubce.com/v2")],
            catalogue: Some("baidu-qianfan"),
        },
        ProviderProfile {
            id: "tencent-tokenhub".into(),
            name: "Tencent TokenHub",
            aliases: &["hunyuan", "tencent"],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://tokenhub.tencentmaas.com/v1", "cn"),
                region(CHAT, "https://tokenhub-intl.tencentmaas.com/v1", "intl"),
            ],
            catalogue: Some("tencent-tokenhub"),
        },
        ProviderProfile {
            id: "volcengine".into(),
            name: "Volcengine Ark",
            aliases: &["ark", "doubao"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://ark.cn-beijing.volces.com/api/v3")],
            catalogue: Some("volcengine"),
        },
        ProviderProfile {
            id: "huaweicloud".into(),
            name: "Huawei Cloud MaaS",
            aliases: &["huawei", "modelarts"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.modelarts-maas.com/v2")],
            catalogue: Some("huaweicloud"),
        },
        ProviderProfile {
            id: "siliconflow".into(),
            name: "SiliconFlow",
            aliases: &["silicon"],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://api.siliconflow.cn/v1", "cn"),
                region(CHAT, "https://api.siliconflow.com/v1", "intl"),
            ],
            catalogue: Some("siliconflow"),
        },
        ProviderProfile {
            id: "modelscope".into(),
            name: "ModelScope",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api-inference.modelscope.cn/v1")],
            catalogue: Some("modelscope"),
        },
        ProviderProfile {
            id: "aihubmix".into(),
            name: "AIHubMix",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://aihubmix.com/v1")],
            catalogue: Some("aihubmix"),
        },
        ProviderProfile {
            id: "302ai".into(),
            name: "302.AI",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.302.ai/v1")],
            catalogue: Some("302ai"),
        },
        ProviderProfile {
            id: "cerebras".into(),
            name: "Cerebras",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.cerebras.ai/v1")],
            catalogue: Some("cerebras"),
        },
        ProviderProfile {
            id: "baseten".into(),
            name: "Baseten",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://inference.baseten.co/v1")],
            catalogue: Some("baseten"),
        },
        ProviderProfile {
            id: "huggingface".into(),
            name: "Hugging Face",
            aliases: &["hf"],
            auth: BEARER.to_vec(),
            targets: vec![target(CHAT, "https://router.huggingface.co/v1")],
            catalogue: Some("huggingface"),
        },
        ProviderProfile {
            id: "deepinfra".into(),
            name: "Deep Infra",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.deepinfra.com/v1/openai")],
            catalogue: Some("deepinfra"),
        },
        ProviderProfile {
            id: "novita".into(),
            name: "Novita AI",
            aliases: &["novita-ai"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.novita.ai/v3/openai")],
            catalogue: Some("novita"),
        },
        ProviderProfile {
            id: "nebius".into(),
            name: "Nebius Token Factory",
            aliases: &["nebius-token-factory"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.tokenfactory.nebius.com/v1")],
            catalogue: Some("nebius"),
        },
        ProviderProfile {
            id: "nvidia".into(),
            name: "NVIDIA NIM",
            aliases: &["nim"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://integrate.api.nvidia.com/v1")],
            catalogue: Some("nvidia"),
        },
        ProviderProfile {
            id: "sambanova".into(),
            name: "SambaNova",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.sambanova.ai/v1")],
            catalogue: Some("sambanova"),
        },
        ProviderProfile {
            id: "upstage".into(),
            name: "Upstage",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.upstage.ai/v1")],
            catalogue: Some("upstage"),
        },
        ProviderProfile {
            id: "perplexity".into(),
            name: "Perplexity",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.perplexity.ai")],
            catalogue: Some("perplexity"),
        },
        ProviderProfile {
            id: "vercel-ai-gateway".into(),
            name: "Vercel AI Gateway",
            aliases: &["vercel", "ai-gateway"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://ai-gateway.vercel.sh/v1")],
            catalogue: Some("vercel"),
        },
        ProviderProfile {
            id: "kilo".into(),
            name: "Kilo Gateway",
            aliases: &["kilocode"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.kilo.ai/api/gateway")],
            catalogue: Some("kilo"),
        },
        ProviderProfile {
            id: "ollama".into(),
            name: "Ollama",
            aliases: &[],
            auth: vec![AuthSpec::None, AuthSpec::ApiKey],
            targets: vec![
                target(native("ollama"), "http://localhost:11434"),
                target(CHAT, "http://localhost:11434/v1"),
            ],
            catalogue: Some("ollama"),
        },
        ProviderProfile {
            id: "ollama-cloud".into(),
            name: "Ollama Cloud",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://ollama.com/v1")],
            catalogue: Some("ollama-cloud"),
        },
        ProviderProfile {
            id: "lmstudio".into(),
            name: "LM Studio",
            aliases: &[],
            auth: vec![AuthSpec::None],
            targets: vec![target(CHAT, "http://localhost:1234/v1")],
            catalogue: Some("lmstudio"),
        },
        ProviderProfile {
            id: "openai-codex".into(),
            name: "OpenAI Codex (ChatGPT account)",
            aliases: &["codex"],
            auth: vec![AuthSpec::OAuth],
            targets: vec![target(RESP, "https://chatgpt.com/backend-api/codex")],
            catalogue: Some("openai"),
        },
        ProviderProfile {
            id: "github-copilot".into(),
            name: "GitHub Copilot",
            aliases: &["copilot"],
            auth: vec![AuthSpec::OAuth],
            targets: vec![target(CHAT, "https://api.githubcopilot.com")],
            catalogue: Some("github-copilot"),
        },
    ]
}
