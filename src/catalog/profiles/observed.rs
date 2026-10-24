//! Profiles reconciled from the frozen reference observations
//! (Magpie presets, Pi factories, Hermes plugins, Magpie community
//! extensions). Entries are added only with documented endpoints.
//!
//! Records that duplicate a `major` provider fold into it; records that
//! are regional/plan variants of one observed provider merge into a
//! single profile here, the source ids kept as aliases. Endpoints that
//! are per-account or per-resource templates (Azure resources,
//! Cloudflare account/gateway ids, user-supplied hosts) have no real
//! base URL and are omitted.

use super::super::{AuthSpec, ProviderProfile, ServiceDef};
use crate::api::Protocol;

const CHAT: Protocol = Protocol::OpenAiChatCompletions;
const RESP: Protocol = Protocol::OpenAiResponses;
const ANTH: Protocol = Protocol::AnthropicMessages;
const GEM: Protocol = Protocol::GeminiGenerateContent;
const BEDROCK: Protocol = Protocol::BedrockConverse;

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

/// Anthropic-Messages surfaces require the protocol version header.
fn anthropic(base_url: &'static str) -> ServiceDef {
    ServiceDef {
        protocol: ANTH,
        base_url,
        region: None,
        headers: &[("anthropic-version", "2023-06-01")],
    }
}

fn anthropic_region(base_url: &'static str, region: &'static str) -> ServiceDef {
    ServiceDef {
        protocol: ANTH,
        base_url,
        region: Some(region),
        headers: &[("anthropic-version", "2023-06-01")],
    }
}

const KEY: &[AuthSpec] = &[AuthSpec::ApiKey];
const OAUTH: &[AuthSpec] = &[AuthSpec::OAuth];

/// Reference-observed providers not already in `major`.
pub fn profiles() -> Vec<ProviderProfile> {
    vec![
        // --- Vendor APIs --------------------------------------------------
        ProviderProfile {
            id: "amazon-bedrock".into(),
            name: "Amazon Bedrock",
            aliases: &["bedrock"],
            auth: vec![AuthSpec::ApiKey, AuthSpec::CloudIdentity],
            // Any commercial region substitutes: bedrock-runtime.{region}.amazonaws.com
            targets: vec![
                region(BEDROCK, "https://bedrock-runtime.us-east-1.amazonaws.com", "us-east-1"),
                region(CHAT, "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1", "us-east-1"),
                anthropic_region("https://bedrock-runtime.us-east-1.amazonaws.com/anthropic", "us-east-1"),
            ],
            catalogue: Some("amazon-bedrock"),
        },
        ProviderProfile {
            id: "bedrock-mantle".into(),
            name: "Amazon Bedrock (Mantle)",
            aliases: &[],
            auth: vec![AuthSpec::ApiKey, AuthSpec::CloudIdentity],
            // Any served region substitutes: bedrock-mantle.{region}.api.aws
            targets: vec![
                region(CHAT, "https://bedrock-mantle.us-east-1.api.aws/v1", "us-east-1"),
                region(RESP, "https://bedrock-mantle.us-east-1.api.aws/v1", "us-east-1"),
                anthropic_region("https://bedrock-mantle.us-east-1.api.aws/anthropic", "us-east-1"),
            ],
            catalogue: Some("bedrock-mantle"),
        },
        ProviderProfile {
            id: "google-vertex".into(),
            name: "Google Vertex AI",
            aliases: &["vertex"],
            auth: vec![AuthSpec::CloudIdentity, AuthSpec::ApiKey],
            // Global endpoint; regional endpoints {location}-aiplatform.googleapis.com substitute.
            targets: vec![region(native("google-vertex"), "https://aiplatform.googleapis.com", "global")],
            catalogue: Some("google-vertex"),
        },
        ProviderProfile {
            id: "ant-ling".into(),
            name: "Ant Ling",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.ant-ling.com/v1")],
            catalogue: Some("ant-ling"),
        },
        ProviderProfile {
            id: "meta".into(),
            name: "Meta",
            aliases: &["meta-ai"],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![target(RESP, "https://api.meta.ai/v1")],
            catalogue: Some("meta"),
        },
        ProviderProfile {
            id: "nous".into(),
            name: "Nous Portal",
            aliases: &["nous-portal"],
            auth: vec![AuthSpec::Bearer, AuthSpec::OAuth],
            targets: vec![target(CHAT, "https://inference-api.nousresearch.com/v1")],
            catalogue: Some("nous"),
        },
        ProviderProfile {
            id: "actual".into(),
            name: "Actual",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.actual.inc/v1")],
            catalogue: Some("actual"),
        },
        ProviderProfile {
            id: "arcee".into(),
            name: "Arcee",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.arcee.ai/api/v1")],
            catalogue: Some("arcee"),
        },
        ProviderProfile {
            id: "gmi".into(),
            name: "GMI Cloud",
            aliases: &["gmi-cloud"],
            auth: KEY.to_vec(),
            targets: vec![target(CHAT, "https://api.gmi-serving.com/v1")],
            catalogue: Some("gmi"),
        },
        // --- Subscription plans and account paths --------------------------
        ProviderProfile {
            id: "qwen-token-plan".into(),
            name: "Alibaba Token Plan (Qwen / Bailian)",
            aliases: &[
                "bailian-token-plan",
                "qwen-token-plan-cn",
                "qwen-token-plan-individual",
            ],
            auth: KEY.to_vec(),
            targets: vec![
                // Sold on the Qwen AI platform…
                region(CHAT, "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1", "qwen-ai"),
                anthropic_region("https://token-plan.maas.qianwenaiapi.com/apps/anthropic", "qwen-ai"),
                // …and on Alibaba Cloud Bailian, Beijing and Singapore.
                region(CHAT, "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "bailian-cn"),
                anthropic_region("https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic", "bailian-cn"),
                region(CHAT, "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "bailian-intl"),
            ],
            catalogue: Some("qwen-token-plan"),
        },
        ProviderProfile {
            id: "alibaba-coding-plan".into(),
            name: "Alibaba Coding Plan",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://coding-intl.dashscope.aliyuncs.com/v1", "intl"),
                region(CHAT, "https://coding.dashscope.aliyuncs.com/v1", "cn"),
            ],
            catalogue: Some("alibaba-coding-plan"),
        },
        ProviderProfile {
            id: "tencent-cloud".into(),
            name: "Tencent Cloud",
            aliases: &["tencent-token-plan", "tencent-tokenhub-cn"],
            auth: KEY.to_vec(),
            targets: vec![
                // Token Plan (LKEAP) and TokenHub plan/pay-as-you-go surfaces.
                region(CHAT, "https://api.lkeap.cloud.tencent.com/plan/v3", "plan-cn"),
                anthropic_region("https://api.lkeap.cloud.tencent.com/plan/anthropic", "plan-cn"),
                region(CHAT, "https://tokenhub.tencentmaas.com/v1", "cn"),
                region(CHAT, "https://tokenhub-intl.tencentmaas.com/v1", "intl"),
                region(CHAT, "https://tokenhub.tencentmaas.com/plan/v3", "enterprise-cn"),
                region(CHAT, "https://tokenhub-intl.tencentmaas.com/plan/v3", "enterprise-intl"),
            ],
            catalogue: Some("tencent-cloud"),
        },
        ProviderProfile {
            id: "chatgpt-api".into(),
            name: "ChatGPT API (subscription)",
            aliases: &["siwc"],
            auth: OAUTH.to_vec(),
            targets: vec![
                target(RESP, "https://api.openai.com/v1"),
                target(CHAT, "https://api.openai.com/v1"),
            ],
            catalogue: Some("chatgpt-api"),
        },
        ProviderProfile {
            id: "gemini-account".into(),
            name: "Gemini (account access)",
            aliases: &["gemini-cli"],
            auth: OAUTH.to_vec(),
            // Gemini CLI's Code Assist (cloudcode-pa) surface.
            targets: vec![target(native("cloudcode-pa"), "https://cloudcode-pa.googleapis.com")],
            catalogue: Some("gemini"),
        },
        ProviderProfile {
            id: "antigravity".into(),
            name: "Antigravity (account access)",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![target(native("cloudcode-pa"), "https://daily-cloudcode-pa.googleapis.com")],
            catalogue: Some("antigravity"),
        },
        ProviderProfile {
            id: "qwen-oauth".into(),
            name: "Qwen Portal (OAuth)",
            aliases: &["qwen-portal", "qwen-cli"],
            auth: vec![AuthSpec::OAuth, AuthSpec::ApiKey],
            targets: vec![target(CHAT, "https://portal.qwen.ai/v1")],
            catalogue: Some("qwen-oauth"),
        },
        ProviderProfile {
            id: "solstice".into(),
            name: "Solstice",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![target(GEM, "https://generativelanguage.googleapis.com/v1alpha")],
            catalogue: Some("solstice"),
        },
        // --- Gateways and relays -------------------------------------------
        ProviderProfile {
            id: "opencode-zen".into(),
            name: "OpenCode Zen",
            aliases: &["opencode", "zen-free"],
            auth: vec![AuthSpec::None, AuthSpec::ApiKey],
            targets: vec![
                target(CHAT, "https://opencode.ai/zen/v1"),
                target(RESP, "https://opencode.ai/zen/v1"),
                anthropic("https://opencode.ai/zen"),
                region(CHAT, "https://opencode.ai/inference/v1", "inference"),
            ],
            catalogue: Some("opencode"),
        },
        ProviderProfile {
            id: "opencode-go".into(),
            name: "OpenCode Go",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                target(CHAT, "https://opencode.ai/zen/go/v1"),
                target(RESP, "https://opencode.ai/zen/go/v1"),
                anthropic("https://opencode.ai/zen/go"),
            ],
            catalogue: Some("opencode-go"),
        },
        ProviderProfile {
            id: "cline".into(),
            name: "Cline",
            aliases: &["clinepass"],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![target(CHAT, "https://api.cline.bot/api/v1")],
            catalogue: Some("cline"),
        },
        ProviderProfile {
            id: "commandcode".into(),
            name: "Command Code",
            aliases: &["command-code"],
            auth: KEY.to_vec(),
            targets: vec![
                target(CHAT, "https://api.commandcode.ai/provider/v1"),
                target(RESP, "https://api.commandcode.ai/provider/v1"),
                anthropic("https://api.commandcode.ai/provider"),
            ],
            catalogue: Some("commandcode"),
        },
        ProviderProfile {
            id: "router".into(),
            name: "Ramp Router",
            aliases: &["ramp-router", "ramp"],
            auth: KEY.to_vec(),
            targets: vec![target(RESP, "https://api.router.com/v1")],
            catalogue: Some("router"),
        },
        ProviderProfile {
            id: "pipellm".into(),
            name: "PipeLLM",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                target(CHAT, "https://api.pipellm.ai/openai/v1"),
                target(RESP, "https://api.pipellm.ai/v1"),
                anthropic("https://api.pipellm.ai"),
            ],
            catalogue: Some("pipellm"),
        },
        ProviderProfile {
            id: "cherryin".into(),
            name: "CherryIN",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                target(CHAT, "https://open.cherryin.ai/v1"),
                target(RESP, "https://open.cherryin.ai/v1"),
                anthropic("https://open.cherryin.ai"),
            ],
            catalogue: Some("cherryin"),
        },
        ProviderProfile {
            id: "yylx".into(),
            name: "Yylx",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![
                region(CHAT, "https://app.yylx.io/v1", "auto"),
                region(CHAT, "https://global.yylx.io/v1", "global"),
                region(CHAT, "https://cn.yylx.io/v1", "cn"),
            ],
            catalogue: Some("yylx"),
        },
        ProviderProfile {
            id: "radius".into(),
            name: "Radius",
            aliases: &[],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![target(native("pi-messages"), "https://radius.pi.dev")],
            catalogue: Some("radius"),
        },
        // --- Decision-model (System One) surfaces ----------------------------
        ProviderProfile {
            id: "typesafe".into(),
            name: "TypeSafe Jev",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(native("typesafe-system-one"), "https://api.typesafe.ai/v1")],
            catalogue: Some("typesafe"),
        },
        ProviderProfile {
            id: "vercel-jev".into(),
            name: "Jev · Vercel AI Gateway",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(native("typesafe-system-one"), "https://ai-gateway.vercel.sh/typesafe")],
            catalogue: Some("vercel-jev"),
        },
        ProviderProfile {
            id: "cloudflare-jev".into(),
            name: "Cloudflare Workers AI · Jev, Clef",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(native("typesafe-system-one"), "https://api.cloudflare.com/client/v4")],
            catalogue: Some("cloudflare-jev"),
        },
        ProviderProfile {
            id: "bailian-decision".into(),
            name: "Bailian Decision Model",
            aliases: &[],
            auth: KEY.to_vec(),
            // Workspaces address System One at {WorkspaceId}.{region}.maas.aliyuncs.com;
            // the Token Plan host is the concrete surface.
            targets: vec![target(native("typesafe-system-one"), "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")],
            catalogue: Some("bailian-decision"),
        },
        ProviderProfile {
            id: "qwen-decision".into(),
            name: "Qwen Decision Model",
            aliases: &[],
            auth: KEY.to_vec(),
            targets: vec![target(native("typesafe-system-one"), "https://maas.qianwenaiapi.com/compatible-mode/v1")],
            catalogue: Some("qwen-decision"),
        },
        // --- Local servers --------------------------------------------------
        ProviderProfile {
            id: "omlx".into(),
            name: "oMLX",
            aliases: &[],
            auth: vec![AuthSpec::None],
            targets: vec![
                target(CHAT, "http://localhost:8000/v1"),
                target(RESP, "http://localhost:8000/v1"),
            ],
            catalogue: Some("omlx"),
        },
        ProviderProfile {
            id: "mlx-serve".into(),
            name: "MLX-Serve",
            aliases: &[],
            auth: vec![AuthSpec::None],
            targets: vec![
                target(CHAT, "http://localhost:11234/v1"),
                target(RESP, "http://localhost:11234/v1"),
            ],
            catalogue: Some("mlx-serve"),
        },
        // --- Agent / IDE subscription paths --------------------------------
        ProviderProfile {
            id: "autoclaw".into(),
            name: "AutoClaw",
            aliases: &["autoglm"],
            auth: OAUTH.to_vec(),
            targets: vec![
                region(CHAT, "https://autoglm-api.autoglm.ai", "global"),
                region(CHAT, "https://autoglm-api.zhipuai.cn", "cn"),
                region(CHAT, "https://autoglm-acceleration-api.zhipuai.cn", "managed"),
            ],
            catalogue: Some("autoclaw"),
        },
        ProviderProfile {
            id: "codearts".into(),
            name: "CodeArts",
            aliases: &["huawei-codearts"],
            auth: vec![AuthSpec::CloudIdentity, AuthSpec::OAuth],
            targets: vec![target(CHAT, "https://snap-access.cn-north-4.myhuaweicloud.com/api/v2")],
            catalogue: Some("codearts"),
        },
        ProviderProfile {
            id: "cursor".into(),
            name: "Cursor",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![target(CHAT, "https://api2.cursor.sh")],
            catalogue: Some("cursor"),
        },
        ProviderProfile {
            id: "devin".into(),
            name: "Devin",
            aliases: &[],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![target(CHAT, "https://server.codeium.com/devin/v1")],
            catalogue: Some("devin"),
        },
        ProviderProfile {
            id: "factory".into(),
            name: "Factory",
            aliases: &["factory-ai", "droid"],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![
                region(CHAT, "https://api.factory.ai/api/llm/o/v1", "global"),
                region(RESP, "https://api.factory.ai/api/llm/o/v1", "global"),
                anthropic_region("https://api.factory.ai/api/llm/a/v1", "global"),
                region(CHAT, "https://api.eu.factory.ai/api/llm/o/v1", "eu"),
                anthropic_region("https://api.eu.factory.ai/api/llm/a/v1", "eu"),
            ],
            catalogue: Some("factory"),
        },
        ProviderProfile {
            id: "grok".into(),
            name: "Grok (X subscription)",
            aliases: &["grok-cli"],
            auth: OAUTH.to_vec(),
            targets: vec![target(CHAT, "https://cli-chat-proxy.grok.com/v1")],
            catalogue: Some("grok"),
        },
        ProviderProfile {
            id: "joycode".into(),
            name: "JoyCode",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![target(CHAT, "https://api-ai.jd.com/api")],
            catalogue: Some("joycode"),
        },
        ProviderProfile {
            id: "kimi-work".into(),
            name: "Kimi Work",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![target(CHAT, "https://agent-gw.kimi.com/coding/v1")],
            catalogue: Some("kimi-work"),
        },
        ProviderProfile {
            id: "kiro".into(),
            name: "Kiro",
            aliases: &[],
            auth: OAUTH.to_vec(),
            // AWS OIDC sign-in; the account's region substitutes runtime.{region}.kiro.dev.
            targets: vec![region(native("kiro"), "https://runtime.us-east-1.kiro.dev", "us-east-1")],
            catalogue: Some("kiro"),
        },
        ProviderProfile {
            id: "mimo-app".into(),
            name: "Xiaomi MiMo (account)",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![
                region(CHAT, "https://mimo-server-sgp.xiaomimimo.com/api", "sgp"),
                region(CHAT, "https://mimo-server-ru.xiaomimimo.com/api", "ru"),
                region(CHAT, "https://mimo-server-in.xiaomimimo.com/api", "in"),
            ],
            catalogue: Some("mimo"),
        },
        ProviderProfile {
            id: "minimax-code".into(),
            name: "MiniMax Code",
            aliases: &["mcode"],
            auth: OAUTH.to_vec(),
            targets: vec![
                anthropic_region("https://agent.minimax.io/mavis/api/v1/llm/v1", "global"),
                anthropic_region("https://agent.minimax.cn/mavis/api/v1/llm/v1", "cn"),
            ],
            catalogue: Some("minimax"),
        },
        ProviderProfile {
            id: "opendesign".into(),
            name: "OpenDesign",
            aliases: &["open-design"],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![
                target(RESP, "https://amr-link.open-design.ai/v1"),
                target(CHAT, "https://amr-link.open-design.ai/v1"),
            ],
            catalogue: Some("opendesign"),
        },
        ProviderProfile {
            id: "qoder".into(),
            name: "Qoder",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![
                region(native("qoder"), "https://api3.qoder.sh", "global"),
                region(native("qoder"), "https://gateway.qoder.com.cn", "cn"),
            ],
            catalogue: Some("qoder"),
        },
        ProviderProfile {
            id: "raccoon".into(),
            name: "Raccoon",
            aliases: &["xiaohuanxiong"],
            auth: OAUTH.to_vec(),
            targets: vec![target(native("raccoon"), "https://xiaohuanxiong.com/api/web/llm/v2")],
            catalogue: Some("raccoon"),
        },
        ProviderProfile {
            id: "trae".into(),
            name: "Trae",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![
                region(CHAT, "https://trae-api-cn.mchost.guru", "cn"),
                region(CHAT, "https://coresg-normal.trae.ai", "sg"),
                region(CHAT, "https://coreva-normal.trae.ai", "us"),
            ],
            catalogue: Some("trae"),
        },
        ProviderProfile {
            id: "workbuddy".into(),
            name: "WorkBuddy",
            aliases: &["workbuddy-ai"],
            auth: OAUTH.to_vec(),
            targets: vec![
                region(CHAT, "https://copilot.tencent.com", "cn"),
                region(CHAT, "https://www.workbuddy.ai", "global"),
            ],
            catalogue: Some("workbuddy"),
        },
        ProviderProfile {
            id: "zcode".into(),
            name: "ZCode",
            aliases: &[],
            auth: vec![AuthSpec::ApiKey, AuthSpec::OAuth],
            targets: vec![
                anthropic_region("https://api.z.ai/api/anthropic", "zai"),
                anthropic_region("https://open.bigmodel.cn/api/anthropic", "bigmodel"),
                anthropic_region("https://zcode.z.ai/api/v1/zcode-plan/anthropic", "start-plan"),
            ],
            catalogue: Some("zcode"),
        },
        ProviderProfile {
            id: "zed".into(),
            name: "Zed",
            aliases: &[],
            auth: OAUTH.to_vec(),
            targets: vec![
                target(CHAT, "https://cloud.zed.dev/v1"),
                target(ANTH, "https://cloud.zed.dev/v1"),
            ],
            catalogue: Some("zed"),
        },
    ]
}
