use async_trait::async_trait;
use harbor_core::{FrameworkError, FrameworkResult, JsonValue, ToolSpec};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    fn as_api_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "assistant",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub response_schema: Option<JsonValue>,
    pub session_id: Option<String>,
    pub tool_choice: ToolChoice,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
    pub structured: Option<JsonValue>,
    pub provider: String,
    pub model: String,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn name(&self) -> &'static str;
    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse>;
}

#[derive(Debug, Default, Clone)]
pub struct MockProvider;

#[async_trait]
impl ModelProvider for MockProvider {
    fn name(&self) -> &'static str {
        "mock"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let latest = request
            .messages
            .last()
            .map(|message| message.content.clone())
            .unwrap_or_else(|| "Hello from Harbor".to_string());

        Ok(CompletionResponse {
            text: format!(
                "[mock:{}] {}",
                request.session_id.unwrap_or_else(|| "sessionless".into()),
                latest
            ),
            structured: request.response_schema.map(|schema| {
                serde_json::json!({
                    "echo": latest,
                    "schema_hint": schema
                })
            }),
            provider: self.name().into(),
            model: request.model,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatibleConfig {
    pub api_key: String,
    pub base_url: String,
    pub default_model: String,
}

impl OpenAICompatibleConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .or_else(|_| env::var("HARBOR_OPENAI_API_KEY"))
            .map_err(|_| {
                FrameworkError::Config(
                    "missing OPENAI_API_KEY (or HARBOR_OPENAI_API_KEY) for OpenAI-compatible provider"
                        .into(),
                )
            })?;

        let base_url = env::var("OPENAI_BASE_URL")
            .or_else(|_| env::var("HARBOR_OPENAI_BASE_URL"))
            .unwrap_or_else(|_| "https://api.openai.com/v1".into());

        let default_model = env::var("OPENAI_MODEL")
            .or_else(|_| env::var("HARBOR_OPENAI_MODEL"))
            .unwrap_or_else(|_| "gpt-4o-mini".into());

        Ok(Self {
            api_key,
            base_url,
            default_model,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OpenAICompatibleProvider {
    client: Client,
    config: OpenAICompatibleConfig,
}

impl OpenAICompatibleProvider {
    pub fn new(config: OpenAICompatibleConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(OpenAICompatibleConfig::from_env()?))
    }

    pub fn with_client(client: Client, config: OpenAICompatibleConfig) -> Self {
        Self { client, config }
    }
}

#[async_trait]
impl ModelProvider for OpenAICompatibleProvider {
    fn name(&self) -> &'static str {
        "openai-compatible"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let CompletionRequest {
            model,
            system_prompt,
            messages,
            ..
        } = request;

        let model = if model.trim().is_empty() {
            self.config.default_model.clone()
        } else {
            model
        };

        let mut api_messages = Vec::new();

        if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
            api_messages.push(OpenAIMessageRequest {
                role: "system".into(),
                content: system_prompt,
            });
        }

        api_messages.extend(messages.into_iter().map(|message| OpenAIMessageRequest {
            role: message.role.as_api_role().into(),
            content: message.content,
        }));

        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.config.api_key)
            .json(&OpenAIChatRequest {
                model: model.clone(),
                messages: api_messages,
            })
            .send()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let response: OpenAIChatResponse = response
            .json()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let text = response
            .choices
            .into_iter()
            .find_map(|choice| {
                choice
                    .message
                    .and_then(|message| message.content)
                    .or(choice.text)
            })
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                FrameworkError::Provider(
                    "OpenAI-compatible provider returned no assistant text response".into(),
                )
            })?;

        Ok(CompletionResponse {
            text,
            structured: None,
            provider: self.name().into(),
            model: response.model.unwrap_or(model),
        })
    }
}

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessageRequest>,
}

#[derive(Debug, Serialize)]
struct OpenAIMessageRequest {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: Option<OpenAIMessageResponse>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessageResponse {
    content: Option<String>,
}
