use async_trait::async_trait;
use harbor_core::{FrameworkResult, JsonValue, ToolSpec};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
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
