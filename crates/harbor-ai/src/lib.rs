use async_trait::async_trait;
use harbor_core::{FrameworkError, FrameworkResult, JsonValue, ToolSpec};
use opentelemetry::{global, propagation::Injector};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue},
    Client,
};
use serde::{Deserialize, Serialize};
use std::{env, sync::Arc, time::Duration};
use tokio::time::{sleep, timeout};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    fn as_openai_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "assistant",
        }
    }

    fn as_anthropic_role(&self) -> Option<&'static str> {
        match self {
            Self::System => None,
            Self::User => Some("user"),
            Self::Assistant | Self::Tool => Some("assistant"),
        }
    }

    fn as_ollama_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant | Self::Tool => "assistant",
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

pub type SharedProvider = Arc<dyn ModelProvider>;

pub fn shared_provider<P>(provider: P) -> SharedProvider
where
    P: ModelProvider + 'static,
{
    Arc::new(provider)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub attempts: usize,
    pub delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 3,
            delay_ms: 100,
        }
    }
}

#[derive(Clone)]
pub struct RetryProvider<P> {
    inner: P,
    policy: RetryPolicy,
}

impl<P> RetryProvider<P> {
    pub fn new(inner: P, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }
}

#[async_trait]
impl<P> ModelProvider for RetryProvider<P>
where
    P: ModelProvider,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let attempts = self.policy.attempts.max(1);
        let mut last_error = None;

        for attempt in 0..attempts {
            match self.inner.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    last_error = Some(error);
                    if attempt + 1 < attempts && self.policy.delay_ms > 0 {
                        sleep(Duration::from_millis(self.policy.delay_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            FrameworkError::Provider("retry provider exhausted without an error".into())
        }))
    }
}

#[derive(Clone)]
pub struct TimeoutProvider<P> {
    inner: P,
    timeout_ms: u64,
}

impl<P> TimeoutProvider<P> {
    pub fn new(inner: P, timeout_ms: u64) -> Self {
        Self {
            inner,
            timeout_ms: timeout_ms.max(1),
        }
    }
}

#[async_trait]
impl<P> ModelProvider for TimeoutProvider<P>
where
    P: ModelProvider,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        timeout(
            Duration::from_millis(self.timeout_ms),
            self.inner.complete(request),
        )
        .await
        .map_err(|_| {
            FrameworkError::Provider(format!(
                "provider '{}' timed out after {} ms",
                self.inner.name(),
                self.timeout_ms
            ))
        })?
    }
}

#[derive(Clone, Default)]
pub struct FallbackProvider {
    providers: Vec<SharedProvider>,
}

impl FallbackProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_provider<P>(mut self, provider: P) -> Self
    where
        P: ModelProvider + 'static,
    {
        self.providers.push(shared_provider(provider));
        self
    }

    pub fn push_shared(&mut self, provider: SharedProvider) {
        self.providers.push(provider);
    }
}

#[async_trait]
impl ModelProvider for FallbackProvider {
    fn name(&self) -> &'static str {
        "fallback"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        if self.providers.is_empty() {
            return Err(FrameworkError::Provider(
                "fallback provider has no configured inner providers".into(),
            ));
        }

        let mut errors = Vec::new();

        for provider in &self.providers {
            match provider.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => errors.push(format!("{}: {}", provider.name(), error)),
            }
        }

        Err(FrameworkError::Provider(format!(
            "all fallback providers failed: {}",
            errors.join(" | ")
        )))
    }
}

#[derive(Clone)]
pub struct StructuredOutputProvider<P> {
    inner: P,
}

impl<P> StructuredOutputProvider<P> {
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<P> ModelProvider for StructuredOutputProvider<P>
where
    P: ModelProvider,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn complete(&self, mut request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let schema = request.response_schema.clone();
        if let Some(schema) = &schema {
            let schema_text = serde_json::to_string_pretty(schema)?;
            let instruction = format!(
                "Return only valid JSON that matches this schema exactly:\n{}",
                schema_text
            );

            request.system_prompt = Some(match request.system_prompt.take() {
                Some(existing) if !existing.trim().is_empty() => {
                    format!("{}\n\n{}", existing, instruction)
                }
                _ => instruction,
            });
        }

        let mut response = self.inner.complete(request).await?;

        if let Some(schema) = &schema {
            let structured = extract_json_value(&response.text)?;
            validate_json_schema(schema, &structured)?;
            response.structured = Some(structured);
        }

        Ok(response)
    }
}

pub fn extract_json_value(text: &str) -> FrameworkResult<JsonValue> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<JsonValue>(trimmed) {
        return Ok(value);
    }

    if let Some(value) = extract_json_from_fence(trimmed) {
        return value;
    }

    if let Some(value) = extract_balanced_json(trimmed, '{', '}') {
        return value;
    }

    if let Some(value) = extract_balanced_json(trimmed, '[', ']') {
        return value;
    }

    Err(FrameworkError::Provider(
        "structured output helper could not find valid JSON in provider response".into(),
    ))
}

pub fn validate_json_schema(schema: &JsonValue, value: &JsonValue) -> FrameworkResult<()> {
    if let Some(enum_values) = schema.get("enum").and_then(|value| value.as_array()) {
        if !enum_values.iter().any(|candidate| candidate == value) {
            return Err(FrameworkError::Provider(
                "structured output does not match enum values".into(),
            ));
        }
    }

    if let Some(schema_type) = schema.get("type").and_then(|value| value.as_str()) {
        match schema_type {
            "object" => {
                let object = value.as_object().ok_or_else(|| {
                    FrameworkError::Provider("structured output is not an object".into())
                })?;

                if let Some(required) = schema.get("required").and_then(|value| value.as_array()) {
                    for key in required.iter().filter_map(|value| value.as_str()) {
                        if !object.contains_key(key) {
                            return Err(FrameworkError::Provider(format!(
                                "structured output is missing required field '{}'",
                                key
                            )));
                        }
                    }
                }

                if let Some(properties) = schema.get("properties").and_then(|value| value.as_object()) {
                    for (key, property_schema) in properties {
                        if let Some(property_value) = object.get(key) {
                            validate_json_schema(property_schema, property_value)?;
                        }
                    }
                }
            }
            "array" => {
                let array = value.as_array().ok_or_else(|| {
                    FrameworkError::Provider("structured output is not an array".into())
                })?;

                if let Some(item_schema) = schema.get("items") {
                    for item in array {
                        validate_json_schema(item_schema, item)?;
                    }
                }
            }
            "string" if !value.is_string() => {
                return Err(FrameworkError::Provider(
                    "structured output is not a string".into(),
                ));
            }
            "number" if !value.is_number() => {
                return Err(FrameworkError::Provider(
                    "structured output is not a number".into(),
                ));
            }
            "integer" if !value.as_i64().is_some() && !value.as_u64().is_some() => {
                return Err(FrameworkError::Provider(
                    "structured output is not an integer".into(),
                ));
            }
            "boolean" if !value.is_boolean() => {
                return Err(FrameworkError::Provider(
                    "structured output is not a boolean".into(),
                ));
            }
            "null" if !value.is_null() => {
                return Err(FrameworkError::Provider(
                    "structured output is not null".into(),
                ));
            }
            _ => {}
        }
    }

    Ok(())
}

fn extract_json_from_fence(text: &str) -> Option<FrameworkResult<JsonValue>> {
    let start = text.find("```")?;
    let rest = &text[start + 3..];
    let after_language = match rest.find('\n') {
        Some(index) => &rest[index + 1..],
        None => rest,
    };
    let end = after_language.find("```")?;
    Some(
        serde_json::from_str::<JsonValue>(after_language[..end].trim()).map_err(Into::into),
    )
}

fn extract_balanced_json(text: &str, open: char, close: char) -> Option<FrameworkResult<JsonValue>> {
    let start = text.find(open)?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let candidate = &text[start..=start + offset];
                    return Some(serde_json::from_str::<JsonValue>(candidate).map_err(Into::into));
                }
            }
            _ => {}
        }
    }

    None
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

        let model = model_or_default(model, &self.config.default_model);
        let mut api_messages = Vec::new();

        if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
            api_messages.push(OpenAIMessageRequest {
                role: "system".into(),
                content: system_prompt,
            });
        }

        api_messages.extend(messages.into_iter().map(|message| OpenAIMessageRequest {
            role: message.role.as_openai_role().into(),
            content: message.content,
        }));

        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);

        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.config.api_key)
            .headers(headers)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub default_model: String,
    pub max_tokens: u32,
}

impl AnthropicConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .or_else(|_| env::var("HARBOR_ANTHROPIC_API_KEY"))
            .map_err(|_| {
                FrameworkError::Config(
                    "missing ANTHROPIC_API_KEY (or HARBOR_ANTHROPIC_API_KEY) for Anthropic provider"
                        .into(),
                )
            })?;

        let base_url = env::var("ANTHROPIC_BASE_URL")
            .or_else(|_| env::var("HARBOR_ANTHROPIC_BASE_URL"))
            .unwrap_or_else(|_| "https://api.anthropic.com/v1".into());

        let default_model = env::var("ANTHROPIC_MODEL")
            .or_else(|_| env::var("HARBOR_ANTHROPIC_MODEL"))
            .unwrap_or_else(|_| "claude-3-5-haiku-latest".into());

        let max_tokens = env::var("ANTHROPIC_MAX_TOKENS")
            .or_else(|_| env::var("HARBOR_ANTHROPIC_MAX_TOKENS"))
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS);

        Ok(Self {
            api_key,
            base_url,
            default_model,
            max_tokens,
        })
    }
}

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: Client,
    config: AnthropicConfig,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(AnthropicConfig::from_env()?))
    }

    pub fn with_client(client: Client, config: AnthropicConfig) -> Self {
        Self { client, config }
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let CompletionRequest {
            model,
            system_prompt,
            messages,
            ..
        } = request;

        let model = model_or_default(model, &self.config.default_model);
        let system_prompt = system_prompt.filter(|value| !value.trim().is_empty());
        let api_messages = messages
            .into_iter()
            .filter_map(|message| {
                message.role.as_anthropic_role().map(|role| AnthropicMessageRequest {
                    role: role.into(),
                    content: vec![AnthropicContentBlockRequest::text(message.content)],
                })
            })
            .collect::<Vec<_>>();

        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);
        headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(&self.config.api_key)
                .map_err(|error| FrameworkError::Config(error.to_string()))?,
        );
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );

        let response = self
            .client
            .post(format!("{}/messages", self.config.base_url.trim_end_matches('/')))
            .headers(headers)
            .json(&AnthropicMessagesRequest {
                model: model.clone(),
                system: system_prompt,
                max_tokens: self.config.max_tokens,
                messages: api_messages,
            })
            .send()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let response: AnthropicMessagesResponse = response
            .json()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let text = response
            .content
            .into_iter()
            .filter(|block| block.kind == "text")
            .filter_map(|block| block.text)
            .filter(|value| !value.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        if text.trim().is_empty() {
            return Err(FrameworkError::Provider(
                "Anthropic provider returned no assistant text response".into(),
            ));
        }

        Ok(CompletionResponse {
            text,
            structured: None,
            provider: self.name().into(),
            model: response.model.unwrap_or(model),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub base_url: String,
    pub default_model: String,
}

impl OllamaConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let base_url = env::var("OLLAMA_BASE_URL")
            .or_else(|_| env::var("HARBOR_OLLAMA_BASE_URL"))
            .unwrap_or_else(|_| "http://127.0.0.1:11434".into());

        let default_model = env::var("OLLAMA_MODEL")
            .or_else(|_| env::var("HARBOR_OLLAMA_MODEL"))
            .unwrap_or_else(|_| "llama3.2".into());

        Ok(Self {
            base_url,
            default_model,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    client: Client,
    config: OllamaConfig,
}

impl OllamaProvider {
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(OllamaConfig::from_env()?))
    }

    pub fn with_client(client: Client, config: OllamaConfig) -> Self {
        Self { client, config }
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "ollama"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let CompletionRequest {
            model,
            system_prompt,
            messages,
            ..
        } = request;

        let model = model_or_default(model, &self.config.default_model);
        let mut api_messages = Vec::new();

        if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
            api_messages.push(OllamaMessageRequest {
                role: "system".into(),
                content: system_prompt,
            });
        }

        api_messages.extend(messages.into_iter().map(|message| OllamaMessageRequest {
            role: message.role.as_ollama_role().into(),
            content: message.content,
        }));

        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);

        let response = self
            .client
            .post(format!("{}/api/chat", self.config.base_url.trim_end_matches('/')))
            .headers(headers)
            .json(&OllamaChatRequest {
                model: model.clone(),
                messages: api_messages,
                stream: false,
            })
            .send()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let response: OllamaChatResponse = response
            .json()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let text = response
            .message
            .and_then(|message| {
                if message.content.trim().is_empty() {
                    None
                } else {
                    Some(message.content)
                }
            })
            .ok_or_else(|| FrameworkError::Provider("Ollama provider returned no assistant text response".into()))?;

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

#[derive(Debug, Serialize)]
struct AnthropicMessagesRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    messages: Vec<AnthropicMessageRequest>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessageRequest {
    role: String,
    content: Vec<AnthropicContentBlockRequest>,
}

#[derive(Debug, Serialize)]
struct AnthropicContentBlockRequest {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

impl AnthropicContentBlockRequest {
    fn text(text: String) -> Self {
        Self {
            kind: "text".into(),
            text,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicMessagesResponse {
    model: Option<String>,
    #[serde(default)]
    content: Vec<AnthropicContentBlockResponse>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlockResponse {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessageRequest>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OllamaMessageRequest {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    model: Option<String>,
    message: Option<OllamaMessageResponse>,
}

#[derive(Debug, Deserialize)]
struct OllamaMessageResponse {
    content: String,
}

fn model_or_default(model: String, default_model: &str) -> String {
    if model.trim().is_empty() {
        default_model.into()
    } else {
        model
    }
}

fn inject_trace_context(headers: &mut HeaderMap) {
    let context = Span::current().context();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}

struct HeaderInjector<'a>(&'a mut HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        extract::State,
        http::HeaderMap,
        response::IntoResponse,
        routing::post,
        Json, Router,
    };
    use opentelemetry::{
        global,
        trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState},
        Context,
    };
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
    use tokio::net::TcpListener;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    #[derive(Clone, Default)]
    struct RequestCapture {
        system_prompts: Arc<Mutex<Vec<Option<String>>>>,
    }

    #[derive(Clone)]
    struct CapturingProvider {
        provider_name: &'static str,
        response_text: String,
        capture: RequestCapture,
    }

    #[async_trait]
    impl ModelProvider for CapturingProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
            self.capture
                .system_prompts
                .lock()
                .unwrap()
                .push(request.system_prompt.clone());

            Ok(CompletionResponse {
                text: self.response_text.clone(),
                structured: None,
                provider: self.provider_name.into(),
                model: request.model,
            })
        }
    }

    #[derive(Clone)]
    struct StaticProvider {
        provider_name: &'static str,
        text: String,
    }

    #[async_trait]
    impl ModelProvider for StaticProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
            Ok(CompletionResponse {
                text: self.text.clone(),
                structured: None,
                provider: self.provider_name.into(),
                model: request.model,
            })
        }
    }

    #[derive(Clone)]
    struct FlakyProvider {
        provider_name: &'static str,
        failures_before_success: usize,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ModelProvider for FlakyProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call <= self.failures_before_success {
                Err(FrameworkError::Provider(format!(
                    "{} failed on call {}",
                    self.provider_name, call
                )))
            } else {
                Ok(CompletionResponse {
                    text: format!("{} ok", self.provider_name),
                    structured: None,
                    provider: self.provider_name.into(),
                    model: request.model,
                })
            }
        }
    }

    #[derive(Clone)]
    struct SlowProvider {
        provider_name: &'static str,
        delay_ms: u64,
    }

    #[async_trait]
    impl ModelProvider for SlowProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            Ok(CompletionResponse {
                text: format!("{} done", self.provider_name),
                structured: None,
                provider: self.provider_name.into(),
                model: request.model,
            })
        }
    }

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".into(),
            system_prompt: None,
            messages: vec![Message {
                role: MessageRole::User,
                content: "hello".into(),
            }],
            tools: vec![],
            response_schema: None,
            session_id: None,
            tool_choice: ToolChoice::Auto,
        }
    }

    #[tokio::test]
    async fn retry_provider_retries_until_success() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = RetryProvider::new(
            FlakyProvider {
                provider_name: "flaky",
                failures_before_success: 2,
                calls: calls.clone(),
            },
            RetryPolicy {
                attempts: 3,
                delay_ms: 0,
            },
        );

        let response = provider.complete(sample_request()).await.unwrap();
        assert_eq!(response.provider, "flaky");
        assert_eq!(response.text, "flaky ok");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn timeout_provider_returns_timeout_error() {
        let provider = TimeoutProvider::new(
            SlowProvider {
                provider_name: "slow",
                delay_ms: 50,
            },
            10,
        );

        let error = provider.complete(sample_request()).await.unwrap_err();
        assert!(error.to_string().contains("timed out after 10 ms"));
    }

    #[tokio::test]
    async fn fallback_provider_uses_second_provider_after_first_failure() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = FallbackProvider::new()
            .with_provider(FlakyProvider {
                provider_name: "first",
                failures_before_success: usize::MAX,
                calls,
            })
            .with_provider(StaticProvider {
                provider_name: "second",
                text: "fallback ok".into(),
            });

        let response = provider.complete(sample_request()).await.unwrap();
        assert_eq!(response.provider, "second");
        assert_eq!(response.text, "fallback ok");
    }

    #[test]
    fn extract_json_value_finds_json_inside_code_fence() {
        let value = extract_json_value("Here you go:\n```json\n{\"answer\":\"hi\"}\n```")
            .unwrap();
        assert_eq!(value["answer"], "hi");
    }

    #[test]
    fn validate_json_schema_rejects_missing_required_field() {
        let schema = json!({
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer": { "type": "string" }
            }
        });

        let error = validate_json_schema(&schema, &json!({ "other": "x" })).unwrap_err();
        assert!(error.to_string().contains("missing required field 'answer'"));
    }

    #[tokio::test]
    async fn structured_output_provider_parses_and_validates_json() {
        let capture = RequestCapture::default();
        let provider = StructuredOutputProvider::new(CapturingProvider {
            provider_name: "capture",
            response_text: "```json\n{\"answer\":\"hi\"}\n```".into(),
            capture: capture.clone(),
        });

        let schema = json!({
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer": { "type": "string" }
            }
        });

        let response = provider
            .complete(CompletionRequest {
                model: "test-model".into(),
                system_prompt: Some("Be precise".into()),
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Say hi".into(),
                }],
                tools: vec![],
                response_schema: Some(schema.clone()),
                session_id: None,
                tool_choice: ToolChoice::Auto,
            })
            .await
            .unwrap();

        assert_eq!(response.structured.unwrap()["answer"], "hi");

        let captured_prompt = capture.system_prompts.lock().unwrap().remove(0).unwrap();
        assert!(captured_prompt.contains("Be precise"));
        assert!(captured_prompt.contains("Return only valid JSON that matches this schema exactly"));
        assert!(captured_prompt.contains("\"answer\""));
    }

    #[derive(Clone, Default)]
    struct CaptureState {
        inner: Arc<Mutex<Vec<CapturedRequest>>>,
    }

    #[derive(Debug, Clone)]
    struct CapturedRequest {
        headers: Vec<(String, String)>,
        body: Value,
    }

    impl CaptureState {
        fn push(&self, captured: CapturedRequest) {
            self.inner.lock().unwrap().push(captured);
        }

        fn take_one(&self) -> CapturedRequest {
            self.inner.lock().unwrap().remove(0)
        }
    }

    #[tokio::test]
    async fn anthropic_provider_maps_request_and_response() {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let state = CaptureState::default();
        let app = Router::new().route("/messages", post(anthropic_handler)).with_state(state.clone());
        let base_url = spawn_router(app).await;

        let provider = AnthropicProvider::new(AnthropicConfig {
            api_key: "anthropic-test-key".into(),
            base_url,
            default_model: "claude-test-model".into(),
            max_tokens: 256,
        });

        let parent_context = Context::new().with_remote_span_context(SpanContext::new(
            TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::from_hex("00f067aa0ba902b7").unwrap(),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        ));
        let span = tracing::info_span!("anthropic_provider_test");
        let _ = span.set_parent(parent_context);
        let _guard = span.enter();

        let response = provider
            .complete(CompletionRequest {
                model: String::new(),
                system_prompt: Some("Be concise".into()),
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Hello from Harbor".into(),
                }],
                tools: vec![],
                response_schema: None,
                session_id: Some("session-1".into()),
                tool_choice: ToolChoice::Auto,
            })
            .await
            .unwrap();

        assert_eq!(response.provider, "anthropic");
        assert_eq!(response.model, "claude-test-model");
        assert_eq!(response.text, "Hi from Anthropic");

        let captured = state.take_one();
        let headers = captured.headers.into_iter().collect::<std::collections::HashMap<_, _>>();
        assert_eq!(headers.get("x-api-key").unwrap(), "anthropic-test-key");
        assert_eq!(headers.get("anthropic-version").unwrap(), ANTHROPIC_VERSION);
        if let Some(traceparent) = headers.get("traceparent") {
            assert!(traceparent.starts_with("00-"));
            assert!(!traceparent.trim().is_empty());
        }

        assert_eq!(captured.body["model"], "claude-test-model");
        assert_eq!(captured.body["system"], "Be concise");
        assert_eq!(captured.body["max_tokens"], 256);
        assert_eq!(captured.body["messages"][0]["role"], "user");
        assert_eq!(captured.body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(captured.body["messages"][0]["content"][0]["text"], "Hello from Harbor");
    }

    #[tokio::test]
    async fn anthropic_config_reads_env() {
        unsafe {
            env::set_var("HARBOR_ANTHROPIC_API_KEY", "secret");
            env::set_var("HARBOR_ANTHROPIC_BASE_URL", "https://anthropic.example/v1");
            env::set_var("HARBOR_ANTHROPIC_MODEL", "claude-env");
            env::set_var("HARBOR_ANTHROPIC_MAX_TOKENS", "777");
        }

        let config = AnthropicConfig::from_env().unwrap();
        assert_eq!(config.api_key, "secret");
        assert_eq!(config.base_url, "https://anthropic.example/v1");
        assert_eq!(config.default_model, "claude-env");
        assert_eq!(config.max_tokens, 777);

        unsafe {
            env::remove_var("HARBOR_ANTHROPIC_API_KEY");
            env::remove_var("HARBOR_ANTHROPIC_BASE_URL");
            env::remove_var("HARBOR_ANTHROPIC_MODEL");
            env::remove_var("HARBOR_ANTHROPIC_MAX_TOKENS");
        }
    }

    async fn anthropic_handler(
        State(state): State<CaptureState>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        state.push(CapturedRequest {
            headers: headers_to_vec(&headers),
            body,
        });

        Json(json!({
            "model": "claude-test-model",
            "content": [
                { "type": "text", "text": "Hi from Anthropic" }
            ]
        }))
    }

    #[tokio::test]
    async fn ollama_provider_maps_request_and_response() {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let state = CaptureState::default();
        let app = Router::new().route("/api/chat", post(ollama_handler)).with_state(state.clone());
        let base_url = spawn_router(app).await;

        let provider = OllamaProvider::new(OllamaConfig {
            base_url,
            default_model: "llama-test-model".into(),
        });

        let parent_context = Context::new().with_remote_span_context(SpanContext::new(
            TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::from_hex("00f067aa0ba902b7").unwrap(),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        ));
        let span = tracing::info_span!("ollama_provider_test");
        let _ = span.set_parent(parent_context);
        let _guard = span.enter();

        let response = provider
            .complete(CompletionRequest {
                model: String::new(),
                system_prompt: Some("Answer plainly".into()),
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Hello from Ollama".into(),
                }],
                tools: vec![],
                response_schema: None,
                session_id: Some("session-2".into()),
                tool_choice: ToolChoice::Auto,
            })
            .await
            .unwrap();

        assert_eq!(response.provider, "ollama");
        assert_eq!(response.model, "llama-test-model");
        assert_eq!(response.text, "Hi from Ollama");

        let captured = state.take_one();
        let headers = captured.headers.into_iter().collect::<std::collections::HashMap<_, _>>();
        if let Some(traceparent) = headers.get("traceparent") {
            assert!(traceparent.starts_with("00-"));
            assert!(!traceparent.trim().is_empty());
        }

        assert_eq!(captured.body["model"], "llama-test-model");
        assert_eq!(captured.body["stream"], false);
        assert_eq!(captured.body["messages"][0]["role"], "system");
        assert_eq!(captured.body["messages"][0]["content"], "Answer plainly");
        assert_eq!(captured.body["messages"][1]["role"], "user");
        assert_eq!(captured.body["messages"][1]["content"], "Hello from Ollama");
    }

    #[tokio::test]
    async fn ollama_config_reads_env() {
        unsafe {
            env::set_var("HARBOR_OLLAMA_BASE_URL", "http://ollama.example:11434");
            env::set_var("HARBOR_OLLAMA_MODEL", "llama-env");
        }

        let config = OllamaConfig::from_env().unwrap();
        assert_eq!(config.base_url, "http://ollama.example:11434");
        assert_eq!(config.default_model, "llama-env");

        unsafe {
            env::remove_var("HARBOR_OLLAMA_BASE_URL");
            env::remove_var("HARBOR_OLLAMA_MODEL");
        }
    }

    async fn ollama_handler(
        State(state): State<CaptureState>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        state.push(CapturedRequest {
            headers: headers_to_vec(&headers),
            body,
        });

        Json(json!({
            "model": "llama-test-model",
            "message": {
                "role": "assistant",
                "content": "Hi from Ollama"
            },
            "done": true
        }))
    }

    async fn spawn_router(router: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{}", address)
    }

    fn headers_to_vec(headers: &HeaderMap) -> Vec<(String, String)> {
        headers
            .iter()
            .filter_map(|(name, value)| value.to_str().ok().map(|value| (name.to_string(), value.to_string())))
            .collect()
    }
}
