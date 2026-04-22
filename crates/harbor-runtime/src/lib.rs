use harbor_ai::{
    CompletionEvent, CompletionEventStream, CompletionRequest, CompletionResponse, Message,
    MessageRole, ModelProvider, ToolChoice,
};
use harbor_core::{FrameworkError, FrameworkResult, ToolRegistry};
use harbor_http::{HarborHttpConfig, HarborHttpMiddlewareConfig, HarborHttpServer, ReadinessGate};
use harbor_memory::{MemoryMessage, SessionMemory};
use harbor_observability::{HarborObservability, HarborObservabilityConfig};
use serde::{Deserialize, Serialize};
use std::env;
use tracing::warn;

#[derive(Clone)]
pub struct AgentRuntime<P, M> {
    provider: P,
    memory: M,
    tools: ToolRegistry,
    default_model: String,
}

pub struct RuntimeEventStream<M> {
    inner: CompletionEventStream,
    memory: M,
    session_id: String,
    user_input: Option<String>,
    assistant_recorded: bool,
    buffered_text: String,
}

impl<M> RuntimeEventStream<M>
where
    M: SessionMemory,
{
    fn new(
        inner: CompletionEventStream,
        memory: M,
        session_id: impl Into<String>,
        user_input: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            memory,
            session_id: session_id.into(),
            user_input: Some(user_input.into()),
            assistant_recorded: false,
            buffered_text: String::new(),
        }
    }

    async fn ensure_user_recorded(&mut self) {
        if let Some(user_input) = self.user_input.take() {
            self.memory
                .append(&self.session_id, MemoryMessage::new("user", user_input))
                .await;
        }
    }

    async fn record_assistant_if_needed(&mut self, text: String) {
        if self.assistant_recorded {
            return;
        }

        self.memory
            .append(&self.session_id, MemoryMessage::new("assistant", text))
            .await;
        self.assistant_recorded = true;
    }

    pub async fn recv(&mut self) -> Option<FrameworkResult<CompletionEvent>> {
        match self.inner.recv().await {
            Some(Ok(CompletionEvent::Started { provider, model })) => {
                self.ensure_user_recorded().await;
                Some(Ok(CompletionEvent::Started { provider, model }))
            }
            Some(Ok(CompletionEvent::Delta { text })) => {
                self.ensure_user_recorded().await;
                self.buffered_text.push_str(&text);
                Some(Ok(CompletionEvent::Delta { text }))
            }
            Some(Ok(CompletionEvent::Finished { response })) => {
                self.ensure_user_recorded().await;

                let assistant_text = if response.text.is_empty() {
                    self.buffered_text.clone()
                } else {
                    response.text.clone()
                };

                self.record_assistant_if_needed(assistant_text).await;
                Some(Ok(CompletionEvent::Finished { response }))
            }
            Some(Err(error)) => Some(Err(error)),
            None => None,
        }
    }
}

impl<P, M> AgentRuntime<P, M>
where
    P: ModelProvider,
    M: SessionMemory,
{
    pub fn new(provider: P, memory: M) -> Self {
        Self {
            provider,
            memory,
            tools: ToolRegistry::new(),
            default_model: "mock-model".into(),
        }
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    async fn build_request(&self, session_id: &str, user_input: String) -> CompletionRequest {
        let history = self.memory.messages(session_id).await;

        let mut messages: Vec<Message> = history
            .into_iter()
            .map(|message| Message {
                role: match message.role.as_str() {
                    "system" => MessageRole::System,
                    "assistant" => MessageRole::Assistant,
                    "tool" => MessageRole::Tool,
                    _ => MessageRole::User,
                },
                content: message.content,
            })
            .collect();

        messages.push(Message {
            role: MessageRole::User,
            content: user_input,
        });

        CompletionRequest {
            model: self.default_model.clone(),
            system_prompt: Some("You are Harbor runtime".into()),
            messages,
            tools: self.tools.list(),
            response_schema: None,
            session_id: Some(session_id.to_string()),
            tool_choice: ToolChoice::Auto,
        }
    }

    pub async fn run_turn(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
    ) -> FrameworkResult<CompletionResponse> {
        let user_input = user_input.into();
        let request = self.build_request(session_id, user_input.clone()).await;
        let response = self.provider.complete(request).await?;

        self.memory
            .append(session_id, MemoryMessage::new("user", user_input))
            .await;
        self.memory
            .append(
                session_id,
                MemoryMessage::new("assistant", response.text.clone()),
            )
            .await;

        Ok(response)
    }

    pub async fn run_turn_stream(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
    ) -> FrameworkResult<RuntimeEventStream<M>> {
        let user_input = user_input.into();
        let request = self.build_request(session_id, user_input.clone()).await;
        let stream = self.provider.complete_stream(request).await?;

        Ok(RuntimeEventStream::new(
            stream,
            self.memory.clone(),
            session_id,
            user_input,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarborAppConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub http_host: String,
    pub http_port: u16,
    pub log_level: String,
    pub json_logs: bool,
    pub metrics_enabled: bool,
    pub otel_enabled: bool,
    pub otel_endpoint: String,
    pub http_timeout_ms: Option<u64>,
    pub http_concurrency_limit: Option<usize>,
    pub http_rate_limit_requests: Option<u64>,
    pub http_rate_limit_window_secs: u64,
    pub http_bearer_token: Option<String>,
}

impl Default for HarborAppConfig {
    fn default() -> Self {
        Self {
            service_name: "harbor-app".into(),
            service_version: "0.1.0".into(),
            environment: "dev".into(),
            http_host: "0.0.0.0".into(),
            http_port: 3000,
            log_level: "info".into(),
            json_logs: false,
            metrics_enabled: true,
            otel_enabled: false,
            otel_endpoint: "http://127.0.0.1:4317".into(),
            http_timeout_ms: None,
            http_concurrency_limit: None,
            http_rate_limit_requests: None,
            http_rate_limit_window_secs: 1,
            http_bearer_token: None,
        }
    }
}

impl HarborAppConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let mut config = Self::default();

        if let Ok(service_name) = env::var("HARBOR_SERVICE_NAME") {
            config.service_name = service_name;
        }

        if let Ok(service_version) = env::var("HARBOR_SERVICE_VERSION") {
            config.service_version = service_version;
        }

        if let Ok(environment) = env::var("HARBOR_ENV") {
            config.environment = environment;
        }

        if let Ok(host) = env::var("HARBOR_HTTP_HOST") {
            config.http_host = host;
        }

        if let Ok(port) = env::var("HARBOR_HTTP_PORT") {
            config.http_port = port.parse().map_err(|error| {
                FrameworkError::Config(format!("invalid HARBOR_HTTP_PORT value: {error}"))
            })?;
        }

        if let Ok(log_level) = env::var("HARBOR_LOG_LEVEL") {
            config.log_level = log_level;
        }

        if let Ok(json_logs) = env::var("HARBOR_JSON_LOGS") {
            config.json_logs = matches!(json_logs.as_str(), "1" | "true" | "TRUE" | "yes" | "YES");
        }

        if let Ok(metrics_enabled) = env::var("HARBOR_METRICS_ENABLED") {
            config.metrics_enabled = !matches!(
                metrics_enabled.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO"
            );
        }

        if let Ok(otel_enabled) = env::var("HARBOR_OTEL_ENABLED") {
            config.otel_enabled =
                matches!(otel_enabled.as_str(), "1" | "true" | "TRUE" | "yes" | "YES");
        }

        if let Ok(otel_endpoint) = env::var("HARBOR_OTEL_ENDPOINT") {
            config.otel_endpoint = otel_endpoint;
        }

        if let Ok(timeout_ms) = env::var("HARBOR_HTTP_TIMEOUT_MS") {
            config.http_timeout_ms = Some(timeout_ms.parse().map_err(|error| {
                FrameworkError::Config(format!("invalid HARBOR_HTTP_TIMEOUT_MS value: {error}"))
            })?);
        }

        if let Ok(concurrency_limit) = env::var("HARBOR_HTTP_CONCURRENCY_LIMIT") {
            config.http_concurrency_limit = Some(concurrency_limit.parse().map_err(|error| {
                FrameworkError::Config(format!(
                    "invalid HARBOR_HTTP_CONCURRENCY_LIMIT value: {error}"
                ))
            })?);
        }

        if let Ok(rate_limit_requests) = env::var("HARBOR_HTTP_RATE_LIMIT_REQUESTS") {
            config.http_rate_limit_requests =
                Some(rate_limit_requests.parse().map_err(|error| {
                    FrameworkError::Config(format!(
                        "invalid HARBOR_HTTP_RATE_LIMIT_REQUESTS value: {error}"
                    ))
                })?);
        }

        if let Ok(rate_limit_window_secs) = env::var("HARBOR_HTTP_RATE_LIMIT_WINDOW_SECS") {
            config.http_rate_limit_window_secs =
                rate_limit_window_secs.parse().map_err(|error| {
                    FrameworkError::Config(format!(
                        "invalid HARBOR_HTTP_RATE_LIMIT_WINDOW_SECS value: {error}"
                    ))
                })?;
        }

        if let Ok(http_bearer_token) = env::var("HARBOR_HTTP_BEARER_TOKEN") {
            if !http_bearer_token.trim().is_empty() {
                config.http_bearer_token = Some(http_bearer_token);
            }
        }

        Ok(config)
    }

    pub fn http_config(&self) -> HarborHttpConfig {
        HarborHttpConfig {
            host: self.http_host.clone(),
            port: self.http_port,
            service_name: self.service_name.clone(),
            service_version: self.service_version.clone(),
            environment: self.environment.clone(),
        }
    }

    pub fn observability_config(&self) -> HarborObservabilityConfig {
        HarborObservabilityConfig {
            service_name: self.service_name.clone(),
            service_version: self.service_version.clone(),
            environment: self.environment.clone(),
            log_level: self.log_level.clone(),
            json_logs: self.json_logs,
            metrics_enabled: self.metrics_enabled,
            otel_enabled: self.otel_enabled,
            otel_endpoint: self.otel_endpoint.clone(),
        }
    }

    pub fn http_middleware_config(&self) -> HarborHttpMiddlewareConfig {
        HarborHttpMiddlewareConfig {
            request_timeout_ms: self.http_timeout_ms,
            concurrency_limit: self.http_concurrency_limit,
            rate_limit_requests: self.http_rate_limit_requests,
            rate_limit_window_secs: self.http_rate_limit_window_secs,
            bearer_token: self.http_bearer_token.clone(),
        }
    }
}

#[derive(Clone)]
pub struct HarborApp {
    config: HarborAppConfig,
    readiness: ReadinessGate,
}

impl HarborApp {
    pub fn new(config: HarborAppConfig) -> Self {
        Self {
            config,
            readiness: ReadinessGate::not_ready(),
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(HarborAppConfig::from_env()?))
    }

    pub fn config(&self) -> &HarborAppConfig {
        &self.config
    }

    pub fn readiness(&self) -> ReadinessGate {
        self.readiness.clone()
    }

    pub fn mark_ready(&self) {
        self.readiness.set_ready(true);
    }

    pub fn mark_not_ready(&self) {
        self.readiness.set_ready(false);
    }

    pub fn http_server(&self) -> HarborHttpServer {
        HarborHttpServer::new(self.config.http_config())
            .with_readiness(self.readiness.clone())
            .with_middleware_config(self.config.http_middleware_config())
    }

    pub async fn run(self) -> FrameworkResult<()> {
        self.run_with_signal(shutdown_signal()).await
    }

    pub async fn run_with_signal<F>(self, shutdown_signal: F) -> FrameworkResult<()>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let observability = HarborObservability::init(&self.config.observability_config())?;
        self.readiness.set_ready(true);

        let mut server = self.http_server();
        if let Some(renderer) = observability.metrics_renderer() {
            server = server.with_metrics_renderer(renderer);
        }

        let result = server.run_with_shutdown(shutdown_signal).await;
        if let Err(error) = observability.shutdown() {
            warn!(error = %error, "harbor observability shutdown reported an error");
        }
        result
    }
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harbor_ai::{CompletionEvent, MockProvider};
    use harbor_memory::{InMemorySessionMemory, SessionMemory};

    #[tokio::test]
    async fn run_turn_stream_emits_events_and_persists_memory() {
        let memory = InMemorySessionMemory::default();
        let runtime =
            AgentRuntime::new(MockProvider, memory.clone()).with_default_model("mock-gpt");

        let mut stream = runtime
            .run_turn_stream("stream-session", "Hello from streaming Harbor")
            .await
            .unwrap();

        let mut saw_started = false;
        let mut delta_text = String::new();
        let mut finished_text = None;

        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                CompletionEvent::Started { provider, model } => {
                    saw_started = true;
                    assert_eq!(provider, "mock");
                    assert_eq!(model, "mock-gpt");
                }
                CompletionEvent::Delta { text } => delta_text.push_str(&text),
                CompletionEvent::Finished { response } => finished_text = Some(response.text),
            }
        }

        let expected = "[mock:stream-session] Hello from streaming Harbor";
        assert!(saw_started);
        assert_eq!(delta_text, expected);
        assert_eq!(finished_text.unwrap(), expected);

        let history = memory.messages("stream-session").await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[0].content, "Hello from streaming Harbor");
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[1].content, expected);
    }
}
