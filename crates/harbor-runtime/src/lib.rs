use harbor_ai::{CompletionRequest, CompletionResponse, Message, MessageRole, ModelProvider, ToolChoice};
use harbor_core::{FrameworkError, FrameworkResult, ToolRegistry};
use harbor_http::{HarborHttpConfig, HarborHttpServer, ReadinessGate};
use harbor_memory::{MemoryMessage, SessionMemory};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Clone)]
pub struct AgentRuntime<P, M> {
    provider: P,
    memory: M,
    tools: ToolRegistry,
    default_model: String,
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

    pub async fn run_turn(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
    ) -> FrameworkResult<CompletionResponse> {
        let user_input = user_input.into();
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
            content: user_input.clone(),
        });

        let response = self
            .provider
            .complete(CompletionRequest {
                model: self.default_model.clone(),
                system_prompt: Some("You are Harbor runtime".into()),
                messages,
                tools: self.tools.list(),
                response_schema: None,
                session_id: Some(session_id.to_string()),
                tool_choice: ToolChoice::Auto,
            })
            .await?;

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarborAppConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub http_host: String,
    pub http_port: u16,
}

impl Default for HarborAppConfig {
    fn default() -> Self {
        Self {
            service_name: "harbor-app".into(),
            service_version: "0.1.0".into(),
            environment: "dev".into(),
            http_host: "0.0.0.0".into(),
            http_port: 3000,
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
}

#[derive(Debug, Clone)]
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
        HarborHttpServer::new(self.config.http_config()).with_readiness(self.readiness.clone())
    }

    pub async fn run(self) -> FrameworkResult<()> {
        self.run_with_signal(shutdown_signal()).await
    }

    pub async fn run_with_signal<F>(self, shutdown_signal: F) -> FrameworkResult<()>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        self.readiness.set_ready(true);
        self.http_server().run_with_shutdown(shutdown_signal).await
    }
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
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
