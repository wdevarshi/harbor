use async_trait::async_trait;
use axum::{routing::post, Json, Router};
use harbor_core::{FrameworkError, FrameworkResult, JsonValue, Tool, ToolRegistry};
use opentelemetry::{global, propagation::Injector};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue},
    Client,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::HashMap,
    env,
    future::Future,
    pin::Pin,
    process::Stdio,
    sync::Arc,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

pub const DEFAULT_HTTP_PATH: &str = "/rpc";

type JsonFuture = Pin<Box<dyn Future<Output = FrameworkResult<JsonValue>> + Send + 'static>>;
type ResourceHandler = Arc<dyn Fn() -> JsonFuture + Send + Sync>;
type PromptHandler = Arc<dyn Fn(JsonValue) -> JsonFuture + Send + Sync>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    pub id: String,
    pub method: String,
    #[serde(default = "default_params")]
    pub params: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

fn default_params() -> JsonValue {
    JsonValue::Object(Default::default())
}

impl RpcResponse {
    pub fn success(id: impl Into<String>, result: JsonValue) -> Self {
        Self {
            id: id.into(),
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: impl Into<String>, code: i64, message: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
            }),
        }
    }

    pub fn into_result(self) -> FrameworkResult<JsonValue> {
        self.result.ok_or_else(|| {
            FrameworkError::Protocol(
                self.error
                    .map(|error| error.message)
                    .unwrap_or_else(|| "missing result".into()),
            )
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

impl ResourceSpec {
    pub fn new(uri: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            name: name.into(),
            description: None,
            mime_type: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgumentSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub required: bool,
}

impl PromptArgumentSpec {
    pub fn new(name: impl Into<String>, required: bool) -> Self {
        Self {
            name: name.into(),
            description: None,
            required,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<PromptArgumentSpec>,
}

impl PromptSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            arguments: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_argument(mut self, argument: PromptArgumentSpec) -> Self {
        self.arguments.push(argument);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    pub tools: bool,
    pub resources: bool,
    pub prompts: bool,
}

#[derive(Clone, Default)]
struct ResourceRegistry {
    specs: Vec<ResourceSpec>,
    handlers: HashMap<String, ResourceHandler>,
}

impl ResourceRegistry {
    fn register<F, Fut>(&mut self, spec: ResourceSpec, handler: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = FrameworkResult<JsonValue>> + Send + 'static,
    {
        self.specs.push(spec.clone());
        self.handlers.insert(
            spec.uri.clone(),
            Arc::new(move || -> JsonFuture { Box::pin(handler()) }),
        );
    }

    fn list(&self) -> Vec<ResourceSpec> {
        self.specs.clone()
    }

    async fn read(&self, uri: &str) -> FrameworkResult<JsonValue> {
        let handler = self
            .handlers
            .get(uri)
            .ok_or_else(|| FrameworkError::Protocol(format!("resource not found: {uri}")))?;
        handler().await
    }

    fn has_any(&self) -> bool {
        !self.specs.is_empty()
    }
}

#[derive(Clone, Default)]
struct PromptRegistry {
    specs: Vec<PromptSpec>,
    handlers: HashMap<String, PromptHandler>,
}

impl PromptRegistry {
    fn register<F, Fut>(&mut self, spec: PromptSpec, handler: F)
    where
        F: Fn(JsonValue) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = FrameworkResult<JsonValue>> + Send + 'static,
    {
        self.specs.push(spec.clone());
        self.handlers.insert(
            spec.name.clone(),
            Arc::new(move |arguments| -> JsonFuture { Box::pin(handler(arguments)) }),
        );
    }

    fn list(&self) -> Vec<PromptSpec> {
        self.specs.clone()
    }

    async fn get(&self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| FrameworkError::Protocol(format!("prompt not found: {name}")))?;
        handler(arguments).await
    }

    fn has_any(&self) -> bool {
        !self.specs.is_empty()
    }
}

#[derive(Clone, Default)]
pub struct McpServerBuilder {
    tools: ToolRegistry,
    resources: ResourceRegistry,
    prompts: PromptRegistry,
}

impl McpServerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.register(tool);
        self
    }

    pub fn resource<F, Fut>(mut self, spec: ResourceSpec, handler: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = FrameworkResult<JsonValue>> + Send + 'static,
    {
        self.resources.register(spec, handler);
        self
    }

    pub fn prompt<F, Fut>(mut self, spec: PromptSpec, handler: F) -> Self
    where
        F: Fn(JsonValue) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = FrameworkResult<JsonValue>> + Send + 'static,
    {
        self.prompts.register(spec, handler);
        self
    }

    pub fn build(self) -> McpServer {
        McpServer {
            tools: self.tools,
            resources: self.resources,
            prompts: self.prompts,
        }
    }
}

#[derive(Clone, Default)]
pub struct McpServer {
    tools: ToolRegistry,
    resources: ResourceRegistry,
    prompts: PromptRegistry,
}

impl McpServer {
    pub async fn handle(&self, request: RpcRequest) -> RpcResponse {
        match self.handle_inner(&request).await {
            Ok(result) => RpcResponse::success(request.id, result),
            Err(error) => RpcResponse::error(request.id, -32000, error.to_string()),
        }
    }

    async fn handle_inner(&self, request: &RpcRequest) -> FrameworkResult<JsonValue> {
        match request.method.as_str() {
            "initialize" => Ok(json!({
                "name": "harbor-server",
                "version": "0.1.0",
                "capabilities": self.capabilities(),
            })),
            "tools/list" => Ok(json!({ "tools": self.tools.list() })),
            "tools/call" => {
                let name = request
                    .params
                    .get("name")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| FrameworkError::InvalidArguments("missing tool name".into()))?;
                let arguments = request
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(default_params);
                self.tools.call(name, arguments).await
            }
            "resources/list" => Ok(json!({ "resources": self.resources.list() })),
            "resources/read" => {
                let uri = request
                    .params
                    .get("uri")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| FrameworkError::InvalidArguments("missing resource uri".into()))?;
                self.resources.read(uri).await
            }
            "prompts/list" => Ok(json!({ "prompts": self.prompts.list() })),
            "prompts/get" => {
                let name = request
                    .params
                    .get("name")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| FrameworkError::InvalidArguments("missing prompt name".into()))?;
                let arguments = request
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(default_params);
                self.prompts.get(name, arguments).await
            }
            method => Err(FrameworkError::Protocol(format!(
                "unsupported method: {method}"
            ))),
        }
    }

    pub fn capabilities(&self) -> McpServerCapabilities {
        McpServerCapabilities {
            tools: !self.tools.list().is_empty(),
            resources: self.resources.has_any(),
            prompts: self.prompts.has_any(),
        }
    }
}

pub fn http_router(server: McpServer) -> Router {
    let server = Arc::new(server);
    Router::new().route(
        DEFAULT_HTTP_PATH,
        post({
            let server = server.clone();
            move |Json(request): Json<RpcRequest>| {
                let server = server.clone();
                async move { Json(server.handle(request).await) }
            }
        }),
    )
}

#[derive(Clone)]
pub struct LocalClient {
    server: McpServer,
}

impl LocalClient {
    pub fn new(server: McpServer) -> Self {
        Self { server }
    }

    pub async fn initialize(&self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "local-init".into(),
            method: "initialize".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn list_tools(&self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "local-tools-list".into(),
            method: "tools/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn call_tool(&self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "local-tool-call".into(),
            method: "tools/call".into(),
            params: json!({
                "name": name,
                "arguments": arguments,
            }),
        })
        .await
    }

    pub async fn list_resources(&self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "local-resources-list".into(),
            method: "resources/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn read_resource(&self, uri: &str) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "local-resource-read".into(),
            method: "resources/read".into(),
            params: json!({ "uri": uri }),
        })
        .await
    }

    pub async fn list_prompts(&self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "local-prompts-list".into(),
            method: "prompts/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn get_prompt(&self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "local-prompt-get".into(),
            method: "prompts/get".into(),
            params: json!({
                "name": name,
                "arguments": arguments,
            }),
        })
        .await
    }

    async fn send(&self, request: RpcRequest) -> FrameworkResult<JsonValue> {
        self.server.handle(request).await.into_result()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpClientConfig {
    pub base_url: String,
    pub path: String,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:3000".into(),
            path: DEFAULT_HTTP_PATH.into(),
        }
    }
}

impl HttpClientConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let mut config = Self::default();

        if let Ok(base_url) = env::var("HARBOR_MCP_BASE_URL") {
            config.base_url = base_url;
        }

        if let Ok(path) = env::var("HARBOR_MCP_PATH") {
            config.path = path;
        }

        Ok(config)
    }

    pub fn endpoint_url(&self) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            self.path.trim_start_matches('/'),
        )
    }
}

#[derive(Clone)]
pub struct HttpClientTransport {
    client: Client,
    config: HttpClientConfig,
}

impl HttpClientTransport {
    pub fn new(config: HttpClientConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(HttpClientConfig::from_env()?))
    }

    pub fn with_client(client: Client, config: HttpClientConfig) -> Self {
        Self { client, config }
    }

    pub async fn initialize(&self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "http-init".into(),
            method: "initialize".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn list_tools(&self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "http-tools-list".into(),
            method: "tools/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn call_tool(&self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "http-tool-call".into(),
            method: "tools/call".into(),
            params: json!({
                "name": name,
                "arguments": arguments,
            }),
        })
        .await
    }

    pub async fn list_resources(&self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "http-resources-list".into(),
            method: "resources/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn read_resource(&self, uri: &str) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "http-resource-read".into(),
            method: "resources/read".into(),
            params: json!({ "uri": uri }),
        })
        .await
    }

    pub async fn list_prompts(&self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "http-prompts-list".into(),
            method: "prompts/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn get_prompt(&self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "http-prompt-get".into(),
            method: "prompts/get".into(),
            params: json!({
                "name": name,
                "arguments": arguments,
            }),
        })
        .await
    }

    async fn send(&self, request: RpcRequest) -> FrameworkResult<JsonValue> {
        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);

        let response = self
            .client
            .post(self.config.endpoint_url())
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|error| FrameworkError::Transport(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Transport(error.to_string()))?;

        response
            .json::<RpcResponse>()
            .await
            .map_err(|error| FrameworkError::Transport(error.to_string()))?
            .into_result()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdioClientConfig {
    pub command: String,
    pub args: Vec<String>,
}

impl StdioClientConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let command = env::var("HARBOR_MCP_STDIO_COMMAND").map_err(|_| {
            FrameworkError::Config(
                "missing HARBOR_MCP_STDIO_COMMAND for spawned MCP stdio client".into(),
            )
        })?;

        let args = env::var("HARBOR_MCP_STDIO_ARGS_JSON")
            .ok()
            .map(|value| serde_json::from_str::<Vec<String>>(&value))
            .transpose()
            .map_err(|error| FrameworkError::Config(error.to_string()))?
            .unwrap_or_default();

        Ok(Self { command, args })
    }
}

pub struct SpawnedStdioClient {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

impl SpawnedStdioClient {
    pub async fn spawn(config: StdioClientConfig) -> FrameworkResult<Self> {
        let mut child = Command::new(&config.command)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| FrameworkError::Transport(error.to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| FrameworkError::Transport("failed to capture child stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| FrameworkError::Transport("failed to capture child stdout".into()))?;

        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    pub async fn spawn_from_env() -> FrameworkResult<Self> {
        Self::spawn(StdioClientConfig::from_env()?).await
    }

    pub async fn initialize(&mut self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "stdio-init".into(),
            method: "initialize".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn list_tools(&mut self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "stdio-tools-list".into(),
            method: "tools/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn call_tool(&mut self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "stdio-tool-call".into(),
            method: "tools/call".into(),
            params: json!({
                "name": name,
                "arguments": arguments,
            }),
        })
        .await
    }

    pub async fn list_resources(&mut self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "stdio-resources-list".into(),
            method: "resources/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn read_resource(&mut self, uri: &str) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "stdio-resource-read".into(),
            method: "resources/read".into(),
            params: json!({ "uri": uri }),
        })
        .await
    }

    pub async fn list_prompts(&mut self) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "stdio-prompts-list".into(),
            method: "prompts/list".into(),
            params: default_params(),
        })
        .await
    }

    pub async fn get_prompt(&mut self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        self.send(RpcRequest {
            id: "stdio-prompt-get".into(),
            method: "prompts/get".into(),
            params: json!({
                "name": name,
                "arguments": arguments,
            }),
        })
        .await
    }

    pub async fn shutdown(&mut self) -> FrameworkResult<()> {
        if self.child.id().is_some() {
            let _ = self.child.kill().await;
        }
        let _ = self.child.wait().await;
        Ok(())
    }

    async fn send(&mut self, request: RpcRequest) -> FrameworkResult<JsonValue> {
        let value = serde_json::to_value(&request)?;
        write_stdio_message(&mut self.stdin, &value)
            .await
            .map_err(|error| FrameworkError::Transport(error.to_string()))?;

        let response = read_stdio_message(&mut self.stdout)
            .await
            .map_err(|error| FrameworkError::Transport(error.to_string()))?
            .ok_or_else(|| {
                FrameworkError::Transport("spawned MCP stdio process closed stdout".into())
            })?;

        serde_json::from_value::<RpcResponse>(response)
            .map_err(|error| FrameworkError::Transport(error.to_string()))?
            .into_result()
    }
}

pub async fn write_stdio_message<W>(writer: &mut W, value: &JsonValue) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let body = serde_json::to_vec(value)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer.write_all(header.as_bytes()).await?;
    writer.write_all(&body).await?;
    writer.flush().await
}

pub async fn read_stdio_message<R>(reader: &mut R) -> std::io::Result<Option<JsonValue>>
where
    R: AsyncRead + Unpin,
{
    let mut header = Vec::new();
    let mut byte = [0u8; 1];

    loop {
        let read = reader.read(&mut byte).await?;
        if read == 0 {
            if header.is_empty() {
                return Ok(None);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading stdio header",
            ));
        }

        header.push(byte[0]);
        if header.ends_with(b"\r\n\r\n") {
            break;
        }
    }

    let header_text = String::from_utf8_lossy(&header);
    let length = header_text
        .lines()
        .find_map(|line| {
            line.strip_prefix("Content-Length:")
                .map(|value| value.trim().parse::<usize>())
        })
        .transpose()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing Content-Length header")
        })?;

    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).await?;

    let value = serde_json::from_slice(&body)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

    Ok(Some(value))
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
    use axum::{extract::State, http::HeaderMap as AxumHeaderMap};
    use std::sync::Mutex;
    use tokio::{net::TcpListener, time::timeout};

    #[derive(Clone, Default)]
    struct CaptureState {
        headers: Arc<Mutex<Vec<(String, String)>>>,
    }

    #[derive(Clone)]
    struct TestState {
        server: McpServer,
        capture: CaptureState,
    }

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> harbor_core::ToolSpec {
            harbor_core::ToolSpec::new(
                "echo",
                "Echo text back",
                json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" }
                    },
                    "required": ["text"]
                }),
            )
        }

        async fn call(&self, args: JsonValue) -> FrameworkResult<JsonValue> {
            Ok(json!({
                "echo": args.get("text").and_then(|value| value.as_str()).unwrap_or_default()
            }))
        }
    }

    fn build_test_server() -> McpServer {
        McpServerBuilder::new()
            .tool(EchoTool)
            .resource(
                ResourceSpec::new("harbor://docs/getting-started", "Getting Started")
                    .with_description("Starter guide")
                    .with_mime_type("text/markdown"),
                || async {
                    Ok(json!({
                        "uri": "harbor://docs/getting-started",
                        "mimeType": "text/markdown",
                        "contents": "# Harbor\nStart here."
                    }))
                },
            )
            .prompt(
                PromptSpec::new("welcome")
                    .with_description("Build a welcome prompt")
                    .with_argument(
                        PromptArgumentSpec::new("name", true).with_description("User name"),
                    ),
                |arguments| async move {
                    let name = arguments
                        .get("name")
                        .and_then(|value| value.as_str())
                        .unwrap_or("friend")
                        .to_string();
                    Ok(json!({
                        "messages": [
                            {
                                "role": "system",
                                "content": format!("Welcome, {}", name)
                            }
                        ]
                    }))
                },
            )
            .build()
    }

    #[tokio::test]
    async fn http_client_transport_calls_remote_mcp_and_injects_trace_context() {
        let server = build_test_server();
        let state = TestState {
            server,
            capture: CaptureState::default(),
        };
        let router = Router::new()
            .route(DEFAULT_HTTP_PATH, post(capturing_http_handler))
            .with_state(state.clone());
        let base_url = spawn_router(router).await;

        let client = HttpClientTransport::new(HttpClientConfig {
            base_url,
            path: DEFAULT_HTTP_PATH.into(),
        });

        let init = client.initialize().await.unwrap();
        assert_eq!(init["name"], "harbor-server");
        assert_eq!(init["capabilities"]["tools"], true);
        assert_eq!(init["capabilities"]["resources"], true);
        assert_eq!(init["capabilities"]["prompts"], true);

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools["tools"][0]["name"], "echo");

        let resources = client.list_resources().await.unwrap();
        assert_eq!(resources["resources"][0]["uri"], "harbor://docs/getting-started");

        let resource = client
            .read_resource("harbor://docs/getting-started")
            .await
            .unwrap();
        assert_eq!(resource["mimeType"], "text/markdown");

        let prompts = client.list_prompts().await.unwrap();
        assert_eq!(prompts["prompts"][0]["name"], "welcome");

        let prompt = client
            .get_prompt("welcome", json!({ "name": "Devarshi" }))
            .await
            .unwrap();
        assert_eq!(prompt["messages"][0]["content"], "Welcome, Devarshi");

        let called = client
            .call_tool("echo", json!({ "text": "hello mcp" }))
            .await
            .unwrap();
        assert_eq!(called["echo"], "hello mcp");

        let headers = state.capture.headers.lock().unwrap().clone();
        let headers = headers
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        if let Some(traceparent) = headers.get("traceparent") {
            assert!(traceparent.starts_with("00-"));
        }
    }

    #[tokio::test]
    async fn http_router_returns_rpc_error_for_unknown_method() {
        let server = build_test_server();
        let base_url = spawn_router(http_router(server)).await;

        let response = reqwest::Client::new()
            .post(format!("{base_url}{DEFAULT_HTTP_PATH}"))
            .json(&RpcRequest {
                id: "bad-1".into(),
                method: "unknown".into(),
                params: default_params(),
            })
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let rpc: RpcResponse = response.json().await.unwrap();
        assert_eq!(rpc.id, "bad-1");
        assert!(rpc.result.is_none());
        assert!(rpc.error.unwrap().message.contains("unsupported method"));
    }

    #[tokio::test]
    async fn spawned_stdio_client_calls_remote_process() {
        let script = r#"
import json, sys

def read_message():
    header = b''
    while not header.endswith(b'\r\n\r\n'):
        chunk = sys.stdin.buffer.read(1)
        if not chunk:
            return None
        header += chunk
    length = 0
    for line in header.decode().split('\r\n'):
        if line.lower().startswith('content-length:'):
            length = int(line.split(':', 1)[1].strip())
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode())

message = read_message()
if message is not None:
    payload = json.dumps({
        'id': message['id'],
        'result': {'name': 'stdio-test-server'}
    }).encode()
    header = f'Content-Length: {len(payload)}\r\n\r\n'.encode()
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
"#;

        let script_path = std::env::temp_dir().join(format!(
            "harbor_mcp_stdio_test_{}.py",
            std::process::id()
        ));
        std::fs::write(&script_path, script).unwrap();

        let mut client = SpawnedStdioClient::spawn(StdioClientConfig {
            command: "/usr/bin/python3".into(),
            args: vec!["-u".into(), script_path.display().to_string()],
        })
        .await
        .unwrap();

        let init = timeout(std::time::Duration::from_secs(5), client.initialize())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(init["name"], "stdio-test-server");

        timeout(std::time::Duration::from_secs(5), client.shutdown())
            .await
            .unwrap()
            .unwrap();

        let _ = std::fs::remove_file(script_path);
    }

    #[tokio::test]
    async fn stdio_client_config_reads_env() {
        unsafe {
            env::set_var("HARBOR_MCP_STDIO_COMMAND", "/usr/bin/python3");
            env::set_var(
                "HARBOR_MCP_STDIO_ARGS_JSON",
                "[\"-u\",\"-c\",\"print('hi')\"]",
            );
        }

        let config = StdioClientConfig::from_env().unwrap();
        assert_eq!(config.command, "/usr/bin/python3");
        assert_eq!(config.args, vec!["-u", "-c", "print('hi')"]);

        unsafe {
            env::remove_var("HARBOR_MCP_STDIO_COMMAND");
            env::remove_var("HARBOR_MCP_STDIO_ARGS_JSON");
        }
    }

    async fn capturing_http_handler(
        State(state): State<TestState>,
        headers: AxumHeaderMap,
        Json(request): Json<RpcRequest>,
    ) -> Json<RpcResponse> {
        {
            let mut guard = state.capture.headers.lock().unwrap();
            for (name, value) in headers.iter() {
                if let Ok(value) = value.to_str() {
                    guard.push((name.to_string(), value.to_string()));
                }
            }
        }

        Json(state.server.handle(request).await)
    }

    async fn spawn_router(router: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{}", address)
    }
}
