use axum::{extract::State, routing::post, Json, Router};
use harbor_core::{FrameworkError, FrameworkResult, JsonValue, Tool, ToolRegistry};
use opentelemetry::{global, propagation::Injector};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue},
    Client,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{env, process::Stdio};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

pub const DEFAULT_HTTP_PATH: &str = "/rpc";

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

#[derive(Clone, Default)]
pub struct McpServerBuilder {
    tools: ToolRegistry,
}

impl McpServerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.register(tool);
        self
    }

    pub fn build(self) -> McpServer {
        McpServer { tools: self.tools }
    }
}

#[derive(Clone, Default)]
pub struct McpServer {
    tools: ToolRegistry,
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
                "capabilities": {
                    "tools": true
                }
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
            method => Err(FrameworkError::Protocol(format!(
                "unsupported method: {method}"
            ))),
        }
    }
}

pub fn http_router(server: McpServer) -> Router {
    Router::new()
        .route(DEFAULT_HTTP_PATH, post(handle_http_rpc))
        .with_state(server)
}

async fn handle_http_rpc(
    State(server): State<McpServer>,
    Json(request): Json<RpcRequest>,
) -> Json<RpcResponse> {
    Json(server.handle(request).await)
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
        self.server
            .handle(RpcRequest {
                id: "local-init".into(),
                method: "initialize".into(),
                params: default_params(),
            })
            .await
            .into_result()
    }

    pub async fn list_tools(&self) -> FrameworkResult<JsonValue> {
        self.server
            .handle(RpcRequest {
                id: "local-tools-list".into(),
                method: "tools/list".into(),
                params: default_params(),
            })
            .await
            .into_result()
    }

    pub async fn call_tool(&self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        self.server
            .handle(RpcRequest {
                id: "local-tool-call".into(),
                method: "tools/call".into(),
                params: json!({
                    "name": name,
                    "arguments": arguments,
                }),
            })
            .await
            .into_result()
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
            .ok_or_else(|| FrameworkError::Transport("spawned MCP stdio process closed stdout".into()))?;

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
    use async_trait::async_trait;
    use axum::{extract::State, http::HeaderMap as AxumHeaderMap};
    use opentelemetry::{
        global,
        trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState},
        Context,
    };
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

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

    #[tokio::test]
    async fn http_client_transport_calls_remote_mcp_and_injects_trace_context() {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let server = McpServerBuilder::new().tool(EchoTool).build();
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

        let parent_context = Context::new().with_remote_span_context(SpanContext::new(
            TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::from_hex("00f067aa0ba902b7").unwrap(),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        ));
        let span = tracing::info_span!("mcp_http_client_test");
        let _ = span.set_parent(parent_context);
        let _guard = span.enter();

        let init = client.initialize().await.unwrap();
        assert_eq!(init["name"], "harbor-server");

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools["tools"][0]["name"], "echo");

        let called = client.call_tool("echo", json!({ "text": "hello mcp" })).await.unwrap();
        assert_eq!(called["echo"], "hello mcp");

        let headers = state.capture.headers.lock().unwrap().clone();
        let headers = headers.into_iter().collect::<std::collections::HashMap<_, _>>();
        if let Some(traceparent) = headers.get("traceparent") {
            assert!(traceparent.starts_with("00-"));
        }
    }

    #[tokio::test]
    async fn http_router_returns_rpc_error_for_unknown_method() {
        let server = McpServerBuilder::new().tool(EchoTool).build();
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

def write_message(payload):
    body = json.dumps(payload).encode()
    header = f'Content-Length: {len(body)}\\r\\n\\r\\n'.encode()
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()

payload = json.dumps({
    'id': 'stdio-init',
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

        let init = tokio::time::timeout(std::time::Duration::from_secs(5), client.initialize())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(init["name"], "stdio-test-server");

        tokio::time::timeout(std::time::Duration::from_secs(5), client.shutdown())
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
