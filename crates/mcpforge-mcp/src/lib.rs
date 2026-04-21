use mcpforge_core::{FrameworkError, FrameworkResult, JsonValue, Tool, ToolRegistry};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

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
                "name": "mcpforge-server",
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

#[derive(Clone)]
pub struct LocalClient {
    server: McpServer,
}

impl LocalClient {
    pub fn new(server: McpServer) -> Self {
        Self { server }
    }

    pub async fn list_tools(&self) -> FrameworkResult<JsonValue> {
        let response = self
            .server
            .handle(RpcRequest {
                id: "local-tools-list".into(),
                method: "tools/list".into(),
                params: default_params(),
            })
            .await;

        response.result.ok_or_else(|| {
            FrameworkError::Protocol(
                response
                    .error
                    .map(|error| error.message)
                    .unwrap_or_else(|| "missing result".into()),
            )
        })
    }

    pub async fn call_tool(&self, name: &str, arguments: JsonValue) -> FrameworkResult<JsonValue> {
        let response = self
            .server
            .handle(RpcRequest {
                id: "local-tool-call".into(),
                method: "tools/call".into(),
                params: json!({
                    "name": name,
                    "arguments": arguments,
                }),
            })
            .await;

        response.result.ok_or_else(|| {
            FrameworkError::Protocol(
                response
                    .error
                    .map(|error| error.message)
                    .unwrap_or_else(|| "missing result".into()),
            )
        })
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
