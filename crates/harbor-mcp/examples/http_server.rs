use async_trait::async_trait;
use harbor_core::{FrameworkResult, JsonValue, Tool, ToolSpec};
use harbor_mcp::{http_router, McpServerBuilder, DEFAULT_HTTP_PATH};
use serde_json::json;
use tokio::net::TcpListener;

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "echo",
            "Echo text back over Harbor MCP HTTP",
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = McpServerBuilder::new().tool(EchoTool).build();
    let listener = TcpListener::bind("127.0.0.1:4001").await?;

    println!(
        "Harbor MCP HTTP server listening on http://127.0.0.1:4001{}",
        DEFAULT_HTTP_PATH
    );

    axum::serve(listener, http_router(server)).await?;
    Ok(())
}
