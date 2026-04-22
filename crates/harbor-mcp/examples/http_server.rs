use async_trait::async_trait;
use harbor_core::{FrameworkResult, JsonValue, Tool, ToolSpec};
use harbor_mcp::{
    http_router, McpServerBuilder, PromptArgumentSpec, PromptSpec, ResourceSpec,
    DEFAULT_HTTP_PATH,
};
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
    let server = McpServerBuilder::new()
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
        .build();

    let listener = TcpListener::bind("127.0.0.1:4001").await?;

    println!(
        "Harbor MCP HTTP server listening on http://127.0.0.1:4001{}",
        DEFAULT_HTTP_PATH
    );

    axum::serve(listener, http_router(server)).await?;
    Ok(())
}
