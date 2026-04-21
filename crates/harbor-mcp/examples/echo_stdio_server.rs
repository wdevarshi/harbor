use async_trait::async_trait;
use harbor_core::{FrameworkResult, JsonValue, Tool, ToolSpec};
use harbor_mcp::{read_stdio_message, write_stdio_message, McpServerBuilder, RpcRequest};
use serde_json::json;
use tokio::io::{stdin, stdout};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "echo",
            "Echo back the provided text",
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
            "echo": args
                .get("text")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = McpServerBuilder::new().tool(EchoTool).build();
    let mut input = stdin();
    let mut output = stdout();

    while let Some(message) = read_stdio_message(&mut input).await? {
        let request: RpcRequest = serde_json::from_value(message)?;
        let response = server.handle(request).await;
        write_stdio_message(&mut output, &serde_json::to_value(response)?).await?;
    }

    Ok(())
}
