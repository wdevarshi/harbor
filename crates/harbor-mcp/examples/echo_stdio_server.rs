use async_trait::async_trait;
use harbor_core::{typed_tool_spec, FrameworkResult, JsonValue, Schema, Tool, ToolSpec, TypedSchema};
use harbor_mcp::{read_stdio_message, write_stdio_message, McpServerBuilder, RpcRequest};
use serde_json::json;
use tokio::io::{stdin, stdout};

struct EchoTool;

struct EchoInputSchema;

impl TypedSchema for EchoInputSchema {
    fn schema() -> JsonValue {
        Schema::object()
            .required_property(
                "text",
                Schema::string().with_description("Text to echo back"),
            )
            .additional_properties(false)
            .build()
            .into_json()
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        typed_tool_spec::<EchoInputSchema>("echo", "Echo back the provided text")
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
