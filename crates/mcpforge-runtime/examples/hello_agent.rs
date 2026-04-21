use mcpforge_ai::MockProvider;
use mcpforge_memory::InMemorySessionMemory;
use mcpforge_runtime::AgentRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AgentRuntime::new(MockProvider, InMemorySessionMemory::default())
        .with_default_model("mock-gpt");

    let response = runtime
        .run_turn("demo-session", "Hello from the MCPForge hello agent example")
        .await?;

    println!("{}", response.text);
    Ok(())
}
