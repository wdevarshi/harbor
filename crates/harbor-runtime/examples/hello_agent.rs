use harbor_ai::MockProvider;
use harbor_memory::InMemorySessionMemory;
use harbor_runtime::AgentRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AgentRuntime::new(MockProvider, InMemorySessionMemory::default())
        .with_default_model("mock-gpt");

    let response = runtime
        .run_turn("demo-session", "Hello from the Harbor hello agent example")
        .await?;

    println!("{}", response.text);
    Ok(())
}
