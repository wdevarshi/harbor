use harbor_ai::{CompletionEvent, MockProvider};
use harbor_memory::InMemorySessionMemory;
use harbor_runtime::AgentRuntime;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AgentRuntime::new(MockProvider, InMemorySessionMemory::default())
        .with_default_model("mock-gpt");

    let mut stream = runtime
        .run_turn_stream("demo-session", "Hello from the Harbor streaming example")
        .await?;

    while let Some(event) = stream.recv().await {
        match event? {
            CompletionEvent::Started { provider, model } => {
                println!("stream started via {provider}:{model}");
            }
            CompletionEvent::Delta { text } => {
                print!("{text}");
                io::stdout().flush()?;
            }
            CompletionEvent::Finished { .. } => {
                println!();
            }
        }
    }

    Ok(())
}
