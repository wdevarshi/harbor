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

    println!(
        "runtime stream ready run_id={} stream_id={}",
        stream.run_id(),
        stream.stream_id()
    );

    while let Some(event) = stream.recv().await {
        match event? {
            CompletionEvent::Started {
                run_id,
                stream_id,
                sequence,
                provider,
                model,
            } => {
                println!(
                    "stream started run_id={run_id} stream_id={stream_id} seq={sequence} via {provider}:{model}"
                );
            }
            CompletionEvent::Delta {
                sequence,
                offset,
                text,
                ..
            } => {
                print!("[seq={sequence} offset={offset}] {text}");
                io::stdout().flush()?;
            }
            CompletionEvent::Finished {
                run_id,
                stream_id,
                sequence,
                response,
            } => {
                println!(
                    "\nstream finished run_id={run_id} stream_id={stream_id} seq={sequence} text_len={}",
                    response.text.len()
                );
            }
        }
    }

    Ok(())
}
