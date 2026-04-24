use harbor_ai::MockProvider;
use harbor_memory::InMemorySessionMemory;
use harbor_rag::{Document, DocumentStore, InMemoryDocumentStore, LexicalRetriever, RetrievalQuery};
use harbor_runtime::AgentRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = AgentRuntime::new(MockProvider, InMemorySessionMemory::default())
        .with_default_model("mock-gpt");

    let store = InMemoryDocumentStore::default();
    store
        .put(Document::new(
            "Operations Runbook",
            "If the queue backs up, inspect worker logs before scaling the service.",
        ))
        .await?;

    let retriever = LexicalRetriever::new(store);
    let response = runtime
        .run_turn_with_retrieval(
            "demo-session",
            "How should I inspect queue logs?",
            &retriever,
            RetrievalQuery::default(),
        )
        .await?;

    println!("{}", response.text);
    Ok(())
}
