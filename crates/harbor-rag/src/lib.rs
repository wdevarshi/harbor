use async_trait::async_trait;
use harbor_core::{FrameworkResult, JsonValue};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    fs,
    sync::{Mutex, RwLock},
};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub text: String,
    pub metadata: JsonValue,
}

impl Document {
    pub fn new(title: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: title.into(),
            text: text.into(),
            metadata: JsonValue::Object(Default::default()),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_metadata(mut self, metadata: JsonValue) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentChunk {
    pub document_id: String,
    pub document_title: String,
    pub chunk_index: usize,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievedChunk {
    pub document_id: String,
    pub document_title: String,
    pub chunk_index: usize,
    pub content: String,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkingConfig {
    pub max_chars: usize,
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chars: 800,
            overlap: 120,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrievalQuery {
    pub limit: usize,
    pub chunking: ChunkingConfig,
}

impl Default for RetrievalQuery {
    fn default() -> Self {
        Self {
            limit: 4,
            chunking: ChunkingConfig::default(),
        }
    }
}

pub fn chunk_document(document: &Document, config: ChunkingConfig) -> Vec<DocumentChunk> {
    if document.text.trim().is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = document.text.chars().collect();
    let mut start = 0usize;
    let mut chunks = Vec::new();
    let step = config.max_chars.saturating_sub(config.overlap).max(1);

    while start < chars.len() {
        let end = (start + config.max_chars.max(1)).min(chars.len());
        let content = chars[start..end].iter().collect::<String>().trim().to_string();
        if !content.is_empty() {
            chunks.push(DocumentChunk {
                document_id: document.id.clone(),
                document_title: document.title.clone(),
                chunk_index: chunks.len(),
                content,
            });
        }

        if end == chars.len() {
            break;
        }
        start = start.saturating_add(step);
    }

    chunks
}

pub fn render_retrieved_context(chunks: &[RetrievedChunk]) -> Option<String> {
    if chunks.is_empty() {
        return None;
    }

    let mut rendered = String::from(
        "Use the following retrieved document context when answering. Cite document titles where helpful:\n",
    );

    for chunk in chunks {
        rendered.push_str(&format!(
            "\n[{} :: chunk {} :: score {:.2}]\n{}\n",
            chunk.document_title, chunk.chunk_index, chunk.score, chunk.content
        ));
    }

    Some(rendered)
}

#[async_trait]
pub trait DocumentStore: Send + Sync + Clone + 'static {
    async fn put(&self, document: Document) -> FrameworkResult<Document>;
    async fn get(&self, id: &str) -> FrameworkResult<Option<Document>>;
    async fn list(&self) -> FrameworkResult<Vec<Document>>;
    async fn delete(&self, id: &str) -> FrameworkResult<()>;
}

#[async_trait]
pub trait Retriever: Send + Sync {
    async fn retrieve(
        &self,
        query: &str,
        options: RetrievalQuery,
    ) -> FrameworkResult<Vec<RetrievedChunk>>;
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryDocumentStore {
    inner: Arc<RwLock<HashMap<String, Document>>>,
}

#[async_trait]
impl DocumentStore for InMemoryDocumentStore {
    async fn put(&self, document: Document) -> FrameworkResult<Document> {
        self.inner
            .write()
            .await
            .insert(document.id.clone(), document.clone());
        Ok(document)
    }

    async fn get(&self, id: &str) -> FrameworkResult<Option<Document>> {
        Ok(self.inner.read().await.get(id).cloned())
    }

    async fn list(&self) -> FrameworkResult<Vec<Document>> {
        let mut documents = self.inner.read().await.values().cloned().collect::<Vec<_>>();
        documents.sort_by(|a, b| a.title.cmp(&b.title));
        Ok(documents)
    }

    async fn delete(&self, id: &str) -> FrameworkResult<()> {
        self.inner.write().await.remove(id);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FileDocumentStore {
    root: Arc<PathBuf>,
    write_lock: Arc<Mutex<()>>,
}

impl FileDocumentStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: Arc::new(root.into()),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn root(&self) -> &Path {
        self.root.as_ref().as_path()
    }

    fn document_path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.json"))
    }
}

#[async_trait]
impl DocumentStore for FileDocumentStore {
    async fn put(&self, document: Document) -> FrameworkResult<Document> {
        let _guard = self.write_lock.lock().await;
        fs::create_dir_all(self.root()).await?;
        let path = self.document_path(&document.id);
        let content = serde_json::to_string_pretty(&document)?;
        fs::write(path, content).await?;
        Ok(document)
    }

    async fn get(&self, id: &str) -> FrameworkResult<Option<Document>> {
        let path = self.document_path(id);
        match fs::read_to_string(path).await {
            Ok(content) => Ok(Some(serde_json::from_str(&content)?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    async fn list(&self) -> FrameworkResult<Vec<Document>> {
        fs::create_dir_all(self.root()).await?;
        let mut entries = fs::read_dir(self.root()).await?;
        let mut documents: Vec<Document> = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                let content = fs::read_to_string(entry.path()).await?;
                documents.push(serde_json::from_str(&content)?);
            }
        }

        documents.sort_by(|a, b| a.title.cmp(&b.title));
        Ok(documents)
    }

    async fn delete(&self, id: &str) -> FrameworkResult<()> {
        let _guard = self.write_lock.lock().await;
        let path = self.document_path(id);
        match fs::remove_file(path).await {
            Ok(_) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LexicalRetriever<S> {
    store: S,
}

impl<S> LexicalRetriever<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }
}

#[async_trait]
impl<S> Retriever for LexicalRetriever<S>
where
    S: DocumentStore,
{
    async fn retrieve(
        &self,
        query: &str,
        options: RetrievalQuery,
    ) -> FrameworkResult<Vec<RetrievedChunk>> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let documents = self.store.list().await?;
        let mut scored = Vec::new();

        for document in documents {
            for chunk in chunk_document(&document, options.chunking) {
                let score = lexical_score(&query_terms, &chunk.content);
                if score > 0.0 {
                    scored.push(RetrievedChunk {
                        document_id: chunk.document_id,
                        document_title: chunk.document_title,
                        chunk_index: chunk.chunk_index,
                        content: chunk.content,
                        score,
                    });
                }
            }
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.document_title.cmp(&b.document_title))
                .then_with(|| a.chunk_index.cmp(&b.chunk_index))
        });
        scored.truncate(options.limit.max(1));
        Ok(scored)
    }
}

fn lexical_score(query_terms: &[String], text: &str) -> f32 {
    let content_terms = tokenize(text);
    if content_terms.is_empty() {
        return 0.0;
    }

    let mut counts = HashMap::<String, usize>::new();
    for term in content_terms {
        *counts.entry(term).or_default() += 1;
    }

    query_terms
        .iter()
        .map(|term| counts.get(term).copied().unwrap_or_default() as f32)
        .sum()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{suffix}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn chunk_document_splits_large_documents() {
        let document = Document::new("Guide", "abcdefghij").with_id("doc-1");
        let chunks = chunk_document(
            &document,
            ChunkingConfig {
                max_chars: 4,
                overlap: 1,
            },
        );

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content, "abcd");
        assert_eq!(chunks[1].content, "defg");
        assert_eq!(chunks[2].content, "ghij");
    }

    #[tokio::test]
    async fn file_document_store_persists_documents() {
        let root = temp_dir("harbor-rag-store-test");
        let store = FileDocumentStore::new(&root);
        let document = Document::new("Runbook", "restart the worker").with_id("doc-1");
        store.put(document.clone()).await.unwrap();

        let restored = FileDocumentStore::new(&root);
        let fetched = restored.get("doc-1").await.unwrap().unwrap();
        assert_eq!(fetched, document);

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn lexical_retriever_returns_best_matching_chunks() {
        let store = InMemoryDocumentStore::default();
        store
            .put(Document::new(
                "Runbook",
                "restart the worker if the queue stalls and inspect the logs",
            ))
            .await
            .unwrap();
        store
            .put(Document::new(
                "Playbook",
                "customer support escalation process for billing requests",
            ))
            .await
            .unwrap();

        let retriever = LexicalRetriever::new(store);
        let results = retriever
            .retrieve(
                "queue logs",
                RetrievalQuery {
                    limit: 2,
                    chunking: ChunkingConfig {
                        max_chars: 120,
                        overlap: 10,
                    },
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_title, "Runbook");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn render_retrieved_context_formats_chunks_for_prompt_injection() {
        let rendered = render_retrieved_context(&[RetrievedChunk {
            document_id: "doc-1".into(),
            document_title: "Runbook".into(),
            chunk_index: 0,
            content: "restart the worker".into(),
            score: 2.0,
        }])
        .unwrap();

        assert!(rendered.contains("Runbook"));
        assert!(rendered.contains("restart the worker"));
    }
}
