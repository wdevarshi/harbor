use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMessage {
    pub role: String,
    pub content: String,
}

impl MemoryMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[async_trait]
pub trait SessionMemory: Send + Sync + Clone + 'static {
    async fn messages(&self, session_id: &str) -> Vec<MemoryMessage>;
    async fn append(&self, session_id: &str, message: MemoryMessage);
    async fn clear(&self, session_id: &str);

    fn create_session(&self) -> String {
        Uuid::new_v4().to_string()
    }
}

#[derive(Debug, Clone, Default)]
pub struct InMemorySessionMemory {
    inner: Arc<RwLock<HashMap<String, Vec<MemoryMessage>>>>,
}

#[async_trait]
impl SessionMemory for InMemorySessionMemory {
    async fn messages(&self, session_id: &str) -> Vec<MemoryMessage> {
        self.inner
            .read()
            .await
            .get(session_id)
            .cloned()
            .unwrap_or_default()
    }

    async fn append(&self, session_id: &str, message: MemoryMessage) {
        let mut guard = self.inner.write().await;
        guard.entry(session_id.to_string()).or_default().push(message);
    }

    async fn clear(&self, session_id: &str) {
        self.inner.write().await.remove(session_id);
    }
}
