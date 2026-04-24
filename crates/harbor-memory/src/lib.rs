use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{fs, sync::{Mutex, RwLock}};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone)]
pub struct FileSessionMemory {
    root: Arc<PathBuf>,
    write_lock: Arc<Mutex<()>>,
}

impl FileSessionMemory {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: Arc::new(root.into()),
            write_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn root(&self) -> &Path {
        self.root.as_ref().as_path()
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.root.join(format!("{session_id}.json"))
    }

    async fn read_messages(&self, session_id: &str) -> Vec<MemoryMessage> {
        let path = self.session_path(session_id);
        match fs::read_to_string(path).await {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    async fn write_messages(&self, session_id: &str, messages: &[MemoryMessage]) {
        let path = self.session_path(session_id);
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent).await;
        }
        if let Ok(content) = serde_json::to_string_pretty(messages) {
            let _ = fs::write(path, content).await;
        }
    }
}

#[async_trait]
impl SessionMemory for FileSessionMemory {
    async fn messages(&self, session_id: &str) -> Vec<MemoryMessage> {
        self.read_messages(session_id).await
    }

    async fn append(&self, session_id: &str, message: MemoryMessage) {
        let _guard = self.write_lock.lock().await;
        let mut messages = self.read_messages(session_id).await;
        messages.push(message);
        self.write_messages(session_id, &messages).await;
    }

    async fn clear(&self, session_id: &str) {
        let _guard = self.write_lock.lock().await;
        let _ = fs::remove_file(self.session_path(session_id)).await;
    }
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

    #[tokio::test]
    async fn file_session_memory_persists_messages_between_instances() {
        let root = temp_dir("harbor-memory-test");
        let session_id = "demo-session";

        let memory = FileSessionMemory::new(&root);
        memory
            .append(session_id, MemoryMessage::new("user", "hello"))
            .await;
        memory
            .append(session_id, MemoryMessage::new("assistant", "world"))
            .await;

        let restored = FileSessionMemory::new(&root);
        let messages = restored.messages(session_id).await;

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0], MemoryMessage::new("user", "hello"));
        assert_eq!(messages[1], MemoryMessage::new("assistant", "world"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn file_session_memory_clear_removes_persisted_state() {
        let root = temp_dir("harbor-memory-clear-test");
        let session_id = "clear-session";

        let memory = FileSessionMemory::new(&root);
        memory
            .append(session_id, MemoryMessage::new("user", "bye"))
            .await;
        memory.clear(session_id).await;

        let restored = FileSessionMemory::new(&root);
        assert!(restored.messages(session_id).await.is_empty());

        let _ = std::fs::remove_dir_all(root);
    }
}
