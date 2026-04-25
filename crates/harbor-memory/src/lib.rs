use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs,
    sync::{Mutex, RwLock},
};
use uuid::Uuid;

pub const FILE_SESSION_MEMORY_SCHEMA_VERSION: u32 = 1;
pub const SESSION_MEMORY_MANIFEST_FILE_NAME: &str = ".harbor-session-memory.json";

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionMemoryManifest {
    pub store_kind: String,
    pub schema_version: u32,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl SessionMemoryManifest {
    fn current(created_at_ms: Option<u64>) -> Self {
        let now = now_millis();
        Self {
            store_kind: "file_session_memory".into(),
            schema_version: FILE_SESSION_MEMORY_SCHEMA_VERSION,
            created_at_ms: created_at_ms.unwrap_or(now),
            updated_at_ms: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileSessionMemoryBootstrap {
    pub created_manifest: bool,
    pub migrated_sessions: usize,
    pub schema_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SessionMemoryFileRecord {
    #[serde(default = "file_session_memory_schema_version")]
    schema_version: u32,
    session_id: String,
    #[serde(default)]
    messages: Vec<MemoryMessage>,
}

fn file_session_memory_schema_version() -> u32 {
    FILE_SESSION_MEMORY_SCHEMA_VERSION
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

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join(SESSION_MEMORY_MANIFEST_FILE_NAME)
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.root.join(format!("{session_id}.json"))
    }

    pub async fn bootstrap(&self) -> io::Result<FileSessionMemoryBootstrap> {
        let _guard = self.write_lock.lock().await;
        self.bootstrap_locked().await
    }

    async fn bootstrap_locked(&self) -> io::Result<FileSessionMemoryBootstrap> {
        fs::create_dir_all(self.root()).await?;

        let manifest_path = self.manifest_path();
        let existing_manifest = self.read_manifest().await?;
        let created_manifest = existing_manifest.is_none();
        let mut manifest = existing_manifest
            .unwrap_or_else(|| SessionMemoryManifest::current(None));
        let mut migrated_sessions = 0usize;

        let mut entries = fs::read_dir(self.root()).await?;
        while let Some(entry) = entries.next_entry().await? {
            if !entry.file_type().await?.is_file() {
                continue;
            }

            let path = entry.path();
            if path == manifest_path {
                continue;
            }

            let Some(session_id) = path
                .file_stem()
                .and_then(|value| value.to_str())
                .map(str::to_string)
            else {
                continue;
            };

            let content = fs::read_to_string(&path).await?;
            let (record, needs_rewrite) = decode_session_record(&session_id, &content)?;
            if needs_rewrite {
                self.write_record(&record).await?;
                migrated_sessions += 1;
            }
        }

        manifest.schema_version = FILE_SESSION_MEMORY_SCHEMA_VERSION;
        manifest.updated_at_ms = now_millis();
        self.write_manifest(&manifest).await?;

        Ok(FileSessionMemoryBootstrap {
            created_manifest,
            migrated_sessions,
            schema_version: FILE_SESSION_MEMORY_SCHEMA_VERSION,
        })
    }

    async fn ensure_bootstrapped(&self) {
        let _ = self.bootstrap().await;
    }

    async fn read_manifest(&self) -> io::Result<Option<SessionMemoryManifest>> {
        match fs::read_to_string(self.manifest_path()).await {
            Ok(content) => Ok(serde_json::from_str(&content).ok()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    async fn write_manifest(&self, manifest: &SessionMemoryManifest) -> io::Result<()> {
        let content = serde_json::to_string_pretty(manifest).map_err(io::Error::other)?;
        fs::write(self.manifest_path(), content).await
    }

    async fn read_record(&self, session_id: &str) -> io::Result<Option<SessionMemoryFileRecord>> {
        match fs::read_to_string(self.session_path(session_id)).await {
            Ok(content) => Ok(Some(decode_session_record(session_id, &content)?.0)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    async fn write_record(&self, record: &SessionMemoryFileRecord) -> io::Result<()> {
        let content = serde_json::to_string_pretty(record).map_err(io::Error::other)?;
        fs::write(self.session_path(&record.session_id), content).await
    }
}

fn decode_session_record(
    session_id: &str,
    content: &str,
) -> io::Result<(SessionMemoryFileRecord, bool)> {
    let value: serde_json::Value = serde_json::from_str(content).map_err(io::Error::other)?;

    if value.is_array() {
        let messages: Vec<MemoryMessage> = serde_json::from_value(value).map_err(io::Error::other)?;
        return Ok((
            SessionMemoryFileRecord {
                schema_version: FILE_SESSION_MEMORY_SCHEMA_VERSION,
                session_id: session_id.to_string(),
                messages,
            },
            true,
        ));
    }

    let mut record: SessionMemoryFileRecord =
        serde_json::from_str(content).map_err(io::Error::other)?;
    let needs_rewrite = value
        .get("schema_version")
        .and_then(|candidate| candidate.as_u64())
        != Some(FILE_SESSION_MEMORY_SCHEMA_VERSION as u64)
        || record.session_id != session_id;

    record.schema_version = FILE_SESSION_MEMORY_SCHEMA_VERSION;
    if record.session_id != session_id {
        record.session_id = session_id.to_string();
    }

    Ok((record, needs_rewrite))
}

#[async_trait]
impl SessionMemory for FileSessionMemory {
    async fn messages(&self, session_id: &str) -> Vec<MemoryMessage> {
        self.ensure_bootstrapped().await;
        self.read_record(session_id)
            .await
            .ok()
            .flatten()
            .map(|record| record.messages)
            .unwrap_or_default()
    }

    async fn append(&self, session_id: &str, message: MemoryMessage) {
        let _guard = self.write_lock.lock().await;
        if self.bootstrap_locked().await.is_err() {
            return;
        }

        let mut record = self.read_record(session_id).await.ok().flatten().unwrap_or(
            SessionMemoryFileRecord {
                schema_version: FILE_SESSION_MEMORY_SCHEMA_VERSION,
                session_id: session_id.to_string(),
                messages: Vec::new(),
            },
        );
        record.messages.push(message);
        let _ = self.write_record(&record).await;
    }

    async fn clear(&self, session_id: &str) {
        let _guard = self.write_lock.lock().await;
        let _ = self.bootstrap_locked().await;
        let _ = fs::remove_file(self.session_path(session_id)).await;
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn file_session_memory_bootstrap_migrates_legacy_sessions() {
        let root = temp_dir("harbor-memory-bootstrap-test");
        let session_id = "legacy-session";
        std::fs::write(
            root.join(format!("{session_id}.json")),
            serde_json::to_string_pretty(&vec![MemoryMessage::new("user", "legacy")]).unwrap(),
        )
        .unwrap();

        let memory = FileSessionMemory::new(&root);
        let bootstrap = memory.bootstrap().await.unwrap();
        assert!(bootstrap.created_manifest);
        assert_eq!(bootstrap.migrated_sessions, 1);

        let manifest: SessionMemoryManifest = serde_json::from_str(
            &std::fs::read_to_string(memory.manifest_path()).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.schema_version, FILE_SESSION_MEMORY_SCHEMA_VERSION);

        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join(format!("{session_id}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(raw["schema_version"], FILE_SESSION_MEMORY_SCHEMA_VERSION);
        assert_eq!(raw["session_id"], session_id);
        assert_eq!(memory.messages(session_id).await[0].content, "legacy");

        let _ = std::fs::remove_dir_all(root);
    }
}
