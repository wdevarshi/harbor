use async_trait::async_trait;
use harbor_ai::{
    CompletionEvent, CompletionEventStream, CompletionRequest, CompletionResponse, Message,
    MessageRole, ModelProvider, ToolChoice,
};
use harbor_core::{FrameworkError, FrameworkResult, JsonValue, ToolRegistry};
use harbor_http::{HarborHttpConfig, HarborHttpMiddlewareConfig, HarborHttpServer, ReadinessGate};
use harbor_memory::{MemoryMessage, SessionMemory};
use harbor_observability::{HarborObservability, HarborObservabilityConfig};
use harbor_rag::{render_retrieved_context, RetrievedChunk, RetrievalQuery, Retriever};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs,
    sync::{Mutex, RwLock},
};
use tracing::warn;
use uuid::Uuid;

#[derive(Clone)]
pub struct AgentRuntime<P, M> {
    provider: P,
    memory: M,
    tools: ToolRegistry,
    default_model: String,
    base_system_prompt: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunContext {
    pub extra_system_prompt: Option<String>,
    pub retrieved_chunks: Vec<RetrievedChunk>,
}

impl RunContext {
    pub fn with_retrieved_chunks(mut self, retrieved_chunks: Vec<RetrievedChunk>) -> Self {
        self.retrieved_chunks = retrieved_chunks;
        self
    }

    pub fn with_extra_system_prompt(mut self, extra_system_prompt: impl Into<String>) -> Self {
        self.extra_system_prompt = Some(extra_system_prompt.into());
        self
    }

    fn merge_system_prompt(&self, base_system_prompt: &str) -> Option<String> {
        let mut sections = Vec::new();
        if !base_system_prompt.trim().is_empty() {
            sections.push(base_system_prompt.trim().to_string());
        }
        if let Some(extra) = self.extra_system_prompt.as_ref().filter(|value| !value.trim().is_empty()) {
            sections.push(extra.trim().to_string());
        }
        if let Some(context) = render_retrieved_context(&self.retrieved_chunks) {
            sections.push(context);
        }

        if sections.is_empty() {
            None
        } else {
            Some(sections.join("\n\n"))
        }
    }
}

pub struct RuntimeEventStream<M> {
    inner: CompletionEventStream,
    memory: M,
    session_id: String,
    user_input: Option<String>,
    assistant_recorded: bool,
    buffered_text: String,
}

impl<M> RuntimeEventStream<M>
where
    M: SessionMemory,
{
    fn new(
        inner: CompletionEventStream,
        memory: M,
        session_id: impl Into<String>,
        user_input: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            memory,
            session_id: session_id.into(),
            user_input: Some(user_input.into()),
            assistant_recorded: false,
            buffered_text: String::new(),
        }
    }

    async fn ensure_user_recorded(&mut self) {
        if let Some(user_input) = self.user_input.take() {
            self.memory
                .append(&self.session_id, MemoryMessage::new("user", user_input))
                .await;
        }
    }

    async fn record_assistant_if_needed(&mut self, text: String) {
        if self.assistant_recorded {
            return;
        }

        self.memory
            .append(&self.session_id, MemoryMessage::new("assistant", text))
            .await;
        self.assistant_recorded = true;
    }

    pub async fn recv(&mut self) -> Option<FrameworkResult<CompletionEvent>> {
        match self.inner.recv().await {
            Some(Ok(CompletionEvent::Started { provider, model })) => {
                self.ensure_user_recorded().await;
                Some(Ok(CompletionEvent::Started { provider, model }))
            }
            Some(Ok(CompletionEvent::Delta { text })) => {
                self.ensure_user_recorded().await;
                self.buffered_text.push_str(&text);
                Some(Ok(CompletionEvent::Delta { text }))
            }
            Some(Ok(CompletionEvent::Finished { response })) => {
                self.ensure_user_recorded().await;

                let assistant_text = if response.text.is_empty() {
                    self.buffered_text.clone()
                } else {
                    response.text.clone()
                };

                self.record_assistant_if_needed(assistant_text).await;
                Some(Ok(CompletionEvent::Finished { response }))
            }
            Some(Err(error)) => Some(Err(error)),
            None => None,
        }
    }
}

impl<P, M> AgentRuntime<P, M>
where
    P: ModelProvider,
    M: SessionMemory,
{
    pub fn new(provider: P, memory: M) -> Self {
        Self {
            provider,
            memory,
            tools: ToolRegistry::new(),
            default_model: "mock-model".into(),
            base_system_prompt: "You are Harbor runtime".into(),
        }
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.base_system_prompt = system_prompt.into();
        self
    }

    async fn build_request(
        &self,
        session_id: &str,
        user_input: String,
        context: &RunContext,
    ) -> CompletionRequest {
        let history = self.memory.messages(session_id).await;

        let mut messages: Vec<Message> = history
            .into_iter()
            .map(|message| Message {
                role: match message.role.as_str() {
                    "system" => MessageRole::System,
                    "assistant" => MessageRole::Assistant,
                    "tool" => MessageRole::Tool,
                    _ => MessageRole::User,
                },
                content: message.content,
            })
            .collect();

        messages.push(Message {
            role: MessageRole::User,
            content: user_input,
        });

        CompletionRequest {
            model: self.default_model.clone(),
            system_prompt: context.merge_system_prompt(&self.base_system_prompt),
            messages,
            tools: self.tools.list(),
            response_schema: None,
            session_id: Some(session_id.to_string()),
            tool_choice: ToolChoice::Auto,
        }
    }

    pub async fn run_turn(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
    ) -> FrameworkResult<CompletionResponse> {
        self.run_turn_with_context(session_id, user_input, RunContext::default())
            .await
    }

    pub async fn run_turn_with_context(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
        context: RunContext,
    ) -> FrameworkResult<CompletionResponse> {
        let user_input = user_input.into();
        let request = self.build_request(session_id, user_input.clone(), &context).await;
        let response = self.provider.complete(request).await?;

        self.memory
            .append(session_id, MemoryMessage::new("user", user_input))
            .await;
        self.memory
            .append(
                session_id,
                MemoryMessage::new("assistant", response.text.clone()),
            )
            .await;

        Ok(response)
    }

    pub async fn run_turn_with_retrieval<R>(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
        retriever: &R,
        query: RetrievalQuery,
    ) -> FrameworkResult<CompletionResponse>
    where
        R: Retriever,
    {
        let user_input = user_input.into();
        let retrieved_chunks = retriever.retrieve(&user_input, query).await?;
        self.run_turn_with_context(
            session_id,
            user_input,
            RunContext::default().with_retrieved_chunks(retrieved_chunks),
        )
        .await
    }

    pub async fn run_turn_stream(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
    ) -> FrameworkResult<RuntimeEventStream<M>> {
        self.run_turn_stream_with_context(session_id, user_input, RunContext::default())
            .await
    }

    pub async fn run_turn_stream_with_context(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
        context: RunContext,
    ) -> FrameworkResult<RuntimeEventStream<M>> {
        let user_input = user_input.into();
        let request = self.build_request(session_id, user_input.clone(), &context).await;
        let stream = self.provider.complete_stream(request).await?;

        Ok(RuntimeEventStream::new(
            stream,
            self.memory.clone(),
            session_id,
            user_input,
        ))
    }

    pub async fn run_turn_stream_with_retrieval<R>(
        &self,
        session_id: &str,
        user_input: impl Into<String>,
        retriever: &R,
        query: RetrievalQuery,
    ) -> FrameworkResult<RuntimeEventStream<M>>
    where
        R: Retriever,
    {
        let user_input = user_input.into();
        let retrieved_chunks = retriever.retrieve(&user_input, query).await?;
        self.run_turn_stream_with_context(
            session_id,
            user_input,
            RunContext::default().with_retrieved_chunks(retrieved_chunks),
        )
        .await
    }
}

pub const TASK_RECORD_SCHEMA_VERSION: u32 = 1;
pub const TASK_STORE_MANIFEST_FILE_NAME: &str = ".harbor-task-store.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskCheckpoint {
    pub label: String,
    pub data: JsonValue,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskLease {
    pub worker_id: String,
    pub claimed_at_ms: u64,
    pub heartbeat_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskEnqueueOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

impl TaskEnqueueOptions {
    pub fn with_idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskStoreManifest {
    pub store_kind: String,
    pub schema_version: u32,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl TaskStoreManifest {
    fn current(created_at_ms: Option<u64>) -> Self {
        let now = now_millis();
        Self {
            store_kind: "task_store".into(),
            schema_version: TASK_RECORD_SCHEMA_VERSION,
            created_at_ms: created_at_ms.unwrap_or(now),
            updated_at_ms: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskStoreBootstrap {
    pub created_manifest: bool,
    pub migrated_tasks: usize,
    pub schema_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskRecord {
    #[serde(default = "task_record_schema_version")]
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub input: JsonValue,
    pub state: TaskState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    pub output: Option<JsonValue>,
    pub error: Option<String>,
    pub attempts: u32,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lease: Option<TaskLease>,
    #[serde(default)]
    pub checkpoints: Vec<TaskCheckpoint>,
}

fn task_record_schema_version() -> u32 {
    TASK_RECORD_SCHEMA_VERSION
}

impl TaskRecord {
    pub fn queued(name: impl Into<String>, input: JsonValue) -> Self {
        Self::queued_with_options(name, input, TaskEnqueueOptions::default())
    }

    pub fn queued_with_options(
        name: impl Into<String>,
        input: JsonValue,
        options: TaskEnqueueOptions,
    ) -> Self {
        let now = now_millis();
        Self {
            schema_version: TASK_RECORD_SCHEMA_VERSION,
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            input,
            state: TaskState::Queued,
            idempotency_key: options.idempotency_key,
            output: None,
            error: None,
            attempts: 0,
            created_at_ms: now,
            updated_at_ms: now,
            lease: None,
            checkpoints: Vec::new(),
        }
    }

    fn is_claimable(&self, now_ms: u64) -> bool {
        match self.state {
            TaskState::Queued => true,
            TaskState::Running => self
                .lease
                .as_ref()
                .map(|lease| lease.expires_at_ms <= now_ms)
                .unwrap_or(true),
            TaskState::Completed | TaskState::Failed | TaskState::Cancelled => false,
        }
    }

    fn claim(&mut self, worker_id: impl Into<String>, lease_ttl_ms: u64) {
        let now = now_millis();
        self.schema_version = TASK_RECORD_SCHEMA_VERSION;
        self.state = TaskState::Running;
        self.attempts += 1;
        self.error = None;
        self.updated_at_ms = now;
        self.lease = Some(TaskLease {
            worker_id: worker_id.into(),
            claimed_at_ms: now,
            heartbeat_at_ms: now,
            expires_at_ms: now.saturating_add(lease_ttl_ms.max(1)),
        });
    }

    fn refresh_lease(&mut self, worker_id: &str, lease_ttl_ms: u64) -> FrameworkResult<()> {
        let now = now_millis();
        let lease = self.lease.as_mut().ok_or_else(|| {
            FrameworkError::Memory(format!("task '{}' has no active lease", self.id))
        })?;

        if lease.worker_id != worker_id {
            return Err(FrameworkError::Memory(format!(
                "task '{}' lease is owned by '{}'",
                self.id, lease.worker_id
            )));
        }

        lease.heartbeat_at_ms = now;
        lease.expires_at_ms = now.saturating_add(lease_ttl_ms.max(1));
        self.updated_at_ms = now;
        Ok(())
    }

    fn clear_lease(&mut self) {
        self.lease = None;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskInsertResult {
    Created(TaskRecord),
    Existing(TaskRecord),
}

impl TaskInsertResult {
    pub fn was_created(&self) -> bool {
        matches!(self, Self::Created(_))
    }

    pub fn task(&self) -> &TaskRecord {
        match self {
            Self::Created(task) | Self::Existing(task) => task,
        }
    }

    pub fn into_task(self) -> TaskRecord {
        match self {
            Self::Created(task) | Self::Existing(task) => task,
        }
    }
}

#[async_trait]
pub trait TaskStore: Send + Sync + Clone + 'static {
    async fn create(&self, task: TaskRecord) -> FrameworkResult<TaskInsertResult>;
    async fn put(&self, task: TaskRecord) -> FrameworkResult<TaskRecord>;
    async fn get(&self, id: &str) -> FrameworkResult<Option<TaskRecord>>;
    async fn list(&self) -> FrameworkResult<Vec<TaskRecord>>;
    async fn delete(&self, id: &str) -> FrameworkResult<()>;
    async fn claim_next(&self, worker_id: &str, lease_ttl_ms: u64)
        -> FrameworkResult<Option<TaskRecord>>;
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryTaskStore {
    inner: Arc<RwLock<HashMap<String, TaskRecord>>>,
}

#[async_trait]
impl TaskStore for InMemoryTaskStore {
    async fn create(&self, task: TaskRecord) -> FrameworkResult<TaskInsertResult> {
        let mut guard = self.inner.write().await;

        if let Some(key) = task.idempotency_key.as_deref() {
            if let Some(existing) = guard
                .values()
                .find(|candidate| task_matches_idempotency(candidate, &task.name, key))
                .cloned()
            {
                return Ok(TaskInsertResult::Existing(existing));
            }
        }

        guard.insert(task.id.clone(), task.clone());
        Ok(TaskInsertResult::Created(task))
    }

    async fn put(&self, task: TaskRecord) -> FrameworkResult<TaskRecord> {
        self.inner
            .write()
            .await
            .insert(task.id.clone(), task.clone());
        Ok(task)
    }

    async fn get(&self, id: &str) -> FrameworkResult<Option<TaskRecord>> {
        Ok(self.inner.read().await.get(id).cloned())
    }

    async fn list(&self) -> FrameworkResult<Vec<TaskRecord>> {
        let mut tasks = self.inner.read().await.values().cloned().collect::<Vec<_>>();
        tasks.sort_by(|a, b| a.created_at_ms.cmp(&b.created_at_ms).then(a.id.cmp(&b.id)));
        Ok(tasks)
    }

    async fn delete(&self, id: &str) -> FrameworkResult<()> {
        self.inner.write().await.remove(id);
        Ok(())
    }

    async fn claim_next(
        &self,
        worker_id: &str,
        lease_ttl_ms: u64,
    ) -> FrameworkResult<Option<TaskRecord>> {
        let now = now_millis();
        let mut guard = self.inner.write().await;

        let Some(task_id) = guard
            .values()
            .filter(|task| task.is_claimable(now))
            .min_by(|a, b| a.created_at_ms.cmp(&b.created_at_ms).then(a.id.cmp(&b.id)))
            .map(|task| task.id.clone())
        else {
            return Ok(None);
        };

        let task = guard
            .get_mut(&task_id)
            .ok_or_else(|| FrameworkError::Memory(format!("task '{}' not found", task_id)))?;
        task.claim(worker_id.to_string(), lease_ttl_ms);
        Ok(Some(task.clone()))
    }
}

#[derive(Debug, Clone)]
pub struct FileTaskStore {
    root: Arc<PathBuf>,
    write_lock: Arc<Mutex<()>>,
}

impl FileTaskStore {
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
        self.root.join(TASK_STORE_MANIFEST_FILE_NAME)
    }

    fn task_path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.json"))
    }

    pub async fn bootstrap(&self) -> FrameworkResult<TaskStoreBootstrap> {
        let _guard = self.write_lock.lock().await;
        self.bootstrap_locked().await
    }

    async fn bootstrap_locked(&self) -> FrameworkResult<TaskStoreBootstrap> {
        fs::create_dir_all(self.root()).await?;

        let existing_manifest = self.read_manifest_locked().await?;
        let created_manifest = existing_manifest.is_none();
        let mut manifest = existing_manifest
            .unwrap_or_else(|| TaskStoreManifest::current(None));
        let mut migrated_tasks = 0usize;

        let mut entries = fs::read_dir(self.root()).await?;
        while let Some(entry) = entries.next_entry().await? {
            if !entry.file_type().await?.is_file() {
                continue;
            }

            let path = entry.path();
            if path == self.manifest_path() {
                continue;
            }

            let content = fs::read_to_string(&path).await?;
            let (task, needs_rewrite) = decode_task_record(&content)?;
            if needs_rewrite {
                self.write_task_locked(&task).await?;
                migrated_tasks += 1;
            }
        }

        manifest.schema_version = TASK_RECORD_SCHEMA_VERSION;
        manifest.updated_at_ms = now_millis();
        self.write_manifest_locked(&manifest).await?;

        Ok(TaskStoreBootstrap {
            created_manifest,
            migrated_tasks,
            schema_version: TASK_RECORD_SCHEMA_VERSION,
        })
    }

    async fn read_manifest_locked(&self) -> FrameworkResult<Option<TaskStoreManifest>> {
        match fs::read_to_string(self.manifest_path()).await {
            Ok(content) => Ok(serde_json::from_str(&content).ok()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    async fn write_manifest_locked(&self, manifest: &TaskStoreManifest) -> FrameworkResult<()> {
        fs::write(self.manifest_path(), serde_json::to_string_pretty(manifest)?).await?;
        Ok(())
    }

    async fn get_locked(&self, id: &str) -> FrameworkResult<Option<TaskRecord>> {
        match fs::read_to_string(self.task_path(id)).await {
            Ok(content) => Ok(Some(decode_task_record(&content)?.0)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    async fn list_locked(&self) -> FrameworkResult<Vec<TaskRecord>> {
        let mut entries = fs::read_dir(self.root()).await?;
        let mut tasks = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            if !entry.file_type().await?.is_file() {
                continue;
            }

            if entry.path() == self.manifest_path() {
                continue;
            }

            let content = fs::read_to_string(entry.path()).await?;
            tasks.push(decode_task_record(&content)?.0);
        }

        tasks.sort_by(|a, b| a.created_at_ms.cmp(&b.created_at_ms).then(a.id.cmp(&b.id)));
        Ok(tasks)
    }

    async fn write_task_locked(&self, task: &TaskRecord) -> FrameworkResult<()> {
        let mut persisted = task.clone();
        persisted.schema_version = TASK_RECORD_SCHEMA_VERSION;
        fs::write(
            self.task_path(&persisted.id),
            serde_json::to_string_pretty(&persisted)?,
        )
        .await?;
        Ok(())
    }
}

fn task_matches_idempotency(task: &TaskRecord, task_name: &str, key: &str) -> bool {
    task.name == task_name && task.idempotency_key.as_deref() == Some(key)
}

fn decode_task_record(content: &str) -> FrameworkResult<(TaskRecord, bool)> {
    let value: JsonValue = serde_json::from_str(content)?;
    let mut task: TaskRecord = serde_json::from_value(value.clone())?;
    let needs_rewrite = value
        .get("schema_version")
        .and_then(|candidate| candidate.as_u64())
        != Some(TASK_RECORD_SCHEMA_VERSION as u64);
    task.schema_version = TASK_RECORD_SCHEMA_VERSION;
    Ok((task, needs_rewrite))
}

#[async_trait]
impl TaskStore for FileTaskStore {
    async fn create(&self, task: TaskRecord) -> FrameworkResult<TaskInsertResult> {
        let _guard = self.write_lock.lock().await;
        self.bootstrap_locked().await?;

        if let Some(key) = task.idempotency_key.as_deref() {
            if let Some(existing) = self
                .list_locked()
                .await?
                .into_iter()
                .find(|candidate| task_matches_idempotency(candidate, &task.name, key))
            {
                return Ok(TaskInsertResult::Existing(existing));
            }
        }

        self.write_task_locked(&task).await?;
        Ok(TaskInsertResult::Created(task))
    }

    async fn put(&self, task: TaskRecord) -> FrameworkResult<TaskRecord> {
        let _guard = self.write_lock.lock().await;
        self.bootstrap_locked().await?;
        self.write_task_locked(&task).await?;
        Ok(task)
    }

    async fn get(&self, id: &str) -> FrameworkResult<Option<TaskRecord>> {
        let _guard = self.write_lock.lock().await;
        self.bootstrap_locked().await?;
        self.get_locked(id).await
    }

    async fn list(&self) -> FrameworkResult<Vec<TaskRecord>> {
        let _guard = self.write_lock.lock().await;
        self.bootstrap_locked().await?;
        self.list_locked().await
    }

    async fn delete(&self, id: &str) -> FrameworkResult<()> {
        let _guard = self.write_lock.lock().await;
        self.bootstrap_locked().await?;
        match fs::remove_file(self.task_path(id)).await {
            Ok(_) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    async fn claim_next(
        &self,
        worker_id: &str,
        lease_ttl_ms: u64,
    ) -> FrameworkResult<Option<TaskRecord>> {
        let _guard = self.write_lock.lock().await;
        self.bootstrap_locked().await?;

        let now = now_millis();
        let Some(mut task) = self
            .list_locked()
            .await?
            .into_iter()
            .find(|candidate| candidate.is_claimable(now))
        else {
            return Ok(None);
        };

        task.claim(worker_id.to_string(), lease_ttl_ms);
        self.write_task_locked(&task).await?;
        Ok(Some(task))
    }
}

#[derive(Debug, Clone)]
pub struct TaskLifecycle<S> {
    store: S,
}

impl<S> TaskLifecycle<S>
where
    S: TaskStore,
{
    pub fn new(store: S) -> Self {
        Self { store }
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub async fn enqueue(
        &self,
        name: impl Into<String>,
        input: JsonValue,
    ) -> FrameworkResult<TaskRecord> {
        self.enqueue_with_options(name, input, TaskEnqueueOptions::default())
            .await
    }

    pub async fn enqueue_with_options(
        &self,
        name: impl Into<String>,
        input: JsonValue,
        options: TaskEnqueueOptions,
    ) -> FrameworkResult<TaskRecord> {
        Ok(self
            .store
            .create(TaskRecord::queued_with_options(name, input, options))
            .await?
            .into_task())
    }

    pub async fn enqueue_idempotent(
        &self,
        name: impl Into<String>,
        input: JsonValue,
        idempotency_key: impl Into<String>,
    ) -> FrameworkResult<TaskRecord> {
        self.enqueue_with_options(
            name,
            input,
            TaskEnqueueOptions::default().with_idempotency_key(idempotency_key),
        )
        .await
    }

    pub async fn claim_next(
        &self,
        worker_id: &str,
        lease_ttl_ms: u64,
    ) -> FrameworkResult<Option<TaskRecord>> {
        self.store.claim_next(worker_id, lease_ttl_ms).await
    }

    pub async fn start(&self, id: &str) -> FrameworkResult<TaskRecord> {
        self.update(id, |task| {
            task.schema_version = TASK_RECORD_SCHEMA_VERSION;
            task.state = TaskState::Running;
            task.attempts += 1;
            task.error = None;
            task.clear_lease();
            task.updated_at_ms = now_millis();
            Ok(())
        })
        .await
    }

    pub async fn heartbeat(
        &self,
        id: &str,
        worker_id: impl Into<String>,
        lease_ttl_ms: u64,
    ) -> FrameworkResult<TaskRecord> {
        let worker_id = worker_id.into();
        self.update(id, move |task| task.refresh_lease(&worker_id, lease_ttl_ms))
            .await
    }

    pub async fn checkpoint(
        &self,
        id: &str,
        label: impl Into<String>,
        data: JsonValue,
    ) -> FrameworkResult<TaskRecord> {
        let label = label.into();
        self.update(id, move |task| {
            task.checkpoints.push(TaskCheckpoint {
                label: label.clone(),
                data: data.clone(),
                created_at_ms: now_millis(),
            });
            task.updated_at_ms = now_millis();
            Ok(())
        })
        .await
    }

    pub async fn complete(&self, id: &str, output: JsonValue) -> FrameworkResult<TaskRecord> {
        self.update(id, move |task| {
            task.state = TaskState::Completed;
            task.output = Some(output.clone());
            task.error = None;
            task.clear_lease();
            task.updated_at_ms = now_millis();
            Ok(())
        })
        .await
    }

    pub async fn fail(&self, id: &str, error: impl Into<String>) -> FrameworkResult<TaskRecord> {
        let error = error.into();
        self.update(id, move |task| {
            task.state = TaskState::Failed;
            task.error = Some(error.clone());
            task.clear_lease();
            task.updated_at_ms = now_millis();
            Ok(())
        })
        .await
    }

    pub async fn cancel(
        &self,
        id: &str,
        reason: impl Into<String>,
    ) -> FrameworkResult<TaskRecord> {
        let reason = reason.into();
        self.update(id, move |task| {
            task.state = TaskState::Cancelled;
            task.error = Some(reason.clone());
            task.clear_lease();
            task.updated_at_ms = now_millis();
            Ok(())
        })
        .await
    }

    async fn update<F>(&self, id: &str, mut mutator: F) -> FrameworkResult<TaskRecord>
    where
        F: FnMut(&mut TaskRecord) -> FrameworkResult<()> + Send,
    {
        let mut task = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| FrameworkError::Memory(format!("task '{id}' not found")))?;
        mutator(&mut task)?;
        self.store.put(task).await
    }
}

#[async_trait]
pub trait BackgroundTaskHandler: Send + Sync {
    fn task_name(&self) -> &'static str;
    async fn run(&self, task: TaskRecord) -> FrameworkResult<JsonValue>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTaskRunnerConfig {
    pub worker_id: String,
    pub poll_interval_ms: u64,
    pub lease_ttl_ms: u64,
    pub heartbeat_interval_ms: u64,
}

impl Default for BackgroundTaskRunnerConfig {
    fn default() -> Self {
        Self {
            worker_id: format!("worker-{}", Uuid::new_v4()),
            poll_interval_ms: 500,
            lease_ttl_ms: 5_000,
            heartbeat_interval_ms: 1_000,
        }
    }
}

pub struct BackgroundTaskRunner<S> {
    lifecycle: TaskLifecycle<S>,
    config: BackgroundTaskRunnerConfig,
    handlers: HashMap<String, Arc<dyn BackgroundTaskHandler>>,
}

impl<S> BackgroundTaskRunner<S>
where
    S: TaskStore,
{
    pub fn new(lifecycle: TaskLifecycle<S>) -> Self {
        Self {
            lifecycle,
            config: BackgroundTaskRunnerConfig::default(),
            handlers: HashMap::new(),
        }
    }

    pub fn with_config(mut self, config: BackgroundTaskRunnerConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_handler<H>(mut self, handler: H) -> Self
    where
        H: BackgroundTaskHandler + 'static,
    {
        self.handlers
            .insert(handler.task_name().to_string(), Arc::new(handler));
        self
    }

    pub fn register_handler_arc(&mut self, handler: Arc<dyn BackgroundTaskHandler>) {
        self.handlers.insert(handler.task_name().to_string(), handler);
    }

    pub fn config(&self) -> &BackgroundTaskRunnerConfig {
        &self.config
    }

    pub async fn run_once(&self) -> FrameworkResult<Option<TaskRecord>> {
        let Some(task) = self
            .lifecycle
            .claim_next(&self.config.worker_id, self.config.lease_ttl_ms)
            .await?
        else {
            return Ok(None);
        };

        let Some(handler) = self.handlers.get(&task.name).cloned() else {
            return Ok(Some(
                self.lifecycle
                    .fail(
                        &task.id,
                        format!(
                            "no background handler registered for task '{}'",
                            task.name.clone()
                        ),
                    )
                    .await?,
            ));
        };

        self.lifecycle
            .checkpoint(
                &task.id,
                "claimed",
                serde_json::json!({
                    "worker_id": self.config.worker_id.clone(),
                    "attempt": task.attempts,
                }),
            )
            .await?;

        let lifecycle = self.lifecycle.clone();
        let task_id = task.id.clone();
        let worker_id = self.config.worker_id.clone();
        let lease_ttl_ms = self.config.lease_ttl_ms;
        let heartbeat_interval_ms = self.config.heartbeat_interval_ms.max(1);
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();

        let heartbeat = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    _ = tokio::time::sleep(Duration::from_millis(heartbeat_interval_ms)) => {
                        let _ = lifecycle.heartbeat(&task_id, &worker_id, lease_ttl_ms).await;
                    }
                }
            }
        });

        let result = handler.run(task.clone()).await;
        let _ = stop_tx.send(());
        let _ = heartbeat.await;

        match result {
            Ok(output) => Ok(Some(self.lifecycle.complete(&task.id, output).await?)),
            Err(error) => Ok(Some(self.lifecycle.fail(&task.id, error.to_string()).await?)),
        }
    }

    pub async fn run_until_shutdown<F>(&self, shutdown_signal: F) -> FrameworkResult<()>
    where
        F: std::future::Future<Output = ()> + Send,
    {
        tokio::pin!(shutdown_signal);

        loop {
            if self.run_once().await?.is_some() {
                continue;
            }

            tokio::select! {
                _ = &mut shutdown_signal => return Ok(()),
                _ = tokio::time::sleep(Duration::from_millis(self.config.poll_interval_ms.max(1))) => {}
            }
        }
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarborAppConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub http_host: String,
    pub http_port: u16,
    pub log_level: String,
    pub json_logs: bool,
    pub metrics_enabled: bool,
    pub otel_enabled: bool,
    pub otel_endpoint: String,
    pub http_timeout_ms: Option<u64>,
    pub http_concurrency_limit: Option<usize>,
    pub http_rate_limit_requests: Option<u64>,
    pub http_rate_limit_window_secs: u64,
    pub http_bearer_token: Option<String>,
}

impl Default for HarborAppConfig {
    fn default() -> Self {
        Self {
            service_name: "harbor-app".into(),
            service_version: "0.1.0".into(),
            environment: "dev".into(),
            http_host: "0.0.0.0".into(),
            http_port: 3000,
            log_level: "info".into(),
            json_logs: false,
            metrics_enabled: true,
            otel_enabled: false,
            otel_endpoint: "http://127.0.0.1:4317".into(),
            http_timeout_ms: None,
            http_concurrency_limit: None,
            http_rate_limit_requests: None,
            http_rate_limit_window_secs: 1,
            http_bearer_token: None,
        }
    }
}

impl HarborAppConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let mut config = Self::default();

        if let Ok(service_name) = env::var("HARBOR_SERVICE_NAME") {
            config.service_name = service_name;
        }

        if let Ok(service_version) = env::var("HARBOR_SERVICE_VERSION") {
            config.service_version = service_version;
        }

        if let Ok(environment) = env::var("HARBOR_ENV") {
            config.environment = environment;
        }

        if let Ok(host) = env::var("HARBOR_HTTP_HOST") {
            config.http_host = host;
        }

        if let Ok(port) = env::var("HARBOR_HTTP_PORT") {
            config.http_port = port.parse().map_err(|error| {
                FrameworkError::Config(format!("invalid HARBOR_HTTP_PORT value: {error}"))
            })?;
        }

        if let Ok(log_level) = env::var("HARBOR_LOG_LEVEL") {
            config.log_level = log_level;
        }

        if let Ok(json_logs) = env::var("HARBOR_JSON_LOGS") {
            config.json_logs = matches!(json_logs.as_str(), "1" | "true" | "TRUE" | "yes" | "YES");
        }

        if let Ok(metrics_enabled) = env::var("HARBOR_METRICS_ENABLED") {
            config.metrics_enabled = !matches!(
                metrics_enabled.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO"
            );
        }

        if let Ok(otel_enabled) = env::var("HARBOR_OTEL_ENABLED") {
            config.otel_enabled =
                matches!(otel_enabled.as_str(), "1" | "true" | "TRUE" | "yes" | "YES");
        }

        if let Ok(otel_endpoint) = env::var("HARBOR_OTEL_ENDPOINT") {
            config.otel_endpoint = otel_endpoint;
        }

        if let Ok(timeout_ms) = env::var("HARBOR_HTTP_TIMEOUT_MS") {
            config.http_timeout_ms = Some(timeout_ms.parse().map_err(|error| {
                FrameworkError::Config(format!("invalid HARBOR_HTTP_TIMEOUT_MS value: {error}"))
            })?);
        }

        if let Ok(concurrency_limit) = env::var("HARBOR_HTTP_CONCURRENCY_LIMIT") {
            config.http_concurrency_limit = Some(concurrency_limit.parse().map_err(|error| {
                FrameworkError::Config(format!(
                    "invalid HARBOR_HTTP_CONCURRENCY_LIMIT value: {error}"
                ))
            })?);
        }

        if let Ok(rate_limit_requests) = env::var("HARBOR_HTTP_RATE_LIMIT_REQUESTS") {
            config.http_rate_limit_requests =
                Some(rate_limit_requests.parse().map_err(|error| {
                    FrameworkError::Config(format!(
                        "invalid HARBOR_HTTP_RATE_LIMIT_REQUESTS value: {error}"
                    ))
                })?);
        }

        if let Ok(rate_limit_window_secs) = env::var("HARBOR_HTTP_RATE_LIMIT_WINDOW_SECS") {
            config.http_rate_limit_window_secs =
                rate_limit_window_secs.parse().map_err(|error| {
                    FrameworkError::Config(format!(
                        "invalid HARBOR_HTTP_RATE_LIMIT_WINDOW_SECS value: {error}"
                    ))
                })?;
        }

        if let Ok(http_bearer_token) = env::var("HARBOR_HTTP_BEARER_TOKEN") {
            if !http_bearer_token.trim().is_empty() {
                config.http_bearer_token = Some(http_bearer_token);
            }
        }

        Ok(config)
    }

    pub fn http_config(&self) -> HarborHttpConfig {
        HarborHttpConfig {
            host: self.http_host.clone(),
            port: self.http_port,
            service_name: self.service_name.clone(),
            service_version: self.service_version.clone(),
            environment: self.environment.clone(),
        }
    }

    pub fn observability_config(&self) -> HarborObservabilityConfig {
        HarborObservabilityConfig {
            service_name: self.service_name.clone(),
            service_version: self.service_version.clone(),
            environment: self.environment.clone(),
            log_level: self.log_level.clone(),
            json_logs: self.json_logs,
            metrics_enabled: self.metrics_enabled,
            otel_enabled: self.otel_enabled,
            otel_endpoint: self.otel_endpoint.clone(),
        }
    }

    pub fn http_middleware_config(&self) -> HarborHttpMiddlewareConfig {
        HarborHttpMiddlewareConfig {
            request_timeout_ms: self.http_timeout_ms,
            concurrency_limit: self.http_concurrency_limit,
            rate_limit_requests: self.http_rate_limit_requests,
            rate_limit_window_secs: self.http_rate_limit_window_secs,
            bearer_token: self.http_bearer_token.clone(),
        }
    }
}

#[derive(Clone)]
pub struct HarborApp {
    config: HarborAppConfig,
    readiness: ReadinessGate,
}

impl HarborApp {
    pub fn new(config: HarborAppConfig) -> Self {
        Self {
            config,
            readiness: ReadinessGate::not_ready(),
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(HarborAppConfig::from_env()?))
    }

    pub fn config(&self) -> &HarborAppConfig {
        &self.config
    }

    pub fn readiness(&self) -> ReadinessGate {
        self.readiness.clone()
    }

    pub fn mark_ready(&self) {
        self.readiness.set_ready(true);
    }

    pub fn mark_not_ready(&self) {
        self.readiness.set_ready(false);
    }

    pub fn http_server(&self) -> HarborHttpServer {
        HarborHttpServer::new(self.config.http_config())
            .with_readiness(self.readiness.clone())
            .with_middleware_config(self.config.http_middleware_config())
    }

    pub async fn run(self) -> FrameworkResult<()> {
        self.run_with_signal(shutdown_signal()).await
    }

    pub async fn run_with_signal<F>(self, shutdown_signal: F) -> FrameworkResult<()>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let observability = HarborObservability::init(&self.config.observability_config())?;
        self.readiness.set_ready(true);

        let mut server = self.http_server();
        if let Some(renderer) = observability.metrics_renderer() {
            server = server.with_metrics_renderer(renderer);
        }

        let result = server.run_with_shutdown(shutdown_signal).await;
        if let Err(error) = observability.shutdown() {
            warn!(error = %error, "harbor observability shutdown reported an error");
        }
        result
    }
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harbor_ai::{CompletionEvent, CompletionRequest, MockProvider};
    use harbor_memory::{InMemorySessionMemory, SessionMemory};
    use harbor_rag::{Document, DocumentStore, InMemoryDocumentStore, LexicalRetriever};
    use std::{sync::{Arc, Mutex}, time::{SystemTime, UNIX_EPOCH}};

    fn temp_dir(prefix: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{suffix}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[derive(Clone)]
    struct CapturingProvider {
        prompts: Arc<Mutex<Vec<Option<String>>>>,
    }

    #[async_trait]
    impl ModelProvider for CapturingProvider {
        fn name(&self) -> &'static str {
            "capture"
        }

        async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
            self.prompts.lock().unwrap().push(request.system_prompt.clone());
            Ok(CompletionResponse {
                text: "captured".into(),
                structured: None,
                provider: self.name().into(),
                model: request.model,
            })
        }
    }

    struct EchoTaskHandler;

    #[async_trait]
    impl BackgroundTaskHandler for EchoTaskHandler {
        fn task_name(&self) -> &'static str {
            "summarize-docs"
        }

        async fn run(&self, task: TaskRecord) -> FrameworkResult<JsonValue> {
            Ok(serde_json::json!({
                "task_id": task.id,
                "doc": task.input["doc"].clone(),
            }))
        }
    }

    #[tokio::test]
    async fn run_turn_stream_emits_events_and_persists_memory() {
        let memory = InMemorySessionMemory::default();
        let runtime =
            AgentRuntime::new(MockProvider, memory.clone()).with_default_model("mock-gpt");

        let mut stream = runtime
            .run_turn_stream("stream-session", "Hello from streaming Harbor")
            .await
            .unwrap();

        let mut saw_started = false;
        let mut delta_text = String::new();
        let mut finished_text = None;

        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                CompletionEvent::Started { provider, model } => {
                    saw_started = true;
                    assert_eq!(provider, "mock");
                    assert_eq!(model, "mock-gpt");
                }
                CompletionEvent::Delta { text } => delta_text.push_str(&text),
                CompletionEvent::Finished { response } => finished_text = Some(response.text),
            }
        }

        let expected = "[mock:stream-session] Hello from streaming Harbor";
        assert!(saw_started);
        assert_eq!(delta_text, expected);
        assert_eq!(finished_text.unwrap(), expected);

        let history = memory.messages("stream-session").await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[0].content, "Hello from streaming Harbor");
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[1].content, expected);
    }

    #[tokio::test]
    async fn run_turn_with_retrieval_injects_rendered_document_context() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let provider = CapturingProvider {
            prompts: prompts.clone(),
        };
        let runtime = AgentRuntime::new(provider, InMemorySessionMemory::default())
            .with_default_model("capture-model");

        let store = InMemoryDocumentStore::default();
        store
            .put(Document::new(
                "Runbook",
                "Restart the worker and inspect the queue logs before scaling.",
            ))
            .await
            .unwrap();

        let retriever = LexicalRetriever::new(store);
        runtime
            .run_turn_with_retrieval(
                "retrieval-session",
                "How do I inspect the queue logs?",
                &retriever,
                RetrievalQuery::default(),
            )
            .await
            .unwrap();

        let prompt = prompts.lock().unwrap().pop().unwrap().unwrap();
        assert!(prompt.contains("You are Harbor runtime"));
        assert!(prompt.contains("Runbook"));
        assert!(prompt.contains("queue logs"));
    }

    #[tokio::test]
    async fn task_lifecycle_tracks_state_checkpoints_and_output() {
        let lifecycle = TaskLifecycle::new(InMemoryTaskStore::default());
        let task = lifecycle
            .enqueue("summarize-docs", serde_json::json!({"doc": "abc"}))
            .await
            .unwrap();
        assert_eq!(task.state, TaskState::Queued);

        let task = lifecycle.start(&task.id).await.unwrap();
        assert_eq!(task.state, TaskState::Running);
        assert_eq!(task.attempts, 1);

        let task = lifecycle
            .checkpoint(&task.id, "retrieved", serde_json::json!({"chunks": 2}))
            .await
            .unwrap();
        assert_eq!(task.checkpoints.len(), 1);
        assert_eq!(task.checkpoints[0].label, "retrieved");

        let task = lifecycle
            .complete(&task.id, serde_json::json!({"status": "ok"}))
            .await
            .unwrap();
        assert_eq!(task.state, TaskState::Completed);
        assert_eq!(task.output.unwrap()["status"], "ok");
    }

    #[tokio::test]
    async fn task_lifecycle_deduplicates_idempotent_enqueue_requests() {
        let lifecycle = TaskLifecycle::new(InMemoryTaskStore::default());

        let first = lifecycle
            .enqueue_idempotent(
                "summarize-docs",
                serde_json::json!({"doc": "abc"}),
                "job-123",
            )
            .await
            .unwrap();

        let second = lifecycle
            .enqueue_idempotent(
                "summarize-docs",
                serde_json::json!({"doc": "different"}),
                "job-123",
            )
            .await
            .unwrap();

        assert_eq!(first.id, second.id);
        assert_eq!(second.input["doc"], "abc");
        assert_eq!(lifecycle.store().list().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn task_store_reclaims_expired_running_leases() {
        let store = InMemoryTaskStore::default();
        let mut task = TaskRecord::queued("sync", serde_json::json!({"kind": "delta"}));
        task.state = TaskState::Running;
        task.attempts = 1;
        task.lease = Some(TaskLease {
            worker_id: "worker-old".into(),
            claimed_at_ms: 1,
            heartbeat_at_ms: 1,
            expires_at_ms: 0,
        });
        let task_id = task.id.clone();
        store.put(task).await.unwrap();

        let claimed = store.claim_next("worker-new", 250).await.unwrap().unwrap();

        assert_eq!(claimed.id, task_id);
        assert_eq!(claimed.state, TaskState::Running);
        assert_eq!(claimed.attempts, 2);
        assert_eq!(claimed.lease.unwrap().worker_id, "worker-new");
    }

    #[tokio::test]
    async fn background_task_runner_executes_claimed_work() {
        let lifecycle = TaskLifecycle::new(InMemoryTaskStore::default());
        let queued = lifecycle
            .enqueue("summarize-docs", serde_json::json!({"doc": "abc"}))
            .await
            .unwrap();

        let runner = BackgroundTaskRunner::new(lifecycle.clone())
            .with_config(BackgroundTaskRunnerConfig {
                worker_id: "worker-test".into(),
                poll_interval_ms: 10,
                lease_ttl_ms: 250,
                heartbeat_interval_ms: 25,
            })
            .with_handler(EchoTaskHandler);

        let finished = runner.run_once().await.unwrap().unwrap();
        assert_eq!(finished.id, queued.id);
        assert_eq!(finished.state, TaskState::Completed);
        assert_eq!(finished.output.unwrap()["doc"], "abc");

        let stored = lifecycle.store().get(&queued.id).await.unwrap().unwrap();
        assert!(stored.lease.is_none());
        assert_eq!(stored.checkpoints.len(), 1);
        assert_eq!(stored.checkpoints[0].label, "claimed");
    }

    #[tokio::test]
    async fn file_task_store_persists_tasks_between_instances() {
        let root = temp_dir("harbor-task-store-test");
        let store = FileTaskStore::new(&root);
        let task = TaskRecord::queued("sync", serde_json::json!({"kind": "delta"}));
        let id = task.id.clone();
        store.put(task.clone()).await.unwrap();

        let restored = FileTaskStore::new(&root);
        let fetched = restored.get(&id).await.unwrap().unwrap();
        assert_eq!(fetched, task);

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn file_task_store_bootstrap_migrates_legacy_task_files() {
        let root = temp_dir("harbor-task-store-bootstrap-test");
        let task_id = "legacy-task";

        std::fs::write(
            root.join(format!("{task_id}.json")),
            serde_json::to_string_pretty(&serde_json::json!({
                "id": task_id,
                "name": "sync",
                "input": {"kind": "delta"},
                "state": "queued",
                "output": null,
                "error": null,
                "attempts": 0,
                "created_at_ms": 1,
                "updated_at_ms": 1,
                "checkpoints": [],
            }))
            .unwrap(),
        )
        .unwrap();

        let store = FileTaskStore::new(&root);
        let bootstrap = store.bootstrap().await.unwrap();
        assert!(bootstrap.created_manifest);
        assert_eq!(bootstrap.migrated_tasks, 1);

        let manifest: TaskStoreManifest = serde_json::from_str(
            &std::fs::read_to_string(store.manifest_path()).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.schema_version, TASK_RECORD_SCHEMA_VERSION);

        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join(format!("{task_id}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(raw["schema_version"], TASK_RECORD_SCHEMA_VERSION);

        let fetched = store.get(task_id).await.unwrap().unwrap();
        assert_eq!(fetched.schema_version, TASK_RECORD_SCHEMA_VERSION);
        assert_eq!(fetched.name, "sync");

        let _ = std::fs::remove_dir_all(root);
    }
}
