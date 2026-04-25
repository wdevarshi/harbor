use async_trait::async_trait;
use harbor_core::{FrameworkError, FrameworkResult, JsonValue, ToolSpec};
use opentelemetry::{global, propagation::Injector};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue},
    Client,
};
use serde::{Deserialize, Serialize};
use std::{
    env,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
use tokio::time::{sleep, timeout};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use uuid::Uuid;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    fn as_openai_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "assistant",
        }
    }

    fn as_anthropic_role(&self) -> Option<&'static str> {
        match self {
            Self::System => None,
            Self::User => Some("user"),
            Self::Assistant | Self::Tool => Some("assistant"),
        }
    }

    fn as_ollama_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant | Self::Tool => "assistant",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub response_schema: Option<JsonValue>,
    pub session_id: Option<String>,
    pub tool_choice: ToolChoice,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<String>,
}

impl CompletionRequest {
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    pub fn with_stream_id(mut self, stream_id: impl Into<String>) -> Self {
        self.stream_id = Some(stream_id.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
    pub structured: Option<JsonValue>,
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompletionEvent {
    Started {
        run_id: String,
        stream_id: String,
        sequence: u64,
        provider: String,
        model: String,
    },
    Delta {
        run_id: String,
        stream_id: String,
        sequence: u64,
        offset: usize,
        text: String,
    },
    Finished {
        run_id: String,
        stream_id: String,
        sequence: u64,
        response: CompletionResponse,
    },
}

impl CompletionEvent {
    pub fn run_id(&self) -> &str {
        match self {
            Self::Started { run_id, .. }
            | Self::Delta { run_id, .. }
            | Self::Finished { run_id, .. } => run_id,
        }
    }

    pub fn stream_id(&self) -> &str {
        match self {
            Self::Started { stream_id, .. }
            | Self::Delta { stream_id, .. }
            | Self::Finished { stream_id, .. } => stream_id,
        }
    }

    pub fn sequence(&self) -> u64 {
        match self {
            Self::Started { sequence, .. }
            | Self::Delta { sequence, .. }
            | Self::Finished { sequence, .. } => *sequence,
        }
    }
}

#[derive(Debug, Clone)]
struct StreamEnvelopeState {
    run_id: String,
    stream_id: String,
    next_sequence: u64,
    accumulated_text: String,
}

impl StreamEnvelopeState {
    fn from_request(request: &CompletionRequest) -> Self {
        Self::from_ids(request.run_id.clone(), request.stream_id.clone())
    }

    fn from_ids(run_id: Option<String>, stream_id: Option<String>) -> Self {
        Self {
            run_id: run_id.unwrap_or_else(new_run_id),
            stream_id: stream_id.unwrap_or_else(new_stream_id),
            next_sequence: 0,
            accumulated_text: String::new(),
        }
    }

    fn from_response(response: &CompletionResponse, stream_id: Option<String>) -> Self {
        Self {
            run_id: response.run_id.clone().unwrap_or_else(new_run_id),
            stream_id: stream_id.unwrap_or_else(new_stream_id),
            next_sequence: 0,
            accumulated_text: String::new(),
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        sequence
    }

    fn started_event(&mut self, provider: String, model: String) -> CompletionEvent {
        CompletionEvent::Started {
            run_id: self.run_id.clone(),
            stream_id: self.stream_id.clone(),
            sequence: self.next_sequence(),
            provider,
            model,
        }
    }

    fn delta_event(&mut self, text: String) -> CompletionEvent {
        let offset = self.accumulated_text.len();
        self.accumulated_text.push_str(&text);

        CompletionEvent::Delta {
            run_id: self.run_id.clone(),
            stream_id: self.stream_id.clone(),
            sequence: self.next_sequence(),
            offset,
            text,
        }
    }

    fn finished_event(&mut self, mut response: CompletionResponse) -> CompletionEvent {
        if response.run_id.is_none() {
            response.run_id = Some(self.run_id.clone());
        }

        if !self.accumulated_text.is_empty() {
            response.text = self.accumulated_text.clone();
        }

        CompletionEvent::Finished {
            run_id: self.run_id.clone(),
            stream_id: self.stream_id.clone(),
            sequence: self.next_sequence(),
            response,
        }
    }
}

pub struct CompletionEventStream {
    receiver: UnboundedReceiver<FrameworkResult<CompletionEvent>>,
}

impl CompletionEventStream {
    pub fn from_response(response: CompletionResponse) -> Self {
        Self::from_response_with_stream_id(response, None)
    }

    pub fn from_response_with_stream_id(
        response: CompletionResponse,
        stream_id: Option<String>,
    ) -> Self {
        let (sender, receiver) = unbounded_channel();
        let mut envelope = StreamEnvelopeState::from_response(&response, stream_id);
        let _ = sender.send(Ok(envelope.started_event(
            response.provider.clone(),
            response.model.clone(),
        )));

        for chunk in split_stream_chunks(&response.text) {
            let _ = sender.send(Ok(envelope.delta_event(chunk)));
        }

        let _ = sender.send(Ok(envelope.finished_event(response)));
        drop(sender);

        Self { receiver }
    }

    fn from_receiver(receiver: UnboundedReceiver<FrameworkResult<CompletionEvent>>) -> Self {
        Self { receiver }
    }

    pub async fn recv(&mut self) -> Option<FrameworkResult<CompletionEvent>> {
        self.receiver.recv().await
    }
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn name(&self) -> &'static str;
    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse>;

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> FrameworkResult<CompletionEventStream> {
        let stream_id = request.stream_id.clone();
        let run_id = request.run_id.clone();
        let response = self.complete(request).await?;
        let response = if response.run_id.is_some() {
            response
        } else {
            CompletionResponse {
                run_id,
                ..response
            }
        };
        Ok(CompletionEventStream::from_response_with_stream_id(
            response,
            stream_id,
        ))
    }
}

pub type SharedProvider = Arc<dyn ModelProvider>;

pub fn shared_provider<P>(provider: P) -> SharedProvider
where
    P: ModelProvider + 'static,
{
    Arc::new(provider)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub attempts: usize,
    pub delay_ms: u64,
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_delay_ms: Option<u64>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 3,
            delay_ms: 100,
            backoff_multiplier: default_backoff_multiplier(),
            max_delay_ms: Some(1_000),
        }
    }
}

impl RetryPolicy {
    fn delay_for_attempt(&self, attempt: usize) -> u64 {
        if self.delay_ms == 0 {
            return 0;
        }

        let multiplier = self.backoff_multiplier.max(1) as u128;
        let mut delay = self.delay_ms as u128;
        for _ in 0..attempt {
            delay = delay.saturating_mul(multiplier);
        }

        let delay = delay.min(self.max_delay_ms.unwrap_or(u64::MAX) as u128);
        delay.min(u64::MAX as u128) as u64
    }
}

fn default_backoff_multiplier() -> u32 {
    2
}

fn is_retryable_provider_error(error: &FrameworkError) -> bool {
    matches!(error, FrameworkError::Provider(_) | FrameworkError::Transport(_))
}

#[derive(Clone)]
pub struct RetryProvider<P> {
    inner: P,
    policy: RetryPolicy,
}

impl<P> RetryProvider<P> {
    pub fn new(inner: P, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }
}

#[async_trait]
impl<P> ModelProvider for RetryProvider<P>
where
    P: ModelProvider,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let attempts = self.policy.attempts.max(1);
        let mut last_error = None;

        for attempt in 0..attempts {
            match self.inner.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    let retryable = is_retryable_provider_error(&error);
                    last_error = Some(error);
                    if !retryable {
                        break;
                    }

                    let delay_ms = self.policy.delay_for_attempt(attempt);
                    if attempt + 1 < attempts && delay_ms > 0 {
                        sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            FrameworkError::Provider("retry provider exhausted without an error".into())
        }))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> FrameworkResult<CompletionEventStream> {
        let attempts = self.policy.attempts.max(1);
        let mut last_error = None;

        for attempt in 0..attempts {
            match self.inner.complete_stream(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(error) => {
                    let retryable = is_retryable_provider_error(&error);
                    last_error = Some(error);
                    if !retryable {
                        break;
                    }

                    let delay_ms = self.policy.delay_for_attempt(attempt);
                    if attempt + 1 < attempts && delay_ms > 0 {
                        sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            FrameworkError::Provider("retry provider exhausted without an error".into())
        }))
    }
}

#[derive(Clone)]
pub struct TimeoutProvider<P> {
    inner: P,
    timeout_ms: u64,
}

impl<P> TimeoutProvider<P> {
    pub fn new(inner: P, timeout_ms: u64) -> Self {
        Self {
            inner,
            timeout_ms: timeout_ms.max(1),
        }
    }
}

#[async_trait]
impl<P> ModelProvider for TimeoutProvider<P>
where
    P: ModelProvider,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        timeout(
            Duration::from_millis(self.timeout_ms),
            self.inner.complete(request),
        )
        .await
        .map_err(|_| {
            FrameworkError::Provider(format!(
                "provider '{}' timed out after {} ms",
                self.inner.name(),
                self.timeout_ms
            ))
        })?
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> FrameworkResult<CompletionEventStream> {
        timeout(
            Duration::from_millis(self.timeout_ms),
            self.inner.complete_stream(request),
        )
        .await
        .map_err(|_| {
            FrameworkError::Provider(format!(
                "provider '{}' stream setup timed out after {} ms",
                self.inner.name(),
                self.timeout_ms
            ))
        })?
    }
}

#[derive(Clone, Default)]
pub struct FallbackProvider {
    providers: Vec<SharedProvider>,
}

impl FallbackProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_provider<P>(mut self, provider: P) -> Self
    where
        P: ModelProvider + 'static,
    {
        self.providers.push(shared_provider(provider));
        self
    }

    pub fn push_shared(&mut self, provider: SharedProvider) {
        self.providers.push(provider);
    }
}

#[async_trait]
impl ModelProvider for FallbackProvider {
    fn name(&self) -> &'static str {
        "fallback"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        if self.providers.is_empty() {
            return Err(FrameworkError::Provider(
                "fallback provider has no configured inner providers".into(),
            ));
        }

        let mut errors = Vec::new();

        for provider in &self.providers {
            match provider.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => errors.push(format!("{}: {}", provider.name(), error)),
            }
        }

        Err(FrameworkError::Provider(format!(
            "all fallback providers failed: {}",
            errors.join(" | ")
        )))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> FrameworkResult<CompletionEventStream> {
        if self.providers.is_empty() {
            return Err(FrameworkError::Provider(
                "fallback provider has no configured inner providers".into(),
            ));
        }

        let mut errors = Vec::new();

        for provider in &self.providers {
            match provider.complete_stream(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(error) => errors.push(format!("{}: {}", provider.name(), error)),
            }
        }

        Err(FrameworkError::Provider(format!(
            "all fallback providers failed: {}",
            errors.join(" | ")
        )))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerPolicy {
    pub failure_threshold: usize,
    pub open_for_ms: u64,
}

impl Default for CircuitBreakerPolicy {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            open_for_ms: 30_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircuitBreakerSnapshot {
    pub consecutive_failures: usize,
    pub is_open: bool,
    pub open_until_ms: Option<u64>,
}

#[derive(Debug, Default)]
struct CircuitBreakerState {
    consecutive_failures: usize,
    open_until_ms: Option<u64>,
}

#[derive(Clone)]
pub struct CircuitBreakerProvider<P> {
    inner: P,
    policy: CircuitBreakerPolicy,
    state: Arc<Mutex<CircuitBreakerState>>,
}

impl<P> CircuitBreakerProvider<P>
where
    P: ModelProvider,
{
    pub fn new(inner: P, policy: CircuitBreakerPolicy) -> Self {
        Self {
            inner,
            policy,
            state: Arc::new(Mutex::new(CircuitBreakerState::default())),
        }
    }

    pub fn snapshot(&self) -> CircuitBreakerSnapshot {
        snapshot_circuit_state(&self.state)
    }

    fn guard(&self) -> FrameworkResult<()> {
        guard_circuit_state(self.inner.name(), &self.state)
    }

    fn record_success(&self) {
        reset_circuit_state(&self.state);
    }

    fn record_failure(&self) {
        record_circuit_failure(&self.state, &self.policy);
    }
}

#[async_trait]
impl<P> ModelProvider for CircuitBreakerProvider<P>
where
    P: ModelProvider,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        self.guard()?;

        match self.inner.complete(request).await {
            Ok(response) => {
                self.record_success();
                Ok(response)
            }
            Err(error) => {
                if is_retryable_provider_error(&error) {
                    self.record_failure();
                }
                Err(error)
            }
        }
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> FrameworkResult<CompletionEventStream> {
        self.guard()?;

        match self.inner.complete_stream(request).await {
            Ok(mut stream) => {
                let (sender, receiver) = unbounded_channel();
                let state = self.state.clone();
                let policy = self.policy.clone();

                tokio::spawn(async move {
                    let mut saw_finished = false;

                    while let Some(item) = stream.recv().await {
                        match &item {
                            Ok(CompletionEvent::Finished { .. }) => {
                                reset_circuit_state(&state);
                                saw_finished = true;
                            }
                            Err(error) if is_retryable_provider_error(error) => {
                                record_circuit_failure(&state, &policy)
                            }
                            _ => {}
                        }

                        if sender.send(item).is_err() {
                            return;
                        }

                        if saw_finished {
                            return;
                        }
                    }

                    if !saw_finished {
                        record_circuit_failure(&state, &policy);
                    }
                });

                Ok(CompletionEventStream::from_receiver(receiver))
            }
            Err(error) => {
                if is_retryable_provider_error(&error) {
                    self.record_failure();
                }
                Err(error)
            }
        }
    }
}

fn snapshot_circuit_state(state: &Arc<Mutex<CircuitBreakerState>>) -> CircuitBreakerSnapshot {
    let now = now_millis();
    let mut state = lock_mutex(state);

    if state.open_until_ms.is_some_and(|until| until <= now) {
        state.consecutive_failures = 0;
        state.open_until_ms = None;
    }

    let open_until_ms = state.open_until_ms.filter(|until| *until > now);

    CircuitBreakerSnapshot {
        consecutive_failures: state.consecutive_failures,
        is_open: open_until_ms.is_some(),
        open_until_ms,
    }
}

fn guard_circuit_state(
    provider_name: &str,
    state: &Arc<Mutex<CircuitBreakerState>>,
) -> FrameworkResult<()> {
    let now = now_millis();
    let mut state = lock_mutex(state);

    if let Some(open_until_ms) = state.open_until_ms {
        if open_until_ms > now {
            let remaining_ms = open_until_ms.saturating_sub(now);
            return Err(FrameworkError::Provider(format!(
                "provider '{}' circuit is open for {} ms after {} consecutive failures",
                provider_name, remaining_ms, state.consecutive_failures
            )));
        }

        state.consecutive_failures = 0;
        state.open_until_ms = None;
    }

    Ok(())
}

fn record_circuit_failure(
    state: &Arc<Mutex<CircuitBreakerState>>,
    policy: &CircuitBreakerPolicy,
) {
    let mut state = lock_mutex(state);
    state.consecutive_failures = state.consecutive_failures.saturating_add(1);

    if state.consecutive_failures >= policy.failure_threshold.max(1) {
        state.open_until_ms = Some(now_millis().saturating_add(policy.open_for_ms.max(1)));
    }
}

fn reset_circuit_state(state: &Arc<Mutex<CircuitBreakerState>>) {
    let mut state = lock_mutex(state);
    state.consecutive_failures = 0;
    state.open_until_ms = None;
}

fn lock_mutex<T>(mutex: &Arc<Mutex<T>>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn new_run_id() -> String {
    format!("run-{}", Uuid::new_v4())
}

fn new_stream_id() -> String {
    format!("stream-{}", Uuid::new_v4())
}

#[derive(Clone)]
pub struct StructuredOutputProvider<P> {
    inner: P,
}

impl<P> StructuredOutputProvider<P> {
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<P> ModelProvider for StructuredOutputProvider<P>
where
    P: ModelProvider,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn complete(
        &self,
        mut request: CompletionRequest,
    ) -> FrameworkResult<CompletionResponse> {
        let schema = request.response_schema.clone();
        if let Some(schema) = &schema {
            let schema_text = serde_json::to_string_pretty(schema)?;
            let instruction = format!(
                "Return only valid JSON that matches this schema exactly:\n{}",
                schema_text
            );

            request.system_prompt = Some(match request.system_prompt.take() {
                Some(existing) if !existing.trim().is_empty() => {
                    format!("{}\n\n{}", existing, instruction)
                }
                _ => instruction,
            });
        }

        let mut response = self.inner.complete(request).await?;

        if let Some(schema) = &schema {
            let structured = extract_json_value(&response.text)?;
            validate_json_schema(schema, &structured)?;
            response.structured = Some(structured);
        }

        Ok(response)
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> FrameworkResult<CompletionEventStream> {
        if request.response_schema.is_none() {
            return self.inner.complete_stream(request).await;
        }

        let stream_id = request.stream_id.clone();
        let response = self.complete(request).await?;
        Ok(CompletionEventStream::from_response_with_stream_id(
            response,
            stream_id,
        ))
    }
}

pub fn extract_json_value(text: &str) -> FrameworkResult<JsonValue> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<JsonValue>(trimmed) {
        return Ok(value);
    }

    if let Some(value) = extract_json_from_fence(trimmed) {
        return value;
    }

    if let Some(value) = extract_balanced_json(trimmed, '{', '}') {
        return value;
    }

    if let Some(value) = extract_balanced_json(trimmed, '[', ']') {
        return value;
    }

    Err(FrameworkError::Provider(
        "structured output helper could not find valid JSON in provider response".into(),
    ))
}

pub fn validate_json_schema(schema: &JsonValue, value: &JsonValue) -> FrameworkResult<()> {
    if let Some(enum_values) = schema.get("enum").and_then(|value| value.as_array()) {
        if !enum_values.iter().any(|candidate| candidate == value) {
            return Err(FrameworkError::Provider(
                "structured output does not match enum values".into(),
            ));
        }
    }

    if let Some(schema_type) = schema.get("type").and_then(|value| value.as_str()) {
        match schema_type {
            "object" => {
                let object = value.as_object().ok_or_else(|| {
                    FrameworkError::Provider("structured output is not an object".into())
                })?;

                if let Some(required) = schema.get("required").and_then(|value| value.as_array()) {
                    for key in required.iter().filter_map(|value| value.as_str()) {
                        if !object.contains_key(key) {
                            return Err(FrameworkError::Provider(format!(
                                "structured output is missing required field '{}'",
                                key
                            )));
                        }
                    }
                }

                if let Some(properties) =
                    schema.get("properties").and_then(|value| value.as_object())
                {
                    for (key, property_schema) in properties {
                        if let Some(property_value) = object.get(key) {
                            validate_json_schema(property_schema, property_value)?;
                        }
                    }
                }
            }
            "array" => {
                let array = value.as_array().ok_or_else(|| {
                    FrameworkError::Provider("structured output is not an array".into())
                })?;

                if let Some(item_schema) = schema.get("items") {
                    for item in array {
                        validate_json_schema(item_schema, item)?;
                    }
                }
            }
            "string" if !value.is_string() => {
                return Err(FrameworkError::Provider(
                    "structured output is not a string".into(),
                ));
            }
            "number" if !value.is_number() => {
                return Err(FrameworkError::Provider(
                    "structured output is not a number".into(),
                ));
            }
            "integer" if !value.as_i64().is_some() && !value.as_u64().is_some() => {
                return Err(FrameworkError::Provider(
                    "structured output is not an integer".into(),
                ));
            }
            "boolean" if !value.is_boolean() => {
                return Err(FrameworkError::Provider(
                    "structured output is not a boolean".into(),
                ));
            }
            "null" if !value.is_null() => {
                return Err(FrameworkError::Provider(
                    "structured output is not null".into(),
                ));
            }
            _ => {}
        }
    }

    Ok(())
}

fn extract_json_from_fence(text: &str) -> Option<FrameworkResult<JsonValue>> {
    let start = text.find("```")?;
    let rest = &text[start + 3..];
    let after_language = match rest.find('\n') {
        Some(index) => &rest[index + 1..],
        None => rest,
    };
    let end = after_language.find("```")?;
    Some(serde_json::from_str::<JsonValue>(after_language[..end].trim()).map_err(Into::into))
}

fn extract_balanced_json(
    text: &str,
    open: char,
    close: char,
) -> Option<FrameworkResult<JsonValue>> {
    let start = text.find(open)?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let candidate = &text[start..=start + offset];
                    return Some(serde_json::from_str::<JsonValue>(candidate).map_err(Into::into));
                }
            }
            _ => {}
        }
    }

    None
}

#[derive(Debug, Default, Clone)]
pub struct MockProvider;

#[async_trait]
impl ModelProvider for MockProvider {
    fn name(&self) -> &'static str {
        "mock"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let latest = request
            .messages
            .last()
            .map(|message| message.content.clone())
            .unwrap_or_else(|| "Hello from Harbor".to_string());

        Ok(CompletionResponse {
            text: format!(
                "[mock:{}] {}",
                request.session_id.unwrap_or_else(|| "sessionless".into()),
                latest
            ),
            structured: request.response_schema.map(|schema| {
                serde_json::json!({
                    "echo": latest,
                    "schema_hint": schema
                })
            }),
            provider: self.name().into(),
            model: request.model,
            run_id: request.run_id,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatibleConfig {
    pub api_key: String,
    pub base_url: String,
    pub default_model: String,
}

impl OpenAICompatibleConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .or_else(|_| env::var("HARBOR_OPENAI_API_KEY"))
            .map_err(|_| {
                FrameworkError::Config(
                    "missing OPENAI_API_KEY (or HARBOR_OPENAI_API_KEY) for OpenAI-compatible provider"
                        .into(),
                )
            })?;

        let base_url = env::var("OPENAI_BASE_URL")
            .or_else(|_| env::var("HARBOR_OPENAI_BASE_URL"))
            .unwrap_or_else(|_| "https://api.openai.com/v1".into());

        let default_model = env::var("OPENAI_MODEL")
            .or_else(|_| env::var("HARBOR_OPENAI_MODEL"))
            .unwrap_or_else(|_| "gpt-4o-mini".into());

        Ok(Self {
            api_key,
            base_url,
            default_model,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OpenAICompatibleProvider {
    client: Client,
    config: OpenAICompatibleConfig,
}

impl OpenAICompatibleProvider {
    pub fn new(config: OpenAICompatibleConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(OpenAICompatibleConfig::from_env()?))
    }

    pub fn with_client(client: Client, config: OpenAICompatibleConfig) -> Self {
        Self { client, config }
    }
}

#[async_trait]
impl ModelProvider for OpenAICompatibleProvider {
    fn name(&self) -> &'static str {
        "openai-compatible"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let CompletionRequest {
            model,
            system_prompt,
            messages,
            run_id,
            ..
        } = request;

        let model = model_or_default(model, &self.config.default_model);
        let mut api_messages = Vec::new();

        if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
            api_messages.push(OpenAIMessageRequest {
                role: "system".into(),
                content: system_prompt,
            });
        }

        api_messages.extend(messages.into_iter().map(|message| OpenAIMessageRequest {
            role: message.role.as_openai_role().into(),
            content: message.content,
        }));

        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);

        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.config.api_key)
            .headers(headers)
            .json(&OpenAIChatRequest {
                model: model.clone(),
                messages: api_messages,
                stream: false,
            })
            .send()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let response: OpenAIChatResponse = response
            .json()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let text = response
            .choices
            .into_iter()
            .find_map(|choice| {
                choice
                    .message
                    .and_then(|message| message.content)
                    .or(choice.text)
            })
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                FrameworkError::Provider(
                    "OpenAI-compatible provider returned no assistant text response".into(),
                )
            })?;

        Ok(CompletionResponse {
            text,
            structured: None,
            provider: self.name().into(),
            model: response.model.unwrap_or(model),
            run_id,
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> FrameworkResult<CompletionEventStream> {
        let CompletionRequest {
            model,
            system_prompt,
            messages,
            run_id,
            stream_id,
            ..
        } = request;

        let model = model_or_default(model, &self.config.default_model);
        let mut api_messages = Vec::new();

        if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
            api_messages.push(OpenAIMessageRequest {
                role: "system".into(),
                content: system_prompt,
            });
        }

        api_messages.extend(messages.into_iter().map(|message| OpenAIMessageRequest {
            role: message.role.as_openai_role().into(),
            content: message.content,
        }));

        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);

        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.config.api_key)
            .headers(headers)
            .json(&OpenAIChatRequest {
                model: model.clone(),
                messages: api_messages,
                stream: true,
            })
            .send()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let (sender, receiver) = unbounded_channel();
        let provider = self.name().to_string();
        let envelope = StreamEnvelopeState::from_ids(run_id, stream_id);
        tokio::spawn(async move {
            stream_openai_response(response, sender, provider, model, envelope).await;
        });

        Ok(CompletionEventStream::from_receiver(receiver))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub default_model: String,
    pub max_tokens: u32,
}

impl AnthropicConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .or_else(|_| env::var("HARBOR_ANTHROPIC_API_KEY"))
            .map_err(|_| {
                FrameworkError::Config(
                    "missing ANTHROPIC_API_KEY (or HARBOR_ANTHROPIC_API_KEY) for Anthropic provider"
                        .into(),
                )
            })?;

        let base_url = env::var("ANTHROPIC_BASE_URL")
            .or_else(|_| env::var("HARBOR_ANTHROPIC_BASE_URL"))
            .unwrap_or_else(|_| "https://api.anthropic.com/v1".into());

        let default_model = env::var("ANTHROPIC_MODEL")
            .or_else(|_| env::var("HARBOR_ANTHROPIC_MODEL"))
            .unwrap_or_else(|_| "claude-3-5-haiku-latest".into());

        let max_tokens = env::var("ANTHROPIC_MAX_TOKENS")
            .or_else(|_| env::var("HARBOR_ANTHROPIC_MAX_TOKENS"))
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS);

        Ok(Self {
            api_key,
            base_url,
            default_model,
            max_tokens,
        })
    }
}

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: Client,
    config: AnthropicConfig,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(AnthropicConfig::from_env()?))
    }

    pub fn with_client(client: Client, config: AnthropicConfig) -> Self {
        Self { client, config }
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let CompletionRequest {
            model,
            system_prompt,
            messages,
            run_id,
            ..
        } = request;

        let model = model_or_default(model, &self.config.default_model);
        let system_prompt = system_prompt.filter(|value| !value.trim().is_empty());
        let api_messages = messages
            .into_iter()
            .filter_map(|message| {
                message
                    .role
                    .as_anthropic_role()
                    .map(|role| AnthropicMessageRequest {
                        role: role.into(),
                        content: vec![AnthropicContentBlockRequest::text(message.content)],
                    })
            })
            .collect::<Vec<_>>();

        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);
        headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(&self.config.api_key)
                .map_err(|error| FrameworkError::Config(error.to_string()))?,
        );
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );

        let response = self
            .client
            .post(format!(
                "{}/messages",
                self.config.base_url.trim_end_matches('/')
            ))
            .headers(headers)
            .json(&AnthropicMessagesRequest {
                model: model.clone(),
                system: system_prompt,
                max_tokens: self.config.max_tokens,
                messages: api_messages,
            })
            .send()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let response: AnthropicMessagesResponse = response
            .json()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let text = response
            .content
            .into_iter()
            .filter(|block| block.kind == "text")
            .filter_map(|block| block.text)
            .filter(|value| !value.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        if text.trim().is_empty() {
            return Err(FrameworkError::Provider(
                "Anthropic provider returned no assistant text response".into(),
            ));
        }

        Ok(CompletionResponse {
            text,
            structured: None,
            provider: self.name().into(),
            model: response.model.unwrap_or(model),
            run_id,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub base_url: String,
    pub default_model: String,
}

impl OllamaConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let base_url = env::var("OLLAMA_BASE_URL")
            .or_else(|_| env::var("HARBOR_OLLAMA_BASE_URL"))
            .unwrap_or_else(|_| "http://127.0.0.1:11434".into());

        let default_model = env::var("OLLAMA_MODEL")
            .or_else(|_| env::var("HARBOR_OLLAMA_MODEL"))
            .unwrap_or_else(|_| "llama3.2".into());

        Ok(Self {
            base_url,
            default_model,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    client: Client,
    config: OllamaConfig,
}

impl OllamaProvider {
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn from_env() -> FrameworkResult<Self> {
        Ok(Self::new(OllamaConfig::from_env()?))
    }

    pub fn with_client(client: Client, config: OllamaConfig) -> Self {
        Self { client, config }
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "ollama"
    }

    async fn complete(&self, request: CompletionRequest) -> FrameworkResult<CompletionResponse> {
        let CompletionRequest {
            model,
            system_prompt,
            messages,
            run_id,
            ..
        } = request;

        let model = model_or_default(model, &self.config.default_model);
        let mut api_messages = Vec::new();

        if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
            api_messages.push(OllamaMessageRequest {
                role: "system".into(),
                content: system_prompt,
            });
        }

        api_messages.extend(messages.into_iter().map(|message| OllamaMessageRequest {
            role: message.role.as_ollama_role().into(),
            content: message.content,
        }));

        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);

        let response = self
            .client
            .post(format!(
                "{}/api/chat",
                self.config.base_url.trim_end_matches('/')
            ))
            .headers(headers)
            .json(&OllamaChatRequest {
                model: model.clone(),
                messages: api_messages,
                stream: false,
            })
            .send()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let response: OllamaChatResponse = response
            .json()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let text = response
            .message
            .and_then(|message| {
                if message.content.trim().is_empty() {
                    None
                } else {
                    Some(message.content)
                }
            })
            .ok_or_else(|| {
                FrameworkError::Provider(
                    "Ollama provider returned no assistant text response".into(),
                )
            })?;

        Ok(CompletionResponse {
            text,
            structured: None,
            provider: self.name().into(),
            model: response.model.unwrap_or(model),
            run_id,
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> FrameworkResult<CompletionEventStream> {
        let CompletionRequest {
            model,
            system_prompt,
            messages,
            run_id,
            stream_id,
            ..
        } = request;

        let model = model_or_default(model, &self.config.default_model);
        let mut api_messages = Vec::new();

        if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
            api_messages.push(OllamaMessageRequest {
                role: "system".into(),
                content: system_prompt,
            });
        }

        api_messages.extend(messages.into_iter().map(|message| OllamaMessageRequest {
            role: message.role.as_ollama_role().into(),
            content: message.content,
        }));

        let mut headers = HeaderMap::new();
        inject_trace_context(&mut headers);

        let response = self
            .client
            .post(format!(
                "{}/api/chat",
                self.config.base_url.trim_end_matches('/')
            ))
            .headers(headers)
            .json(&OllamaChatRequest {
                model: model.clone(),
                messages: api_messages,
                stream: true,
            })
            .send()
            .await
            .map_err(|error| FrameworkError::Provider(error.to_string()))?
            .error_for_status()
            .map_err(|error| FrameworkError::Provider(error.to_string()))?;

        let (sender, receiver) = unbounded_channel();
        let provider = self.name().to_string();
        let envelope = StreamEnvelopeState::from_ids(run_id, stream_id);
        tokio::spawn(async move {
            stream_ollama_response(response, sender, provider, model, envelope).await;
        });

        Ok(CompletionEventStream::from_receiver(receiver))
    }
}

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessageRequest>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessageRequest {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: Option<OpenAIMessageResponse>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessageResponse {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: Option<OpenAIDeltaResponse>,
    text: Option<String>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIDeltaResponse {
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessagesRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    messages: Vec<AnthropicMessageRequest>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessageRequest {
    role: String,
    content: Vec<AnthropicContentBlockRequest>,
}

#[derive(Debug, Serialize)]
struct AnthropicContentBlockRequest {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

impl AnthropicContentBlockRequest {
    fn text(text: String) -> Self {
        Self {
            kind: "text".into(),
            text,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicMessagesResponse {
    model: Option<String>,
    #[serde(default)]
    content: Vec<AnthropicContentBlockResponse>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlockResponse {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessageRequest>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OllamaMessageRequest {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    model: Option<String>,
    message: Option<OllamaMessageResponse>,
    done: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct OllamaMessageResponse {
    content: String,
}

async fn stream_openai_response(
    mut response: reqwest::Response,
    sender: tokio::sync::mpsc::UnboundedSender<FrameworkResult<CompletionEvent>>,
    provider: String,
    requested_model: String,
    mut envelope: StreamEnvelopeState,
) {
    let _ = sender.send(Ok(envelope.started_event(
        provider.clone(),
        requested_model.clone(),
    )));

    let mut buffer = String::new();
    let mut response_model = requested_model;

    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(frame) = take_sse_frame(&mut buffer) {
                    for line in frame.lines() {
                        let Some(data) = line.strip_prefix("data:") else {
                            continue;
                        };

                        let data = data.trim();
                        if data.is_empty() {
                            continue;
                        }

                        if data == "[DONE]" {
                            let _ = finish_stream(
                                &sender,
                                &mut envelope,
                                provider.clone(),
                                response_model,
                            );
                            return;
                        }

                        match serde_json::from_str::<OpenAIStreamResponse>(data) {
                            Ok(event) => {
                                if let Some(model) = event.model {
                                    response_model = model;
                                }

                                for choice in event.choices {
                                    if let Some(text) = choice
                                        .delta
                                        .and_then(|delta| delta.content)
                                        .or(choice.text)
                                        .filter(|text| !text.is_empty())
                                    {
                                        let _ = sender.send(Ok(envelope.delta_event(text)));
                                    }

                                    if choice.finish_reason.is_some() {
                                        let _ = finish_stream(
                                            &sender,
                                            &mut envelope,
                                            provider.clone(),
                                            response_model.clone(),
                                        );
                                        return;
                                    }
                                }
                            }
                            Err(error) => {
                                let _ = sender.send(Err(FrameworkError::Provider(error.to_string())));
                                return;
                            }
                        }
                    }
                }
            }
            Ok(None) => {
                let _ = finish_stream(&sender, &mut envelope, provider.clone(), response_model);
                return;
            }
            Err(error) => {
                let _ = sender.send(Err(FrameworkError::Provider(error.to_string())));
                return;
            }
        }
    }
}

async fn stream_ollama_response(
    mut response: reqwest::Response,
    sender: tokio::sync::mpsc::UnboundedSender<FrameworkResult<CompletionEvent>>,
    provider: String,
    requested_model: String,
    mut envelope: StreamEnvelopeState,
) {
    let _ = sender.send(Ok(envelope.started_event(
        provider.clone(),
        requested_model.clone(),
    )));

    let mut buffer = String::new();
    let mut response_model = requested_model;

    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(line) = take_line(&mut buffer) {
                    if handle_ollama_stream_line(
                        line,
                        &sender,
                        &mut envelope,
                        &provider,
                        &mut response_model,
                    ) {
                        return;
                    }
                }
            }
            Ok(None) => {
                if !buffer.trim().is_empty()
                    && handle_ollama_stream_line(
                        std::mem::take(&mut buffer),
                        &sender,
                        &mut envelope,
                        &provider,
                        &mut response_model,
                    )
                {
                    return;
                }

                let _ = finish_stream(&sender, &mut envelope, provider.clone(), response_model);
                return;
            }
            Err(error) => {
                let _ = sender.send(Err(FrameworkError::Provider(error.to_string())));
                return;
            }
        }
    }
}

fn handle_ollama_stream_line(
    line: String,
    sender: &tokio::sync::mpsc::UnboundedSender<FrameworkResult<CompletionEvent>>,
    envelope: &mut StreamEnvelopeState,
    provider: &str,
    response_model: &mut String,
) -> bool {
    let line = line.trim();
    if line.is_empty() {
        return false;
    }

    match serde_json::from_str::<OllamaChatResponse>(line) {
        Ok(event) => {
            if let Some(model) = event.model {
                *response_model = model;
            }

            if let Some(text) = event
                .message
                .and_then(|message| if message.content.is_empty() { None } else { Some(message.content) })
            {
                let _ = sender.send(Ok(envelope.delta_event(text)));
            }

            if event.done.unwrap_or(false) {
                let _ = finish_stream(sender, envelope, provider.to_string(), response_model.clone());
                return true;
            }
        }
        Err(error) => {
            let _ = sender.send(Err(FrameworkError::Provider(error.to_string())));
            return true;
        }
    }

    false
}

fn finish_stream(
    sender: &tokio::sync::mpsc::UnboundedSender<FrameworkResult<CompletionEvent>>,
    envelope: &mut StreamEnvelopeState,
    provider: String,
    model: String,
) -> Result<(), tokio::sync::mpsc::error::SendError<FrameworkResult<CompletionEvent>>> {
    sender.send(Ok(envelope.finished_event(CompletionResponse {
        text: String::new(),
        structured: None,
        provider,
        model,
        run_id: Some(envelope.run_id.clone()),
    })))
}

fn take_sse_frame(buffer: &mut String) -> Option<String> {
    if let Some(index) = buffer.find("\r\n\r\n") {
        let frame = buffer[..index].to_string();
        buffer.replace_range(..index + 4, "");
        return Some(frame);
    }

    if let Some(index) = buffer.find("\n\n") {
        let frame = buffer[..index].to_string();
        buffer.replace_range(..index + 2, "");
        return Some(frame);
    }

    None
}

fn take_line(buffer: &mut String) -> Option<String> {
    if let Some(index) = buffer.find('\n') {
        let line = buffer[..index].trim_end_matches('\r').to_string();
        buffer.replace_range(..index + 1, "");
        Some(line)
    } else {
        None
    }
}

fn model_or_default(model: String, default_model: &str) -> String {
    if model.trim().is_empty() {
        default_model.into()
    } else {
        model
    }
}

fn inject_trace_context(headers: &mut HeaderMap) {
    let context = Span::current().context();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}

fn split_stream_chunks(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let chars = text.chars().collect::<Vec<_>>();
    chars
        .chunks(16)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect()
}

struct HeaderInjector<'a>(&'a mut HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        extract::State,
        http::{header::CONTENT_TYPE, HeaderMap},
        response::IntoResponse,
        routing::post,
        Json, Router,
    };
    use opentelemetry::{
        global,
        trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState},
        Context,
    };
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use serde_json::{json, Value};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };
    use tokio::net::TcpListener;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    #[derive(Clone, Default)]
    struct RequestCapture {
        system_prompts: Arc<Mutex<Vec<Option<String>>>>,
    }

    #[derive(Clone)]
    struct CapturingProvider {
        provider_name: &'static str,
        response_text: String,
        capture: RequestCapture,
    }

    #[async_trait]
    impl ModelProvider for CapturingProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> FrameworkResult<CompletionResponse> {
            self.capture
                .system_prompts
                .lock()
                .unwrap()
                .push(request.system_prompt.clone());

            Ok(CompletionResponse {
                text: self.response_text.clone(),
                structured: None,
                provider: self.provider_name.into(),
                model: request.model,
                run_id: request.run_id,
            })
        }
    }

    #[derive(Clone)]
    struct StaticProvider {
        provider_name: &'static str,
        text: String,
    }

    #[async_trait]
    impl ModelProvider for StaticProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> FrameworkResult<CompletionResponse> {
            Ok(CompletionResponse {
                text: self.text.clone(),
                structured: None,
                provider: self.provider_name.into(),
                model: request.model,
                run_id: request.run_id,
            })
        }
    }

    #[derive(Clone)]
    struct FlakyProvider {
        provider_name: &'static str,
        failures_before_success: usize,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ModelProvider for FlakyProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> FrameworkResult<CompletionResponse> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call <= self.failures_before_success {
                Err(FrameworkError::Provider(format!(
                    "{} failed on call {}",
                    self.provider_name, call
                )))
            } else {
                Ok(CompletionResponse {
                    text: format!("{} ok", self.provider_name),
                    structured: None,
                    provider: self.provider_name.into(),
                    model: request.model,
                    run_id: request.run_id,
                })
            }
        }
    }

    #[derive(Clone)]
    struct SlowProvider {
        provider_name: &'static str,
        delay_ms: u64,
    }

    #[async_trait]
    impl ModelProvider for SlowProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> FrameworkResult<CompletionResponse> {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            Ok(CompletionResponse {
                text: format!("{} done", self.provider_name),
                structured: None,
                provider: self.provider_name.into(),
                model: request.model,
                run_id: request.run_id,
            })
        }
    }

    #[derive(Clone)]
    struct NativeStreamingProvider {
        provider_name: &'static str,
        complete_text: &'static str,
        stream_chunks: Vec<&'static str>,
        failures_before_stream_success: usize,
        stream_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ModelProvider for NativeStreamingProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> FrameworkResult<CompletionResponse> {
            Ok(CompletionResponse {
                text: self.complete_text.into(),
                structured: None,
                provider: self.provider_name.into(),
                model: request.model,
                run_id: request.run_id,
            })
        }

        async fn complete_stream(
            &self,
            request: CompletionRequest,
        ) -> FrameworkResult<CompletionEventStream> {
            let call = self.stream_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call <= self.failures_before_stream_success {
                return Err(FrameworkError::Provider(format!(
                    "{} stream failed on call {}",
                    self.provider_name, call
                )));
            }

            Ok(native_stream(
                &request,
                self.provider_name,
                self.stream_chunks.clone(),
            ))
        }
    }

    fn native_stream(
        request: &CompletionRequest,
        provider_name: &str,
        stream_chunks: Vec<&'static str>,
    ) -> CompletionEventStream {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut envelope = StreamEnvelopeState::from_request(request);
        let provider = provider_name.to_string();
        let model = request.model.clone();

        let _ = sender.send(Ok(envelope.started_event(provider.clone(), model.clone())));

        for chunk in stream_chunks {
            let _ = sender.send(Ok(envelope.delta_event(chunk.to_string())));
        }

        let _ = sender.send(Ok(envelope.finished_event(CompletionResponse {
            text: String::new(),
            structured: None,
            provider,
            model,
            run_id: Some(envelope.run_id.clone()),
        })));
        drop(sender);

        CompletionEventStream::from_receiver(receiver)
    }

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".into(),
            system_prompt: None,
            messages: vec![Message {
                role: MessageRole::User,
                content: "hello".into(),
            }],
            tools: vec![],
            response_schema: None,
            session_id: None,
            tool_choice: ToolChoice::Auto,
            run_id: None,
            stream_id: None,
        }
    }

    #[tokio::test]
    async fn retry_provider_retries_until_success() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = RetryProvider::new(
            FlakyProvider {
                provider_name: "flaky",
                failures_before_success: 2,
                calls: calls.clone(),
            },
            RetryPolicy {
                attempts: 3,
                delay_ms: 0,
                ..RetryPolicy::default()
            },
        );

        let response = provider.complete(sample_request()).await.unwrap();
        assert_eq!(response.provider, "flaky");
        assert_eq!(response.text, "flaky ok");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn retry_policy_uses_exponential_backoff_with_cap() {
        let policy = RetryPolicy {
            attempts: 4,
            delay_ms: 10,
            backoff_multiplier: 3,
            max_delay_ms: Some(50),
        };

        assert_eq!(policy.delay_for_attempt(0), 10);
        assert_eq!(policy.delay_for_attempt(1), 30);
        assert_eq!(policy.delay_for_attempt(2), 50);
        assert_eq!(policy.delay_for_attempt(3), 50);
    }

    #[tokio::test]
    async fn retry_provider_retries_stream_setup_and_preserves_native_stream() {
        let stream_calls = Arc::new(AtomicUsize::new(0));
        let provider = RetryProvider::new(
            NativeStreamingProvider {
                provider_name: "native-stream",
                complete_text: "complete-path",
                stream_chunks: vec!["native ", "stream"],
                failures_before_stream_success: 1,
                stream_calls: stream_calls.clone(),
            },
            RetryPolicy {
                attempts: 2,
                delay_ms: 0,
                ..RetryPolicy::default()
            },
        );

        let mut stream = provider
            .complete_stream(
                sample_request()
                    .with_run_id("run-native-retry")
                    .with_stream_id("stream-native-retry"),
            )
            .await
            .unwrap();

        let mut deltas = String::new();
        let mut finished = None;

        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                CompletionEvent::Started { .. } => {}
                CompletionEvent::Delta { text, .. } => deltas.push_str(&text),
                CompletionEvent::Finished { response, .. } => finished = Some(response),
            }
        }

        assert_eq!(stream_calls.load(Ordering::SeqCst), 2);
        assert_eq!(deltas, "native stream");

        let finished = finished.unwrap();
        assert_eq!(finished.text, "native stream");
        assert_eq!(finished.run_id.as_deref(), Some("run-native-retry"));
    }

    #[tokio::test]
    async fn timeout_provider_returns_timeout_error() {
        let provider = TimeoutProvider::new(
            SlowProvider {
                provider_name: "slow",
                delay_ms: 50,
            },
            10,
        );

        let error = provider.complete(sample_request()).await.unwrap_err();
        assert!(error.to_string().contains("timed out after 10 ms"));
    }

    #[tokio::test]
    async fn fallback_provider_uses_second_provider_after_first_failure() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = FallbackProvider::new()
            .with_provider(FlakyProvider {
                provider_name: "first",
                failures_before_success: usize::MAX,
                calls,
            })
            .with_provider(StaticProvider {
                provider_name: "second",
                text: "fallback ok".into(),
            });

        let response = provider.complete(sample_request()).await.unwrap();
        assert_eq!(response.provider, "second");
        assert_eq!(response.text, "fallback ok");
    }

    #[tokio::test]
    async fn fallback_provider_stream_uses_second_provider_after_first_failure() {
        let provider = FallbackProvider::new()
            .with_provider(NativeStreamingProvider {
                provider_name: "first-stream",
                complete_text: "first-complete",
                stream_chunks: vec!["first"],
                failures_before_stream_success: usize::MAX,
                stream_calls: Arc::new(AtomicUsize::new(0)),
            })
            .with_provider(NativeStreamingProvider {
                provider_name: "second-stream",
                complete_text: "second-complete",
                stream_chunks: vec!["fallback ", "stream"],
                failures_before_stream_success: 0,
                stream_calls: Arc::new(AtomicUsize::new(0)),
            });

        let mut stream = provider
            .complete_stream(
                sample_request()
                    .with_run_id("run-fallback-stream")
                    .with_stream_id("stream-fallback-stream"),
            )
            .await
            .unwrap();

        let mut deltas = String::new();
        let mut finished = None;

        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                CompletionEvent::Started { provider, .. } => assert_eq!(provider, "second-stream"),
                CompletionEvent::Delta { text, .. } => deltas.push_str(&text),
                CompletionEvent::Finished { response, .. } => finished = Some(response),
            }
        }

        assert_eq!(deltas, "fallback stream");

        let finished = finished.unwrap();
        assert_eq!(finished.text, "fallback stream");
        assert_eq!(finished.run_id.as_deref(), Some("run-fallback-stream"));
    }

    #[tokio::test]
    async fn circuit_breaker_opens_after_failures_and_recovers_after_cooldown() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = CircuitBreakerProvider::new(
            FlakyProvider {
                provider_name: "breaker",
                failures_before_success: 2,
                calls: calls.clone(),
            },
            CircuitBreakerPolicy {
                failure_threshold: 2,
                open_for_ms: 25,
            },
        );

        provider.complete(sample_request()).await.unwrap_err();
        assert_eq!(provider.snapshot().consecutive_failures, 1);
        assert!(!provider.snapshot().is_open);

        provider.complete(sample_request()).await.unwrap_err();
        assert!(provider.snapshot().is_open);
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        let open_error = provider.complete(sample_request()).await.unwrap_err();
        assert!(open_error.to_string().contains("circuit is open"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        let response = provider.complete(sample_request()).await.unwrap();
        assert_eq!(response.text, "breaker ok");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(provider.snapshot().consecutive_failures, 0);
        assert!(!provider.snapshot().is_open);
    }

    #[test]
    fn extract_json_value_finds_json_inside_code_fence() {
        let value = extract_json_value("Here you go:\n```json\n{\"answer\":\"hi\"}\n```").unwrap();
        assert_eq!(value["answer"], "hi");
    }

    #[test]
    fn validate_json_schema_rejects_missing_required_field() {
        let schema = json!({
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer": { "type": "string" }
            }
        });

        let error = validate_json_schema(&schema, &json!({ "other": "x" })).unwrap_err();
        assert!(error
            .to_string()
            .contains("missing required field 'answer'"));
    }

    #[tokio::test]
    async fn completion_event_stream_emits_started_delta_and_finished() {
        let response = CompletionResponse {
            text: "stream me please".into(),
            structured: None,
            provider: "mock".into(),
            model: "mock-model".into(),
            run_id: Some("run-stream-test".into()),
        };
        let expected_text = response.text.clone();

        let mut stream = CompletionEventStream::from_response(response.clone());
        let mut saw_started = false;
        let mut deltas = String::new();
        let mut finished = None;
        let mut seen_sequences = Vec::new();
        let mut seen_stream_id = None;

        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                CompletionEvent::Started {
                    run_id,
                    stream_id,
                    sequence,
                    provider,
                    model,
                } => {
                    saw_started = true;
                    assert_eq!(run_id, "run-stream-test");
                    assert_eq!(provider, "mock");
                    assert_eq!(model, "mock-model");
                    seen_sequences.push(sequence);
                    seen_stream_id = Some(stream_id);
                }
                CompletionEvent::Delta {
                    run_id,
                    stream_id,
                    sequence,
                    offset,
                    text,
                } => {
                    assert_eq!(run_id, "run-stream-test");
                    assert_eq!(stream_id, seen_stream_id.clone().unwrap());
                    assert_eq!(offset, deltas.len());
                    deltas.push_str(&text);
                    seen_sequences.push(sequence);
                }
                CompletionEvent::Finished {
                    run_id,
                    stream_id,
                    sequence,
                    response,
                } => {
                    assert_eq!(run_id, "run-stream-test");
                    assert_eq!(stream_id, seen_stream_id.clone().unwrap());
                    seen_sequences.push(sequence);
                    finished = Some(response);
                }
            }
        }

        assert!(saw_started);
        assert_eq!(deltas, expected_text);
        assert_eq!(seen_sequences, (0..seen_sequences.len() as u64).collect::<Vec<_>>());

        let finished = finished.unwrap();
        assert_eq!(finished.text, response.text);
        assert_eq!(finished.run_id.as_deref(), Some("run-stream-test"));
    }

    #[tokio::test]
    async fn default_complete_stream_uses_complete_response() {
        let provider = StaticProvider {
            provider_name: "static",
            text: "hello from stream".into(),
        };

        let mut stream = provider
            .complete_stream(
                sample_request()
                    .with_run_id("run-static-stream")
                    .with_stream_id("stream-static-stream"),
            )
            .await
            .unwrap();
        let mut saw_started = false;
        let mut deltas = String::new();
        let mut finished = None;
        let mut seen_sequences = Vec::new();

        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                CompletionEvent::Started {
                    run_id,
                    stream_id,
                    sequence,
                    provider,
                    model,
                } => {
                    saw_started = true;
                    assert_eq!(run_id, "run-static-stream");
                    assert_eq!(stream_id, "stream-static-stream");
                    assert_eq!(provider, "static");
                    assert_eq!(model, "test-model");
                    seen_sequences.push(sequence);
                }
                CompletionEvent::Delta {
                    run_id,
                    stream_id,
                    sequence,
                    offset,
                    text,
                } => {
                    assert_eq!(run_id, "run-static-stream");
                    assert_eq!(stream_id, "stream-static-stream");
                    assert_eq!(offset, deltas.len());
                    deltas.push_str(&text);
                    seen_sequences.push(sequence);
                }
                CompletionEvent::Finished {
                    run_id,
                    stream_id,
                    sequence,
                    response,
                } => {
                    assert_eq!(run_id, "run-static-stream");
                    assert_eq!(stream_id, "stream-static-stream");
                    seen_sequences.push(sequence);
                    finished = Some(response);
                }
            }
        }

        assert!(saw_started);
        assert_eq!(deltas, "hello from stream");
        assert_eq!(seen_sequences, (0..seen_sequences.len() as u64).collect::<Vec<_>>());

        let finished = finished.unwrap();
        assert_eq!(finished.provider, "static");
        assert_eq!(finished.model, "test-model");
        assert_eq!(finished.text, "hello from stream");
        assert_eq!(finished.run_id.as_deref(), Some("run-static-stream"));
    }

    #[tokio::test]
    async fn structured_output_provider_parses_and_validates_json() {
        let capture = RequestCapture::default();
        let provider = StructuredOutputProvider::new(CapturingProvider {
            provider_name: "capture",
            response_text: "```json\n{\"answer\":\"hi\"}\n```".into(),
            capture: capture.clone(),
        });

        let schema = json!({
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer": { "type": "string" }
            }
        });

        let response = provider
            .complete(CompletionRequest {
                model: "test-model".into(),
                system_prompt: Some("Be precise".into()),
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Say hi".into(),
                }],
                tools: vec![],
                response_schema: Some(schema.clone()),
                session_id: None,
                tool_choice: ToolChoice::Auto,
                run_id: None,
                stream_id: None,
            })
            .await
            .unwrap();

        assert_eq!(response.structured.unwrap()["answer"], "hi");

        let captured_prompt = capture.system_prompts.lock().unwrap().remove(0).unwrap();
        assert!(captured_prompt.contains("Be precise"));
        assert!(captured_prompt.contains("Return only valid JSON that matches this schema exactly"));
        assert!(captured_prompt.contains("\"answer\""));
    }

    #[derive(Clone, Default)]
    struct CaptureState {
        inner: Arc<Mutex<Vec<CapturedRequest>>>,
    }

    #[derive(Debug, Clone)]
    struct CapturedRequest {
        headers: Vec<(String, String)>,
        body: Value,
    }

    impl CaptureState {
        fn push(&self, captured: CapturedRequest) {
            self.inner.lock().unwrap().push(captured);
        }

        fn take_one(&self) -> CapturedRequest {
            self.inner.lock().unwrap().remove(0)
        }
    }

    #[tokio::test]
    async fn openai_provider_maps_request_and_response() {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let state = CaptureState::default();
        let app = Router::new()
            .route("/chat/completions", post(openai_handler))
            .with_state(state.clone());
        let base_url = spawn_router(app).await;

        let provider = OpenAICompatibleProvider::new(OpenAICompatibleConfig {
            api_key: "openai-test-key".into(),
            base_url,
            default_model: "gpt-test-model".into(),
        });

        let response = provider
            .complete(CompletionRequest {
                model: String::new(),
                system_prompt: Some("Be direct".into()),
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Hello from OpenAI".into(),
                }],
                tools: vec![],
                response_schema: None,
                session_id: Some("session-openai".into()),
                tool_choice: ToolChoice::Auto,
                run_id: None,
                stream_id: None,
            })
            .await
            .unwrap();

        assert_eq!(response.provider, "openai-compatible");
        assert_eq!(response.model, "gpt-test-model");
        assert_eq!(response.text, "Hi from OpenAI");

        let captured = state.take_one();
        let headers = captured
            .headers
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        if let Some(traceparent) = headers.get("traceparent") {
            assert!(traceparent.starts_with("00-"));
            assert!(!traceparent.trim().is_empty());
        }

        assert_eq!(captured.body["model"], "gpt-test-model");
        assert_eq!(captured.body["stream"], false);
        assert_eq!(captured.body["messages"][0]["role"], "system");
        assert_eq!(captured.body["messages"][0]["content"], "Be direct");
        assert_eq!(captured.body["messages"][1]["role"], "user");
        assert_eq!(captured.body["messages"][1]["content"], "Hello from OpenAI");
    }

    #[tokio::test]
    async fn openai_provider_streams_native_events() {
        let state = CaptureState::default();
        let app = Router::new()
            .route("/chat/completions", post(openai_handler))
            .with_state(state.clone());
        let base_url = spawn_router(app).await;

        let provider = OpenAICompatibleProvider::new(OpenAICompatibleConfig {
            api_key: "openai-test-key".into(),
            base_url,
            default_model: "gpt-test-model".into(),
        });

        let mut stream = provider
            .complete_stream(CompletionRequest {
                model: String::new(),
                system_prompt: None,
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Stream it".into(),
                }],
                tools: vec![],
                response_schema: None,
                session_id: Some("session-openai-stream".into()),
                tool_choice: ToolChoice::Auto,
                run_id: Some("run-openai-stream".into()),
                stream_id: Some("stream-openai-stream".into()),
            })
            .await
            .unwrap();

        let mut saw_started = false;
        let mut deltas = String::new();
        let mut finished = None;
        let mut seen_sequences = Vec::new();

        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                CompletionEvent::Started {
                    run_id,
                    stream_id,
                    sequence,
                    provider,
                    model,
                } => {
                    saw_started = true;
                    assert_eq!(run_id, "run-openai-stream");
                    assert_eq!(stream_id, "stream-openai-stream");
                    assert_eq!(provider, "openai-compatible");
                    assert_eq!(model, "gpt-test-model");
                    seen_sequences.push(sequence);
                }
                CompletionEvent::Delta {
                    run_id,
                    stream_id,
                    sequence,
                    offset,
                    text,
                } => {
                    assert_eq!(run_id, "run-openai-stream");
                    assert_eq!(stream_id, "stream-openai-stream");
                    assert_eq!(offset, deltas.len());
                    deltas.push_str(&text);
                    seen_sequences.push(sequence);
                }
                CompletionEvent::Finished {
                    run_id,
                    stream_id,
                    sequence,
                    response,
                } => {
                    assert_eq!(run_id, "run-openai-stream");
                    assert_eq!(stream_id, "stream-openai-stream");
                    seen_sequences.push(sequence);
                    finished = Some(response);
                }
            }
        }

        assert!(saw_started);
        assert_eq!(seen_sequences, (0..seen_sequences.len() as u64).collect::<Vec<_>>());
        assert_eq!(deltas, "Hi from OpenAI stream");
        let finished = finished.unwrap();
        assert_eq!(finished.text, "Hi from OpenAI stream");
        assert_eq!(finished.run_id.as_deref(), Some("run-openai-stream"));

        let captured = state.take_one();
        assert_eq!(captured.body["stream"], true);
    }

    #[tokio::test]
    async fn openai_config_reads_env() {
        unsafe {
            env::set_var("HARBOR_OPENAI_API_KEY", "secret");
            env::set_var("HARBOR_OPENAI_BASE_URL", "https://openai.example/v1");
            env::set_var("HARBOR_OPENAI_MODEL", "gpt-env");
        }

        let config = OpenAICompatibleConfig::from_env().unwrap();
        assert_eq!(config.api_key, "secret");
        assert_eq!(config.base_url, "https://openai.example/v1");
        assert_eq!(config.default_model, "gpt-env");

        unsafe {
            env::remove_var("HARBOR_OPENAI_API_KEY");
            env::remove_var("HARBOR_OPENAI_BASE_URL");
            env::remove_var("HARBOR_OPENAI_MODEL");
        }
    }

    async fn openai_handler(
        State(state): State<CaptureState>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        state.push(CapturedRequest {
            headers: headers_to_vec(&headers),
            body: body.clone(),
        });

        if body.get("stream").and_then(|value| value.as_bool()) == Some(true) {
            return (
                [(CONTENT_TYPE, "text/event-stream")],
                concat!(
                    "data: {\"model\":\"gpt-test-model\",\"choices\":[{\"delta\":{\"content\":\"Hi from \"}}]}\n\n",
                    "data: {\"model\":\"gpt-test-model\",\"choices\":[{\"delta\":{\"content\":\"OpenAI stream\"}}]}\n\n",
                    "data: [DONE]\n\n"
                ),
            )
                .into_response();
        }

        Json(json!({
            "model": "gpt-test-model",
            "choices": [
                {
                    "message": { "content": "Hi from OpenAI" }
                }
            ]
        }))
        .into_response()
    }

    #[tokio::test]
    async fn anthropic_provider_maps_request_and_response() {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let state = CaptureState::default();
        let app = Router::new()
            .route("/messages", post(anthropic_handler))
            .with_state(state.clone());
        let base_url = spawn_router(app).await;

        let provider = AnthropicProvider::new(AnthropicConfig {
            api_key: "anthropic-test-key".into(),
            base_url,
            default_model: "claude-test-model".into(),
            max_tokens: 256,
        });

        let parent_context = Context::new().with_remote_span_context(SpanContext::new(
            TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::from_hex("00f067aa0ba902b7").unwrap(),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        ));
        let span = tracing::info_span!("anthropic_provider_test");
        let _ = span.set_parent(parent_context);
        let _guard = span.enter();

        let response = provider
            .complete(CompletionRequest {
                model: String::new(),
                system_prompt: Some("Be concise".into()),
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Hello from Harbor".into(),
                }],
                tools: vec![],
                response_schema: None,
                session_id: Some("session-1".into()),
                tool_choice: ToolChoice::Auto,
                run_id: None,
                stream_id: None,
            })
            .await
            .unwrap();

        assert_eq!(response.provider, "anthropic");
        assert_eq!(response.model, "claude-test-model");
        assert_eq!(response.text, "Hi from Anthropic");

        let captured = state.take_one();
        let headers = captured
            .headers
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(headers.get("x-api-key").unwrap(), "anthropic-test-key");
        assert_eq!(headers.get("anthropic-version").unwrap(), ANTHROPIC_VERSION);
        if let Some(traceparent) = headers.get("traceparent") {
            assert!(traceparent.starts_with("00-"));
            assert!(!traceparent.trim().is_empty());
        }

        assert_eq!(captured.body["model"], "claude-test-model");
        assert_eq!(captured.body["system"], "Be concise");
        assert_eq!(captured.body["max_tokens"], 256);
        assert_eq!(captured.body["messages"][0]["role"], "user");
        assert_eq!(captured.body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(
            captured.body["messages"][0]["content"][0]["text"],
            "Hello from Harbor"
        );
    }

    #[tokio::test]
    async fn anthropic_config_reads_env() {
        unsafe {
            env::set_var("HARBOR_ANTHROPIC_API_KEY", "secret");
            env::set_var("HARBOR_ANTHROPIC_BASE_URL", "https://anthropic.example/v1");
            env::set_var("HARBOR_ANTHROPIC_MODEL", "claude-env");
            env::set_var("HARBOR_ANTHROPIC_MAX_TOKENS", "777");
        }

        let config = AnthropicConfig::from_env().unwrap();
        assert_eq!(config.api_key, "secret");
        assert_eq!(config.base_url, "https://anthropic.example/v1");
        assert_eq!(config.default_model, "claude-env");
        assert_eq!(config.max_tokens, 777);

        unsafe {
            env::remove_var("HARBOR_ANTHROPIC_API_KEY");
            env::remove_var("HARBOR_ANTHROPIC_BASE_URL");
            env::remove_var("HARBOR_ANTHROPIC_MODEL");
            env::remove_var("HARBOR_ANTHROPIC_MAX_TOKENS");
        }
    }

    async fn anthropic_handler(
        State(state): State<CaptureState>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        state.push(CapturedRequest {
            headers: headers_to_vec(&headers),
            body,
        });

        Json(json!({
            "model": "claude-test-model",
            "content": [
                { "type": "text", "text": "Hi from Anthropic" }
            ]
        }))
    }

    #[tokio::test]
    async fn ollama_provider_maps_request_and_response() {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let state = CaptureState::default();
        let app = Router::new()
            .route("/api/chat", post(ollama_handler))
            .with_state(state.clone());
        let base_url = spawn_router(app).await;

        let provider = OllamaProvider::new(OllamaConfig {
            base_url,
            default_model: "llama-test-model".into(),
        });

        let parent_context = Context::new().with_remote_span_context(SpanContext::new(
            TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::from_hex("00f067aa0ba902b7").unwrap(),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        ));
        let span = tracing::info_span!("ollama_provider_test");
        let _ = span.set_parent(parent_context);
        let _guard = span.enter();

        let response = provider
            .complete(CompletionRequest {
                model: String::new(),
                system_prompt: Some("Answer plainly".into()),
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Hello from Ollama".into(),
                }],
                tools: vec![],
                response_schema: None,
                session_id: Some("session-2".into()),
                tool_choice: ToolChoice::Auto,
                run_id: None,
                stream_id: None,
            })
            .await
            .unwrap();

        assert_eq!(response.provider, "ollama");
        assert_eq!(response.model, "llama-test-model");
        assert_eq!(response.text, "Hi from Ollama");

        let captured = state.take_one();
        let headers = captured
            .headers
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        if let Some(traceparent) = headers.get("traceparent") {
            assert!(traceparent.starts_with("00-"));
            assert!(!traceparent.trim().is_empty());
        }

        assert_eq!(captured.body["model"], "llama-test-model");
        assert_eq!(captured.body["stream"], false);
        assert_eq!(captured.body["messages"][0]["role"], "system");
        assert_eq!(captured.body["messages"][0]["content"], "Answer plainly");
        assert_eq!(captured.body["messages"][1]["role"], "user");
        assert_eq!(captured.body["messages"][1]["content"], "Hello from Ollama");
    }

    #[tokio::test]
    async fn ollama_config_reads_env() {
        unsafe {
            env::set_var("HARBOR_OLLAMA_BASE_URL", "http://ollama.example:11434");
            env::set_var("HARBOR_OLLAMA_MODEL", "llama-env");
        }

        let config = OllamaConfig::from_env().unwrap();
        assert_eq!(config.base_url, "http://ollama.example:11434");
        assert_eq!(config.default_model, "llama-env");

        unsafe {
            env::remove_var("HARBOR_OLLAMA_BASE_URL");
            env::remove_var("HARBOR_OLLAMA_MODEL");
        }
    }

    async fn ollama_handler(
        State(state): State<CaptureState>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        state.push(CapturedRequest {
            headers: headers_to_vec(&headers),
            body: body.clone(),
        });

        if body.get("stream").and_then(|value| value.as_bool()) == Some(true) {
            return (
                [(CONTENT_TYPE, "application/x-ndjson")],
                concat!(
                    "{\"model\":\"llama-test-model\",\"message\":{\"role\":\"assistant\",\"content\":\"Hi from \"},\"done\":false}\n",
                    "{\"model\":\"llama-test-model\",\"message\":{\"role\":\"assistant\",\"content\":\"Ollama stream\"},\"done\":false}\n",
                    "{\"model\":\"llama-test-model\",\"done\":true}\n"
                ),
            )
                .into_response();
        }

        Json(json!({
            "model": "llama-test-model",
            "message": {
                "role": "assistant",
                "content": "Hi from Ollama"
            },
            "done": true
        }))
        .into_response()
    }

    #[tokio::test]
    async fn ollama_provider_streams_native_events() {
        let state = CaptureState::default();
        let app = Router::new()
            .route("/api/chat", post(ollama_handler))
            .with_state(state.clone());
        let base_url = spawn_router(app).await;

        let provider = OllamaProvider::new(OllamaConfig {
            base_url,
            default_model: "llama-test-model".into(),
        });

        let mut stream = provider
            .complete_stream(CompletionRequest {
                model: String::new(),
                system_prompt: None,
                messages: vec![Message {
                    role: MessageRole::User,
                    content: "Stream from Ollama".into(),
                }],
                tools: vec![],
                response_schema: None,
                session_id: Some("session-ollama-stream".into()),
                tool_choice: ToolChoice::Auto,
                run_id: Some("run-ollama-stream".into()),
                stream_id: Some("stream-ollama-stream".into()),
            })
            .await
            .unwrap();

        let mut saw_started = false;
        let mut deltas = String::new();
        let mut finished = None;
        let mut seen_sequences = Vec::new();

        while let Some(event) = stream.recv().await {
            match event.unwrap() {
                CompletionEvent::Started {
                    run_id,
                    stream_id,
                    sequence,
                    provider,
                    model,
                } => {
                    saw_started = true;
                    assert_eq!(run_id, "run-ollama-stream");
                    assert_eq!(stream_id, "stream-ollama-stream");
                    assert_eq!(provider, "ollama");
                    assert_eq!(model, "llama-test-model");
                    seen_sequences.push(sequence);
                }
                CompletionEvent::Delta {
                    run_id,
                    stream_id,
                    sequence,
                    offset,
                    text,
                } => {
                    assert_eq!(run_id, "run-ollama-stream");
                    assert_eq!(stream_id, "stream-ollama-stream");
                    assert_eq!(offset, deltas.len());
                    deltas.push_str(&text);
                    seen_sequences.push(sequence);
                }
                CompletionEvent::Finished {
                    run_id,
                    stream_id,
                    sequence,
                    response,
                } => {
                    assert_eq!(run_id, "run-ollama-stream");
                    assert_eq!(stream_id, "stream-ollama-stream");
                    seen_sequences.push(sequence);
                    finished = Some(response);
                }
            }
        }

        assert!(saw_started);
        assert_eq!(seen_sequences, (0..seen_sequences.len() as u64).collect::<Vec<_>>());
        assert_eq!(deltas, "Hi from Ollama stream");
        let finished = finished.unwrap();
        assert_eq!(finished.text, "Hi from Ollama stream");
        assert_eq!(finished.run_id.as_deref(), Some("run-ollama-stream"));

        let captured = state.take_one();
        assert_eq!(captured.body["stream"], true);
    }

    async fn spawn_router(router: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{}", address)
    }

    fn headers_to_vec(headers: &HeaderMap) -> Vec<(String, String)> {
        headers
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.to_string(), value.to_string()))
            })
            .collect()
    }
}
