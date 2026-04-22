use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

pub type JsonValue = Value;
pub type FrameworkResult<T> = Result<T, FrameworkError>;

#[derive(Debug, Error)]
pub enum FrameworkError {
    #[error("tool not found: {0}")]
    ToolNotFound(String),
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("memory error: {0}")]
    Memory(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: JsonValue,
}

impl ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>, input_schema: JsonValue) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, args: JsonValue) -> FrameworkResult<JsonValue>;
}

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.register_arc(Arc::new(tool));
    }

    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.spec().name.clone(), tool);
    }

    pub fn list(&self) -> Vec<ToolSpec> {
        let mut specs: Vec<_> = self.tools.values().map(|tool| tool.spec()).collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub async fn call(&self, name: &str, args: JsonValue) -> FrameworkResult<JsonValue> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| FrameworkError::ToolNotFound(name.to_string()))?;
        tool.call(args).await
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub tags: Vec<String>,
}

#[derive(Clone)]
pub struct AppBlueprint {
    pub metadata: AppMetadata,
    pub tools: ToolRegistry,
    pub default_config: JsonValue,
}

impl AppBlueprint {
    pub fn builder(name: impl Into<String>) -> AppBlueprintBuilder {
        AppBlueprintBuilder {
            metadata: AppMetadata {
                name: name.into(),
                version: "0.1.0".into(),
                description: "AI-first Rust solution".into(),
                tags: vec!["ai".into(), "mcp".into()],
            },
            tools: ToolRegistry::new(),
            default_config: json!({}),
        }
    }
}

pub struct AppBlueprintBuilder {
    metadata: AppMetadata,
    tools: ToolRegistry,
    default_config: JsonValue,
}

impl AppBlueprintBuilder {
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.metadata.version = version.into();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.metadata.description = description.into();
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.metadata.tags.push(tag.into());
        self
    }

    pub fn config(mut self, config: JsonValue) -> Self {
        self.default_config = config;
        self
    }

    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.register(tool);
        self
    }

    pub fn build(self) -> AppBlueprint {
        AppBlueprint {
            metadata: self.metadata,
            tools: self.tools,
            default_config: self.default_config,
        }
    }
}

pub mod schema;
pub use schema::*;
