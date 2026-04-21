use harbor_core::{FrameworkError, FrameworkResult};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde::{Deserialize, Serialize};
use std::{sync::{Arc, OnceLock},};
use tracing_subscriber::{fmt, EnvFilter};

static TRACING_INITIALIZED: OnceLock<()> = OnceLock::new();
static METRICS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarborObservabilityConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub log_level: String,
    pub json_logs: bool,
    pub metrics_enabled: bool,
}

impl Default for HarborObservabilityConfig {
    fn default() -> Self {
        Self {
            service_name: "harbor-app".into(),
            service_version: "0.1.0".into(),
            environment: "dev".into(),
            log_level: "info".into(),
            json_logs: false,
            metrics_enabled: true,
        }
    }
}

#[derive(Clone)]
pub struct HarborObservability {
    metrics: Option<PrometheusHandle>,
}

impl HarborObservability {
    pub fn init(config: &HarborObservabilityConfig) -> FrameworkResult<Self> {
        setup_tracing(config)?;

        let metrics = if config.metrics_enabled {
            Some(metrics_handle()?)
        } else {
            None
        };

        Ok(Self { metrics })
    }

    pub fn metrics_renderer(&self) -> Option<MetricsRenderer> {
        self.metrics.clone().map(|handle| {
            let handle = Arc::new(handle);
            Arc::new(move || handle.render()) as MetricsRenderer
        })
    }
}

pub type MetricsRenderer = Arc<dyn Fn() -> String + Send + Sync>;

pub fn setup_tracing(config: &HarborObservabilityConfig) -> FrameworkResult<()> {
    if TRACING_INITIALIZED.get().is_some() {
        return Ok(());
    }

    let env_filter = EnvFilter::try_new(config.log_level.clone())
        .unwrap_or_else(|_| EnvFilter::new("info"));

    if config.json_logs {
        fmt()
            .json()
            .with_env_filter(env_filter)
            .with_target(false)
            .try_init()
            .map_err(|error| FrameworkError::Config(error.to_string()))?;
    } else {
        fmt()
            .compact()
            .with_env_filter(env_filter)
            .with_target(false)
            .try_init()
            .map_err(|error| FrameworkError::Config(error.to_string()))?;
    }

    let _ = TRACING_INITIALIZED.set(());
    Ok(())
}

fn metrics_handle() -> FrameworkResult<PrometheusHandle> {
    if let Some(handle) = METRICS_HANDLE.get() {
        return Ok(handle.clone());
    }

    let handle = PrometheusBuilder::new()
        .install_recorder()
        .map_err(|error| FrameworkError::Config(error.to_string()))?;

    let _ = METRICS_HANDLE.set(handle.clone());
    Ok(handle)
}
