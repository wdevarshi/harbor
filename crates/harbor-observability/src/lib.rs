use harbor_core::{FrameworkError, FrameworkResult};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::{propagation::TraceContextPropagator, trace::SdkTracerProvider, Resource};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

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
    pub otel_enabled: bool,
    pub otel_endpoint: String,
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
            otel_enabled: false,
            otel_endpoint: "http://127.0.0.1:4317".into(),
        }
    }
}

#[derive(Clone)]
pub struct HarborObservability {
    metrics: Option<PrometheusHandle>,
    tracer_provider: Option<SdkTracerProvider>,
}

impl HarborObservability {
    pub fn init(config: &HarborObservabilityConfig) -> FrameworkResult<Self> {
        let tracer_provider = setup_tracing(config)?;

        let metrics = if config.metrics_enabled {
            Some(metrics_handle()?)
        } else {
            None
        };

        Ok(Self {
            metrics,
            tracer_provider,
        })
    }

    pub fn metrics_renderer(&self) -> Option<MetricsRenderer> {
        self.metrics.clone().map(|handle| {
            let handle = Arc::new(handle);
            Arc::new(move || handle.render()) as MetricsRenderer
        })
    }

    pub fn shutdown(&self) -> FrameworkResult<()> {
        if let Some(provider) = &self.tracer_provider {
            provider
                .shutdown()
                .map_err(|error| FrameworkError::Transport(error.to_string()))?;
        }

        Ok(())
    }
}

pub type MetricsRenderer = Arc<dyn Fn() -> String + Send + Sync>;

pub fn setup_tracing(
    config: &HarborObservabilityConfig,
) -> FrameworkResult<Option<SdkTracerProvider>> {
    if TRACING_INITIALIZED.get().is_some() {
        return Ok(None);
    }

    let env_filter =
        EnvFilter::try_new(config.log_level.clone()).unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = if config.json_logs {
        fmt::layer().json().with_target(false).boxed()
    } else {
        fmt::layer().compact().with_target(false).boxed()
    };

    let tracer_provider = if config.otel_enabled {
        Some(build_otel_provider(config)?)
    } else {
        None
    };

    let subscriber = tracing_subscriber::registry().with(env_filter).with(fmt_layer);

    if let Some(provider) = tracer_provider.as_ref() {
        let tracer = provider.tracer(config.service_name.clone());
        subscriber
            .with(tracing_opentelemetry::layer().with_tracer(tracer).boxed())
            .try_init()
            .map_err(|error| FrameworkError::Config(error.to_string()))?;
    } else {
        subscriber
            .try_init()
            .map_err(|error| FrameworkError::Config(error.to_string()))?;
    }

    let _ = TRACING_INITIALIZED.set(());
    Ok(tracer_provider)
}

fn build_otel_provider(config: &HarborObservabilityConfig) -> FrameworkResult<SdkTracerProvider> {
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(config.otel_endpoint.clone())
        .build()
        .map_err(|error| FrameworkError::Config(error.to_string()))?;

    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .with_attributes(vec![
            KeyValue::new("service.version", config.service_version.clone()),
            KeyValue::new("deployment.environment.name", config.environment.clone()),
        ])
        .build();

    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(exporter)
        .build();

    global::set_text_map_propagator(TraceContextPropagator::new());
    let _ = global::set_tracer_provider(provider.clone());

    Ok(provider)
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
