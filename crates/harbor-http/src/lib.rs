use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use harbor_core::{FrameworkError, FrameworkResult};
use serde::{Deserialize, Serialize};
use std::{
    env,
    future::Future,
    net::{IpAddr, SocketAddr},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::net::TcpListener;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarborHttpConfig {
    pub host: String,
    pub port: u16,
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
}

impl Default for HarborHttpConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 3000,
            service_name: "harbor-app".into(),
            service_version: "0.1.0".into(),
            environment: "dev".into(),
        }
    }
}

impl HarborHttpConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let mut config = Self::default();

        if let Ok(host) = env::var("HARBOR_HTTP_HOST") {
            config.host = host;
        }

        if let Ok(port) = env::var("HARBOR_HTTP_PORT") {
            config.port = port.parse().map_err(|error| {
                FrameworkError::Config(format!("invalid HARBOR_HTTP_PORT value: {error}"))
            })?;
        }

        if let Ok(service_name) = env::var("HARBOR_SERVICE_NAME") {
            config.service_name = service_name;
        }

        if let Ok(service_version) = env::var("HARBOR_SERVICE_VERSION") {
            config.service_version = service_version;
        }

        if let Ok(environment) = env::var("HARBOR_ENV") {
            config.environment = environment;
        }

        Ok(config)
    }

    pub fn socket_addr(&self) -> FrameworkResult<SocketAddr> {
        let ip: IpAddr = self
            .host
            .parse()
            .map_err(|error| FrameworkError::Config(format!("invalid HTTP host '{}': {error}", self.host)))?;
        Ok(SocketAddr::new(ip, self.port))
    }
}

#[derive(Debug, Clone)]
pub struct ReadinessGate {
    ready: Arc<AtomicBool>,
}

impl ReadinessGate {
    pub fn ready() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn not_ready() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::SeqCst);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }
}

impl Default for ReadinessGate {
    fn default() -> Self {
        Self::ready()
    }
}

#[derive(Clone)]
struct HarborHttpState {
    config: HarborHttpConfig,
    readiness: ReadinessGate,
}

pub fn router(config: HarborHttpConfig, readiness: ReadinessGate) -> Router {
    Router::new()
        .route("/healthcheck", get(healthcheck))
        .route("/readycheck", get(readycheck))
        .with_state(HarborHttpState { config, readiness })
}

#[derive(Debug, Clone)]
pub struct HarborHttpServer {
    config: HarborHttpConfig,
    readiness: ReadinessGate,
}

impl HarborHttpServer {
    pub fn new(config: HarborHttpConfig) -> Self {
        Self {
            config,
            readiness: ReadinessGate::default(),
        }
    }

    pub fn with_readiness(mut self, readiness: ReadinessGate) -> Self {
        self.readiness = readiness;
        self
    }

    pub fn readiness(&self) -> ReadinessGate {
        self.readiness.clone()
    }

    pub fn config(&self) -> &HarborHttpConfig {
        &self.config
    }

    pub async fn run(self) -> FrameworkResult<()> {
        self.run_with_shutdown(async {}).await
    }

    pub async fn run_with_shutdown<F>(self, shutdown_signal: F) -> FrameworkResult<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let listener = TcpListener::bind(self.config.socket_addr()?)
            .await
            .map_err(|error| FrameworkError::Transport(error.to_string()))?;

        axum::serve(listener, router(self.config, self.readiness))
            .with_graceful_shutdown(shutdown_signal)
            .await
            .map_err(|error| FrameworkError::Transport(error.to_string()))
    }
}

async fn healthcheck(State(state): State<HarborHttpState>) -> Json<ServiceStatusPayload> {
    Json(ServiceStatusPayload::from_config(&state.config, "ok", None))
}

async fn readycheck(State(state): State<HarborHttpState>) -> impl IntoResponse {
    let ready = state.readiness.is_ready();
    let status_code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let status = if ready { "ready" } else { "not_ready" };

    (
        status_code,
        Json(ServiceStatusPayload::from_config(
            &state.config,
            status,
            Some(ready),
        )),
    )
}

#[derive(Debug, Serialize)]
struct ServiceStatusPayload {
    status: String,
    service: String,
    version: String,
    environment: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ready: Option<bool>,
}

impl ServiceStatusPayload {
    fn from_config(config: &HarborHttpConfig, status: impl Into<String>, ready: Option<bool>) -> Self {
        Self {
            status: status.into(),
            service: config.service_name.clone(),
            version: config.service_version.clone(),
            environment: config.environment.clone(),
            ready,
        }
    }
}
