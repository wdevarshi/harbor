use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use harbor_core::{FrameworkError, FrameworkResult};
use metrics::{counter, gauge, histogram};
use opentelemetry::{global, propagation::Extractor};
use serde::{Deserialize, Serialize};
use std::{
    env,
    future::Future,
    net::{IpAddr, SocketAddr},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
use tokio::net::TcpListener;
use tracing::{field, info, info_span, Instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use uuid::Uuid;

pub const REQUEST_ID_HEADER: &str = "x-request-id";
pub type MetricsRenderer = Arc<dyn Fn() -> String + Send + Sync>;

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
        let ip: IpAddr = self.host.parse().map_err(|error| {
            FrameworkError::Config(format!("invalid HTTP host '{}': {error}", self.host))
        })?;
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

#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
}

#[derive(Clone)]
struct HarborHttpState {
    config: HarborHttpConfig,
    readiness: ReadinessGate,
    metrics_renderer: Option<MetricsRenderer>,
}

pub fn router(config: HarborHttpConfig, readiness: ReadinessGate) -> Router {
    router_with_metrics(config, readiness, None)
}

pub fn router_with_metrics(
    config: HarborHttpConfig,
    readiness: ReadinessGate,
    metrics_renderer: Option<MetricsRenderer>,
) -> Router {
    let state = HarborHttpState {
        config,
        readiness,
        metrics_renderer,
    };

    Router::new()
        .route("/healthcheck", get(healthcheck))
        .route("/readycheck", get(readycheck))
        .route("/metrics", get(metrics_endpoint))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            request_context_middleware,
        ))
        .with_state(state)
}

#[derive(Clone)]
pub struct HarborHttpServer {
    config: HarborHttpConfig,
    readiness: ReadinessGate,
    metrics_renderer: Option<MetricsRenderer>,
}

impl HarborHttpServer {
    pub fn new(config: HarborHttpConfig) -> Self {
        Self {
            config,
            readiness: ReadinessGate::default(),
            metrics_renderer: None,
        }
    }

    pub fn with_readiness(mut self, readiness: ReadinessGate) -> Self {
        self.readiness = readiness;
        self
    }

    pub fn with_metrics_renderer(mut self, metrics_renderer: MetricsRenderer) -> Self {
        self.metrics_renderer = Some(metrics_renderer);
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

        axum::serve(
            listener,
            router_with_metrics(self.config, self.readiness, self.metrics_renderer),
        )
        .with_graceful_shutdown(shutdown_signal)
        .await
        .map_err(|error| FrameworkError::Transport(error.to_string()))
    }
}

async fn request_context_middleware(
    State(state): State<HarborHttpState>,
    mut request: Request,
    next: Next,
) -> Response {
    let request_id = request_id_from_headers(request.headers());
    request.extensions_mut().insert(RequestContext {
        request_id: request_id.clone(),
    });

    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let started_at = Instant::now();

    let span = info_span!(
        "http.request",
        service = %state.config.service_name.as_str(),
        environment = %state.config.environment.as_str(),
        request_id = %request_id,
        method = %method,
        path = %path,
        status = field::Empty,
        duration_ms = field::Empty,
        "otel.kind" = "server",
        "http.request.method" = %method,
        "url.path" = %path,
        "service.name" = %state.config.service_name.as_str(),
        "service.version" = %state.config.service_version.as_str(),
        "deployment.environment.name" = %state.config.environment.as_str(),
        "http.response.status_code" = field::Empty,
    );

    let parent_context =
        global::get_text_map_propagator(|propagator| propagator.extract(&HeaderExtractor(request.headers())));
    let _ = span.set_parent(parent_context);

    let mut response = next.run(request).instrument(span.clone()).await;
    let duration_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let status = response.status().as_u16().to_string();

    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    counter!(
        "harbor_http_requests_total",
        "route" => path.clone(),
        "method" => method.clone(),
        "status" => status.clone()
    )
    .increment(1);

    histogram!(
        "harbor_http_request_duration_ms",
        "route" => path.clone(),
        "method" => method.clone(),
        "status" => status.clone()
    )
    .record(duration_ms);

    span.record("status", field::display(status.as_str()));
    span.record(
        "http.response.status_code",
        field::display(status.as_str()),
    );
    span.record("duration_ms", duration_ms);

    span.in_scope(|| {
        info!(
            status = %status,
            duration_ms = duration_ms,
            "harbor http request"
        );
    });

    response
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

    gauge!("harbor_ready_state").set(if ready { 1.0 } else { 0.0 });

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

async fn metrics_endpoint(State(state): State<HarborHttpState>) -> Response {
    match &state.metrics_renderer {
        Some(render) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
            render(),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            "metrics disabled\n",
        )
            .into_response(),
    }
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
    fn from_config(
        config: &HarborHttpConfig,
        status: impl Into<String>,
        ready: Option<bool>,
    ) -> Self {
        Self {
            status: status.into(),
            service: config.service_name.clone(),
            version: config.service_version.clone(),
            environment: config.environment.clone(),
            ready,
        }
    }
}

fn request_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

struct HeaderExtractor<'a>(&'a HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .map(|name| name.as_str())
            .collect::<Vec<_>>()
    }
}
