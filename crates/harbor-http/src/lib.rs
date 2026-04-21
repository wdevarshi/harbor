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
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::{net::TcpListener, sync::Semaphore, time::timeout};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarborHttpMiddlewareConfig {
    pub request_timeout_ms: Option<u64>,
    pub concurrency_limit: Option<usize>,
    pub rate_limit_requests: Option<u64>,
    pub rate_limit_window_secs: u64,
    pub bearer_token: Option<String>,
}

impl Default for HarborHttpMiddlewareConfig {
    fn default() -> Self {
        Self {
            request_timeout_ms: None,
            concurrency_limit: None,
            rate_limit_requests: None,
            rate_limit_window_secs: 1,
            bearer_token: None,
        }
    }
}

impl HarborHttpMiddlewareConfig {
    pub fn from_env() -> FrameworkResult<Self> {
        let mut config = Self::default();

        if let Ok(timeout_ms) = env::var("HARBOR_HTTP_TIMEOUT_MS") {
            config.request_timeout_ms = Some(timeout_ms.parse().map_err(|error| {
                FrameworkError::Config(format!("invalid HARBOR_HTTP_TIMEOUT_MS value: {error}"))
            })?);
        }

        if let Ok(concurrency_limit) = env::var("HARBOR_HTTP_CONCURRENCY_LIMIT") {
            config.concurrency_limit = Some(concurrency_limit.parse().map_err(|error| {
                FrameworkError::Config(format!(
                    "invalid HARBOR_HTTP_CONCURRENCY_LIMIT value: {error}"
                ))
            })?);
        }

        if let Ok(rate_limit_requests) = env::var("HARBOR_HTTP_RATE_LIMIT_REQUESTS") {
            config.rate_limit_requests = Some(rate_limit_requests.parse().map_err(|error| {
                FrameworkError::Config(format!(
                    "invalid HARBOR_HTTP_RATE_LIMIT_REQUESTS value: {error}"
                ))
            })?);
        }

        if let Ok(rate_limit_window_secs) = env::var("HARBOR_HTTP_RATE_LIMIT_WINDOW_SECS") {
            config.rate_limit_window_secs = rate_limit_window_secs.parse().map_err(|error| {
                FrameworkError::Config(format!(
                    "invalid HARBOR_HTTP_RATE_LIMIT_WINDOW_SECS value: {error}"
                ))
            })?;
        }

        if let Ok(bearer_token) = env::var("HARBOR_HTTP_BEARER_TOKEN") {
            if !bearer_token.trim().is_empty() {
                config.bearer_token = Some(bearer_token);
            }
        }

        Ok(config)
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
    middleware: Arc<MiddlewareRuntime>,
}

#[derive(Debug)]
struct MiddlewareRuntime {
    config: HarborHttpMiddlewareConfig,
    concurrency: Option<Arc<Semaphore>>,
    rate_limiter: Option<Arc<FixedWindowRateLimiter>>,
}

impl MiddlewareRuntime {
    fn new(config: HarborHttpMiddlewareConfig) -> Self {
        let concurrency = config
            .concurrency_limit
            .map(|limit| Arc::new(Semaphore::new(limit.max(1))));
        let rate_limiter = config.rate_limit_requests.map(|limit| {
            Arc::new(FixedWindowRateLimiter::new(
                limit.max(1),
                Duration::from_secs(config.rate_limit_window_secs.max(1)),
            ))
        });

        Self {
            config,
            concurrency,
            rate_limiter,
        }
    }
}

#[derive(Debug)]
struct FixedWindowRateLimiter {
    max_requests: u64,
    window: Duration,
    state: Mutex<RateLimitState>,
}

#[derive(Debug)]
struct RateLimitState {
    started_at: Instant,
    count: u64,
}

impl FixedWindowRateLimiter {
    fn new(max_requests: u64, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            state: Mutex::new(RateLimitState {
                started_at: Instant::now(),
                count: 0,
            }),
        }
    }

    fn allow(&self) -> bool {
        let mut guard = self.state.lock().unwrap();
        if guard.started_at.elapsed() >= self.window {
            guard.started_at = Instant::now();
            guard.count = 0;
        }

        if guard.count < self.max_requests {
            guard.count += 1;
            true
        } else {
            false
        }
    }
}

pub fn router(config: HarborHttpConfig, readiness: ReadinessGate) -> Router {
    router_with_stack(
        config,
        readiness,
        None,
        HarborHttpMiddlewareConfig::default(),
        Router::new(),
    )
}

pub fn router_with_metrics(
    config: HarborHttpConfig,
    readiness: ReadinessGate,
    metrics_renderer: Option<MetricsRenderer>,
) -> Router {
    router_with_stack(
        config,
        readiness,
        metrics_renderer,
        HarborHttpMiddlewareConfig::default(),
        Router::new(),
    )
}

pub fn router_with_stack(
    config: HarborHttpConfig,
    readiness: ReadinessGate,
    metrics_renderer: Option<MetricsRenderer>,
    middleware_config: HarborHttpMiddlewareConfig,
    app_router: Router,
) -> Router {
    let state = HarborHttpState {
        config,
        readiness,
        metrics_renderer,
        middleware: Arc::new(MiddlewareRuntime::new(middleware_config)),
    };

    let protected_router = app_router.layer(middleware::from_fn_with_state(
        state.clone(),
        app_policy_middleware,
    ));

    let health_state = state.clone();
    let ready_state = state.clone();
    let metrics_state = state.clone();

    Router::new()
        .merge(protected_router)
        .route(
            "/healthcheck",
            get(move || healthcheck(health_state.clone())),
        )
        .route(
            "/readycheck",
            get(move || readycheck(ready_state.clone())),
        )
        .route(
            "/metrics",
            get(move || metrics_endpoint(metrics_state.clone())),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            request_context_middleware,
        ))
}

#[derive(Clone)]
pub struct HarborHttpServer {
    config: HarborHttpConfig,
    readiness: ReadinessGate,
    metrics_renderer: Option<MetricsRenderer>,
    middleware_config: HarborHttpMiddlewareConfig,
    app_router: Router,
}

impl HarborHttpServer {
    pub fn new(config: HarborHttpConfig) -> Self {
        Self {
            config,
            readiness: ReadinessGate::default(),
            metrics_renderer: None,
            middleware_config: HarborHttpMiddlewareConfig::default(),
            app_router: Router::new(),
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

    pub fn with_middleware_config(mut self, middleware_config: HarborHttpMiddlewareConfig) -> Self {
        self.middleware_config = middleware_config;
        self
    }

    pub fn with_app_router(mut self, app_router: Router) -> Self {
        self.app_router = app_router;
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
            router_with_stack(
                self.config,
                self.readiness,
                self.metrics_renderer,
                self.middleware_config,
                self.app_router,
            ),
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

    let parent_context = global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(request.headers()))
    });
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
    span.record("http.response.status_code", field::display(status.as_str()));
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

async fn app_policy_middleware(
    State(state): State<HarborHttpState>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(expected) = state.middleware.config.bearer_token.as_ref() {
        match bearer_token(request.headers()) {
            Some(token) if token == expected => {}
            _ => {
                return (
                    StatusCode::UNAUTHORIZED,
                    [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                    "missing or invalid bearer token\n",
                )
                    .into_response();
            }
        }
    }

    if let Some(rate_limiter) = &state.middleware.rate_limiter {
        if !rate_limiter.allow() {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                "rate limit exceeded\n",
            )
                .into_response();
        }
    }

    let _permit = match &state.middleware.concurrency {
        Some(semaphore) => match semaphore.clone().try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                    "concurrency limit exceeded\n",
                )
                    .into_response();
            }
        },
        None => None,
    };

    let future = next.run(request);
    match state.middleware.config.request_timeout_ms {
        Some(timeout_ms) => match timeout(Duration::from_millis(timeout_ms.max(1)), future).await {
            Ok(response) => response,
            Err(_) => (
                StatusCode::REQUEST_TIMEOUT,
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                "request timed out\n",
            )
                .into_response(),
        },
        None => future.await,
    }
}

async fn healthcheck(state: HarborHttpState) -> Json<ServiceStatusPayload> {
    Json(ServiceStatusPayload::from_config(&state.config, "ok", None))
}

async fn readycheck(state: HarborHttpState) -> impl IntoResponse {
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

async fn metrics_endpoint(state: HarborHttpState) -> Response {
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

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

struct HeaderExtractor<'a>(&'a HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|name| name.as_str()).collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode as HttpStatusCode;
    use tokio::{net::TcpListener, sync::oneshot, time::sleep};

    #[tokio::test]
    async fn readycheck_returns_503_and_echoes_request_id() {
        let config = HarborHttpConfig {
            host: "127.0.0.1".into(),
            port: 0,
            service_name: "harbor-http-test".into(),
            service_version: "0.1.0".into(),
            environment: "test".into(),
        };
        let readiness = ReadinessGate::not_ready();
        let base_url = spawn_router(router(config, readiness)).await;

        let response = reqwest::Client::new()
            .get(format!("{base_url}/readycheck"))
            .header(REQUEST_ID_HEADER, "req-http-test-1")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), HttpStatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers().get(REQUEST_ID_HEADER).unwrap(),
            "req-http-test-1"
        );
        let body = response.text().await.unwrap();
        assert!(body.contains("not_ready"));
        assert!(body.contains("harbor-http-test"));
    }

    #[tokio::test]
    async fn metrics_returns_404_when_disabled() {
        let config = HarborHttpConfig {
            host: "127.0.0.1".into(),
            port: 0,
            service_name: "harbor-http-test".into(),
            service_version: "0.1.0".into(),
            environment: "test".into(),
        };
        let base_url = spawn_router(router(config, ReadinessGate::ready())).await;

        let response = reqwest::get(format!("{base_url}/metrics")).await.unwrap();

        assert_eq!(response.status(), HttpStatusCode::NOT_FOUND);
        assert_eq!(response.text().await.unwrap(), "metrics disabled\n");
    }

    #[tokio::test]
    async fn app_router_requires_bearer_token() {
        let base_url = spawn_router(router_with_stack(
            test_config(),
            ReadinessGate::ready(),
            None,
            HarborHttpMiddlewareConfig {
                bearer_token: Some("secret-token".into()),
                ..HarborHttpMiddlewareConfig::default()
            },
            Router::new().route("/protected", get(|| async { "ok" })),
        ))
        .await;

        let unauthorized = reqwest::get(format!("{base_url}/protected")).await.unwrap();
        assert_eq!(unauthorized.status(), HttpStatusCode::UNAUTHORIZED);

        let authorized = reqwest::Client::new()
            .get(format!("{base_url}/protected"))
            .header(header::AUTHORIZATION, "Bearer secret-token")
            .send()
            .await
            .unwrap();
        assert_eq!(authorized.status(), HttpStatusCode::OK);
        assert_eq!(authorized.text().await.unwrap(), "ok");

        let health = reqwest::get(format!("{base_url}/healthcheck")).await.unwrap();
        assert_eq!(health.status(), HttpStatusCode::OK);
    }

    #[tokio::test]
    async fn app_router_times_out() {
        let base_url = spawn_router(router_with_stack(
            test_config(),
            ReadinessGate::ready(),
            None,
            HarborHttpMiddlewareConfig {
                request_timeout_ms: Some(25),
                ..HarborHttpMiddlewareConfig::default()
            },
            Router::new().route(
                "/slow",
                get(|| async {
                    sleep(Duration::from_millis(75)).await;
                    "slow"
                }),
            ),
        ))
        .await;

        let response = reqwest::get(format!("{base_url}/slow")).await.unwrap();
        assert_eq!(response.status(), HttpStatusCode::REQUEST_TIMEOUT);
        assert_eq!(response.text().await.unwrap(), "request timed out\n");
    }

    #[tokio::test]
    async fn app_router_rate_limits_requests() {
        let base_url = spawn_router(router_with_stack(
            test_config(),
            ReadinessGate::ready(),
            None,
            HarborHttpMiddlewareConfig {
                rate_limit_requests: Some(1),
                rate_limit_window_secs: 60,
                ..HarborHttpMiddlewareConfig::default()
            },
            Router::new().route("/limited", get(|| async { "ok" })),
        ))
        .await;

        let first = reqwest::get(format!("{base_url}/limited")).await.unwrap();
        assert_eq!(first.status(), HttpStatusCode::OK);

        let second = reqwest::get(format!("{base_url}/limited")).await.unwrap();
        assert_eq!(second.status(), HttpStatusCode::TOO_MANY_REQUESTS);
        assert_eq!(second.text().await.unwrap(), "rate limit exceeded\n");
    }

    #[tokio::test]
    async fn app_router_enforces_concurrency_limit() {
        let (tx, rx) = oneshot::channel::<()>();
        let rx = Arc::new(Mutex::new(Some(rx)));
        let base_url = spawn_router(router_with_stack(
            test_config(),
            ReadinessGate::ready(),
            None,
            HarborHttpMiddlewareConfig {
                concurrency_limit: Some(1),
                ..HarborHttpMiddlewareConfig::default()
            },
            Router::new().route(
                "/block",
                get(move || {
                    let rx = rx.clone();
                    async move {
                        let receiver = rx.lock().unwrap().take();
                        if let Some(receiver) = receiver {
                            let _ = receiver.await;
                        }
                        "done"
                    }
                }),
            ),
        ))
        .await;

        let client = reqwest::Client::new();
        let first = tokio::spawn({
            let client = client.clone();
            let url = format!("{base_url}/block");
            async move { client.get(url).send().await.unwrap() }
        });

        sleep(Duration::from_millis(25)).await;

        let second = client.get(format!("{base_url}/block")).send().await.unwrap();
        assert_eq!(second.status(), HttpStatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(second.text().await.unwrap(), "concurrency limit exceeded\n");

        let _ = tx.send(());
        assert_eq!(first.await.unwrap().status(), HttpStatusCode::OK);
    }

    async fn spawn_router(router: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{}", address)
    }

    fn test_config() -> HarborHttpConfig {
        HarborHttpConfig {
            host: "127.0.0.1".into(),
            port: 0,
            service_name: "harbor-http-test".into(),
            service_version: "0.1.0".into(),
            environment: "test".into(),
        }
    }
}
