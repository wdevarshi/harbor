# Harbor Roadmap

## Phase 1 - Foundation (implemented)
- workspace split into reusable crates
- provider abstraction
- mock provider
- OpenAI-compatible provider client
- Anthropic provider client
- Ollama provider client
- provider retry / timeout / fallback helpers
- provider/runtime event streaming hooks
- native OpenAI-compatible and Ollama streaming adapters
- typed schema helpers for tool definitions
- structured output helpers
- tool registry
- session memory abstraction
- file-backed session memory persistence
- document store + lexical retrieval + prompt injection helpers
- agent runtime
- lifecycle task primitives with checkpoints
- workflow engine
- shared `HarborApp` runtime/bootstrap entrypoint
- signal-driven graceful shutdown wiring for the shared app runtime
- env-first app/bootstrap config
- MCP server builder
- MCP local integration client
- spawned MCP stdio client transport
- MCP HTTP transport
- MCP resource and prompt endpoints
- MCP capability reporting
- HTTP ops surface with `/healthcheck`, `/readycheck`, and `/metrics`
- request ID propagation via `x-request-id`
- incoming trace-context extraction / parent span wiring
- middleware-based request logging
- timeout/auth/rate-limit middleware for app routes
- tracing bootstrap
- structured logging bootstrap
- Prometheus recorder bootstrap
- OTEL exporter bootstrap
- CLI scaffolding

## Phase 2 - Runtime and ops
- richer lifecycle hooks for background workers and subsystems
- OTEL log/metric exporter options
- broader behavioral test coverage across crates

## Phase 3 - Richer AI integrations
- native Anthropic streaming adapter

## Phase 4 - Richer MCP support
- richer auth/retry configuration for HTTP transport

## Phase 5 - Developer ergonomics
- derive macro for tools
- schema generation from Rust types
- test harness for agent workflows
- deployment templates (Docker / K8s)

## Phase 6 - Data + memory
- vector store integrations
- Redis / Postgres memory backends
- vector retrieval backends
- richer document ingestion adapters
