# Harbor

Harbor is an **AI-first Rust framework** for building reusable AI solutions with **first-class MCP support**.

It is inspired by modern AI application platforms and documentation patterns like the ones in Coldbrew Cloud: a single developer surface for
- model providers
- tools and integrations
- memory/session state
- workflows and agents
- MCP server creation
- MCP client integration
- CLI scaffolding for new AI projects

## What this v0.1 gives you

- **`harbor-core`**
  - tool traits
  - tool registry
  - application blueprint/builder
  - shared framework errors
- **`harbor-ai`**
  - provider abstraction
  - chat/message types
  - structured completion request/response model
  - provider event streaming hooks
  - mock provider for local development and tests
  - OpenAI-compatible provider client
  - Anthropic provider client
  - Ollama provider client
  - provider retry / timeout / fallback helpers
  - structured output helpers
  - outbound trace-context injection on provider HTTP calls
- **`harbor-memory`**
  - session memory trait
  - in-memory implementation
  - file-backed persistent session memory
- **`harbor-rag`**
  - document store abstraction
  - in-memory + file-backed document stores
  - document chunking helpers
  - lexical retrieval
  - prompt injection helpers for retrieved context
- **`harbor-runtime`**
  - agent runtime
  - streaming turn API
  - retrieval-aware turn execution
  - lifecycle task primitives
  - in-memory + file-backed task stores
  - workflow engine
  - reusable execution context
  - shared `HarborApp` bootstrap entrypoint
  - signal-driven shutdown wiring
  - app-level observability integration
- **`harbor-mcp`**
  - JSON-RPC + MCP-inspired protocol types
  - stdio framing (`Content-Length`)
  - MCP server builder
  - spawned stdio client transport
  - HTTP transport for remote MCP integration
  - local + HTTP integration clients
  - resource + prompt endpoints
  - capability reporting
  - outbound trace-context injection for MCP HTTP calls
- **`harbor-http`**
  - Axum-based HTTP ops surface
  - `/healthcheck`, `/readycheck`, and `/metrics`
  - request ID propagation via `x-request-id`
  - request logging middleware
  - configurable timeout/auth/rate-limit middleware for app routes
  - env-driven HTTP config
  - graceful shutdown hook support
- **`harbor-observability`**
  - tracing/log bootstrap
  - Prometheus recorder setup
  - OTEL trace exporter bootstrap
  - metrics renderer for the HTTP surface
- **`harbor-cli`**
  - `new` command to scaffold a new AI solution
  - `doctor` command to explain workspace capabilities

## Workspace layout

```text
harbor/
  crates/
    harbor-core/
    harbor-ai/
    harbor-memory/
    harbor-rag/
    harbor-runtime/
    harbor-mcp/
    harbor-http/
    harbor-observability/
    harbor-cli/
  docs/
    ARCHITECTURE.md
    ROADMAP.md
```

## Requirements

- Rust **1.86+**
- `rust-toolchain.toml` is included so contributors land on a compatible toolchain by default

## Example usage

### Run the agent example

```bash
cargo run -p harbor-runtime --example hello_agent
```

### Run the streaming agent example

```bash
cargo run -p harbor-runtime --example streaming_agent
```

### Run the retrieval-aware agent example

```bash
cargo run -p harbor-runtime --example retrieval_agent
```

### Run the stdio MCP server example

```bash
cargo run -p harbor-mcp --example echo_stdio_server
```

### Run the MCP HTTP server example

```bash
cargo run -p harbor-mcp --example http_server
```

### Run the HTTP ops server example

```bash
cargo run -p harbor-http --example minimal_server
```

### Run the shared Harbor bootstrap example

```bash
cargo run -p harbor-runtime --example bootstrap_http
```

This boots Harbor with:
- `/healthcheck`
- `/readycheck`
- `/metrics`
- request ID propagation via `x-request-id`
- incoming `traceparent` extraction when OTEL is enabled
- request logging middleware
- optional timeout/auth/rate-limit middleware for app routes
- signal-driven shutdown
- env-driven tracing/logging + metrics bootstrap

Optional OTEL envs:
- `HARBOR_OTEL_ENABLED=true`
- `HARBOR_OTEL_ENDPOINT=http://127.0.0.1:4317`

### Scaffold a new project

```bash
cargo run -p harbor-cli -- new my-ai-app --with-mcp-server
```

## Design goals

1. **AI-first**: providers, prompts, tools, memory, and workflows are first-class.
2. **MCP-native**: easy to create MCP servers and integrate external MCP servers.
3. **Reusable**: common building blocks for real AI products, not one-off demos.
4. **Rust-first**: strongly typed, composable, and suitable for production-grade services.

## CI

GitHub Actions now runs:
- `cargo check --workspace --all-targets`
- `cargo test --workspace --no-run`

## Near-term roadmap

- typed tool schemas via derive macros
- richer observability hooks
- Anthropic native streaming adapter
- Redis / Postgres state backends
- vector retrieval backends
- deployment templates for containerized AI services
