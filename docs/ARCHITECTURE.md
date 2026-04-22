# Harbor Architecture

## Core idea

Harbor is a workspace of focused crates that compose into an AI application platform:

- **core**: shared primitives and tool contracts
- **ai**: provider abstractions and model-facing types
- **memory**: session and state retention
- **runtime**: agents and workflows
- **mcp**: MCP server/client protocol and transports
- **http**: health/readiness/metrics and ops-facing HTTP surface
- **observability**: tracing/log bootstrap and Prometheus setup
- **cli**: project scaffolding and developer ergonomics

## Capability map

### 1. Providers
Provider abstraction is intentionally separate from runtime so the framework can swap:
- mock providers
- local models
- hosted API providers
- future routing/fallback layers

The current baseline includes:
- OpenAI-compatible HTTP providers
- Anthropic HTTP providers
- Ollama HTTP providers

Hosted HTTP providers can also inherit the current trace context so model/API calls participate in the same distributed trace when Harbor is running with OTEL enabled.

The provider layer also includes lightweight reliability wrappers for retry, timeout, fallback, and structured-output parsing/validation so apps can compose resilience without rewriting provider orchestration each time.

### 2. Tools
Tools are defined once and reused in:
- native runtime workflows
- MCP exposure
- future HTTP/gateway layers

### 3. Memory
Memory is session-scoped and provider-agnostic.
The first implementation is in-memory, but the interface allows:
- Redis
- Postgres
- vector stores
- hybrid memory

### 4. Runtime
Runtime orchestrates:
- prompt/system state
- memory loading
- provider invocation
- tool registry access
- workflow state transitions
- shared application bootstrap (`HarborApp`)
- signal-driven shutdown and readiness wiring

### 5. MCP
The MCP layer turns the same tool registry into an MCP server.
This keeps the framework from duplicating tool definitions between:
- internal runtime use
- external MCP integrations

The current MCP baseline includes:
- in-process/local client usage
- stdio framing for process-based transports
- spawned stdio client transport for subprocess-backed MCP servers
- HTTP transport for remote MCP-style integration
- resource and prompt endpoints
- capability reporting via `initialize`
- outbound trace-context injection for MCP HTTP client calls

### 6. HTTP / ops
The HTTP layer provides the operational surface around Harbor runtimes:
- `/healthcheck`
- `/readycheck`
- `/metrics`
- request ID propagation via `x-request-id`
- incoming trace-context extraction from request headers
- request logging middleware
- timeout/auth/rate-limit middleware for application routes
- env-driven bind configuration
- graceful shutdown hooks

This is Harbor's equivalent of the production defaults that frameworks like ColdBrew expose for service operations.

### 7. Observability
The observability layer bootstraps the global cross-cutting runtime concerns:
- tracing/log subscriber setup
- log level + JSON log configuration
- Prometheus recorder initialization
- OTLP trace exporter initialization
- global W3C trace-context propagator setup
- metrics rendering into the HTTP surface
- structured request logs emitted by the HTTP middleware

## Why this shape

This design is meant to solve the most common AI framework failure mode:
> every new AI solution reimplements providers, tools, memory, execution flow, and integrations from scratch.

Instead, Harbor gives one reusable substrate that can be adapted into:
- assistants
- vertical AI apps
- internal copilots
- automations
- MCP servers for external agent ecosystems
