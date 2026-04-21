# Harbor Roadmap

## Phase 1 - Foundation (implemented)
- workspace split into reusable crates
- provider abstraction
- mock provider
- OpenAI-compatible provider client
- tool registry
- session memory abstraction
- agent runtime
- workflow engine
- MCP server builder
- MCP local integration client
- HTTP ops surface with `/healthcheck` and `/readycheck`
- CLI scaffolding

## Phase 2 - Runtime and ops
- shared runtime/bootstrap entrypoint
- graceful shutdown wiring across runtimes
- env-first config helpers across crates
- metrics surface
- tracing bootstrap
- structured logging bootstrap

## Phase 3 - Richer AI integrations
- Anthropic adapter
- Ollama / local model adapter
- retry, timeout, and fallback policies
- structured output helpers

## Phase 4 - Richer MCP support
- stdio client for spawned MCP servers
- HTTP transport
- resource and prompt endpoints
- server capability negotiation helpers

## Phase 5 - Developer ergonomics
- derive macro for tools
- schema generation from Rust types
- test harness for agent workflows
- deployment templates (Docker / K8s)

## Phase 6 - Data + memory
- vector store integrations
- Redis / Postgres memory backends
- RAG pipeline helpers
- document ingestion adapters
