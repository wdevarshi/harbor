# MCPForge Roadmap

## Phase 1 - Foundation (implemented in this scaffold)
- workspace split into reusable crates
- provider abstraction
- tool registry
- session memory abstraction
- agent runtime
- workflow engine
- MCP server builder
- MCP local integration client
- CLI scaffolding

## Phase 2 - Production AI integrations
- OpenAI adapter
- Anthropic adapter
- Ollama / local model adapter
- retry, timeout, and fallback policies
- structured output helpers

## Phase 3 - Richer MCP support
- stdio client for spawned MCP servers
- HTTP transport
- resource and prompt endpoints
- server capability negotiation helpers

## Phase 4 - Developer ergonomics
- derive macro for tools
- schema generation from Rust types
- tracing + metrics hooks
- test harness for agent workflows
- deployment templates (Docker / K8s)

## Phase 5 - Data + memory
- vector store integrations
- Redis / Postgres memory backends
- RAG pipeline helpers
- document ingestion adapters
