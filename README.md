# MCPForge

MCPForge is an **AI-first Rust framework** for building reusable AI solutions with **first-class MCP support**.

It is inspired by modern AI application platforms and documentation patterns like the ones in Coldbrew Cloud: a single developer surface for
- model providers
- tools and integrations
- memory/session state
- workflows and agents
- MCP server creation
- MCP client integration
- CLI scaffolding for new AI projects

## What this v0.1 gives you

- **`mcpforge-core`**
  - tool traits
  - tool registry
  - application blueprint/builder
  - shared framework errors
- **`mcpforge-ai`**
  - provider abstraction
  - chat/message types
  - structured completion request/response model
  - mock provider for local development and tests
- **`mcpforge-memory`**
  - session memory trait
  - in-memory implementation
- **`mcpforge-runtime`**
  - agent runtime
  - workflow engine
  - reusable execution context
- **`mcpforge-mcp`**
  - JSON-RPC + MCP-inspired protocol types
  - stdio framing (`Content-Length`)
  - MCP server builder
  - local integration client for tests and embedding
- **`mcpforge-cli`**
  - `new` command to scaffold a new AI solution
  - `doctor` command to explain workspace capabilities

## Workspace layout

```text
mcpforge-rs/
  crates/
    mcpforge-core/
    mcpforge-ai/
    mcpforge-memory/
    mcpforge-runtime/
    mcpforge-mcp/
    mcpforge-cli/
  docs/
    ARCHITECTURE.md
    ROADMAP.md
```

## Example usage

### Run the agent example

```bash
cargo run -p mcpforge-runtime --example hello_agent
```

### Run the stdio MCP server example

```bash
cargo run -p mcpforge-mcp --example echo_stdio_server
```

### Scaffold a new project

```bash
cargo run -p mcpforge-cli -- new my-ai-app --with-mcp-server
```

## Design goals

1. **AI-first**: providers, prompts, tools, memory, and workflows are first-class.
2. **MCP-native**: easy to create MCP servers and integrate external MCP servers.
3. **Reusable**: common building blocks for real AI products, not one-off demos.
4. **Rust-first**: strongly typed, composable, and suitable for production-grade services.

## Near-term roadmap

- OpenAI / Anthropic provider adapters
- HTTP transport for MCP integration
- typed tool schemas via derive macros
- event streaming + observability hooks
- vector memory backends
- deployment templates for containerized AI services
