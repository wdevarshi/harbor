# Harbor Architecture

## Core idea

Harbor is a workspace of focused crates that compose into an AI application platform:

- **core**: shared primitives and tool contracts
- **ai**: provider abstractions and model-facing types
- **memory**: session and state retention
- **runtime**: agents and workflows
- **mcp**: MCP server/client protocol and transports
- **cli**: project scaffolding and developer ergonomics

## Capability map

### 1. Providers
Provider abstraction is intentionally separate from runtime so the framework can swap:
- mock providers
- local models
- hosted API providers
- future routing/fallback layers

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

### 5. MCP
The MCP layer turns the same tool registry into an MCP server.
This keeps the framework from duplicating tool definitions between:
- internal runtime use
- external MCP integrations

## Why this shape

This design is meant to solve the most common AI framework failure mode:
> every new AI solution reimplements providers, tools, memory, execution flow, and integrations from scratch.

Instead, Harbor gives one reusable substrate that can be adapted into:
- assistants
- vertical AI apps
- internal copilots
- automations
- MCP servers for external agent ecosystems
