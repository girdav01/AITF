# MCP Span Conventions

AITF defines semantic conventions for the Model Context Protocol (MCP), covering server lifecycle, tool discovery and invocation, resource access, prompt management, and sampling.

## Overview

MCP is a protocol that enables AI models to interact with external systems through a standardized interface. AITF provides first-class telemetry for all MCP operations:

```
MCP Server Lifecycle:
  connect → initialize → discover (tools/resources/prompts) → use → disconnect

MCP Operations:
  - Tool Discovery & Invocation
  - Resource Read & Subscribe
  - Prompt Get & Execute
  - Sampling (server-initiated LLM requests)
  - Root Management
```

---

## Span: `aitf.mcp.server.connect`

Represents establishing a connection to an MCP server.

### Span Name

Format: `mcp.server.connect {aitf.mcp.server.name}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.server.name` | string | Server name |
| `aitf.mcp.server.transport` | string | `"stdio"`, `"sse"`, `"streamable_http"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.server.version` | string | Server version |
| `aitf.mcp.server.url` | string | Server URL (if network transport) |
| `aitf.mcp.protocol.version` | string | MCP protocol version |

### Events

#### `mcp.server.capabilities`

Emitted after successful initialization.

| Attribute | Type | Description |
|-----------|------|-------------|
| `mcp.capabilities.tools` | boolean | Server supports tools |
| `mcp.capabilities.resources` | boolean | Server supports resources |
| `mcp.capabilities.prompts` | boolean | Server supports prompts |
| `mcp.capabilities.sampling` | boolean | Server supports sampling |
| `mcp.capabilities.roots` | boolean | Server supports roots |

---

## Span: `aitf.mcp.server.disconnect`

Represents disconnecting from an MCP server.

### Span Name

Format: `mcp.server.disconnect {aitf.mcp.server.name}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.server.name` | string | Server name |

---

## Span: `aitf.mcp.tool.discover`

Represents discovering available tools from an MCP server.

### Span Name

Format: `mcp.tool.discover {aitf.mcp.server.name}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.server.name` | string | Server name |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.tool.count` | int | Number of tools discovered |
| `aitf.mcp.tool.names` | string[] | Names of discovered tools |

---

## Span: `aitf.mcp.tool.invoke`

Represents invoking a tool on an MCP server. This is the primary MCP telemetry span.

### Span Name

Format: `mcp.tool.invoke {aitf.mcp.tool.name}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.tool.name` | string | Tool name |
| `aitf.mcp.tool.server` | string | Source MCP server name |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.tool.input` | string | Input parameters (JSON) |
| `aitf.mcp.tool.output` | string | Tool output (may be redacted) |
| `aitf.mcp.tool.is_error` | boolean | Whether tool returned error |
| `aitf.mcp.tool.duration_ms` | double | Execution duration (ms) |
| `aitf.mcp.tool.approval_required` | boolean | Human approval needed |
| `aitf.mcp.tool.approved` | boolean | Whether approved |
| `aitf.mcp.server.transport` | string | Transport type |

### Events

#### `mcp.tool.input`

Emitted with tool input (can be sampled/redacted for sensitive inputs).

| Attribute | Type | Description |
|-----------|------|-------------|
| `mcp.tool.input.content` | string | Full input content |

#### `mcp.tool.output`

Emitted with tool output (can be sampled/redacted).

| Attribute | Type | Description |
|-----------|------|-------------|
| `mcp.tool.output.content` | string | Full output content |
| `mcp.tool.output.type` | string | `"text"`, `"image"`, `"resource"` |

#### `mcp.tool.approval`

Emitted when human approval is requested/granted.

| Attribute | Type | Description |
|-----------|------|-------------|
| `mcp.tool.approval.status` | string | `"requested"`, `"approved"`, `"denied"` |
| `mcp.tool.approval.approver` | string | Who approved (if applicable) |

---

## Span: `aitf.mcp.resource.read`

Represents reading a resource from an MCP server.

### Span Name

Format: `mcp.resource.read {aitf.mcp.resource.uri}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.resource.uri` | string | Resource URI |
| `aitf.mcp.server.name` | string | Server name |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.resource.name` | string | Resource display name |
| `aitf.mcp.resource.mime_type` | string | Content MIME type |
| `aitf.mcp.resource.size_bytes` | int | Content size |

---

## Span: `aitf.mcp.resource.subscribe`

Represents subscribing to resource updates from an MCP server.

### Span Name

Format: `mcp.resource.subscribe {aitf.mcp.resource.uri}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.resource.uri` | string | Resource URI |
| `aitf.mcp.server.name` | string | Server name |

---

## Span: `aitf.mcp.prompt.get`

Represents retrieving a prompt template from an MCP server.

### Span Name

Format: `mcp.prompt.get {aitf.mcp.prompt.name}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.prompt.name` | string | Prompt name |
| `aitf.mcp.server.name` | string | Server name |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.prompt.arguments` | string | Prompt arguments (JSON) |
| `aitf.mcp.prompt.description` | string | Prompt description |

---

## Span: `aitf.mcp.sampling.request`

Represents a server-initiated sampling (LLM) request via MCP.

### Span Name

Format: `mcp.sampling.request {aitf.mcp.server.name}`

### Span Kind

`SERVER` (server is requesting from the client)

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.server.name` | string | Requesting server |
| `aitf.mcp.sampling.model` | string | Requested model hint |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.sampling.max_tokens` | int | Max tokens |
| `aitf.mcp.sampling.include_context` | string | Context scope |
| `gen_ai.usage.input_tokens` | int | Input tokens used |
| `gen_ai.usage.output_tokens` | int | Output tokens used |

---

## Example: MCP Tool Discovery and Invocation

```
Span: mcp.server.connect filesystem
  aitf.mcp.server.transport: "stdio"
  aitf.mcp.protocol.version: "2025-03-26"
  │
  ├─ Event: mcp.server.capabilities
  │    mcp.capabilities.tools: true
  │    mcp.capabilities.resources: true
  │
  ├─ Span: mcp.tool.discover filesystem
  │    aitf.mcp.tool.count: 5
  │    aitf.mcp.tool.names: ["read_file", "write_file", "list_dir", "search", "move_file"]
  │
  ├─ Span: mcp.tool.invoke read_file
  │    aitf.mcp.tool.server: "filesystem"
  │    aitf.mcp.tool.input: "{\"path\":\"/data/config.yaml\"}"
  │    aitf.mcp.tool.is_error: false
  │    aitf.mcp.tool.duration_ms: 12.5
  │    Events:
  │      mcp.tool.input: {content: "{\"path\":\"/data/config.yaml\"}"}
  │      mcp.tool.output: {content: "server:\n  port: 8080\n...", type: "text"}
  │
  └─ Span: mcp.tool.invoke write_file
       aitf.mcp.tool.server: "filesystem"
       aitf.mcp.tool.approval_required: true
       aitf.mcp.tool.approved: true
       aitf.mcp.tool.duration_ms: 25.0
       Events:
         mcp.tool.approval: {status: "approved", approver: "user@example.com"}
         mcp.tool.input: {content: "{\"path\":\"/data/output.txt\",\"content\":\"...\"}"}
         mcp.tool.output: {content: "File written successfully", type: "text"}
```

## Example: MCP Sampling (Server-Initiated LLM Request)

```
Span: mcp.sampling.request code-analyzer
  aitf.mcp.server.name: "code-analyzer"
  aitf.mcp.sampling.model: "claude-sonnet-4-5-20250929"
  aitf.mcp.sampling.max_tokens: 1024
  aitf.mcp.sampling.include_context: "thisServer"
  gen_ai.usage.input_tokens: 200
  gen_ai.usage.output_tokens: 150
  │
  └─ Span: chat claude-sonnet-4-5-20250929
       gen_ai.system: "anthropic"
       gen_ai.operation.name: "chat"
       ...
```

## Security Considerations

MCP tool invocations should be monitored for:

1. **Unauthorized file access** — Tools accessing files outside allowed paths
2. **Command injection** — Malicious input parameters
3. **Data exfiltration** — Tools sending sensitive data to external systems
4. **Privilege escalation** — Tools operating beyond granted permissions

AITF's Security Processor automatically flags suspicious MCP tool invocations based on configurable policies.
