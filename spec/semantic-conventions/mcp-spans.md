# MCP Span Conventions (MCP_ACTIVITY)

Status: **Normative** | CoSAI WS2 Alignment: **MCP_ACTIVITY** | OCSF Class: **7003 Tool Execution**

AITF defines semantic conventions for the Model Context Protocol (MCP), covering server lifecycle, tool discovery and invocation, resource access, prompt management, and sampling. This specification defines the normative field requirements aligned with CoSAI Working Stream 2 (Telemetry for AI) and mapped to applicable compliance and threat frameworks.

Key words "MUST", "SHOULD", "MAY" follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## Overview

MCP is a protocol that enables AI models to interact with external systems through a standardized interface. AITF provides first-class telemetry for all MCP operations:

```
MCP Server Lifecycle:
  connect -> initialize -> discover (tools/resources/prompts) -> use -> disconnect

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

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.server.name` | string | **Required** | MCP server name | OWASP LLM06 (Excessive Agency), MITRE ATLAS [AML.T0040](https://atlas.mitre.org/techniques/AML.T0040) |
| `aitf.mcp.server.transport` | string | **Required** | Transport type: `"stdio"`, `"sse"`, `"streamable_http"` | MITRE ATLAS [AML.T0040](https://atlas.mitre.org/techniques/AML.T0040) (ML Supply Chain) |
| `aitf.mcp.connection.id` | string | **Recommended** | Unique connection identifier for session correlation | NIST AI RMF GOVERN-1.2 |
| `aitf.mcp.server.version` | string | **Recommended** | MCP server version | NIST AI RMF MAP-1.1 |
| `aitf.mcp.server.url` | string | **Recommended** | Server URL (if network transport) | MITRE ATLAS AML.T0040 |
| `aitf.mcp.protocol.version` | string | **Recommended** | MCP protocol version (e.g. `"2025-03-26"`) | NIST AI RMF MAP-1.1 |

### Events

#### `mcp.server.capabilities`

Emitted after successful initialization.

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `mcp.capabilities.tools` | boolean | **Recommended** | Server supports tools | OWASP LLM06 |
| `mcp.capabilities.resources` | boolean | **Recommended** | Server supports resources | — |
| `mcp.capabilities.prompts` | boolean | **Optional** | Server supports prompts | — |
| `mcp.capabilities.sampling` | boolean | **Optional** | Server supports sampling | — |
| `mcp.capabilities.roots` | boolean | **Optional** | Server supports roots | — |

---

## Span: `aitf.mcp.server.disconnect`

Represents disconnecting from an MCP server.

### Span Name

Format: `mcp.server.disconnect {aitf.mcp.server.name}`

### Span Kind

`CLIENT`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.server.name` | string | **Required** | Server name | OWASP LLM06 |
| `aitf.mcp.connection.id` | string | **Recommended** | Connection identifier | NIST AI RMF GOVERN-1.2 |

---

## Span: `aitf.mcp.tool.discover`

Represents discovering available tools from an MCP server.

### Span Name

Format: `mcp.tool.discover {aitf.mcp.server.name}`

### Span Kind

`CLIENT`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.server.name` | string | **Required** | Server name | OWASP LLM06 |
| `aitf.mcp.tool.count` | int | **Recommended** | Number of tools discovered | OWASP LLM06 |
| `aitf.mcp.tool.names` | string[] | **Recommended** | Names of discovered tools | OWASP LLM06, MITRE ATLAS AML.T0048 |
| `aitf.mcp.connection.id` | string | **Optional** | Connection identifier | NIST AI RMF GOVERN-1.2 |

---

## Span: `aitf.mcp.tool.invoke`

Represents invoking a tool on an MCP server. This is the primary MCP telemetry span.

### Span Name

Format: `mcp.tool.invoke {aitf.mcp.tool.name}`

### Span Kind

`CLIENT`

### Normative Field Table

#### Tool Identification

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.tool.name` | string | **Required** | Tool name | OWASP LLM06 (Excessive Agency), MITRE ATLAS [AML.T0048](https://atlas.mitre.org/techniques/AML.T0048) |
| `aitf.mcp.tool.server` | string | **Required** | Source MCP server name | OWASP LLM06, MITRE ATLAS [AML.T0040](https://atlas.mitre.org/techniques/AML.T0040) |
| `aitf.mcp.connection.id` | string | **Recommended** | Connection identifier for session correlation | NIST AI RMF GOVERN-1.2 |

#### Request & Response (CoSAI WS2)

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.tool.input` | string | **Recommended** | Input parameters (JSON) | OWASP LLM01 (Prompt Injection via tools), MITRE ATLAS [AML.T0051](https://atlas.mitre.org/techniques/AML.T0051) |
| `aitf.mcp.tool.output` | string | **Recommended** | Tool output (may be redacted for sensitive content) | OWASP LLM05 (Improper Output), OWASP LLM02 (Sensitive Info) |
| `aitf.mcp.tool.response_error` | string | **Recommended** | Error message content when tool execution fails | NIST AI RMF MEASURE-2.5 |
| `aitf.mcp.tool.is_error` | boolean | **Recommended** | Whether tool returned an error | NIST AI RMF MEASURE-2.5 |

#### Execution Metadata

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.tool.duration_ms` | double | **Recommended** | Execution duration in milliseconds | NIST AI RMF MEASURE-2.5, OWASP LLM10 |
| `aitf.mcp.server.transport` | string | **Recommended** | Transport type | MITRE ATLAS AML.T0040 |

#### Human Approval

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.tool.approval_required` | boolean | **Recommended** | Whether human approval was needed | EU AI Act Art.14 (Human Oversight) |
| `aitf.mcp.tool.approved` | boolean | **Recommended** | Whether the tool was approved (if required) | EU AI Act Art.14 |

### Events

#### `mcp.tool.input`

Emitted with tool input (can be sampled/redacted for sensitive inputs).

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `mcp.tool.input.content` | string | **Optional** | Full input content | OWASP LLM01 |

#### `mcp.tool.output`

Emitted with tool output (can be sampled/redacted).

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `mcp.tool.output.content` | string | **Optional** | Full output content | OWASP LLM05, OWASP LLM02 |
| `mcp.tool.output.type` | string | **Optional** | Output type: `"text"`, `"image"`, `"resource"` | — |

#### `mcp.tool.approval`

Emitted when human approval is requested/granted.

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `mcp.tool.approval.status` | string | **Required** | `"requested"`, `"approved"`, `"denied"` | EU AI Act Art.14 |
| `mcp.tool.approval.approver` | string | **Recommended** | Who approved (if applicable) | EU AI Act Art.14, NIST AI RMF GOVERN-1.7 |

---

## Span: `aitf.mcp.resource.read`

Represents reading a resource from an MCP server.

### Span Name

Format: `mcp.resource.read {aitf.mcp.resource.uri}`

### Span Kind

`CLIENT`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.resource.uri` | string | **Required** | Resource URI | OWASP LLM02 (Sensitive Info) |
| `aitf.mcp.server.name` | string | **Required** | Server name | OWASP LLM06 |
| `aitf.mcp.connection.id` | string | **Optional** | Connection identifier | NIST AI RMF GOVERN-1.2 |
| `aitf.mcp.resource.name` | string | **Recommended** | Resource display name | — |
| `aitf.mcp.resource.mime_type` | string | **Recommended** | Content MIME type | — |
| `aitf.mcp.resource.size_bytes` | int | **Optional** | Content size in bytes | OWASP LLM10 |

---

## Span: `aitf.mcp.resource.subscribe`

Represents subscribing to resource updates from an MCP server.

### Span Name

Format: `mcp.resource.subscribe {aitf.mcp.resource.uri}`

### Span Kind

`CLIENT`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.resource.uri` | string | **Required** | Resource URI | OWASP LLM02 |
| `aitf.mcp.server.name` | string | **Required** | Server name | OWASP LLM06 |
| `aitf.mcp.connection.id` | string | **Optional** | Connection identifier | NIST AI RMF GOVERN-1.2 |

---

## Span: `aitf.mcp.prompt.get`

Represents retrieving a prompt template from an MCP server.

### Span Name

Format: `mcp.prompt.get {aitf.mcp.prompt.name}`

### Span Kind

`CLIENT`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.prompt.name` | string | **Required** | Prompt template name | OWASP LLM01 (Prompt Injection) |
| `aitf.mcp.server.name` | string | **Required** | Server name | OWASP LLM06 |
| `aitf.mcp.prompt.arguments` | string | **Recommended** | Prompt arguments (JSON) | OWASP LLM01 |
| `aitf.mcp.prompt.description` | string | **Optional** | Prompt description | EU AI Act Art.13 |

---

## Span: `aitf.mcp.sampling.request`

Represents a server-initiated sampling (LLM) request via MCP.

### Span Name

Format: `mcp.sampling.request {aitf.mcp.server.name}`

### Span Kind

`SERVER` (server is requesting from the client)

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.mcp.server.name` | string | **Required** | Requesting server name | OWASP LLM06, MITRE ATLAS AML.T0048 |
| `aitf.mcp.sampling.model` | string | **Required** | Requested model hint | MITRE ATLAS AML.T0044 |
| `aitf.mcp.sampling.max_tokens` | int | **Recommended** | Max tokens | OWASP LLM10 |
| `aitf.mcp.sampling.include_context` | string | **Optional** | Context scope: `"thisServer"`, `"allServers"` | OWASP LLM02 |
| `gen_ai.usage.input_tokens` | int | **Recommended** | Input tokens used | OWASP LLM10 |
| `gen_ai.usage.output_tokens` | int | **Recommended** | Output tokens used | OWASP LLM10 |

---

## CoSAI WS2 Field Mapping

Cross-reference between CoSAI WS2 `MCP_ACTIVITY` field names and AITF attribute keys:

| CoSAI WS2 Field | AITF Attribute | Notes |
|---|---|---|
| `mcp.server.name` | `aitf.mcp.server.name` | Direct match |
| `mcp.tool.name` | `aitf.mcp.tool.name` | Direct match |
| `mcp.request.args` | `aitf.mcp.tool.input` | JSON-encoded input parameters |
| `mcp.response.result` | `aitf.mcp.tool.output` | May be redacted |
| `mcp.response.error` | `aitf.mcp.tool.response_error` | New in CoSAI WS2 alignment |
| `mcp.transport` | `aitf.mcp.server.transport` | On server/tool spans |
| `mcp.connection.id` | `aitf.mcp.connection.id` | New in CoSAI WS2 alignment |

---

## Security Considerations

MCP tool invocations should be monitored for:

1. **Unauthorized file access** -- Tools accessing files outside allowed paths
2. **Command injection** -- Malicious input parameters
3. **Data exfiltration** -- Tools sending sensitive data to external systems
4. **Privilege escalation** -- Tools operating beyond granted permissions

AITF's Security Processor automatically flags suspicious MCP tool invocations based on configurable policies.

---

## Example: MCP Tool Discovery and Invocation

```
Span: mcp.server.connect filesystem
  aitf.mcp.server.transport: "stdio"
  aitf.mcp.protocol.version: "2025-03-26"
  aitf.mcp.connection.id: "conn-fs-abc123"
  |
  +- Event: mcp.server.capabilities
  |    mcp.capabilities.tools: true
  |    mcp.capabilities.resources: true
  |
  +- Span: mcp.tool.discover filesystem
  |    aitf.mcp.tool.count: 5
  |    aitf.mcp.tool.names: ["read_file", "write_file", "list_dir", "search", "move_file"]
  |
  +- Span: mcp.tool.invoke read_file
  |    aitf.mcp.tool.server: "filesystem"
  |    aitf.mcp.tool.input: "{\"path\":\"/data/config.yaml\"}"
  |    aitf.mcp.tool.is_error: false
  |    aitf.mcp.tool.response_error: ""
  |    aitf.mcp.tool.duration_ms: 12.5
  |    aitf.mcp.connection.id: "conn-fs-abc123"
  |    Events:
  |      mcp.tool.input: {content: "{\"path\":\"/data/config.yaml\"}"}
  |      mcp.tool.output: {content: "server:\n  port: 8080\n...", type: "text"}
  |
  +- Span: mcp.tool.invoke write_file
       aitf.mcp.tool.server: "filesystem"
       aitf.mcp.tool.approval_required: true
       aitf.mcp.tool.approved: true
       aitf.mcp.tool.duration_ms: 25.0
       aitf.mcp.connection.id: "conn-fs-abc123"
       Events:
         mcp.tool.approval: {status: "approved", approver: "user@example.com"}
         mcp.tool.input: {content: "{\"path\":\"/data/output.txt\",\"content\":\"...\"}"}
         mcp.tool.output: {content: "File written successfully", type: "text"}
```
