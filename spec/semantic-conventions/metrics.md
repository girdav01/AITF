# AITF Metrics Conventions

AITF defines metrics for monitoring AI system performance, cost, security, and quality.

## OTel GenAI Metrics (Preserved)

These metrics follow OpenTelemetry GenAI semantic conventions.

### `gen_ai.client.token.usage`

**Type:** Histogram
**Unit:** `{token}`
**Description:** Number of tokens used per request.

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider |
| `gen_ai.request.model` | string | Model ID |
| `gen_ai.token.type` | string | `"input"`, `"output"` |
| `gen_ai.operation.name` | string | Operation name |

### `gen_ai.client.operation.duration`

**Type:** Histogram
**Unit:** `s`
**Description:** Duration of GenAI operations.

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider |
| `gen_ai.request.model` | string | Model ID |
| `gen_ai.operation.name` | string | Operation name |
| `gen_ai.response.finish_reasons` | string[] | Finish reasons |
| `error.type` | string | Error type (if error) |

---

## AITF Extended Metrics

### Inference Metrics

#### `aitf.inference.requests`

**Type:** Counter
**Unit:** `{request}`
**Description:** Total number of inference requests.

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider |
| `gen_ai.request.model` | string | Model ID |
| `gen_ai.operation.name` | string | Operation name |
| `aitf.cost.attribution.user` | string | User ID |
| `aitf.cost.attribution.project` | string | Project ID |

#### `aitf.inference.errors`

**Type:** Counter
**Unit:** `{error}`
**Description:** Total inference errors.

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider |
| `gen_ai.request.model` | string | Model ID |
| `error.type` | string | Error type |

#### `aitf.inference.time_to_first_token`

**Type:** Histogram
**Unit:** `ms`
**Description:** Time to first token in streaming responses.

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider |
| `gen_ai.request.model` | string | Model ID |

#### `aitf.inference.tokens_per_second`

**Type:** Histogram
**Unit:** `{token}/s`
**Description:** Token generation rate.

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider |
| `gen_ai.request.model` | string | Model ID |

---

### Agent Metrics

#### `aitf.agent.sessions`

**Type:** Counter
**Unit:** `{session}`
**Description:** Total agent sessions started.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Agent name |
| `aitf.agent.framework` | string | Framework |

#### `aitf.agent.steps`

**Type:** Counter
**Unit:** `{step}`
**Description:** Total agent steps executed.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Agent name |
| `aitf.agent.step.type` | string | Step type |

#### `aitf.agent.session.duration`

**Type:** Histogram
**Unit:** `s`
**Description:** Duration of agent sessions.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Agent name |
| `aitf.agent.framework` | string | Framework |

#### `aitf.agent.steps_per_session`

**Type:** Histogram
**Unit:** `{step}`
**Description:** Number of steps per agent session.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Agent name |

#### `aitf.agent.delegations`

**Type:** Counter
**Unit:** `{delegation}`
**Description:** Total agent delegations.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Delegating agent |
| `aitf.agent.delegation.target_agent` | string | Target agent |
| `aitf.agent.delegation.strategy` | string | Strategy |

---

### MCP Metrics

#### `aitf.mcp.tool.invocations`

**Type:** Counter
**Unit:** `{invocation}`
**Description:** Total MCP tool invocations.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.tool.name` | string | Tool name |
| `aitf.mcp.tool.server` | string | Server name |
| `aitf.mcp.tool.is_error` | boolean | Error status |

#### `aitf.mcp.tool.duration`

**Type:** Histogram
**Unit:** `ms`
**Description:** MCP tool invocation duration.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.tool.name` | string | Tool name |
| `aitf.mcp.tool.server` | string | Server name |

#### `aitf.mcp.server.connections`

**Type:** UpDownCounter
**Unit:** `{connection}`
**Description:** Active MCP server connections.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.server.name` | string | Server name |
| `aitf.mcp.server.transport` | string | Transport type |

#### `aitf.mcp.tool.approvals`

**Type:** Counter
**Unit:** `{approval}`
**Description:** MCP tool approval requests.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.mcp.tool.name` | string | Tool name |
| `aitf.mcp.tool.approved` | boolean | Whether approved |

---

### Skill Metrics

#### `aitf.skill.invocations`

**Type:** Counter
**Unit:** `{invocation}`
**Description:** Total skill invocations.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.skill.name` | string | Skill name |
| `aitf.skill.category` | string | Skill category |
| `aitf.skill.status` | string | Execution status |

#### `aitf.skill.duration`

**Type:** Histogram
**Unit:** `ms`
**Description:** Skill execution duration.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.skill.name` | string | Skill name |
| `aitf.skill.category` | string | Skill category |

---

### Cost Metrics

#### `aitf.cost.total`

**Type:** Counter
**Unit:** `USD`
**Description:** Cumulative cost of AI operations.

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider |
| `gen_ai.request.model` | string | Model ID |
| `aitf.cost.attribution.user` | string | User ID |
| `aitf.cost.attribution.team` | string | Team ID |
| `aitf.cost.attribution.project` | string | Project ID |

#### `aitf.cost.budget.utilization`

**Type:** Gauge
**Unit:** `%`
**Description:** Budget utilization percentage.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.cost.attribution.project` | string | Project ID |
| `aitf.cost.attribution.team` | string | Team ID |

---

### Security Metrics

#### `aitf.security.threats_detected`

**Type:** Counter
**Unit:** `{threat}`
**Description:** Total threats detected.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.security.threat_type` | string | Threat type |
| `aitf.security.owasp_category` | string | OWASP category |
| `aitf.security.risk_level` | string | Risk level |

#### `aitf.security.requests_blocked`

**Type:** Counter
**Unit:** `{request}`
**Description:** Total requests blocked by security.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.security.threat_type` | string | Threat type |

#### `aitf.security.pii_detected`

**Type:** Counter
**Unit:** `{detection}`
**Description:** Total PII detections.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.security.pii.types` | string[] | PII types |
| `aitf.security.pii.action` | string | Action taken |

#### `aitf.security.guardrail.checks`

**Type:** Counter
**Unit:** `{check}`
**Description:** Total guardrail checks.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.security.guardrail.name` | string | Guardrail name |
| `aitf.security.guardrail.result` | string | Result |

---

### RAG Metrics

#### `aitf.rag.retrievals`

**Type:** Counter
**Unit:** `{retrieval}`
**Description:** Total RAG retrieval operations.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.rag.retrieve.database` | string | Vector DB |
| `aitf.rag.pipeline.name` | string | Pipeline name |

#### `aitf.rag.retrieval.duration`

**Type:** Histogram
**Unit:** `ms`
**Description:** RAG retrieval duration.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.rag.retrieve.database` | string | Vector DB |

#### `aitf.rag.retrieval.results`

**Type:** Histogram
**Unit:** `{document}`
**Description:** Number of documents retrieved.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.rag.retrieve.database` | string | Vector DB |
| `aitf.rag.pipeline.name` | string | Pipeline name |

---

### Quality Metrics

#### `aitf.quality.hallucination`

**Type:** Histogram
**Unit:** `1`
**Description:** Hallucination scores (0-1 scale, lower is better).

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.request.model` | string | Model ID |

#### `aitf.quality.user_rating`

**Type:** Histogram
**Unit:** `{rating}`
**Description:** User rating scores (1-5).

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.request.model` | string | Model ID |
| `aitf.agent.name` | string | Agent name (if applicable) |
