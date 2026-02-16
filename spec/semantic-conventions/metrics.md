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

---

### Model Operations (LLMOps/MLOps) Metrics

#### `aitf.model_ops.training.runs`

**Type:** Counter
**Unit:** `{run}`
**Description:** Total training/fine-tuning runs.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.training.type` | string | Training type |
| `aitf.model_ops.training.status` | string | Run status |
| `aitf.model_ops.training.base_model` | string | Base model |

#### `aitf.model_ops.training.duration`

**Type:** Histogram
**Unit:** `s`
**Description:** Training run duration.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.training.type` | string | Training type |
| `aitf.model_ops.training.base_model` | string | Base model |

#### `aitf.model_ops.training.gpu_hours`

**Type:** Counter
**Unit:** `h`
**Description:** Cumulative GPU hours consumed by training.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.training.compute.gpu_type` | string | GPU type |
| `aitf.model_ops.training.type` | string | Training type |

#### `aitf.model_ops.evaluation.runs`

**Type:** Counter
**Unit:** `{run}`
**Description:** Total evaluation runs.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.evaluation.type` | string | Evaluation type |
| `aitf.model_ops.evaluation.pass` | boolean | Pass/fail |

#### `aitf.model_ops.evaluation.score`

**Type:** Histogram
**Unit:** `1`
**Description:** Evaluation metric scores (0-1 normalized).

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.evaluation.model_id` | string | Model evaluated |
| `aitf.model_ops.evaluation.type` | string | Evaluation type |
| `aitf.model_ops.monitoring.metric_name` | string | Metric name |

#### `aitf.model_ops.registry.operations`

**Type:** Counter
**Unit:** `{operation}`
**Description:** Total model registry operations.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.registry.operation` | string | Operation type |
| `aitf.model_ops.registry.stage` | string | Target stage |

#### `aitf.model_ops.deployment.count`

**Type:** Counter
**Unit:** `{deployment}`
**Description:** Total model deployments.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.deployment.strategy` | string | Deployment strategy |
| `aitf.model_ops.deployment.environment` | string | Target environment |
| `aitf.model_ops.deployment.status` | string | Deployment status |

#### `aitf.model_ops.serving.cache_hit_rate`

**Type:** Gauge
**Unit:** `%`
**Description:** Serving cache hit rate.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.serving.cache.type` | string | Cache type |

#### `aitf.model_ops.serving.fallback.count`

**Type:** Counter
**Unit:** `{fallback}`
**Description:** Total fallback events.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.serving.fallback.trigger` | string | Trigger reason |

#### `aitf.model_ops.serving.circuit_breaker.transitions`

**Type:** Counter
**Unit:** `{transition}`
**Description:** Circuit breaker state transitions.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.serving.circuit_breaker.state` | string | New state |
| `aitf.model_ops.serving.circuit_breaker.model` | string | Affected model |

#### `aitf.model_ops.monitoring.drift_score`

**Type:** Histogram
**Unit:** `1`
**Description:** Drift detection scores (0-1).

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.monitoring.model_id` | string | Monitored model |
| `aitf.model_ops.monitoring.drift_type` | string | Drift type |

#### `aitf.model_ops.monitoring.alerts`

**Type:** Counter
**Unit:** `{alert}`
**Description:** Total monitoring alerts triggered.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.monitoring.check_type` | string | Check type |
| `aitf.model_ops.monitoring.result` | string | Alert level |
| `aitf.model_ops.monitoring.action_triggered` | string | Action taken |

#### `aitf.model_ops.prompt.versions`

**Type:** Counter
**Unit:** `{version}`
**Description:** Total prompt versions created.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.model_ops.prompt.name` | string | Prompt name |
| `aitf.model_ops.prompt.operation` | string | Operation type |

---

### Identity Metrics

#### `aitf.identity.authentications`

**Type:** Counter
**Unit:** `{authentication}`
**Description:** Total agent authentication attempts.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.auth.method` | string | Auth method |
| `aitf.identity.auth.result` | string | Auth result |
| `aitf.identity.provider` | string | Identity provider |

#### `aitf.identity.authentication.duration`

**Type:** Histogram
**Unit:** `ms`
**Description:** Authentication operation duration.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.auth.method` | string | Auth method |
| `aitf.identity.provider` | string | Identity provider |

#### `aitf.identity.authorizations`

**Type:** Counter
**Unit:** `{authorization}`
**Description:** Total authorization decisions.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.authz.decision` | string | Decision |
| `aitf.identity.authz.policy_engine` | string | Policy engine |

#### `aitf.identity.delegations`

**Type:** Counter
**Unit:** `{delegation}`
**Description:** Total credential delegations.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.delegation.type` | string | Delegation type |
| `aitf.identity.delegation.result` | string | Delegation result |

#### `aitf.identity.delegation.chain_depth`

**Type:** Histogram
**Unit:** `{depth}`
**Description:** Delegation chain depth distribution.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.delegation.type` | string | Delegation type |

#### `aitf.identity.active_sessions`

**Type:** UpDownCounter
**Unit:** `{session}`
**Description:** Active identity sessions.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.type` | string | Identity type |
| `aitf.identity.provider` | string | Provider |

#### `aitf.identity.lifecycle.events`

**Type:** Counter
**Unit:** `{event}`
**Description:** Total identity lifecycle events.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.lifecycle.operation` | string | Lifecycle operation |
| `aitf.identity.type` | string | Identity type |

#### `aitf.identity.trust.establishments`

**Type:** Counter
**Unit:** `{trust}`
**Description:** Total trust establishment attempts.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.trust.method` | string | Trust method |
| `aitf.identity.trust.result` | string | Trust result |
| `aitf.identity.trust.cross_domain` | boolean | Cross-domain |

#### `aitf.identity.auth_failures`

**Type:** Counter
**Unit:** `{failure}`
**Description:** Total authentication and authorization failures (security signal).

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.agent_name` | string | Agent name |
| `aitf.identity.auth.method` | string | Auth method |
| `aitf.identity.auth.failure_reason` | string | Failure reason |
