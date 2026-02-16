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

---

### Asset Inventory Metrics

#### `aitf.asset.registered`

**Type:** Counter
**Unit:** `{asset}`
**Description:** Total assets registered in inventory.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.type` | string | Asset type |
| `aitf.asset.deployment_environment` | string | Environment |
| `aitf.asset.risk_classification` | string | Risk level |

#### `aitf.asset.discovery.scans`

**Type:** Counter
**Unit:** `{scan}`
**Description:** Total asset discovery scans performed.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.discovery.scope` | string | Discovery scope |
| `aitf.asset.discovery.method` | string | Discovery method |

#### `aitf.asset.discovery.shadow_assets`

**Type:** Counter
**Unit:** `{asset}`
**Description:** Cumulative shadow (unregistered) AI assets detected.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.discovery.scope` | string | Discovery scope |

#### `aitf.asset.audit.runs`

**Type:** Counter
**Unit:** `{audit}`
**Description:** Total asset audits performed.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.audit.type` | string | Audit type |
| `aitf.asset.audit.result` | string | Audit result |
| `aitf.asset.audit.framework` | string | Compliance framework |

#### `aitf.asset.audit.risk_score`

**Type:** Histogram
**Unit:** `1`
**Description:** Distribution of asset audit risk scores (0-100).

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.type` | string | Asset type |
| `aitf.asset.audit.framework` | string | Framework |

#### `aitf.asset.classification.changes`

**Type:** Counter
**Unit:** `{change}`
**Description:** Total risk classification changes.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.risk_classification` | string | New classification |
| `aitf.asset.classification.previous` | string | Previous classification |
| `aitf.asset.classification.framework` | string | Framework |

#### `aitf.asset.audit.overdue`

**Type:** Gauge
**Unit:** `{asset}`
**Description:** Number of assets with overdue audits.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.type` | string | Asset type |
| `aitf.asset.deployment_environment` | string | Environment |

---

### Drift Detection Metrics

#### `aitf.drift.detections`

**Type:** Counter
**Unit:** `{detection}`
**Description:** Total drift detections performed.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.type` | string | Drift type |
| `aitf.drift.result` | string | Detection result |
| `aitf.drift.detection_method` | string | Statistical method |

#### `aitf.drift.score`

**Type:** Histogram
**Unit:** `1`
**Description:** Distribution of drift scores (0.0–1.0).

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.model_id` | string | Monitored model |
| `aitf.drift.type` | string | Drift type |

#### `aitf.drift.alerts`

**Type:** Counter
**Unit:** `{alert}`
**Description:** Total drift alerts triggered (warning, alert, critical).

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.type` | string | Drift type |
| `aitf.drift.result` | string | Alert level |
| `aitf.drift.action_triggered` | string | Action taken |

#### `aitf.drift.remediations`

**Type:** Counter
**Unit:** `{remediation}`
**Description:** Total drift remediation actions taken.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.remediation.action` | string | Action type |
| `aitf.drift.remediation.automated` | boolean | Was automated |
| `aitf.drift.remediation.status` | string | Outcome |

#### `aitf.drift.time_to_detect`

**Type:** Histogram
**Unit:** `s`
**Description:** Time from drift onset to detection.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.type` | string | Drift type |
| `aitf.drift.detection_method` | string | Detection method |

#### `aitf.drift.time_to_remediate`

**Type:** Histogram
**Unit:** `s`
**Description:** Time from drift detection to completed remediation.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.remediation.action` | string | Remediation action |
| `aitf.drift.remediation.automated` | boolean | Was automated |

---

### Memory Security Metrics

#### `aitf.memory.security.mutations`

**Type:** Counter
**Unit:** `{mutation}`
**Description:** Total memory mutations tracked.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.memory.operation` | string | Operation type |
| `aitf.memory.store` | string | Memory store |

#### `aitf.memory.security.poisoning_detections`

**Type:** Counter
**Unit:** `{detection}`
**Description:** Total memory poisoning detections.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.memory.store` | string | Memory store |
| `aitf.memory.provenance` | string | Content provenance |

#### `aitf.memory.security.integrity_violations`

**Type:** Counter
**Unit:** `{violation}`
**Description:** Total memory integrity hash mismatches.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.memory.store` | string | Memory store |

#### `aitf.memory.security.cross_session_accesses`

**Type:** Counter
**Unit:** `{access}`
**Description:** Total cross-session memory access attempts.

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.memory.store` | string | Memory store |

#### `aitf.memory.security.session_size`

**Type:** Histogram
**Unit:** `By`
**Description:** Memory size per session (bytes).

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.memory.store` | string | Memory store |
