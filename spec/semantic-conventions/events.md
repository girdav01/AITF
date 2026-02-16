# AITF Event Conventions

AITF defines events for security findings, compliance actions, and other discrete occurrences that need to be captured alongside trace spans.

## Security Events

### `aitf.security.threat_detected`

Emitted when a security threat is detected in AI input or output.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.security.threat_type` | string | Type of threat | Yes |
| `aitf.security.owasp_category` | string | OWASP LLM category | Yes |
| `aitf.security.risk_level` | string | Risk level | Yes |
| `aitf.security.risk_score` | double | Risk score (0-100) | Yes |
| `aitf.security.confidence` | double | Detection confidence (0-1) | Yes |
| `aitf.security.detection_method` | string | How detected | Recommended |
| `aitf.security.blocked` | boolean | Whether blocked | Recommended |
| `aitf.security.details` | string | Threat details (JSON) | Recommended |

#### Threat Types

| Value | OWASP | Description |
|-------|-------|-------------|
| `prompt_injection` | LLM01 | Direct or indirect prompt injection |
| `sensitive_data_exposure` | LLM02 | Sensitive information disclosure |
| `supply_chain` | LLM03 | Compromised training data or models |
| `data_poisoning` | LLM04 | Data poisoning attempts |
| `improper_output` | LLM05 | Improper output handling |
| `excessive_agency` | LLM06 | Excessive autonomy or permissions |
| `system_prompt_leak` | LLM07 | System prompt leakage |
| `vector_data_weakness` | LLM08 | Vector/embedding weaknesses |
| `misinformation` | LLM09 | Generated misinformation |
| `unbounded_consumption` | LLM10 | Resource exhaustion / DoS |
| `jailbreak` | LLM01 | Jailbreak attempt |
| `data_exfiltration` | LLM02 | Data exfiltration attempt |
| `model_theft` | LLM03 | Model extraction attempt |

### `aitf.security.pii_detected`

Emitted when PII is detected in content.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.security.pii.types` | string[] | Types of PII found | Yes |
| `aitf.security.pii.count` | int | Number of instances | Yes |
| `aitf.security.pii.action` | string | Action taken | Yes |
| `aitf.security.pii.location` | string | `"input"`, `"output"`, `"tool_result"` | Recommended |

#### PII Types

| Value | Description |
|-------|-------------|
| `email` | Email addresses |
| `phone` | Phone numbers |
| `ssn` | Social Security Numbers |
| `credit_card` | Credit card numbers |
| `api_key` | API keys and tokens |
| `password` | Passwords |
| `address` | Physical addresses |
| `ip_address` | IP addresses |
| `name` | Personal names |
| `dob` | Dates of birth |
| `passport` | Passport numbers |
| `driver_license` | Driver's license numbers |
| `jwt` | JWT tokens |

### `aitf.security.guardrail_triggered`

Emitted when a guardrail check produces a result.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.security.guardrail.name` | string | Guardrail name | Yes |
| `aitf.security.guardrail.type` | string | `"input"`, `"output"`, `"both"` | Yes |
| `aitf.security.guardrail.result` | string | `"pass"`, `"fail"`, `"warn"` | Yes |
| `aitf.security.guardrail.provider` | string | Guardrail provider | Recommended |
| `aitf.security.guardrail.policy` | string | Policy name | Recommended |
| `aitf.security.guardrail.details` | string | Details (JSON) | Recommended |

---

## Compliance Events

### `aitf.compliance.control_mapped`

Emitted when an AI event is mapped to compliance controls.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.compliance.frameworks` | string[] | Mapped frameworks | Yes |
| `aitf.compliance.event_type` | string | Type of AI event | Yes |
| `aitf.compliance.controls` | string | All controls (JSON) | Recommended |

### `aitf.compliance.violation_detected`

Emitted when a potential compliance violation is detected.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.compliance.framework` | string | Framework | Yes |
| `aitf.compliance.control` | string | Violated control | Yes |
| `aitf.compliance.severity` | string | Violation severity | Yes |
| `aitf.compliance.details` | string | Violation details | Recommended |
| `aitf.compliance.remediation` | string | Suggested remediation | Recommended |

---

## Cost Events

### `aitf.cost.budget_warning`

Emitted when budget utilization reaches a threshold.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.cost.budget.limit` | double | Budget limit (USD) | Yes |
| `aitf.cost.budget.used` | double | Budget used (USD) | Yes |
| `aitf.cost.budget.remaining` | double | Budget remaining (USD) | Yes |
| `aitf.cost.budget.utilization_pct` | double | Utilization percentage | Yes |
| `aitf.cost.budget.threshold` | string | `"75%"`, `"90%"`, `"100%"` | Yes |
| `aitf.cost.attribution.project` | string | Project | Recommended |

### `aitf.cost.budget_exceeded`

Emitted when a budget limit is exceeded.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.cost.budget.limit` | double | Budget limit (USD) | Yes |
| `aitf.cost.budget.used` | double | Budget used (USD) | Yes |
| `aitf.cost.budget.overage` | double | Amount over budget (USD) | Yes |
| `aitf.cost.attribution.project` | string | Project | Recommended |
| `aitf.cost.action` | string | `"warn"`, `"throttle"`, `"block"` | Recommended |

---

## Quality Events

### `aitf.quality.low_confidence`

Emitted when model confidence is below threshold.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.quality.confidence` | double | Confidence score (0-1) | Yes |
| `aitf.quality.threshold` | double | Threshold value | Yes |
| `gen_ai.request.model` | string | Model ID | Recommended |

### `aitf.quality.hallucination_detected`

Emitted when potential hallucination is detected.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.quality.hallucination_score` | double | Score (0-1) | Yes |
| `aitf.quality.threshold` | double | Threshold value | Yes |
| `aitf.quality.details` | string | Details (JSON) | Recommended |

### `aitf.quality.user_feedback`

Emitted when user provides feedback.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.quality.feedback.rating` | double | Rating (1-5) | Conditional |
| `aitf.quality.feedback.thumbs` | string | `"up"`, `"down"` | Conditional |
| `aitf.quality.feedback.comment` | string | Free-text comment | Recommended |

---

## Agent Events

### `aitf.agent.error_recovery`

Emitted when an agent encounters and recovers from an error.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.agent.name` | string | Agent name | Yes |
| `aitf.agent.error.type` | string | Error type | Yes |
| `aitf.agent.error.message` | string | Error message | Yes |
| `aitf.agent.error.recovery_action` | string | Recovery action taken | Recommended |
| `aitf.agent.error.retry_count` | int | Retry count | Recommended |

### `aitf.agent.human_approval`

Emitted when agent requests/receives human approval.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.agent.name` | string | Agent name | Yes |
| `aitf.agent.approval.action` | string | Action requiring approval | Yes |
| `aitf.agent.approval.status` | string | `"requested"`, `"approved"`, `"denied"` | Yes |
| `aitf.agent.approval.approver` | string | Approver identity | Recommended |
| `aitf.agent.approval.reason` | string | Reason for decision | Recommended |

---

## Model Operations (LLMOps/MLOps) Events

### `aitf.model_ops.training_completed`

Emitted when a training or fine-tuning run completes.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.model_ops.training.run_id` | string | Training run ID | Yes |
| `aitf.model_ops.training.type` | string | Training type | Yes |
| `aitf.model_ops.training.base_model` | string | Base model | Yes |
| `aitf.model_ops.training.status` | string | `"completed"`, `"failed"`, `"cancelled"` | Yes |
| `aitf.model_ops.training.loss_final` | double | Final training loss | Recommended |
| `aitf.model_ops.training.output_model.id` | string | Output model ID | Recommended |
| `aitf.model_ops.training.compute.gpu_hours` | double | GPU hours consumed | Recommended |

### `aitf.model_ops.evaluation_completed`

Emitted when a model evaluation run completes.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.model_ops.evaluation.run_id` | string | Evaluation run ID | Yes |
| `aitf.model_ops.evaluation.model_id` | string | Model evaluated | Yes |
| `aitf.model_ops.evaluation.type` | string | Evaluation type | Yes |
| `aitf.model_ops.evaluation.pass` | boolean | Passed quality gates | Yes |
| `aitf.model_ops.evaluation.metrics` | string | JSON metric results | Recommended |
| `aitf.model_ops.evaluation.regression_detected` | boolean | Regression found | Recommended |

### `aitf.model_ops.model_promoted`

Emitted when a model is promoted through lifecycle stages.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.model_ops.registry.model_id` | string | Model ID | Yes |
| `aitf.model_ops.registry.stage` | string | New stage | Yes |
| `aitf.model_ops.registry.previous_stage` | string | Previous stage | Yes |
| `aitf.model_ops.registry.model_alias` | string | Alias assigned | Recommended |
| `aitf.model_ops.registry.approval.approver` | string | Approver | Recommended |

### `aitf.model_ops.deployment_completed`

Emitted when a model deployment completes (success or failure).

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.model_ops.deployment.id` | string | Deployment ID | Yes |
| `aitf.model_ops.deployment.model_id` | string | Model deployed | Yes |
| `aitf.model_ops.deployment.strategy` | string | Deployment strategy | Yes |
| `aitf.model_ops.deployment.status` | string | `"completed"`, `"failed"`, `"rolled_back"` | Yes |
| `aitf.model_ops.deployment.environment` | string | Target environment | Recommended |
| `aitf.model_ops.deployment.canary_percent` | double | Canary traffic % | Recommended |

### `aitf.model_ops.drift_detected`

Emitted when model monitoring detects drift above threshold.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.model_ops.monitoring.model_id` | string | Monitored model | Yes |
| `aitf.model_ops.monitoring.drift_type` | string | Type of drift | Yes |
| `aitf.model_ops.monitoring.drift_score` | double | Drift magnitude (0-1) | Yes |
| `aitf.model_ops.monitoring.result` | string | Alert level | Yes |
| `aitf.model_ops.monitoring.baseline_value` | double | Baseline value | Recommended |
| `aitf.model_ops.monitoring.metric_value` | double | Current value | Recommended |
| `aitf.model_ops.monitoring.action_triggered` | string | Automated action | Recommended |

### `aitf.model_ops.fallback_triggered`

Emitted when a model serving fallback occurs.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.model_ops.serving.fallback.original_model` | string | Original model | Yes |
| `aitf.model_ops.serving.fallback.final_model` | string | Fallback model used | Yes |
| `aitf.model_ops.serving.fallback.trigger` | string | Trigger reason | Yes |
| `aitf.model_ops.serving.fallback.depth` | int | Fallback depth | Recommended |
| `aitf.model_ops.serving.cache.cost_saved_usd` | double | Cost saved by cache | Recommended |

### `aitf.model_ops.prompt_promoted`

Emitted when a prompt version is promoted to a deployment label.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.model_ops.prompt.name` | string | Prompt name | Yes |
| `aitf.model_ops.prompt.version` | string | Promoted version | Yes |
| `aitf.model_ops.prompt.label` | string | Target label | Yes |
| `aitf.model_ops.prompt.previous_version` | string | Previous version at label | Recommended |
| `aitf.model_ops.prompt.evaluation.score` | double | Evaluation score | Recommended |

---

## Identity Events

### `aitf.identity.created`

Emitted when a new agent identity is created.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.agent_id` | string | Agent identity ID | Yes |
| `aitf.identity.agent_name` | string | Agent name | Yes |
| `aitf.identity.type` | string | Identity type | Yes |
| `aitf.identity.provider` | string | Identity provider | Recommended |
| `aitf.identity.owner` | string | Identity owner | Recommended |
| `aitf.identity.credential_type` | string | Credential type | Recommended |
| `aitf.identity.ttl_seconds` | int | TTL | Recommended |

### `aitf.identity.auth_failed`

Emitted when agent authentication fails.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.agent_id` | string | Agent identity ID | Yes |
| `aitf.identity.agent_name` | string | Agent name | Yes |
| `aitf.identity.auth.method` | string | Auth method used | Yes |
| `aitf.identity.auth.result` | string | Failure type | Yes |
| `aitf.identity.auth.failure_reason` | string | Failure reason | Yes |
| `aitf.identity.auth.target_service` | string | Target service | Recommended |

### `aitf.identity.authz_denied`

Emitted when an authorization request is denied.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.agent_id` | string | Agent identity ID | Yes |
| `aitf.identity.agent_name` | string | Agent name | Yes |
| `aitf.identity.authz.resource` | string | Resource requested | Yes |
| `aitf.identity.authz.action` | string | Action requested | Yes |
| `aitf.identity.authz.deny_reason` | string | Denial reason | Yes |
| `aitf.identity.authz.policy_id` | string | Policy that denied | Recommended |

### `aitf.identity.delegation_created`

Emitted when credentials are delegated between agents.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.delegation.delegator` | string | Delegating agent | Yes |
| `aitf.identity.delegation.delegatee` | string | Receiving agent | Yes |
| `aitf.identity.delegation.type` | string | Delegation type | Yes |
| `aitf.identity.delegation.scope_delegated` | string[] | Delegated scopes | Yes |
| `aitf.identity.delegation.chain_depth` | int | Chain depth | Recommended |
| `aitf.identity.delegation.scope_attenuated` | boolean | Scope was reduced | Recommended |
| `aitf.identity.delegation.ttl_seconds` | int | Delegation TTL | Recommended |

### `aitf.identity.privilege_escalation`

Emitted when potential privilege escalation is detected — an agent attempts to access resources beyond its delegated scope.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.agent_id` | string | Agent identity ID | Yes |
| `aitf.identity.agent_name` | string | Agent name | Yes |
| `aitf.identity.authz.resource` | string | Resource attempted | Yes |
| `aitf.identity.authz.action` | string | Action attempted | Yes |
| `aitf.identity.authz.scope_required` | string[] | Scopes required | Yes |
| `aitf.identity.authz.scope_present` | string[] | Scopes present | Yes |
| `aitf.identity.delegation.chain` | string[] | Delegation chain | Recommended |

### `aitf.identity.credential_rotated`

Emitted when agent credentials are rotated.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.agent_id` | string | Agent identity ID | Yes |
| `aitf.identity.credential_type` | string | Credential type | Yes |
| `aitf.identity.auto_rotate` | boolean | Was auto-rotation | Yes |
| `aitf.identity.expires_at` | string | New expiration | Recommended |

### `aitf.identity.revoked`

Emitted when an agent identity is revoked.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.agent_id` | string | Agent identity ID | Yes |
| `aitf.identity.agent_name` | string | Agent name | Yes |
| `aitf.identity.status` | string | New status (`"revoked"`) | Yes |
| `aitf.identity.previous_status` | string | Previous status | Yes |
| `aitf.identity.lifecycle.operation` | string | `"revoke"` | Yes |

### `aitf.identity.trust_established`

Emitted when trust is established between two agents.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.agent_name` | string | This agent | Yes |
| `aitf.identity.trust.peer_agent` | string | Peer agent | Yes |
| `aitf.identity.trust.method` | string | Trust method | Yes |
| `aitf.identity.trust.result` | string | Trust result | Yes |
| `aitf.identity.trust.trust_level` | string | Trust level | Recommended |
| `aitf.identity.trust.cross_domain` | boolean | Cross-domain | Recommended |

### `aitf.identity.session_hijack_detected`

Emitted when potential session hijacking is detected.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.identity.agent_id` | string | Agent identity ID | Yes |
| `aitf.identity.session.id` | string | Session ID | Yes |
| `aitf.identity.session.ip_address` | string | Anomalous source IP | Recommended |
| `aitf.identity.session.user_agent` | string | Anomalous user agent | Recommended |

---

## Asset Inventory Events

### `aitf.asset.registered`

Emitted when a new AI asset is registered in the inventory.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.asset.id` | string | Asset identifier | Yes |
| `aitf.asset.name` | string | Asset name | Yes |
| `aitf.asset.type` | string | Asset type | Yes |
| `aitf.asset.owner` | string | Asset owner | Yes |
| `aitf.asset.deployment_environment` | string | Environment | Recommended |
| `aitf.asset.risk_classification` | string | Risk classification | Recommended |

### `aitf.asset.shadow_detected`

Emitted when a shadow (unregistered) AI asset is discovered during a scan.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.asset.type` | string | Discovered asset type | Yes |
| `aitf.asset.discovery.scope` | string | Discovery scope | Yes |
| `aitf.asset.discovery.method` | string | Discovery method | Yes |
| `aitf.asset.name` | string | Discovered asset name | Recommended |
| `aitf.asset.deployment_environment` | string | Where found | Recommended |

### `aitf.asset.audit_failed`

Emitted when an AI asset fails a compliance or integrity audit.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.asset.id` | string | Asset identifier | Yes |
| `aitf.asset.audit.type` | string | Audit type | Yes |
| `aitf.asset.audit.result` | string | `"fail"` | Yes |
| `aitf.asset.audit.framework` | string | Framework | Recommended |
| `aitf.asset.audit.findings` | string | JSON findings | Recommended |
| `aitf.asset.audit.risk_score` | double | Risk score | Recommended |

### `aitf.asset.risk_reclassified`

Emitted when an asset's risk classification changes.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.asset.id` | string | Asset identifier | Yes |
| `aitf.asset.risk_classification` | string | New classification | Yes |
| `aitf.asset.classification.previous` | string | Previous classification | Yes |
| `aitf.asset.classification.framework` | string | Framework | Yes |
| `aitf.asset.classification.reason` | string | Reason for change | Recommended |

### `aitf.asset.decommissioned`

Emitted when an AI asset is decommissioned.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.asset.id` | string | Asset identifier | Yes |
| `aitf.asset.type` | string | Asset type | Yes |
| `aitf.asset.decommission.reason` | string | Decommission reason | Yes |
| `aitf.asset.decommission.replacement_id` | string | Replacement asset | Recommended |

### `aitf.asset.audit_overdue`

Emitted when an asset's audit is overdue.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.asset.id` | string | Asset identifier | Yes |
| `aitf.asset.audit.next_audit_due` | string | Overdue audit date | Yes |
| `aitf.asset.risk_classification` | string | Risk level | Recommended |
| `aitf.asset.deployment_environment` | string | Environment | Recommended |

---

## Drift Detection Events

### `aitf.drift.detected`

Emitted when model drift is detected above threshold. Provides structured forensic-quality drift analysis beyond the basic `aitf.model_ops.drift_detected`.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.drift.model_id` | string | Monitored model | Yes |
| `aitf.drift.type` | string | Drift type | Yes |
| `aitf.drift.score` | double | Drift magnitude (0–1) | Yes |
| `aitf.drift.result` | string | Alert level | Yes |
| `aitf.drift.detection_method` | string | Statistical method | Yes |
| `aitf.drift.baseline_metric` | double | Baseline value | Recommended |
| `aitf.drift.current_metric` | double | Current value | Recommended |
| `aitf.drift.p_value` | double | Statistical significance | Recommended |
| `aitf.drift.affected_segments` | string[] | Impacted segments | Recommended |
| `aitf.drift.reference_dataset` | string | Reference dataset | Recommended |
| `aitf.drift.action_triggered` | string | Automated action | Recommended |

### `aitf.drift.baseline_updated`

Emitted when a drift baseline is created or refreshed.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.drift.model_id` | string | Model | Yes |
| `aitf.drift.baseline.operation` | string | `"create"` or `"refresh"` | Yes |
| `aitf.drift.baseline.id` | string | Baseline identifier | Yes |
| `aitf.drift.baseline.dataset` | string | Baseline dataset | Recommended |
| `aitf.drift.baseline.sample_size` | int | Sample size | Recommended |

### `aitf.drift.investigation_completed`

Emitted when a drift investigation completes with root cause analysis.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.drift.model_id` | string | Model investigated | Yes |
| `aitf.drift.investigation.root_cause_category` | string | Root cause category | Yes |
| `aitf.drift.investigation.severity` | string | Severity | Yes |
| `aitf.drift.investigation.blast_radius` | string | Impact scope | Yes |
| `aitf.drift.investigation.affected_users_estimate` | int | Affected users | Recommended |
| `aitf.drift.investigation.recommendation` | string | Recommendation | Recommended |

### `aitf.drift.remediation_completed`

Emitted when a drift remediation action completes.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.drift.model_id` | string | Model remediated | Yes |
| `aitf.drift.remediation.action` | string | Action taken | Yes |
| `aitf.drift.remediation.status` | string | Outcome status | Yes |
| `aitf.drift.remediation.automated` | boolean | Was automated | Yes |
| `aitf.drift.remediation.validation_passed` | boolean | Post-validation passed | Recommended |

---

## Memory Security Events

### `aitf.memory.poisoning_detected`

Emitted when memory poisoning is detected (unexpected content injection).

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.memory.key` | string | Affected memory key | Yes |
| `aitf.memory.store` | string | Memory store | Yes |
| `aitf.memory.security.poisoning_score` | double | Poisoning score (0-1) | Yes |
| `aitf.memory.provenance` | string | Content provenance | Yes |
| `aitf.agent.session.id` | string | Session ID | Recommended |

### `aitf.memory.integrity_violation`

Emitted when memory content hash does not match expected integrity hash.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.memory.key` | string | Affected memory key | Yes |
| `aitf.memory.store` | string | Memory store | Yes |
| `aitf.memory.security.integrity_hash` | string | Expected hash | Yes |
| `aitf.memory.security.content_hash` | string | Actual hash | Yes |
| `aitf.agent.session.id` | string | Session ID | Recommended |

### `aitf.memory.cross_session_access`

Emitted when a session accesses memory belonging to another session.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.memory.key` | string | Accessed memory key | Yes |
| `aitf.memory.store` | string | Memory store | Yes |
| `aitf.agent.session.id` | string | Accessing session | Yes |

### `aitf.memory.growth_anomaly`

Emitted when session memory growth exceeds configured thresholds.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.agent.session.id` | string | Session ID | Yes |
| `aitf.memory.security.mutation_count` | int | Current entry count | Yes |
| `aitf.memory.security.content_size` | int | Current total size | Recommended |

### `aitf.memory.untrusted_provenance`

Emitted when memory is written from an untrusted provenance source.

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `aitf.memory.key` | string | Memory key | Yes |
| `aitf.memory.store` | string | Memory store | Yes |
| `aitf.memory.provenance` | string | Untrusted provenance | Yes |
| `aitf.agent.session.id` | string | Session ID | Recommended |
