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
