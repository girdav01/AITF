# OCSF Category 7: AI Event Classes

AITF defines OCSF event classes for AI-specific security telemetry, enabling SIEM/XDR integration.

## Overview

These event classes extend the OCSF v1.1.0 specification with a new Category 7 for AI systems. They enable AI telemetry to be consumed by security tools using the same schema as traditional security events.

### Category: AI System Activity (7)

| Class UID | Event Class | Description |
|-----------|-------------|-------------|
| 7001 | AI Model Inference | LLM/model inference requests and responses |
| 7002 | AI Agent Activity | Agent lifecycle, reasoning, delegation |
| 7003 | AI Tool Execution | Tool/function calls including MCP tools |
| 7004 | AI Data Retrieval | RAG, vector search, knowledge retrieval |
| 7005 | AI Security Finding | Security events, guardrails, policy violations |
| 7006 | AI Supply Chain | Model provenance, AI-BOM, integrity |
| 7007 | AI Governance | Compliance, audit, regulatory events |
| 7008 | AI Identity | Agent identity, credential delegation, auth |

---

## Class 7001: AI Model Inference

Represents an AI model inference operation (request + response).

### Type UIDs

| type_uid | Activity | Description |
|----------|----------|-------------|
| 700101 | Chat Completion | Chat/conversation inference |
| 700102 | Text Completion | Text generation inference |
| 700103 | Embedding Generation | Embedding computation |
| 700104 | Image Generation | Image creation/editing |
| 700105 | Audio Processing | Speech-to-text, text-to-speech |
| 700199 | Other | Other inference types |

### Fields

| Field | Type | Requirement | Description |
|-------|------|-------------|-------------|
| `model` | `AIModelInfo` | Required | Model information |
| `token_usage` | `AITokenUsage` | Required | Token usage statistics |
| `latency` | `AILatencyMetrics` | Recommended | Latency metrics |
| `request_content` | string | Optional | Request content (may be redacted) |
| `response_content` | string | Optional | Response content (may be redacted) |
| `streaming` | boolean | Optional | Whether streaming mode |
| `tools_provided` | int | Optional | Number of tools provided |
| `finish_reason` | string | Required | Completion finish reason |
| `cost` | `AICostInfo` | Recommended | Cost information |
| `error` | `AIErrorInfo` | Conditional | Error details (if failed) |

### Objects

#### `AIModelInfo`

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Model identifier |
| `name` | string | Model display name |
| `version` | string | Model version |
| `provider` | string | Model provider (openai, anthropic, etc.) |
| `type` | string | Model type (llm, embedding, image, audio) |
| `parameters` | object | Model parameters (temperature, etc.) |

#### `AITokenUsage`

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | int | Input/prompt tokens |
| `output_tokens` | int | Output/completion tokens |
| `total_tokens` | int | Total tokens |
| `cached_tokens` | int | Cached/prefix tokens |
| `reasoning_tokens` | int | Reasoning/thinking tokens |
| `estimated_cost_usd` | double | Estimated cost in USD |

#### `AILatencyMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `total_ms` | double | Total latency (ms) |
| `time_to_first_token_ms` | double | Time to first token (ms) |
| `tokens_per_second` | double | Token generation rate |
| `queue_time_ms` | double | Time in queue (ms) |
| `inference_time_ms` | double | Inference processing time (ms) |

#### `AICostInfo`

| Field | Type | Description |
|-------|------|-------------|
| `input_cost_usd` | double | Input token cost |
| `output_cost_usd` | double | Output token cost |
| `total_cost_usd` | double | Total cost |
| `currency` | string | Currency code |

---

## Class 7002: AI Agent Activity

Represents an AI agent lifecycle event.

### Type UIDs

| type_uid | Activity | Description |
|----------|----------|-------------|
| 700201 | Session Start | Agent session begins |
| 700202 | Session End | Agent session ends |
| 700203 | Step Execute | Agent executes a step |
| 700204 | Delegation | Agent delegates to another |
| 700205 | Memory Access | Agent accesses memory |
| 700206 | Error Recovery | Agent recovers from error |
| 700207 | Human Approval | Agent requests human input |
| 700299 | Other | Other agent activity |

### Fields

| Field | Type | Requirement | Description |
|-------|------|-------------|-------------|
| `agent_name` | string | Required | Agent name |
| `agent_id` | string | Required | Agent instance ID |
| `agent_type` | string | Required | Agent type |
| `framework` | string | Recommended | Agent framework |
| `session_id` | string | Required | Session ID |
| `step_type` | string | Conditional | Step type (for step events) |
| `step_index` | int | Conditional | Step index |
| `thought` | string | Optional | Agent reasoning |
| `action` | string | Optional | Planned/executed action |
| `observation` | string | Optional | Observation result |
| `delegation_target` | string | Conditional | Target agent (for delegation) |
| `team_info` | `AITeamInfo` | Optional | Team information |

#### `AITeamInfo`

| Field | Type | Description |
|-------|------|-------------|
| `team_name` | string | Team name |
| `team_id` | string | Team ID |
| `topology` | string | Team topology |
| `members` | string[] | Member agent names |
| `coordinator` | string | Coordinator agent |

---

## Class 7003: AI Tool Execution

Represents a tool/function execution, including MCP tools and skills.

### Type UIDs

| type_uid | Activity | Description |
|----------|----------|-------------|
| 700301 | Function Call | LLM function/tool call |
| 700302 | MCP Tool Invoke | MCP tool invocation |
| 700303 | Skill Invoke | Skill invocation |
| 700304 | API Call | External API call |
| 700305 | Code Execute | Code execution |
| 700399 | Other | Other tool execution |

### Fields

| Field | Type | Requirement | Description |
|-------|------|-------------|-------------|
| `tool_name` | string | Required | Tool/function name |
| `tool_type` | string | Required | `"function"`, `"mcp_tool"`, `"skill"`, `"api"` |
| `tool_input` | string | Recommended | Input parameters (JSON) |
| `tool_output` | string | Optional | Output (may be redacted) |
| `is_error` | boolean | Required | Whether tool returned error |
| `duration_ms` | double | Recommended | Execution duration |
| `mcp_server` | string | Conditional | MCP server (for MCP tools) |
| `mcp_transport` | string | Conditional | MCP transport type |
| `skill_category` | string | Conditional | Skill category |
| `skill_version` | string | Conditional | Skill version |
| `approval_required` | boolean | Optional | Whether human approval needed |
| `approved` | boolean | Conditional | Whether approved |

---

## Class 7004: AI Data Retrieval

Represents RAG and vector search operations.

### Type UIDs

| type_uid | Activity | Description |
|----------|----------|-------------|
| 700401 | Vector Search | Vector similarity search |
| 700402 | Document Retrieval | Document retrieval |
| 700403 | Knowledge Graph Query | Knowledge graph query |
| 700404 | Embedding Generation | Embedding creation for retrieval |
| 700405 | Reranking | Result reranking |
| 700499 | Other | Other retrieval |

### Fields

| Field | Type | Requirement | Description |
|-------|------|-------------|-------------|
| `database_name` | string | Required | Database/index name |
| `database_type` | string | Required | `"pinecone"`, `"chromadb"`, `"weaviate"`, etc. |
| `query` | string | Optional | Query text (may be redacted) |
| `top_k` | int | Recommended | Results requested |
| `results_count` | int | Required | Results returned |
| `min_score` | double | Recommended | Minimum similarity score |
| `max_score` | double | Recommended | Maximum similarity score |
| `filter` | string | Optional | Metadata filter (JSON) |
| `embedding_model` | string | Recommended | Embedding model used |
| `embedding_dimensions` | int | Optional | Embedding dimensions |
| `pipeline_name` | string | Optional | RAG pipeline name |
| `pipeline_stage` | string | Optional | Pipeline stage |
| `quality_scores` | `AIQualityScores` | Optional | Quality metrics |

---

## Class 7005: AI Security Finding

Represents a security finding in AI operations.

### Type UIDs

| type_uid | Activity | Description |
|----------|----------|-------------|
| 700501 | Threat Detection | Threat detected |
| 700502 | PII Detection | PII found in content |
| 700503 | Guardrail Trigger | Guardrail check result |
| 700504 | Policy Violation | Policy violation |
| 700505 | Anomaly Detection | Anomalous behavior |
| 700599 | Other | Other security finding |

### Fields

| Field | Type | Requirement | Description |
|-------|------|-------------|-------------|
| `finding_type` | string | Required | Type of finding |
| `owasp_category` | string | Conditional | OWASP LLM category |
| `risk_level` | string | Required | `"critical"`, `"high"`, `"medium"`, `"low"`, `"info"` |
| `risk_score` | double | Required | Risk score (0-100) |
| `confidence` | double | Required | Detection confidence (0-1) |
| `detection_method` | string | Required | `"pattern"`, `"ml_model"`, `"guardrail"`, `"policy"` |
| `blocked` | boolean | Required | Whether action was blocked |
| `details` | string | Recommended | Finding details (JSON) |
| `remediation` | string | Optional | Suggested remediation |
| `guardrail_name` | string | Conditional | Guardrail name |
| `guardrail_provider` | string | Conditional | Guardrail provider |
| `pii_types` | string[] | Conditional | PII types found |
| `pii_count` | int | Conditional | PII instance count |
| `matched_patterns` | string[] | Optional | Matched detection patterns |

---

## Class 7006: AI Supply Chain

Represents AI supply chain events (model provenance, integrity).

### Type UIDs

| type_uid | Activity | Description |
|----------|----------|-------------|
| 700601 | Model Registration | New model registered |
| 700602 | Model Verification | Model integrity check |
| 700603 | AI-BOM Generation | AI Bill of Materials created |
| 700604 | Dependency Check | Dependency security check |
| 700699 | Other | Other supply chain event |

### Fields

| Field | Type | Requirement | Description |
|-------|------|-------------|-------------|
| `model_source` | string | Required | Model source |
| `model_hash` | string | Recommended | Model hash |
| `model_license` | string | Recommended | Model license |
| `model_signed` | boolean | Recommended | Whether signed |
| `model_signer` | string | Conditional | Signer identity |
| `verification_result` | string | Conditional | `"pass"`, `"fail"`, `"unknown"` |
| `ai_bom_id` | string | Conditional | AI-BOM ID |
| `ai_bom_components` | string | Conditional | Components (JSON) |

---

## Class 7007: AI Governance

Represents compliance and governance events.

### Type UIDs

| type_uid | Activity | Description |
|----------|----------|-------------|
| 700701 | Compliance Check | Compliance verification |
| 700702 | Audit Record | Audit trail entry |
| 700703 | Policy Evaluation | Policy evaluation |
| 700704 | Risk Assessment | Risk assessment |
| 700799 | Other | Other governance event |

### Fields

| Field | Type | Requirement | Description |
|-------|------|-------------|-------------|
| `frameworks` | string[] | Required | Compliance frameworks |
| `controls` | string | Required | Mapped controls (JSON) |
| `event_type` | string | Required | Source event type |
| `violation_detected` | boolean | Required | Whether violation found |
| `severity` | string | Conditional | Violation severity |
| `remediation` | string | Optional | Suggested remediation |
| `audit_id` | string | Recommended | Audit record ID |

---

## Class 7008: AI Identity

Represents agent identity and authentication events.

### Type UIDs

| type_uid | Activity | Description |
|----------|----------|-------------|
| 700801 | Agent Authentication | Agent authenticates |
| 700802 | Credential Delegation | Credentials delegated |
| 700803 | Permission Check | Permission verified |
| 700804 | Token Issuance | Auth token issued |
| 700899 | Other | Other identity event |

### Fields

| Field | Type | Requirement | Description |
|-------|------|-------------|-------------|
| `agent_name` | string | Required | Agent name |
| `agent_id` | string | Required | Agent ID |
| `auth_method` | string | Required | `"api_key"`, `"oauth"`, `"mtls"`, `"jwt"` |
| `auth_result` | string | Required | `"success"`, `"failure"`, `"denied"` |
| `permissions` | string[] | Recommended | Granted permissions |
| `credential_type` | string | Conditional | Delegated credential type |
| `delegation_chain` | string[] | Conditional | Chain of delegation |
| `scope` | string | Optional | Auth scope |

---

## Common OCSF Fields

All AI event classes inherit standard OCSF base fields:

| Field | Type | Description |
|-------|------|-------------|
| `activity_id` | int | Activity type identifier |
| `category_uid` | int | Category UID (7 for AI) |
| `class_uid` | int | Event class UID |
| `type_uid` | int | `class_uid * 100 + activity_id` |
| `time` | timestamp | Event timestamp |
| `severity_id` | int | 0-6 (Unknown to Fatal) |
| `status_id` | int | 0-99 (Unknown, Success, Failure, Other) |
| `message` | string | Human-readable description |
| `metadata` | `OCSFMetadata` | Event metadata |
| `actor` | `OCSFActor` | Actor information |
| `device` | `OCSFDevice` | Device/host information |
| `observables` | `OCSFObservable[]` | Observable values |
| `enrichments` | `OCSFEnrichment[]` | Enrichment data |
| `compliance` | `ComplianceMetadata` | Compliance mappings |
