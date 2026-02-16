# AI Asset Inventory Span Conventions

AITF defines semantic conventions for AI asset inventory management — covering the registration, discovery, audit, and risk classification of all AI system components. These conventions align with CoSAI's AI Incident Response requirement for maintaining complete inventories of models, datasets, prompts, and infrastructure dependencies.

## Overview

The `aitf.asset.*` namespace covers the complete AI asset lifecycle:

| Stage | Span Name | Description |
|-------|-----------|-------------|
| Registration | `aitf.asset.register` | Asset registration into inventory |
| Discovery | `aitf.asset.discover` | Automated asset discovery and scanning |
| Audit | `aitf.asset.audit` | Periodic audit and compliance verification |
| Risk Classification | `aitf.asset.classify` | Risk classification (EU AI Act, internal policy) |
| Dependency Mapping | `aitf.asset.dependency` | Dependency graph resolution |
| Decommission | `aitf.asset.decommission` | Asset retirement and decommissioning |

---

## Span: `aitf.asset.register`

Represents the registration of an AI asset (model, dataset, prompt template, vector DB, MCP server, agent) into the organization's AI asset inventory.

### Span Name

Format: `asset.register {aitf.asset.type} {aitf.asset.name}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.id` | string | Unique asset identifier |
| `aitf.asset.name` | string | Human-readable asset name |
| `aitf.asset.type` | string | Asset type (see below) |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.version` | string | Asset version (SemVer or hash) |
| `aitf.asset.hash` | string | Content hash for integrity verification |
| `aitf.asset.owner` | string | Asset owner (team or individual) |
| `aitf.asset.owner_type` | string | `"team"`, `"individual"`, `"organization"` |
| `aitf.asset.deployment_environment` | string | Deployment environment |
| `aitf.asset.risk_classification` | string | Risk classification |
| `aitf.asset.description` | string | Asset description |
| `aitf.asset.tags` | string[] | Searchable tags |
| `aitf.asset.source_repository` | string | Source code/model repository URL |
| `aitf.asset.created_at` | string | Creation timestamp (ISO 8601) |

### Asset Types

| Value | Description |
|-------|-------------|
| `model` | ML/LLM model artifact |
| `dataset` | Training, evaluation, or reference dataset |
| `prompt_template` | Prompt template or system prompt |
| `vector_db` | Vector database or embedding index |
| `mcp_server` | MCP server or tool provider |
| `agent` | Autonomous AI agent |
| `pipeline` | ML/data pipeline definition |
| `guardrail` | Safety guardrail configuration |
| `embedding_model` | Embedding model (distinct from inference model) |
| `knowledge_base` | RAG knowledge base or document collection |

### Deployment Environments

| Value | Description |
|-------|-------------|
| `production` | Production environment |
| `staging` | Pre-production staging |
| `development` | Development/testing |
| `shadow` | Shadow/dark-launch environment |

### Risk Classifications (EU AI Act aligned)

| Value | Description |
|-------|-------------|
| `unacceptable` | Unacceptable risk — prohibited under EU AI Act |
| `high_risk` | High risk — subject to conformity assessment |
| `limited_risk` | Limited risk — transparency obligations |
| `minimal_risk` | Minimal risk — no specific obligations |
| `systemic` | Systemic risk — GPAI models with systemic risk |
| `not_classified` | Not yet classified |

---

## Span: `aitf.asset.discover`

Represents automated asset discovery — scanning infrastructure for unregistered or shadow AI assets.

### Span Name

Format: `asset.discover {aitf.asset.discovery.scope}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.discovery.scope` | string | Discovery scope (`"cluster"`, `"namespace"`, `"environment"`, `"organization"`) |
| `aitf.asset.discovery.method` | string | `"api_scan"`, `"network_scan"`, `"registry_sync"`, `"log_analysis"`, `"manual"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.discovery.assets_found` | int | Number of assets discovered |
| `aitf.asset.discovery.new_assets` | int | Number of previously unknown assets |
| `aitf.asset.discovery.shadow_assets` | int | Number of unregistered (shadow) AI assets |
| `aitf.asset.discovery.status` | string | `"completed"`, `"partial"`, `"failed"` |

---

## Span: `aitf.asset.audit`

Represents a periodic audit of an AI asset — verifying integrity, compliance, and operational status.

### Span Name

Format: `asset.audit {aitf.asset.type} {aitf.asset.id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.id` | string | Asset being audited |
| `aitf.asset.audit.type` | string | Audit type (see below) |
| `aitf.asset.audit.result` | string | `"pass"`, `"fail"`, `"warning"`, `"not_applicable"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.audit.auditor` | string | Auditor identity (human or automated) |
| `aitf.asset.audit.framework` | string | Compliance framework (`"eu_ai_act"`, `"nist_ai_rmf"`, `"iso_42001"`, `"soc2"`) |
| `aitf.asset.audit.findings` | string | JSON array of audit findings |
| `aitf.asset.audit.last_audit_time` | string | Previous audit timestamp |
| `aitf.asset.audit.next_audit_due` | string | Next scheduled audit timestamp |
| `aitf.asset.audit.risk_score` | double | Calculated risk score (0-100) |
| `aitf.asset.audit.integrity_verified` | boolean | Whether integrity hash matches |
| `aitf.asset.audit.compliance_status` | string | `"compliant"`, `"non_compliant"`, `"partially_compliant"` |

### Audit Types

| Value | Description |
|-------|-------------|
| `integrity` | Hash/signature verification |
| `compliance` | Regulatory compliance check |
| `access_review` | Access control and permission audit |
| `drift` | Model drift and performance audit |
| `security` | Security vulnerability assessment |
| `lineage` | Data/model lineage verification |
| `full` | Comprehensive full audit |

---

## Span: `aitf.asset.classify`

Represents risk classification of an AI asset — assigning regulatory risk levels per EU AI Act, internal policy, or other frameworks.

### Span Name

Format: `asset.classify {aitf.asset.id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.id` | string | Asset being classified |
| `aitf.asset.risk_classification` | string | Assigned risk level (see Risk Classifications above) |
| `aitf.asset.classification.framework` | string | Framework used for classification |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.classification.previous` | string | Previous risk classification |
| `aitf.asset.classification.reason` | string | Reason for classification |
| `aitf.asset.classification.assessor` | string | Person/system performing classification |
| `aitf.asset.classification.use_case` | string | Intended use case (affects risk level) |
| `aitf.asset.classification.affected_persons` | string | `"employees"`, `"consumers"`, `"public"`, `"children"` |
| `aitf.asset.classification.sector` | string | Deployment sector (affects risk — healthcare, finance, etc.) |
| `aitf.asset.classification.biometric` | boolean | Uses biometric data |
| `aitf.asset.classification.autonomous_decision` | boolean | Makes autonomous decisions affecting rights |

---

## Span: `aitf.asset.dependency`

Represents dependency mapping — resolving the dependency graph of an AI asset.

### Span Name

Format: `asset.dependency {aitf.asset.id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.id` | string | Asset whose dependencies are being resolved |
| `aitf.asset.dependency.operation` | string | `"resolve"`, `"update"`, `"validate"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.dependency.count` | int | Total dependency count |
| `aitf.asset.dependency.direct_count` | int | Direct dependency count |
| `aitf.asset.dependency.transitive_count` | int | Transitive dependency count |
| `aitf.asset.dependency.vulnerable_count` | int | Dependencies with known vulnerabilities |
| `aitf.asset.dependency.graph` | string | JSON dependency graph |

---

## Span: `aitf.asset.decommission`

Represents the decommissioning/retirement of an AI asset.

### Span Name

Format: `asset.decommission {aitf.asset.type} {aitf.asset.id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.id` | string | Asset being decommissioned |
| `aitf.asset.decommission.reason` | string | `"replaced"`, `"deprecated"`, `"security_risk"`, `"compliance"`, `"end_of_life"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.asset.decommission.replacement_id` | string | Replacement asset ID (if applicable) |
| `aitf.asset.decommission.data_retention` | string | `"purge"`, `"archive"`, `"retain"` |
| `aitf.asset.decommission.approved_by` | string | Approver identity |
| `aitf.asset.decommission.effective_date` | string | Effective decommission date |

---

## Examples

### Example 1: Registering a production model

```
Span: asset.register model customer-support-llama-70b
  aitf.asset.id: "model-cs-llama70b-v3"
  aitf.asset.name: "customer-support-llama-70b"
  aitf.asset.type: "model"
  aitf.asset.version: "3.1.0"
  aitf.asset.hash: "sha256:abc123def456"
  aitf.asset.owner: "ml-platform-team"
  aitf.asset.owner_type: "team"
  aitf.asset.deployment_environment: "production"
  aitf.asset.risk_classification: "high_risk"
  aitf.asset.source_repository: "https://registry.internal/models/cs-llama70b"
  aitf.asset.tags: ["customer-support", "llama", "fine-tuned"]
```

### Example 2: Shadow AI discovery scan

```
Span: asset.discover organization
  aitf.asset.discovery.scope: "organization"
  aitf.asset.discovery.method: "api_scan"
  aitf.asset.discovery.assets_found: 47
  aitf.asset.discovery.new_assets: 3
  aitf.asset.discovery.shadow_assets: 2
  aitf.asset.discovery.status: "completed"
```

### Example 3: EU AI Act risk classification

```
Span: asset.classify model-hiring-screener-v1
  aitf.asset.id: "model-hiring-screener-v1"
  aitf.asset.risk_classification: "high_risk"
  aitf.asset.classification.framework: "eu_ai_act"
  aitf.asset.classification.reason: "Employment context — CV screening affects natural persons"
  aitf.asset.classification.use_case: "automated resume screening"
  aitf.asset.classification.affected_persons: "consumers"
  aitf.asset.classification.sector: "hr_recruitment"
  aitf.asset.classification.autonomous_decision: true
```

### Example 4: Periodic compliance audit

```
Span: asset.audit model model-cs-llama70b-v3
  aitf.asset.id: "model-cs-llama70b-v3"
  aitf.asset.audit.type: "compliance"
  aitf.asset.audit.result: "warning"
  aitf.asset.audit.framework: "eu_ai_act"
  aitf.asset.audit.risk_score: 72.5
  aitf.asset.audit.integrity_verified: true
  aitf.asset.audit.compliance_status: "partially_compliant"
  aitf.asset.audit.last_audit_time: "2026-01-15T10:00:00Z"
  aitf.asset.audit.next_audit_due: "2026-04-15T10:00:00Z"
  aitf.asset.audit.findings: "[{\"finding\": \"Missing bias evaluation for protected groups\", \"severity\": \"high\"}]"
```
