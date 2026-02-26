# Extending OpenTelemetry for the AI Ecosystem

**AITF (AI Telemetry Framework) — Architecture & Design**

## 1. Why Extend OpenTelemetry for AI?

Traditional observability tools were built for request-response web services. AI systems introduce fundamentally different telemetry challenges:

- **Multi-step reasoning** — An agent may plan, retrieve context, call tools, delegate to sub-agents, and reflect — all within a single user request. Standard HTTP spans cannot capture this structure.
- **Token economics** — AI cost is measured in tokens, not compute time. Input tokens, output tokens, cached tokens, and reasoning tokens each have different pricing.
- **Security surface** — Prompt injection, data exfiltration, jailbreaking, and model theft are new threat categories that don't exist in traditional web applications.
- **Tool orchestration** — MCP servers, skills, function calling, and multi-agent delegation create complex execution graphs that need purpose-built instrumentation.
- **Compliance requirements** — EU AI Act, NIST AI RMF, ISO 42001, and CSA AICM all require specific audit trails that generic logging cannot provide.

AITF addresses these challenges by **extending** OpenTelemetry — not forking it. All AITF instrumentation produces standard OTel signals (spans, metrics, events) that flow over standard OTLP. The extension adds AI-specific semantic conventions, security processors, and a parallel OCSF export pipeline for SIEM integration.

## 2. Architecture: The Dual-Pipeline Model

The central innovation in AITF is the dual-pipeline architecture. A single instrumentation pass produces two simultaneous outputs:

```
                    ┌──────────────────────────────────────┐
                    │       AITF Instrumentation            │
                    │  (produces standard OTel spans)       │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │     TracerProvider                     │
                    │     (DualPipelineProvider)             │
                    │                                       │
                    │  ┌─────────────┐  ┌───────────────┐  │
                    │  │  BatchSpan  │  │  BatchSpan    │  │
                    │  │  Processor  │  │  Processor    │  │
                    │  │  (OTLP)     │  │  (OCSF)       │  │
                    │  └──────┬──────┘  └───────┬───────┘  │
                    └─────────┼─────────────────┼──────────┘
                              │                 │
            ┌─────────────────▼──┐    ┌────────▼───────────────┐
            │  OTLP Exporter     │    │  OCSF Exporter         │
            │  (gRPC / HTTP)     │    │  (JSON → File / HTTP)  │
            └─────────┬──────────┘    └────────┬───────────────┘
                      │                        │
            ┌─────────▼──────────┐    ┌────────▼───────────────┐
            │  Observability     │    │  Security / Compliance  │
            │  Jaeger, Tempo,    │    │  Splunk, Security Lake, │
            │  Datadog, etc.     │    │  QRadar, S3, etc.       │
            └────────────────────┘    └────────────────────────┘
```

### How it works

1. **Single instrumentation** — AITF instrumentors create standard OTel spans with `gen_ai.*` and `aitf.*` attributes.
2. **Shared TracerProvider** — One `TracerProvider` with multiple `SpanProcessor` chains attached.
3. **Parallel export** — Every span flows through all processors and exporters simultaneously:
   - The **OTLP exporter** sends the raw OTel span to observability backends (Jaeger, Tempo, Datadog).
   - The **OCSF exporter** converts the span to an OCSF Category 7 AI event and delivers it to SIEM/XDR platforms.
   - Optional **CEF/Syslog** and **immutable log** exporters add legacy SIEM and tamper-evident audit trail support.

### Pipeline options

| Pipeline | Format | Purpose | Backends |
|----------|--------|---------|----------|
| Observability | OTLP (gRPC/HTTP) | Distributed tracing, latency analysis, dependency maps | Jaeger, Grafana Tempo, Datadog, Honeycomb |
| Security | OCSF Category 7 (JSON) | Security events, compliance, threat detection | Splunk, AWS Security Lake, QRadar, Sentinel |
| Security (legacy) | CEF over Syslog | Legacy SIEM integration | ArcSight, LogRhythm, QRadar |
| Audit | Hash-chained JSONL | Tamper-evident audit trail (EU AI Act Art. 12, SOC 2 CC8.1) | File-based, S3, compliance archives |

### Setup

```python
from aitf import AITFInstrumentor, create_dual_pipeline_provider

# Create the dual pipeline
pipeline = create_dual_pipeline_provider(
    otlp_endpoint="http://localhost:4317",         # -> Jaeger/Tempo
    ocsf_output_file="/var/log/aitf_events.jsonl", # -> SIEM
    compliance_frameworks=["nist_ai_rmf", "eu_ai_act", "csa_aicm"],
    service_name="my-ai-app",
)

# Attach security and cost processors
from aitf.processors.security_processor import SecurityProcessor
from aitf.processors.cost_processor import CostProcessor

pipeline.tracer_provider.add_span_processor(SecurityProcessor())
pipeline.tracer_provider.add_span_processor(CostProcessor(budget_limit=100.0))
pipeline.set_as_global()

# Enable all instrumentors
instrumentor = AITFInstrumentor(tracer_provider=pipeline.tracer_provider)
instrumentor.instrument_all()
```

## 3. Extending OTel Semantic Conventions for AI

AITF preserves the existing OpenTelemetry `gen_ai.*` namespace for backward compatibility and extends it with the `aitf.*` namespace for AI-specific concerns.

### 3.1 Preserved OTel GenAI Conventions (`gen_ai.*`)

These are the standard OpenTelemetry GenAI semantic conventions, used as-is:

| Attribute | Example | Purpose |
|-----------|---------|---------|
| `gen_ai.system` | `"openai"`, `"anthropic"` | LLM provider identifier |
| `gen_ai.operation.name` | `"chat"`, `"embeddings"` | Operation type |
| `gen_ai.request.model` | `"gpt-4o"` | Requested model |
| `gen_ai.request.temperature` | `0.7` | Sampling temperature |
| `gen_ai.request.max_tokens` | `4096` | Token limit |
| `gen_ai.response.id` | `"chatcmpl-abc123"` | Response identifier |
| `gen_ai.response.finish_reasons` | `["stop"]` | Completion reasons |
| `gen_ai.usage.input_tokens` | `150` | Input token count |
| `gen_ai.usage.output_tokens` | `320` | Output token count |

### 3.2 AITF Extensions (`aitf.*`)

AITF adds 15 attribute namespaces covering the full AI ecosystem:

| Namespace | Domain | Key Attributes |
|-----------|--------|----------------|
| `aitf.agent.*` | Agent lifecycle | `name`, `id`, `type`, `framework`, `session.id`, `step.type/thought/action/observation`, `delegation.target_agent`, `team.topology` |
| `aitf.mcp.*` | Model Context Protocol | `server.name/transport`, `tool.name/input/output/duration_ms/approval_required`, `resource.uri`, `sampling.model` |
| `aitf.skill.*` | Skills framework | `name`, `version`, `category`, `provider`, `input/output`, `compose.pattern` |
| `aitf.rag.*` | Retrieval-Augmented Generation | `pipeline.name/stage`, `retrieve.database/top_k/results_count`, `doc.id/score/provenance`, `quality.faithfulness/groundedness` |
| `aitf.security.*` | Threat detection | `threat_detected`, `threat_type`, `owasp_category`, `risk_score`, `blocked`, `guardrail.name/result`, `pii.types/action` |
| `aitf.compliance.*` | Regulatory mapping | `frameworks`, `nist_ai_rmf.controls`, `eu_ai_act.articles`, `csa_aicm.controls` |
| `aitf.cost.*` | Token economics | `input_cost`, `output_cost`, `total_cost`, `budget.limit/used/remaining`, `attribution.user/team/project` |
| `aitf.quality.*` | Output quality | `hallucination_score`, `confidence`, `factuality`, `toxicity_score`, `bias_score` |
| `aitf.identity.*` | Agent identity | `agent_id`, `auth.method/result`, `authz.decision/resource`, `delegation.chain/scope_attenuated`, `trust.method/level` |
| `aitf.model_ops.*` | LLMOps/MLOps lifecycle | `training.run_id/type/base_model/loss_final`, `evaluation.metrics/pass`, `deployment.strategy/environment`, `monitoring.drift_score` |
| `aitf.asset.*` | AI asset inventory | `id`, `name`, `type`, `risk_classification`, `discovery.shadow_assets`, `audit.result/framework` |
| `aitf.drift.*` | Model drift detection | `model_id`, `type`, `score`, `detection_method`, `p_value`, `remediation.action` |
| `aitf.supply_chain.*` | Model provenance | `model.source/hash/license/signed`, `ai_bom.id/components` |
| `aitf.a2a.*` | Google A2A protocol | `agent.name/skills`, `task.id/state`, `message.role`, `stream.event_type` |
| `aitf.acp.*` | Agent Communication Protocol | `run.id/mode/status`, `message.role/parts_count`, `await.active/duration_ms` |
| `aitf.memory.*` | Agent memory | `operation`, `store` (short_term/long_term/episodic), `key`, `security.poisoning_score/isolation_verified` |
| `aitf.agentic_log.*` | Agentic audit log | `event_id`, `agent_id`, `goal_id`, `tool_used`, `outcome`, `confidence_score`, `anomaly_score`, `policy_evaluation` |
| `aitf.latency.*` | Performance metrics | `total_ms`, `time_to_first_token_ms`, `tokens_per_second` |

## 4. Instrumentation Layer: 12 Instrumentors

Each instrumentor creates OTel spans with a dedicated tracer name and domain-specific span naming convention.

### 4.1 Span Naming Convention

All spans follow the pattern: `"{domain}.{operation} {entity}"`

```
chat gpt-4o                              # LLM inference
agent.session research-bot               # Agent session
agent.step.planning research-bot         # Agent reasoning step
mcp.tool.invoke search_docs              # MCP tool call
rag.retrieve pinecone                    # Vector search
skill.invoke code_review                  # Skill execution
identity.auth orchestrator               # Agent authentication
model_ops.training run-001               # Training run
asset.register model customer-llm        # Asset registration
drift.detect data_distribution model-01  # Drift detection
```

### 4.2 Instrumentor Summary

| # | Instrumentor | Tracer Name | What It Traces | Key Span Names |
|---|-------------|-------------|----------------|----------------|
| 1 | `LLMInstrumentor` | `aitf.instrumentation.llm` | LLM inference (chat, embeddings, completions) | `chat {model}`, `embeddings {model}` |
| 2 | `AgentInstrumentor` | `aitf.instrumentation.agent` | Agent sessions, steps, delegation, memory | `agent.session {name}`, `agent.step.{type} {name}`, `agent.delegate {a} -> {b}` |
| 3 | `MCPInstrumentor` | `aitf.instrumentation.mcp` | MCP server connections, tool invocations, resources | `mcp.server.connect {server}`, `mcp.tool.invoke {tool}` |
| 4 | `RAGInstrumentor` | `aitf.instrumentation.rag` | RAG pipeline stages (retrieve, rerank, evaluate) | `rag.pipeline {name}`, `rag.retrieve {database}` |
| 5 | `SkillInstrumentor` | `aitf.instrumentation.skills` | Skill invocations, discovery, composition | `skill.invoke {name}`, `skill.compose {workflow}` |
| 6 | `ModelOpsInstrumentor` | `aitf.instrumentation.model_ops` | Training, evaluation, deployment, serving, monitoring | `model_ops.training {run_id}`, `model_ops.deployment {id}` |
| 7 | `IdentityInstrumentor` | `aitf.instrumentation.identity` | Authentication, authorization, delegation, trust | `identity.auth {name}`, `identity.delegate {a} -> {b}` |
| 8 | `AssetInventoryInstrumentor` | `aitf.instrumentation.asset_inventory` | Asset registration, discovery, audit, classification | `asset.register {type} {name}`, `asset.discover {scope}` |
| 9 | `DriftDetectionInstrumentor` | `aitf.instrumentation.drift_detection` | Drift detection, baseline management, remediation | `drift.detect {type} {model}`, `drift.remediate {action} {model}` |
| 10 | `A2AInstrumentor` | `aitf.instrumentation.a2a` | Google A2A protocol (Agent Cards, tasks, streaming) | `a2a.agent.discover`, `a2a.task.send` |
| 11 | `ACPInstrumentor` | `aitf.instrumentation.acp` | Agent Communication Protocol (runs, messages, await) | `acp.run.create {agent}`, `acp.message.send` |
| 12 | `AgenticLogInstrumentor` | `aitf.instrumentation.agentic_log` | Structured security audit log (Table 10.1 fields) | `agentic_log.action {agent_id}` |

### 4.3 Span Hierarchy

Spans nest to capture the full execution tree of an AI agent interaction:

```
agent.session research-bot
├── identity.auth research-bot                    (OCSF 7008)
├── agent.step.planning research-bot
│    └── chat gpt-4o                              (OCSF 7001)
├── agent.step.tool_use research-bot
│    ├── identity.authz research-bot -> customer-db  (OCSF 7008)
│    └── mcp.tool.invoke search_docs              (OCSF 7003)
│         └── skill.invoke vector_search           (OCSF 7003)
├── agent.step.rag research-bot
│    └── rag.pipeline knowledge-retrieval
│         ├── rag.retrieve pinecone               (OCSF 7004)
│         └── chat gpt-4o                         (OCSF 7001)
├── agent.step.response research-bot
│    └── chat gpt-4o                              (OCSF 7001)
└── agent.step.delegation research-bot
     ├── identity.delegate research-bot -> writer  (OCSF 7008)
     └── agent.session writer                     (recursive)
```

Every span in this tree flows simultaneously to both the OTLP pipeline (for tracing) and the OCSF pipeline (for security events).

## 5. Processors: In-Flight Span Enrichment

Processors implement the OTel `SpanProcessor` interface and operate on spans before they reach exporters. AITF provides five processors:

### 5.1 SecurityProcessor

Detects OWASP LLM Top 10 threats in real time by scanning span content:

| Detection | OWASP Category | What It Catches |
|-----------|---------------|-----------------|
| Prompt injection | LLM01 | "ignore all previous instructions", role overrides, delimiter attacks |
| Jailbreak | LLM01 | "DAN mode", "bypass safety", roleplay exploits |
| System prompt leak | LLM07 | "reveal your system prompt", instruction extraction |
| Data exfiltration | LLM02 | Encoded data transfer, URL exfiltration patterns |
| Command injection | LLM05 | Shell commands, backtick execution, pipe chains |
| SQL injection | LLM05 | UNION SELECT, OR 1=1, DROP TABLE patterns |

Findings are emitted with risk scores and confidence levels. The processor can optionally block critical threats.

### 5.2 PIIProcessor

Detects and handles PII in prompts, completions, and tool I/O with three modes:

- **Flag** — Detect and count PII types (email, phone, SSN, credit card, API key, JWT)
- **Redact** — Replace with `[EMAIL_REDACTED]`, `[SSN_REDACTED]`, etc.
- **Hash** — Replace with HMAC-SHA256 pseudonyms: `[EMAIL:a1b2c3d4]` (keyed per processor instance for consistency)

### 5.3 CostProcessor

Tracks token-level cost across every LLM call:

- Built-in pricing table for ~25 models (OpenAI, Anthropic, Google, Mistral, Meta, Cohere)
- Calculates `aitf.cost.input_cost`, `aitf.cost.output_cost`, `aitf.cost.total_cost`
- Budget tracking with `budget_limit`, `budget_used`, `budget_remaining`
- Cost attribution by user, team, and project

### 5.4 ComplianceProcessor

Maps AI event types to compliance framework controls:

- Eight frameworks: NIST AI RMF, MITRE ATLAS, ISO 42001, EU AI Act, SOC 2, GDPR, CCPA, CSA AICM
- Classifies spans by name prefix and attaches `aitf.compliance.*` attributes
- Provides `get_coverage_matrix()` for audit reporting

### 5.5 MemoryStateProcessor

Tracks agent memory mutations for security:

- Captures before/after memory snapshots with content hashes
- Detects memory poisoning (unexpected content injection)
- Verifies cross-session memory isolation
- Monitors long-term memory growth anomalies

## 6. OCSF Mapping: OTel Spans to Security Events

The `OCSFMapper` converts OTel spans to OCSF Category 7 (AI System Activity) events. This is the bridge between the observability world and the security world.

### 6.1 OCSF Category 7 Event Classes

| Class UID | Event Class | OTel Span Triggers | What It Captures |
|-----------|-------------|-------------------|------------------|
| 7001 | `AIModelInferenceEvent` | `chat *`, `embeddings *`, `gen_ai.system` attr | Model, tokens, latency, cost, finish reason |
| 7002 | `AIAgentActivityEvent` | `agent.*`, `aitf.agent.name` attr | Agent identity, step type, thought/action/observation, delegation |
| 7003 | `AIToolExecutionEvent` | `mcp.tool.*`, `skill.invoke*` | Tool name/type, input/output, MCP server, approval status |
| 7004 | `AIDataRetrievalEvent` | `rag.*`, `aitf.rag.retrieve.database` attr | Database, query, top_k, results count, scores |
| 7005 | `AISecurityFindingEvent` | `aitf.security.threat_detected` attr | Finding type, OWASP category, risk score, confidence, blocked |
| 7006 | `AISupplyChainEvent` | `supply_chain.*`, `aitf.supply_chain.model.source` attr | Model source/hash/license, signature verification, AI BOM |
| 7007 | `AIGovernanceEvent` | `governance.*`, `compliance.*` | Compliance frameworks, controls, violations, audit ID |
| 7008 | `AIIdentityEvent` | `identity.*`, `aitf.identity.agent_id` attr | Auth method/result, credential type, delegation chain, scope |
| 7009 | `AIModelOpsEvent` | `model_ops.*`, `drift.*` | Training, evaluation, deployment, serving, monitoring, drift |
| 7010 | `AIAssetInventoryEvent` | `asset.*`, `aitf.asset.id` attr | Asset type, owner, risk classification, discovery, audit |

### 6.2 Mapping Flow

```
OTel Span (ReadableSpan)
    │
    ▼
OCSFMapper.map_span()
    │
    ├── Classify by span name prefix + attributes
    ├── Extract fields from aitf.*/gen_ai.* attributes
    ├── Determine activity_id from span name keywords
    │
    ▼
AIBaseEvent (Pydantic model)
    │
    ├── class_uid: 7001-7010
    ├── category_uid: 7 (AI System Activity)
    ├── type_uid: class_uid * 100 + activity_id
    ├── time: span start time (ISO 8601)
    ├── metadata: OCSF v1.1.0 + AITF product info
    ├── compliance: mapped framework controls
    │
    ▼
JSON serialization → SIEM / XDR / Data Lake
```

### 6.3 OCSF Event Example

An OTel span named `chat gpt-4o` with `gen_ai.*` attributes produces:

```json
{
  "class_uid": 7001,
  "category_uid": 7,
  "type_uid": 700101,
  "activity_id": 1,
  "time": "2026-02-26T10:30:00Z",
  "severity_id": 1,
  "status_id": 1,
  "message": "chat gpt-4o",
  "metadata": {
    "version": "1.1.0",
    "product": {"name": "AITF", "vendor_name": "AITF", "version": "1.0.0"},
    "uid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  },
  "model": {
    "model_id": "gpt-4o",
    "provider": "openai",
    "type": "llm"
  },
  "token_usage": {
    "input_tokens": 150,
    "output_tokens": 320,
    "total_tokens": 470
  },
  "cost": {
    "input_cost_usd": 0.000375,
    "output_cost_usd": 0.0032,
    "total_cost_usd": 0.003575
  },
  "finish_reason": "stop",
  "streaming": false,
  "compliance": {
    "nist_ai_rmf": {"controls": ["MEASURE-2.6", "MANAGE-3.2"], "function": "Measure"},
    "eu_ai_act": {"articles": ["Article 13", "Article 14"], "risk_level": "limited"},
    "csa_aicm": {"controls": ["MDS-01", "AIS-04", "LOG-14"], "domain": "Model Security"}
  }
}
```

## 7. Exporters

### 7.1 OCSF Exporter

The primary security pipeline exporter. For each span:
1. Calls `OCSFMapper.map_span()` to produce an OCSF event (or `None` for non-AI spans)
2. Enriches with compliance framework mappings
3. Serializes to JSON and writes to JSONL file and/or POSTs to HTTP endpoint

Security hardening: HTTPS enforcement for non-localhost endpoints, path traversal prevention, file rotation at 500 MB, TLS verification.

### 7.2 CEF/Syslog Exporter

Converts OCSF events to CEF (Common Event Format) syslog messages for legacy SIEMs:

```
CEF:0|AITF|AI-Telemetry-Framework|1.0.0|700101|AI Model Inference|3|
  rt=2026-02-26T10:30:00Z msg=chat gpt-4o cs1=7001 cs1Label=ocsf_class_uid ...
```

Supports TCP with TLS (RFC 5425), TCP without TLS (development), and UDP (RFC 3164).

### 7.3 Immutable Log Exporter

Writes hash-chained JSONL entries for tamper-evident audit trails:

```json
{"seq": 1, "timestamp": "...", "prev_hash": "0000...", "hash": "a1b2...", "event": {...}}
{"seq": 2, "timestamp": "...", "prev_hash": "a1b2...", "hash": "c3d4...", "event": {...}}
```

Each hash is `SHA-256("{seq}|{timestamp}|{prev_hash}|{event_json}")`. Any modification to any historical entry breaks all subsequent hashes, providing forensic-quality tamper detection.

## 8. Compliance Framework Integration

Every OCSF event is automatically enriched with mappings to eight compliance frameworks:

| Framework | Coverage | Example Controls |
|-----------|----------|-----------------|
| NIST AI RMF | GOVERN, MAP, MEASURE, MANAGE | GOVERN-1.2, MEASURE-2.6, MANAGE-3.2 |
| MITRE ATLAS | Reconnaissance through Impact | AML.T0043, AML.T0051 |
| ISO/IEC 42001 | AI Management System | A.6.2.4, A.8.4 |
| EU AI Act | Articles 9-15, 52, 68-69 | Article 13 (Transparency), Article 12 (Record-keeping) |
| SOC 2 | Trust Service Criteria | CC6.1, CC7.2, CC8.1 |
| GDPR | Articles 5-6, 13-14, 22, 25, 30, 35 | Article 22 (Automated Decisions), Article 35 (DPIA) |
| CCPA | Sections 1798.100-1798.199 | 1798.100 (Right to Know), 1798.150 (Data Breaches) |
| CSA AICM | 18 domains, 243 controls | MDS-01, AIS-04, LOG-14, GRC-13, TVM-11 |

The `ComplianceMapper` maps each event type (inference, agent activity, tool execution, etc.) to the relevant controls from each framework, producing a `ComplianceMetadata` object attached to every OCSF event.

## 9. Usage Examples

### 9.1 LLM Inference Tracing

```python
with instrumentor.llm.trace_inference(model="gpt-4o", system="openai") as span:
    span.set_prompt("Summarize the quarterly report...")
    # ... call actual LLM API ...
    span.set_completion("The quarterly report shows...")
    span.set_usage(input_tokens=150, output_tokens=320)
    span.set_cost(input_cost=0.000375, output_cost=0.0032)
    span.set_latency(total_ms=680.0, time_to_first_token_ms=120.0)
```

### 9.2 Agent Session with Tools

```python
with instrumentor.agent.trace_session(
    agent_name="research-bot", agent_id="agent-001", framework="langgraph"
) as session:
    # Planning step
    with session.step("planning") as step:
        with instrumentor.llm.trace_inference(model="gpt-4o") as llm:
            llm.set_prompt("Plan research on AI security...")
            llm.set_usage(input_tokens=100, output_tokens=200)

    # Tool use step
    with session.step("tool_use") as step:
        with instrumentor.mcp.trace_tool_invoke(
            tool_name="search_docs", server_name="knowledge-base"
        ) as tool:
            tool.set_input('{"query": "AI security best practices"}')
            tool.set_output('{"results": [...]}')

    # RAG retrieval step
    with session.step("rag") as step:
        with instrumentor.rag.trace_pipeline("knowledge-retrieval") as pipeline:
            with pipeline.retrieve(database="pinecone", top_k=10) as retrieval:
                retrieval.set_results(count=8, min_score=0.72, max_score=0.95)
```

### 9.3 Model Operations

```python
with instrumentor.model_ops.trace_training(
    training_type="fine_tuning", base_model="meta-llama/Llama-3.1-70B",
    dataset_id="customer-support-v3"
) as run:
    # ... perform training ...
    run.set_loss(0.42)
    run.set_output_model("cs-llama-70b-lora-v3", "sha256:abc123")

with instrumentor.model_ops.trace_deployment(
    model_id="cs-llama-70b-lora-v3", strategy="canary", environment="production"
) as deployment:
    deployment.set_endpoint("https://models.example.com/cs-llama-v3")
    deployment.set_canary_percent(10)
```

## 10. Cross-SDK Support

AITF provides consistent implementations across three language SDKs:

| Component | Python | TypeScript | Go |
|-----------|--------|------------|-----|
| OCSF Schema (class UIDs, base events) | Pydantic models | TypeScript interfaces + enums | Go structs + constants |
| Event Classes (7001-7010) | Pydantic models with validators | Interfaces + factory functions | Structs + constructors |
| OCSFMapper | `OCSFMapper.map_span()` | `OCSFMapper.mapSpan()` | Not yet implemented |
| Compliance Mapper | `ComplianceMapper` | `ComplianceMapper` | Not yet implemented |
| Semantic Conventions | Python constants | TypeScript constants | Go constants |
| Instrumentors | 12 instrumentors | Subset (LLM, Agent, MCP) | Subset (LLM, Agent) |

All SDKs share the same OCSF JSON Schema (`spec/schema/aitf-ocsf-schema.json`) and semantic convention specifications (`spec/semantic-conventions/`).

## 11. Design Decisions

### Why extend OTel instead of building from scratch?

- OTel has mature SDKs in every major language with battle-tested context propagation, batching, and retry logic.
- OTLP is a universal wire format supported by every major observability vendor.
- The `gen_ai.*` namespace already covers basic LLM inference — AITF builds on this rather than competing.
- Teams can adopt AITF incrementally: start with OTel-only, add OCSF export later.

### Why OCSF for the security pipeline?

- OCSF is an open standard (by AWS, Splunk, IBM, etc.) specifically designed for security event normalization.
- Category 7 (AI System Activity) provides a natural home for AI-specific security events.
- OCSF events are directly ingestible by major SIEMs (Splunk, AWS Security Lake, QRadar, Sentinel).
- The `class_uid` / `activity_id` / `type_uid` structure maps cleanly to AI operation taxonomies.

### Why dual-pipeline instead of post-processing?

- Post-processing adds latency and requires separate infrastructure for security teams.
- Dual-pipeline ensures security events are generated in real time, in the same process.
- A single instrumentation pass eliminates the risk of observability/security data divergence.
- Security processors (threat detection, PII redaction) operate on raw spans before any data leaves the process.
