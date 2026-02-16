# AITF Specification Overview

**Version:** 1.0.0-draft
**Status:** Draft
**Date:** 2026-02-15

## 1. Introduction

The AI Telemetry Framework (AITF) is a comprehensive telemetry specification and SDK for AI systems. It extends OpenTelemetry's GenAI semantic conventions with security-first design, native support for agentic patterns, MCP protocol instrumentation, and OCSF-based security event normalization.

### 1.1 Goals

1. **Extend OpenTelemetry GenAI** — Build on existing `gen_ai.*` conventions rather than replacing them
2. **Bridge Observability and Security** — Dual-pipeline producing both OTel traces and OCSF security events
3. **Support Modern AI Patterns** — First-class support for agentic AI, MCP, skills, multi-agent orchestration
4. **Compliance by Design** — Automatic mapping to regulatory frameworks
5. **Production Ready** — Stable conventions with clear migration from OTel GenAI experimental

### 1.2 Non-Goals

- Replacing OpenTelemetry — AITF is an extension, not a fork
- Defining new wire protocols — Uses standard OTLP
- Application-specific schemas — AITF is framework-agnostic

## 2. Design Principles

### 2.1 OpenTelemetry-Native

AITF is built on OpenTelemetry primitives:
- **Spans** for distributed tracing of AI operations
- **Metrics** for quantitative measurements
- **Logs/Events** for discrete occurrences
- **Resources** for entity identification

All AITF instrumentation produces standard OTel signals. The `gen_ai.*` namespace is preserved for OTel-compatible attributes. AITF-specific extensions use the `aitf.*` namespace.

### 2.2 Security-First

Every AI telemetry event is enriched with security context:
- OWASP LLM Top 10 detection (prompt injection, data leakage, etc.)
- PII detection and redaction
- Threat indicators and risk scoring
- OCSF-formatted security events for SIEM/XDR integration

### 2.3 Compliance by Design

Telemetry events are automatically mapped to compliance frameworks:
- NIST AI RMF, MITRE ATLAS, ISO/IEC 42001, EU AI Act, SOC 2, GDPR, CCPA

### 2.4 Minimal Overhead

AITF instrumentation is designed for production use:
- Configurable sampling and redaction
- Async-first implementation
- Lazy attribute evaluation
- Context propagation via standard OTel mechanisms

## 3. Architecture

### 3.1 Four-Layer Pipeline

```
Layer 1: Instrumentation   — OTel GenAI SDK + AITF extension instrumentation
Layer 2: Collection         — OTel Collector + AITF processors
Layer 3: Normalization      — OCSF mapper + compliance mapper
Layer 4: Analytics          — SIEM, XDR, dashboards, compliance reporting
```

### 3.2 Signal Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  AI Application                                                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │  LLM    │ │  Agent  │ │   MCP   │ │   RAG   │ │  Skills │  │
│  │  Calls  │ │  Steps  │ │ Server  │ │Pipeline │ │ Invoke  │  │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │
│       │           │           │           │           │         │
│  ┌────▼───────────▼───────────▼───────────▼───────────▼────┐   │
│  │              AITF Instrumentation Layer                  │   │
│  │  (OTel TracerProvider + AITF SpanProcessors)             │   │
│  └────┬────────────────────────────────────────────────┬────┘   │
│       │ OTel Spans/Metrics/Logs                        │        │
└───────┼────────────────────────────────────────────────┼────────┘
        │                                                │
        ▼                                                ▼
┌───────────────────┐                          ┌──────────────────┐
│  OTel Collector   │                          │  AITF Processors │
│  (OTLP Receiver)  │◄────────────────────────►│  - Security      │
│                   │                          │  - PII            │
│  Standard OTel    │                          │  - Compliance     │
│  Processing       │                          │  - Cost           │
└───────┬───────────┘                          └────────┬─────────┘
        │                                               │
        ▼                                               ▼
┌───────────────────┐                          ┌──────────────────┐
│  OTel Backends    │                          │  OCSF Exporter   │
│  - Jaeger         │                          │  - SIEM/XDR      │
│  - Grafana Tempo  │                          │  - S3 Data Lake  │
│  - Datadog        │                          │  - Syslog        │
└───────────────────┘                          └──────────────────┘
```

### 3.3 Span Hierarchy

AITF defines a clear span hierarchy for AI operations:

```
[Agent Session]                         aitf.agent.session
  └─[Identity Auth]                    aitf.identity.authentication
  └─[Agent Step: planning]              aitf.agent.step
      └─[LLM Inference]                gen_ai.inference
          ├─[gen_ai.content.prompt]     (event)
          └─[gen_ai.content.completion] (event)
  └─[Agent Step: tool_use]             aitf.agent.step
      ├─[Identity Authz]              aitf.identity.authorization
      └─[MCP Tool Call]                aitf.mcp.tool.invoke
          └─[Skill Execution]          aitf.skill.invoke
  └─[Agent Step: rag]                  aitf.agent.step
      └─[RAG Pipeline]                aitf.rag.pipeline
          ├─[Vector Search]            aitf.rag.retrieve
          └─[LLM Generation]          gen_ai.inference
  └─[Agent Step: delegation]           aitf.agent.step
      ├─[Identity Delegation]         aitf.identity.delegation
      ├─[Trust Establishment]         aitf.identity.trust
      └─[Sub-Agent Session]           aitf.agent.session
          └─ ...

[Model Operations Pipeline]            aitf.model_ops.*
  └─[Training Run]                    aitf.model_ops.training
      └─[Evaluation]                  aitf.model_ops.evaluation
          └─[Registry: register]      aitf.model_ops.registry
              └─[Deployment]          aitf.model_ops.deployment
                  └─[Monitoring]      aitf.model_ops.monitoring

[Serving Request]                       aitf.model_ops.serving
  ├─[Route Decision]                  aitf.model_ops.serving (route)
  ├─[Cache Lookup]                    aitf.model_ops.serving (cache_lookup)
  ├─[LLM Inference]                   gen_ai.inference
  └─[Fallback]                        aitf.model_ops.serving (fallback)

[Asset Inventory]                       aitf.asset.*
  ├─[Register]                        aitf.asset.register
  ├─[Discover]                        aitf.asset.discover
  ├─[Audit]                           aitf.asset.audit
  ├─[Classify]                        aitf.asset.classify
  └─[Decommission]                    aitf.asset.decommission

[Drift Detection]                       aitf.drift.*
  ├─[Detect]                          aitf.drift.detect
  ├─[Baseline]                        aitf.drift.baseline
  ├─[Investigate]                     aitf.drift.investigate
  └─[Remediate]                       aitf.drift.remediate
```

## 4. Namespace Registry

### 4.1 OTel-Compatible (Preserved)

| Namespace | Description |
|-----------|-------------|
| `gen_ai.*` | Standard OTel GenAI attributes |
| `gen_ai.system` | AI system identifier (e.g., "openai", "anthropic") |
| `gen_ai.request.*` | Request parameters (model, temperature, etc.) |
| `gen_ai.response.*` | Response attributes (finish_reason, id) |
| `gen_ai.usage.*` | Token usage (input_tokens, output_tokens) |
| `gen_ai.token.*` | Token-level metrics |

### 4.2 AITF Extensions

| Namespace | Description |
|-----------|-------------|
| `aitf.agent.*` | Agent lifecycle, reasoning, memory, delegation |
| `aitf.mcp.*` | MCP protocol (servers, tools, resources, prompts) |
| `aitf.skill.*` | Skill discovery, invocation, results, registry |
| `aitf.rag.*` | RAG pipeline stages, quality metrics |
| `aitf.security.*` | Security events, threat detection, guardrails |
| `aitf.compliance.*` | Regulatory framework mapping, audit |
| `aitf.cost.*` | Token costs, budget tracking, attribution |
| `aitf.quality.*` | Output quality (hallucination, confidence) |
| `aitf.supply_chain.*` | Model provenance, AI-BOM, integrity |
| `aitf.identity.*` | Agent identity lifecycle, authentication, authorization, delegation, trust |
| `aitf.model_ops.*` | LLMOps/MLOps lifecycle (training, evaluation, registry, deployment, serving, monitoring, prompts) |
| `aitf.asset.*` | AI asset inventory (registration, discovery, audit, risk classification) |
| `aitf.drift.*` | Structured model drift detection (baseline, investigation, remediation) |
| `aitf.guardrail.*` | Content filtering, safety checks, policies |
| `aitf.memory.*` | Agent memory operations (store, retrieve) |
| `aitf.memory.security.*` | Memory security (poisoning detection, integrity, isolation) |

## 5. OCSF Integration

### 5.1 OCSF Category 7: AI Events

AITF defines a new OCSF category (Category 7) for AI-specific security events. Each AITF span can be mapped to an OCSF event for consumption by SIEM/XDR platforms.

See [OCSF Event Classes](ocsf-mapping/event-classes.md) for detailed class definitions.

### 5.2 Compliance Mapping

Every OCSF event is automatically enriched with compliance metadata mapping to seven regulatory frameworks.

See [Compliance Mapping](ocsf-mapping/compliance-mapping.md) for framework details.

## 6. Comparison with OpenTelemetry GenAI SIG

### 6.1 What AITF Preserves

- All `gen_ai.*` semantic conventions
- Standard OTel span kinds and status codes
- OTLP wire format and transport
- OTel Collector compatibility
- Resource and context propagation semantics

### 6.2 What AITF Adds

1. **Agentic AI Telemetry** — Full agent lifecycle with reasoning traces, memory operations, delegation chains, and multi-agent coordination
2. **MCP Protocol Telemetry** — Native instrumentation for MCP server connections, tool discovery/invocation, resource access, and prompt management
3. **Skills Framework** — Skill registry, discovery, version negotiation, invocation tracing, and capability reporting
4. **Security Event Pipeline** — Real-time OWASP LLM Top 10 detection, PII detection/redaction, threat indicators, and OCSF security event export
5. **Compliance Automation** — Automatic mapping to NIST AI RMF, MITRE ATLAS, ISO 42001, EU AI Act, SOC 2, GDPR, CCPA
6. **Cost Attribution** — Per-token cost tracking with model pricing tables, budget enforcement, and multi-tenant attribution
7. **Quality Metrics** — Hallucination scoring, confidence estimation, factuality checking, and output quality tracking
8. **Supply Chain Telemetry** — Model provenance tracking, AI Bill of Materials, integrity verification, and model signing
9. **Guardrail Telemetry** — Content filter results, safety check outcomes, policy enforcement, and risk scoring
10. **Agent Memory** — Memory operation tracing (store, retrieve, update, delete) with provenance and TTL tracking
11. **Agentic Identity** — Full agent identity lifecycle (creation, rotation, revocation), authentication (OAuth 2.1, SPIFFE, mTLS, DID/VC), authorization with policy-as-code, delegation chains with scope attenuation, agent-to-agent trust establishment, and identity session management
12. **Model Operations (LLMOps/MLOps)** — Training/fine-tuning runs, model evaluation and benchmarking, model registry with lineage tracking, deployment strategies (canary, blue-green, A/B), serving infrastructure (routing, fallback chains, caching, circuit breakers), drift detection and monitoring, and prompt versioning lifecycle
13. **AI Asset Inventory (CoSAI IR Preparation)** — Complete AI asset registration and discovery (models, datasets, prompts, vector DBs, MCP servers, agents), EU AI Act risk classification, periodic compliance auditing with configurable frameworks, dependency mapping, shadow AI detection, and asset decommissioning
14. **Structured Drift Detection** — Forensic-quality drift analysis with statistical test results (PSI, KS, Jensen-Shannon, Wasserstein, ADWIN), segment-level impact assessment, reference dataset comparisons, root cause investigation with blast radius estimation, and automated remediation tracking
15. **Memory Security (CoSAI IR Preparation)** — Memory state tracking processor with before/after mutation snapshots, memory poisoning detection (aligned with CoSAI MINJA/AGENTPOISON case studies), content integrity verification, session memory isolation enforcement, untrusted provenance alerting, and memory growth anomaly detection

## 7. Versioning and Stability

AITF follows semantic versioning:
- **Stable** attributes: No breaking changes within a major version
- **Experimental** attributes: May change across minor versions (prefixed with documentation note)
- **Deprecated** attributes: Maintained for one major version after deprecation

Current status:
- `gen_ai.*` — Stable (following OTel)
- `aitf.agent.*` — Stable
- `aitf.mcp.*` — Stable
- `aitf.skill.*` — Stable
- `aitf.rag.*` — Stable
- `aitf.security.*` — Stable
- `aitf.compliance.*` — Stable
- `aitf.cost.*` — Stable
- `aitf.quality.*` — Experimental
- `aitf.supply_chain.*` — Experimental
- `aitf.identity.*` — Stable
- `aitf.model_ops.*` — Stable
- `aitf.asset.*` — Stable
- `aitf.drift.*` — Stable
- `aitf.memory.*` — Experimental
- `aitf.memory.security.*` — Stable

## 8. Related Documents

- [Attributes Registry](semantic-conventions/attributes-registry.md)
- [GenAI Spans](semantic-conventions/gen-ai-spans.md)
- [Agent Spans](semantic-conventions/agent-spans.md)
- [MCP Spans](semantic-conventions/mcp-spans.md)
- [Skills](semantic-conventions/skills.md)
- [Model Operations Spans](semantic-conventions/model-ops-spans.md)
- [Identity Spans](semantic-conventions/identity-spans.md)
- [Asset Inventory Spans](semantic-conventions/asset-inventory-spans.md)
- [Drift Detection Spans](semantic-conventions/drift-detection-spans.md)
- [Metrics](semantic-conventions/metrics.md)
- [Events](semantic-conventions/events.md)
- [OCSF Event Classes](ocsf-mapping/event-classes.md)
- [Compliance Mapping](ocsf-mapping/compliance-mapping.md)
