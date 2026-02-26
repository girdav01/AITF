# AITF - AI Telemetry Framework

> **Version 0.1 — Proposal**
> Proposed by David Girard (TrendAI) to [CoSAI](https://www.coalitionforsafeai.org/) WS2.
> This is a proposal for discussion and feedback — not a final standard.

**A comprehensive, security-first telemetry framework for AI systems built on OpenTelemetry and OCSF.**

AITF extends OpenTelemetry's GenAI semantic conventions with first-class support for agentic AI, MCP (Model Context Protocol), Skills, multi-agent orchestration, and security/compliance telemetry — bridging the gap between AI observability and cybersecurity.

## Why AITF?

OpenTelemetry's GenAI SIG provides foundational semantic conventions for AI observability, but the rapidly evolving AI landscape — particularly agentic systems, tool-use protocols like MCP, and regulatory requirements — demands a more comprehensive approach. AITF builds on OTel GenAI as its foundation while adding:

| Capability | OTel GenAI SIG | AITF |
|---|---|---|
| LLM Inference Tracing | Experimental | Stable + Enhanced |
| Agent Spans | Basic (experimental) | Full lifecycle, delegation, memory |
| MCP Protocol | Not covered | Native `aitf.mcp.*` namespace |
| Skills / Tool Registry | Not covered | Full `aitf.skill.*` namespace |
| Multi-Agent Orchestration | Not covered | Teams, delegation chains, consensus |
| Security Events | Not covered | OWASP LLM Top 10, threat detection |
| OCSF Integration | Not covered | Native OCSF Category 7 export |
| Compliance Mapping | Not covered | 8 frameworks (NIST, MITRE, EU AI Act, CSA AICM, ...) |
| Cost Attribution | Not covered | Per-request, per-user, per-model |
| Quality Metrics | Not covered | Hallucination, confidence, factuality |
| PII Detection | Not covered | Built-in processor |
| Supply Chain | Not covered | Model provenance, AI-BOM |
| Guardrail Telemetry | Not covered | Content filtering, safety checks |
| Agentic Identity | Not covered | Full lifecycle, OAuth 2.1, SPIFFE, delegation chains, trust |
| LLMOps/MLOps | Not covered | Training, evaluation, registry, deployment, drift, prompts |

## Architecture

AITF follows a four-layer pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Layer 4: Analytics                           │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │   SIEM   │  │   XDR    │  │ Grafana  │  │  Compliance Rpt  │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    Layer 3: Normalization                           │
│   ┌─────────────────────┐  ┌────────────────────────────────────┐  │
│   │   OCSF Mapper       │  │  Compliance Mapper (8 frameworks) │  │
│   │   (Category 7: AI)  │  │  NIST·MITRE·ISO·EU·SOC2·GDPR·CCPA·CSA│
│   └─────────────────────┘  └────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     Layer 2: Collection                            │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              OTel Collector + AITF Processors               │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌────────┐ │
│   │  │ Security │ │   PII    │ │   Cost   │ │ Compliance │ │ Memory │ │
│   │  │Processor │ │Processor │ │Processor │ │ Processor  │ │ State  │ │
│   │  └──────────┘ └──────────┘ └──────────┘ └────────────┘ └────────┘ │
│   └─────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                   Layer 1: Instrumentation                         │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ ┌──────────┐  │
│   │ LLM │ │Agent│ │ MCP │ │RAG│ │Skil│ │MOps│ │Iden│ │Asst│ │Drft│  │
│   │Inst.│ │Inst.│ │Inst.│ │Ins│ │Ins.│ │Ins.│ │Ins.│ │Inv.│ │Det.│  │
│   └─────┘ └─────┘ └─────┘ └───┘ └────┘ └────┘ └────┘ └────┘ └────┘  │
│                    OTel GenAI SDK + AITF Extensions                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Dual-Standard Pipeline

AITF uniquely bridges two standards:

- **OpenTelemetry** — For distributed tracing, metrics, and logs (observability)
- **OCSF** — For security event normalization (cybersecurity / SIEM / XDR)

Every AI telemetry event flows through both pipelines:

```
AI System → OTel SDK (traces/metrics/logs) → AITF Processors → OTel Collector
                                                    ↓
                                              OCSF Mapper → SIEM/XDR
                                                    ↓
                                           Compliance Tagger → Audit
```

## OCSF Category 7: AI Event Classes

AITF defines ten OCSF event classes for AI systems:

| Class UID | Event Class | Description |
|-----------|-------------|-------------|
| 7001 | AI Model Inference | LLM/model inference requests and responses |
| 7002 | AI Agent Activity | Agent lifecycle, reasoning, delegation |
| 7003 | AI Tool Execution | Tool/function calls including MCP tools |
| 7004 | AI Data Retrieval | RAG, vector search, knowledge retrieval |
| 7005 | AI Security Finding | Security events, guardrails, policy violations |
| 7006 | AI Supply Chain | Model provenance, AI-BOM, integrity |
| 7007 | AI Governance | Compliance, audit, regulatory events |
| 7008 | AI Identity | Agent identity, authentication, authorization, delegation, trust |
| 7009 | AI Model Operations | Model lifecycle: training, evaluation, deployment, monitoring, serving |
| 7010 | AI Asset Inventory | Asset registration, discovery, audit, risk classification, drift, memory security, shadow AI |

## SDK Language Support

| Language | Status | Package |
|----------|--------|---------|
| **Python** | Stable | `pip install aitf` |
| **Go** | Stable | `go get github.com/girdav01/AITF/sdk/go` |
| **TypeScript/JavaScript** | Stable | `npm install @aitf/sdk` |
| **Java** | Roadmap | Planned — contributions welcome |
| **C++** | Roadmap | Planned — contributions welcome |

## Quick Start

### Python

```bash
pip install aitf
```

```python
from aitf.instrumentation.llm import LLMInstrumentor
from aitf.instrumentation.agent import AgentInstrumentor, agent_span
from aitf.instrumentation.mcp import MCPInstrumentor

# Instrument LLM inference
llm = LLMInstrumentor()
llm.instrument(provider="openai")

# Instrument agents with decorator
@agent_span(agent_name="research-agent", framework="langchain")
async def research_agent(query: str):
    return await llm.invoke(query)

# Instrument MCP
mcp = MCPInstrumentor()
mcp.instrument()  # Captures tool.invoke, resource.read, prompt.get, etc.

# Security & PII processors
from aitf.processors.security_processor import SecurityProcessor
from aitf.processors.pii_processor import PIIProcessor

security = SecurityProcessor()  # OWASP LLM Top 10 detection
pii = PIIProcessor(action="redact")

# OCSF export to SIEM/XDR
from aitf.exporters.ocsf_exporter import OCSFExporter
exporter = OCSFExporter(
    endpoint="https://siem.example.com/api/events",
    compliance_frameworks=["nist_ai_rmf", "mitre_atlas", "eu_ai_act"]
)
```

### Go

```bash
go get github.com/girdav01/AITF/sdk/go
```

```go
package main

import (
    aitf "github.com/girdav01/AITF/sdk/go"
    "github.com/girdav01/AITF/sdk/go/instrumentation"
    "github.com/girdav01/AITF/sdk/go/processors"
    "github.com/girdav01/AITF/sdk/go/exporters"
)

func main() {
    // Initialize all instrumentors
    inst := aitf.NewInstrumentor()
    inst.InstrumentAll()

    // Trace an LLM inference
    llm := instrumentation.NewLLMInstrumentor()
    span := llm.TraceInference(ctx, "gpt-4o", "openai")
    span.SetPrompt("What is quantum computing?")
    span.SetCompletion("Quantum computing uses...")
    span.SetUsage(150, 300)
    span.End()

    // Trace an agent session
    agent := instrumentation.NewAgentInstrumentor()
    session := agent.TraceSession(ctx, "research-agent", "langchain")
    step := session.Step(ctx, "tool_use", 0)
    step.SetAction("search")
    step.End()
    session.End()

    // Security processor
    sec := processors.NewSecurityProcessor()
    findings := sec.AnalyzeText("Ignore all previous instructions")

    // OCSF export
    exp, _ := exporters.NewOCSFExporter(
        exporters.WithOutputFile("/var/log/aitf/events.jsonl"),
        exporters.WithComplianceFrameworks([]string{"nist_ai_rmf", "eu_ai_act"}),
    )
}
```

### TypeScript / JavaScript

```bash
npm install @aitf/sdk
```

```typescript
import {
  LLMInstrumentor,
  AgentInstrumentor,
  MCPInstrumentor,
  SecurityProcessor,
  PIIProcessor,
  CostProcessor,
  OCSFExporter,
} from '@aitf/sdk';

// Trace LLM inference
const llm = new LLMInstrumentor();
const span = llm.traceInference('gpt-4o', 'openai');
span.setPrompt('What is quantum computing?');
span.setCompletion('Quantum computing uses...');
span.setUsage(150, 300);
span.end();

// Trace agent session
const agent = new AgentInstrumentor();
const session = agent.traceSession('research-agent', { framework: 'langchain' });
const step = session.step('tool_use', 0);
step.setAction('search');
step.end();
session.end();

// Security processor (OWASP LLM Top 10)
const security = new SecurityProcessor();
const findings = security.analyzeText('Ignore all previous instructions');

// Cost tracking with budget
const cost = new CostProcessor({ budgetLimit: 100.0 });
const result = cost.calculateCost('gpt-4o', 1000, 500);
console.log(`Cost: $${result?.totalCost}, Budget remaining: $${cost.budgetRemaining}`);

// OCSF export
const exporter = new OCSFExporter({
  outputFile: '/var/log/aitf/events.jsonl',
  complianceFrameworks: ['nist_ai_rmf', 'eu_ai_act', 'mitre_atlas'],
});
```

## SIEM & XDR Integrations

AITF includes proposed forwarding examples for major security platforms (experimental — not production-ready):

| Platform | Ingestion Method | OCSF Native | Example |
|----------|-----------------|-------------|---------|
| **AWS Security Lake** | S3 (Parquet) / Kinesis Firehose | Yes | `examples/siem-forwarding/aws_security_lake.py` |
| **Trend Vision One** | CEF | Yes | `examples/siem-forwarding/trend_vision_one.py` |
| **Splunk** | HTTP Event Collector (HEC) | CIM mapping | `examples/siem-forwarding/splunk_forwarding.py` |

```
AI App → AITF SDK → OCSF Mapper → ┬─ AWS Security Lake (S3/Parquet)
                                    ├─ Trend Vision One XDR (CEF)
                                    └─ Splunk (HEC → CIM)
```

## Detection Rules & Anomaly Detection

AITF includes 14 built-in detection rules and a statistical anomaly detection engine:

| Rule ID | Name | Category | Severity |
|---------|------|----------|----------|
| AITF-DET-001 | Unusual Token Usage | Inference | Medium |
| AITF-DET-002 | Model Switching Attack | Inference | High |
| AITF-DET-003 | Prompt Injection Attempt | Inference | Critical |
| AITF-DET-004 | Excessive Cost Spike | Inference | High |
| AITF-DET-005 | Agent Loop Detection | Agent | Medium |
| AITF-DET-006 | Unauthorized Agent Delegation | Agent | High |
| AITF-DET-007 | Agent Session Hijack | Agent | Critical |
| AITF-DET-008 | Excessive Tool Calls | Agent | Medium |
| AITF-DET-009 | MCP Server Impersonation | MCP/Tool | Critical |
| AITF-DET-010 | Tool Permission Bypass | MCP/Tool | High |
| AITF-DET-011 | Data Exfiltration via Tools | MCP/Tool | Critical |
| AITF-DET-012 | PII Exfiltration Chain | Security | Critical |
| AITF-DET-013 | Jailbreak Escalation | Security | High |
| AITF-DET-014 | Supply Chain Compromise | Security | Critical |

Also includes:
- **Sigma rules** (YAML) for cross-SIEM deployment
- **Splunk SPL queries** for AI threat dashboards and agent behavioral analysis
- **Statistical anomaly detector** with z-score, IQR, and behavioral analysis

## OWASP Coverage Mapping

AITF's telemetry, detection rules, and semantic conventions map to the **OWASP Top 10 for LLM Applications (2025)**, the **OWASP Top 10 for Agentic Applications (2026)**, and the **OWASP Top 10 for MCP Security (2025)**. The tables below show which AITF mechanisms provide observability and detection for each risk.

### OWASP Top 10 for LLM Applications (2025)

| # | Risk | AITF Coverage | Key Mechanisms |
|---|------|:---:|----------------|
| LLM01 | Prompt Injection | **AITF-DET-003**, **AITF-DET-013** | Pattern-based detection (direct & indirect injection, jailbreak escalation); `aitf.security.threat_type` = `prompt_injection`; Sigma rule `aitf_prompt_injection.yml`; Guardrail processor (input/output) |
| LLM02 | Sensitive Information Disclosure | **AITF-DET-011**, **AITF-DET-012** | PII Processor (email, phone, SSN, credit card, API key detection & redaction); `aitf.security.pii.*` attributes; data-exfiltration read-then-send pattern detection; Sigma rule `aitf_data_exfiltration.yml` |
| LLM03 | Supply Chain | **AITF-DET-014**, **AITF-DET-009** | Model provenance verification (hash, source, signer); AI-BOM tracking (`aitf.supply_chain.ai_bom.*`); MCP server allow-list validation; OCSF event class 7006 (AI Supply Chain) |
| LLM04 | Data and Model Poisoning | Memory security events | `aitf.memory.security.poisoning_score`; content-hash integrity verification; provenance tracking (`aitf.memory.provenance`); RAG document provenance (`aitf.rag.doc.provenance`); OCSF event class 7004 |
| LLM05 | Improper Output Handling | Guardrail integration | `aitf.security.guardrail.type` = `output`; guardrail result pass/fail/warn; `aitf.security.blocked`; OCSF event 700503 (Guardrail Trigger) |
| LLM06 | Excessive Agency | **AITF-DET-005**, **006**, **008**, **010** | Agent loop detection (identical-call & cyclic patterns); unauthorized delegation validation; excessive tool-call thresholds; tool permission bypass detection; `aitf.agent.session.turn_count` |
| LLM07 | System Prompt Leakage | Security processor | `gen_ai.system_prompt.hash` (leak detection without storing content); jailbreak/extraction pattern matching; `aitf.security.threat_type` = `prompt_leakage` |
| LLM08 | Vector and Embedding Weaknesses | RAG telemetry | `aitf.rag.retrieve.*` attributes (database, filter, top_k); `aitf.rag.doc.provenance`; quality metrics (`faithfulness`, `groundedness`); OCSF event class 7004 (AI Data Retrieval) |
| LLM09 | Misinformation | Quality metrics | `aitf.quality.hallucination_score`; `aitf.quality.confidence`; `aitf.quality.factuality`; RAG source provenance verification |
| LLM10 | Unbounded Consumption | **AITF-DET-001**, **AITF-DET-004** | Token usage anomaly detection (z-score on EMA); cost-spike detection (5x rolling avg or $1.00 absolute); budget tracking (`aitf.cost.budget.*`); Sigma rule `aitf_cost_anomaly.yml` |

### OWASP Top 10 for Agentic Applications (2026)

| # | Risk | AITF Coverage | Key Mechanisms |
|---|------|:---:|----------------|
| ASI01 | Agent Goal Hijack | Agent telemetry | `aitf.agent.step.thought`, `aitf.agent.next_action`, `aitf.agent.state`; behavioral deviation detection via ReAct scratchpad and action-sequence analysis |
| ASI02 | Tool Misuse & Exploitation | **AITF-DET-010**, **AITF-DET-011** | Tool permission bypass detection (`aitf.mcp.tool.approval_required` / `approved`); data exfiltration via tools; tool I/O logging |
| ASI03 | Identity & Privilege Abuse | Identity spans | `aitf.identity.auth.*` (method, result, scope_granted); credential lifecycle tracking (create/rotate/revoke); scope-creep detection; OCSF event class 7008 (AI Identity) |
| ASI04 | Agentic Supply Chain Vulnerabilities | **AITF-DET-009**, **AITF-DET-014** | MCP server impersonation detection (allow-list, transport mismatch, protocol downgrade); model provenance & AI-BOM; Sigma rule `aitf_mcp_server_anomaly.yml` |
| ASI05 | Unexpected Code Execution | MCP tool telemetry | Tool execution logging via `aitf.mcp.tool.*`; suspicious tool-name detection (exec, shell, eval); sandbox monitoring |
| ASI06 | Memory & Context Poisoning | Memory security | `aitf.memory.security.poisoning_score`; content-hash mutation detection; cross-session access flags; provenance verification; OCSF memory-poisoning events |
| ASI07 | Insecure Inter-Agent Communication | Identity & delegation spans | `aitf.identity.trust.method` (mTLS, SPIFFE, DID-VC); agent delegation logging; `aitf.identity.auth.method` for agent-to-agent auth |
| ASI08 | Cascading Failures | Agent lifecycle telemetry | `aitf.agent.step.status` (success/error/retry/skipped); `aitf.agent.session.turn_count`; loop detection (AITF-DET-005); OCSF event class 7002 (AI Agent Activity) |
| ASI09 | Human-Agent Trust Exploitation | Human-in-loop attributes | `aitf.agent.step.type` = `human_in_loop`; tool approval tracking; EU AI Act Art. 14 compliance telemetry |
| ASI10 | Rogue Agents | Behavioral analysis | Agent behavioral baselines via action-sequence Markov modeling; `aitf.agent.state` lifecycle monitoring; **AITF-DET-007** (Agent Session Hijack); self-replication detection |

### OWASP Top 10 for MCP Security (2025)

| # | Risk | AITF Coverage | Key Mechanisms |
|---|------|:---:|----------------|
| MCP01 | Token Mismanagement & Secret Exposure | PII Processor, Identity spans | API-key / credential detection in tool I/O (`aitf.security.pii.types`); credential lifecycle tracking (`aitf.identity.lifecycle.operation`); `aitf.identity.ttl_seconds` for expiry monitoring |
| MCP02 | Privilege Escalation via Scope Creep | Identity spans | `aitf.identity.auth.scope_requested` vs `scope_granted` comparison; permission drift detection; credential rotation tracking |
| MCP03 | Tool Poisoning | **AITF-DET-009** | MCP server allow-list validation; tool output integrity monitoring (`aitf.mcp.tool.output`); suspicious tool-name detection; Sigma rule `aitf_mcp_server_anomaly.yml` |
| MCP04 | Software Supply Chain Attacks | **AITF-DET-014** | Model/component provenance verification; AI-BOM component tracking; cryptographic signature validation (`aitf.supply_chain.model.signed`) |
| MCP05 | Command Injection & Execution | MCP tool telemetry | Tool input logging (`aitf.mcp.tool.input`); suspicious tool-name blocklist (exec, shell, eval, reverse_shell); `aitf.mcp.tool.is_error` tracking |
| MCP06 | Prompt Injection via Contextual Payloads | **AITF-DET-003** | Indirect injection detection via `aitf.mcp.tool.output`; prompt-injection pattern matching in tool results and retrieved documents |
| MCP07 | Insufficient Authentication & Authorization | Identity spans, **AITF-DET-010** | `aitf.identity.auth.method` and `auth.result`; tool approval enforcement (`aitf.mcp.tool.approval_required`); OCSF event class 7008 |
| MCP08 | Lack of Audit and Telemetry | Full AITF pipeline | AITF's core purpose — comprehensive OTel spans + OCSF events for all MCP operations; spans for server connections, tool invocations, resource reads, prompt fetches |
| MCP09 | Shadow MCP Servers | **AITF-DET-009** | Server name allow-list; external URL flagging (`aitf.mcp.server.url`); transport validation; protocol version tracking; OCSF event class 7003 |
| MCP10 | Context Injection & Over-Sharing | Memory security, PII Processor | `aitf.memory.security.cross_session` flag; session isolation verification; PII detection and redaction in context; `aitf.memory.security.isolation_verified` |

## Project Structure

```
AITF/
├── spec/                              # Formal specifications
│   ├── overview.md                    # Architecture and design principles
│   ├── semantic-conventions/          # AITF semantic conventions
│   │   ├── attributes-registry.md     # Complete attribute registry
│   │   ├── gen-ai-spans.md            # Extended GenAI span conventions
│   │   ├── agent-spans.md             # Agent-specific span conventions
│   │   ├── mcp-spans.md               # MCP protocol span conventions
│   │   ├── skills.md                  # Skills semantic conventions
│   │   ├── metrics.md                 # Metrics conventions
│   │   └── events.md                  # Security and compliance events
│   ├── ocsf-mapping/                  # OCSF integration specs
│   │   ├── event-classes.md           # OCSF Category 7 event classes
│   │   └── compliance-mapping.md      # Multi-framework compliance mapping
│   └── schema/                        # JSON Schema definitions
│       ├── aitf-trace-schema.json     # Trace attribute schemas
│       └── aitf-ocsf-schema.json      # OCSF extension schemas
├── sdk/
│   ├── python/                        # Python SDK
│   │   ├── src/aitf/
│   │   │   ├── semantic_conventions/  # Attribute constants
│   │   │   ├── instrumentation/       # LLM, Agent, MCP, RAG, Skills, AgenticLog
│   │   │   ├── processors/            # Security, PII, Compliance, Cost, Memory
│   │   │   ├── generators/            # AI-BOM generator
│   │   │   ├── exporters/             # OCSF, Immutable Log, CEF Syslog
│   │   │   │   └── ocsf/                  # OCSF schema, mapper, vendor mappings
│   │   │       └── vendor_mappings/   # LangChain, CrewAI, OpenRouter JSON
│   │   └── tests/                     # Test suite (235 tests)
│   ├── go/                            # Go SDK
│   │   ├── semconv/                   # Attribute constants & metrics
│   │   ├── instrumentation/           # LLM, Agent, MCP, RAG, Skills
│   │   ├── processors/                # Security, PII, Compliance, Cost
│   │   ├── exporters/                 # OCSF exporter
│   │   └── ocsf/                      # OCSF schema, events & mapper
│   └── typescript/                    # TypeScript/JavaScript SDK
│       ├── src/
│       │   ├── semantic-conventions/  # Attribute constants & metrics
│       │   ├── instrumentation/       # LLM, Agent, MCP, RAG, Skills
│       │   ├── processors/            # Security, PII, Compliance, Cost
│       │   ├── exporters/             # OCSF exporter
│       │   └── ocsf/                  # OCSF schema, events & mapper
│       ├── package.json
│       └── tsconfig.json
├── examples/
│   ├── basic_llm_tracing.py           # Basic LLM tracing example
│   ├── agent_tracing.py               # Agent lifecycle example
│   ├── mcp_tracing.py                 # MCP protocol example
│   ├── rag_pipeline_tracing.py        # RAG pipeline example
│   ├── agentic_log_tracing.py         # Agentic log (Table 10.1) example
│   ├── ai_bom_generation.py           # AI-BOM generation example
│   ├── vendor_mapping_tracing.py      # Vendor mapping pipeline example
│   ├── openrouter_tracing.py         # OpenRouter multi-provider routing example
│   ├── aitf_colab_demo.ipynb          # Interactive Google Colab notebook
│   ├── siem-forwarding/               # SIEM integration examples
│   │   ├── aws_security_lake.py       # AWS Security Lake (OCSF native)
│   │   ├── trend_vision_one.py        # Trend Vision One XDR
│   │   └── splunk_forwarding.py       # Splunk HEC + CIM mapping
│   └── detection-rules/               # Detection & anomaly rules
│       ├── aitf_detection_rules.py    # 14 built-in detection rules
│       ├── anomaly_detector.py        # Statistical anomaly engine
│       ├── sigma_rules/               # Sigma-format rules (YAML)
│       └── splunk_queries/            # SPL queries for dashboards
├── collector/                         # OTel Collector configuration
└── dashboards/                        # Pre-built Grafana dashboards
```

## Semantic Convention Namespaces

AITF extends OTel GenAI with additional namespaces:

| Namespace | Scope | Standard |
|-----------|-------|----------|
| `gen_ai.*` | LLM inference (OTel-compatible) | OTel GenAI |
| `aitf.agent.*` | Agent lifecycle, reasoning, memory | AITF |
| `aitf.mcp.*` | MCP servers, tools, resources, prompts | AITF |
| `aitf.skill.*` | Skill discovery, invocation, registry | AITF |
| `aitf.rag.*` | RAG pipeline, retrieval, generation | AITF |
| `aitf.security.*` | Threat detection, guardrails, policy | AITF |
| `aitf.compliance.*` | Regulatory framework mapping | AITF |
| `aitf.cost.*` | Token costs, budget, attribution | AITF |
| `aitf.quality.*` | Hallucination, confidence, factuality | AITF |
| `aitf.supply_chain.*` | Model provenance, AI-BOM, integrity | AITF |
| `aitf.identity.*` | Agent identity, auth, delegation, trust | AITF |
| `aitf.model_ops.*` | LLMOps/MLOps lifecycle | AITF |
| `aitf.asset.*` | AI asset inventory, audit, risk classification | AITF + CoSAI |
| `aitf.drift.*` | Structured drift detection, investigation, remediation | AITF + CoSAI |
| `aitf.memory.security.*` | Memory poisoning, integrity, isolation | AITF + CoSAI |

## Compliance Frameworks

AITF automatically maps telemetry events to eight regulatory and security frameworks.

### Framework Coverage

| Framework | Version | AITF Mapped | Total Controls | Coverage |
|-----------|---------|:-----------:|:--------------:|:--------:|
| CSA AICM | v1.0.3 | 243 | 243 | **100.0%** |
| EU AI Act | 2024 | 9 | 15 | **60.0%** |
| GDPR | 2016/679 | 7 | 20 | **35.0%** |
| CCPA | CPRA 2023 | 4 | 12 | **33.3%** |
| ISO/IEC 42001 | 2023 | 12 | 39 | **30.8%** |
| MITRE ATLAS | 2024 | 7 | 30 | **23.3%** |
| NIST AI RMF | 1.0 | 16 | 72 | **22.2%** |
| SOC 2 | TSC 2017 | 7 | 33 | **21.2%** |
| **Total** | | **305** | **464** | **65.7%** |

> For regulatory frameworks (EU AI Act, GDPR, CCPA), "Total Controls" reflects
> AI-relevant operational provisions, not all articles in the regulation.
> See [compliance-mapping.md](spec/ocsf-mapping/compliance-mapping.md) for detailed per-event-type mappings.

## Requirements Documents

- [Framework Telemetry Requirements](framework_telemetry_requirements.md)
- [Telemetry Gaps Analysis](telemetry_gaps_analysis.md)
- [Defining AI Stack Telemetry](DefiningAIStackTelemetry-GDF-DGMarch04.pdf)

## Security Hardening

The AITF SDK is built with defense-in-depth principles. Key security measures include:

| Category | Measure | Location |
|----------|---------|----------|
| **ReDoS Prevention** | All regex patterns use bounded quantifiers; vendor-supplied patterns validated for length and correctness | `security_processor.py`, `pii_processor.py`, `vendor_mapper.py` |
| **Path Traversal** | File output paths reject `..` components to prevent directory escape | `ocsf_exporter.py`, `immutable_log.py` |
| **Input Validation** | Content length limits (100KB), token count limits (10M), score clamping (0.0-1.0), attribute type/length validation | All processors, instrumentors |
| **Thread Safety** | All shared mutable state protected by threading locks | `cost_processor.py`, `memory_state.py`, `ai_bom.py`, `ocsf_exporter.py` |
| **Credential Safety** | HTTPS enforced for remote endpoints with API keys; HMAC-SHA256 with random keys for PII hashing | `ocsf_exporter.py`, `pii_processor.py` |
| **DoS Prevention** | File size limits on log resume, max event/snapshot bounds, max components bound | `immutable_log.py`, `memory_state.py`, `ai_bom.py` |
| **Log Injection** | CEF field values sanitized; OCSF/MITRE/compliance data use separate fields | `cef_syslog_exporter.py` |
| **Hash Chain Integrity** | SHA-256 hash chains with genesis hash, sequence numbers, and file rotation | `immutable_log.py` |

## Roadmap

- **Java SDK** — Full AITF SDK for Java/JVM ecosystem (Spring Boot, Quarkus integrations)
- **C++ SDK** — High-performance SDK for edge AI and embedded inference systems
- **OpenTelemetry Collector Processor** — Native AITF processor plugin for the OTel Collector
- **Kubernetes Operator** — Auto-instrument AI workloads in K8s clusters
- **AI-BOM Generator** — Automated AI Bill of Materials from telemetry data
- **Real-time Threat Dashboard** — Pre-built Grafana dashboards for AI security operations

## Based On

- [AITelemetry](https://github.com/girdav01/AITelemetry) — Reference implementation and OCSF schema extensions
- [OpenTelemetry GenAI SIG](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai) — Foundation semantic conventions
- [OCSF](https://schema.ocsf.io/) — Open Cybersecurity Schema Framework v1.1.0

## License

Apache 2.0 — See [LICENSE](LICENSE)
