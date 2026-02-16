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
| 7010 | AI Asset Inventory | Asset registration, discovery, audit, risk classification, drift, memory security |

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

AITF ships with production-ready forwarding examples for major security platforms:

| Platform | Ingestion Method | OCSF Native | Example |
|----------|-----------------|-------------|---------|
| **AWS Security Lake** | S3 (Parquet) / Kinesis Firehose | Yes | `examples/siem-forwarding/aws_security_lake.py` |
| **Trend Vision One** | REST API + Workbench | Yes | `examples/siem-forwarding/trend_vision_one.py` |
| **Splunk** | HTTP Event Collector (HEC) | CIM mapping | `examples/siem-forwarding/splunk_forwarding.py` |

```
AI App → AITF SDK → OCSF Mapper → ┬─ AWS Security Lake (S3/Parquet)
                                    ├─ Trend Vision One XDR (REST API)
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
│   │   │   ├── instrumentation/       # LLM, Agent, MCP, RAG, Skills
│   │   │   ├── processors/            # Security, PII, Compliance, Cost
│   │   │   ├── exporters/             # OCSF exporter
│   │   │   └── ocsf/                  # OCSF schema & mapper
│   │   └── tests/                     # Test suite
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
