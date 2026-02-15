# AITF - AI Telemetry Framework

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
| Compliance Mapping | Not covered | 7 frameworks (NIST, MITRE, EU AI Act, ...) |
| Cost Attribution | Not covered | Per-request, per-user, per-model |
| Quality Metrics | Not covered | Hallucination, confidence, factuality |
| PII Detection | Not covered | Built-in processor |
| Supply Chain | Not covered | Model provenance, AI-BOM |
| Guardrail Telemetry | Not covered | Content filtering, safety checks |

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
│   │   OCSF Mapper       │  │  Compliance Mapper (7 frameworks) │  │
│   │   (Category 7: AI)  │  │  NIST·MITRE·ISO·EU·SOC2·GDPR·CCPA│  │
│   └─────────────────────┘  └────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     Layer 2: Collection                            │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              OTel Collector + AITF Processors               │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │  │
│   │  │ Security │ │   PII    │ │   Cost   │ │  Compliance  │  │  │
│   │  │Processor │ │Processor │ │Processor │ │  Processor   │  │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │  │
│   └─────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                   Layer 1: Instrumentation                         │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ ┌──────────┐  │
│   │   LLM    │ │  Agent   │ │   MCP    │ │  RAG  │ │  Skills  │  │
│   │Instrumtn │ │Instrumtn │ │Instrumtn │ │Instr. │ │Instrumtn │  │
│   └──────────┘ └──────────┘ └──────────┘ └───────┘ └──────────┘  │
│                  OTel GenAI SDK + AITF Extensions                  │
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

AITF defines eight OCSF event classes for AI systems:

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

## Quick Start

### Installation

```bash
pip install aitf
```

### Basic LLM Tracing

```python
from aitf.instrumentation import AITFInstrumentor

# Auto-instrument supported libraries
AITFInstrumentor().instrument_all()

# Or instrument specific components
from aitf.instrumentation.llm import LLMInstrumentor
LLMInstrumentor().instrument(provider="openai")
```

### Agent Tracing

```python
from aitf.instrumentation.agent import AgentInstrumentor, agent_span

instrumentor = AgentInstrumentor()

@agent_span(agent_name="research-agent", framework="langchain")
async def research_agent(query: str):
    # Agent logic here - automatically traced
    result = await llm.invoke(query)
    return result
```

### MCP Protocol Tracing

```python
from aitf.instrumentation.mcp import MCPInstrumentor

mcp_instrumentor = MCPInstrumentor()
mcp_instrumentor.instrument()

# Automatically captures:
# - mcp.server.connect / disconnect
# - mcp.tool.discover / invoke
# - mcp.resource.read / subscribe
# - mcp.prompt.get / execute
```

### Skills Tracing

```python
from aitf.instrumentation.skills import SkillInstrumentor, skill_span

@skill_span(skill_name="web-search", version="1.0")
async def web_search(query: str):
    # Skill execution automatically traced
    return await search_api.search(query)
```

### Security Processing

```python
from aitf.processors.security_processor import SecurityProcessor
from aitf.processors.pii_processor import PIIProcessor

# Add to your OTel pipeline
security = SecurityProcessor(
    detect_prompt_injection=True,
    detect_data_exfiltration=True,
    owasp_checks=True
)

pii = PIIProcessor(
    detect_types=["email", "ssn", "credit_card", "api_key"],
    action="redact"  # or "flag", "hash"
)
```

### OCSF Export

```python
from aitf.exporters.ocsf_exporter import OCSFExporter

exporter = OCSFExporter(
    endpoint="https://siem.example.com/api/events",
    compliance_frameworks=["nist_ai_rmf", "mitre_atlas", "eu_ai_act"]
)
```

## Project Structure

```
AITF/
├── spec/                           # Formal specifications
│   ├── overview.md                 # Architecture and design principles
│   ├── semantic-conventions/       # AITF semantic conventions
│   │   ├── attributes-registry.md  # Complete attribute registry
│   │   ├── gen-ai-spans.md         # Extended GenAI span conventions
│   │   ├── agent-spans.md          # Agent-specific span conventions
│   │   ├── mcp-spans.md            # MCP protocol span conventions
│   │   ├── skills.md               # Skills semantic conventions
│   │   ├── metrics.md              # Metrics conventions
│   │   └── events.md               # Security and compliance events
│   ├── ocsf-mapping/               # OCSF integration specs
│   │   ├── event-classes.md        # OCSF Category 7 event classes
│   │   └── compliance-mapping.md   # Multi-framework compliance mapping
│   └── schema/                     # JSON Schema definitions
│       ├── aitf-trace-schema.json  # Trace attribute schemas
│       └── aitf-ocsf-schema.json   # OCSF extension schemas
├── sdk/python/                     # Python SDK
│   ├── src/aitf/
│   │   ├── semantic_conventions/   # Attribute constants
│   │   ├── instrumentation/        # Auto-instrumentation (LLM, Agent, MCP, RAG, Skills)
│   │   ├── exporters/              # OCSF exporter
│   │   ├── processors/             # Security, PII, Compliance, Cost processors
│   │   └── ocsf/                   # OCSF schema (based on AITelemetry)
│   └── tests/                      # Test suite
├── collector/                      # OTel Collector configuration
├── examples/                       # Usage examples
└── dashboards/                     # Pre-built Grafana dashboards
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

## Compliance Frameworks

AITF automatically maps telemetry events to seven regulatory frameworks:

- **NIST AI RMF** — Risk management controls
- **MITRE ATLAS** — AI attack techniques
- **ISO/IEC 42001** — AI management system
- **EU AI Act** — European AI regulation
- **SOC 2** — Service organization controls
- **GDPR** — Data protection regulation
- **CCPA** — California consumer privacy

## Requirements Documents

- [Framework Telemetry Requirements](framework_telemetry_requirements.md)
- [Telemetry Gaps Analysis](telemetry_gaps_analysis.md)
- [Defining AI Stack Telemetry](DefiningAIStackTelemetry-GDF-DGMarch04.pdf)

## Based On

- [AITelemetry](https://github.com/girdav01/AITelemetry) — Reference implementation and OCSF schema extensions
- [OpenTelemetry GenAI SIG](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai) — Foundation semantic conventions
- [OCSF](https://schema.ocsf.io/) — Open Cybersecurity Schema Framework v1.1.0

## License

Apache 2.0 — See [LICENSE](LICENSE)
