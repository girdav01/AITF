"""AITF Example: AI-BOM Generation.

Demonstrates how to generate an AI Bill of Materials (AI-BOM) from
telemetry data, both through automatic span collection and manual
component registration. Shows export in AITF, CycloneDX, and SPDX formats.
"""

import json

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from aitf.generators.ai_bom import AIBOMGenerator
from aitf.instrumentation.llm import LLMInstrumentor
from aitf.instrumentation.agent import AgentInstrumentor
from aitf.instrumentation.mcp import MCPInstrumentor
from aitf.instrumentation.rag import RAGInstrumentor

# --- Setup ---

provider = TracerProvider()

# Create AI-BOM generator as a SpanExporter — it passively collects
# component information from every traced span.
bom_generator = AIBOMGenerator(
    system_name="customer-support-ai",
    system_version="2.1.0",
)
provider.add_span_processor(SimpleSpanProcessor(bom_generator))
trace.set_tracer_provider(provider)

# Initialize instrumentors
llm = LLMInstrumentor(tracer_provider=provider)
agent = AgentInstrumentor(tracer_provider=provider)
mcp = MCPInstrumentor(tracer_provider=provider)
rag = RAGInstrumentor(tracer_provider=provider)

llm.instrument()
agent.instrument()
mcp.instrument()
rag.instrument()

# --- Simulate AI Workload (traces are collected automatically) ---

print("=== Simulating AI Workload ===\n")

# 1. LLM inference
with llm.trace_inference(
    model="claude-sonnet-4-20250514",
    system="anthropic",
    operation="chat",
    temperature=0.3,
    max_tokens=4096,
) as span:
    span.set_attribute("gen_ai.usage.input_tokens", 500)
    span.set_attribute("gen_ai.usage.output_tokens", 200)

# 2. Agent session with tool use
with agent.trace_session(
    agent_name="support-agent",
    agent_type="autonomous",
    framework="langchain",
) as session:
    with session.step("reasoning") as step:
        step.set_thought("Customer needs help with billing")

# 3. MCP tool invocation
with mcp.trace_tool(
    tool_name="search_knowledge_base",
    server_name="kb-server",
    server_version="1.2.0",
    transport="stdio",
) as tool:
    tool.set_output("Found 3 relevant articles")

# 4. RAG retrieval
with rag.trace_retrieval(
    database="pinecone",
    query="billing dispute resolution",
    top_k=5,
    embedding_model="text-embedding-3-small",
) as retrieval:
    retrieval.set_results_count(5)
    retrieval.set_max_score(0.95)

# --- Manual Component Registration ---

print("=== Adding Manual Components ===\n")

# Add components that aren't captured via tracing
bom_generator.add_component(
    component_type="framework",
    name="langchain",
    version="0.3.0",
    provider="LangChain Inc.",
    license="MIT",
    source="https://pypi.org/project/langchain/",
)

bom_generator.add_component(
    component_type="guardrail",
    name="content-filter-v2",
    version="2.1",
    provider="internal",
    description="Production content safety filter",
)

bom_generator.add_component(
    component_type="prompt_template",
    name="support-agent-system-prompt",
    version="3.2",
    provider="internal",
    properties={"category": "customer_support", "reviewed": True},
)

# Add a known vulnerability
bom_generator.add_vulnerability(
    component_type="framework",
    component_name="langchain",
    vulnerability_id="CVE-2024-EXAMPLE",
    severity="medium",
    description="Example vulnerability for demonstration",
    source="NVD",
)

# --- Generate AI-BOM in Multiple Formats ---

print("=== Component Summary ===\n")
summary = bom_generator.get_component_summary()
print(json.dumps(summary, indent=2))

# AITF format
print("\n=== AI-BOM (AITF Format) ===\n")
bom = bom_generator.generate(bom_id="bom-cs-ai-2025-001")
print(bom.to_json())

# CycloneDX format
print("\n=== AI-BOM (CycloneDX Format) ===\n")
cdx = bom.to_cyclonedx()
print(json.dumps(cdx, indent=2, default=str))

# SPDX format
print("\n=== AI-BOM (SPDX Format) ===\n")
spdx = bom.to_spdx()
print(json.dumps(spdx, indent=2, default=str))

# --- Summary Statistics ---

print("\n=== Final Statistics ===\n")
print(f"Total components discovered: {bom.component_count}")
print(f"Component types: {bom.component_types}")
print(f"Vulnerabilities: {len(bom.vulnerabilities)}")
print(f"Vulnerability summary: {bom.vulnerability_summary}")

provider.shutdown()
