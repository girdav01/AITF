"""AITF Example: Basic LLM Tracing.

Demonstrates how to trace LLM inference operations with AITF,
including cost tracking and security processing.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from aitf.instrumentation.llm import LLMInstrumentor
from aitf.processors.cost_processor import CostProcessor
from aitf.processors.security_processor import SecurityProcessor
from aitf.exporters.ocsf_exporter import OCSFExporter

# --- Setup ---

# Create TracerProvider with AITF processors
provider = TracerProvider()

# Add security processor (detects OWASP LLM Top 10 threats)
provider.add_span_processor(SecurityProcessor(
    detect_prompt_injection=True,
    detect_jailbreak=True,
    detect_data_exfiltration=True,
))

# Add cost processor (tracks token costs)
provider.add_span_processor(CostProcessor(
    default_project="demo-app",
    budget_limit=10.0,  # $10 budget
))

# Add OCSF exporter (exports to SIEM-compatible format)
ocsf_exporter = OCSFExporter(
    output_file="/tmp/aitf_ocsf_events.jsonl",
    compliance_frameworks=["nist_ai_rmf", "mitre_atlas", "eu_ai_act"],
)
provider.add_span_processor(SimpleSpanProcessor(ocsf_exporter))

# Also export to console for demo
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(provider)

# --- Tracing LLM Calls ---

llm = LLMInstrumentor(tracer_provider=provider)
llm.instrument()

# Example 1: Basic chat completion
print("=== Example 1: Basic Chat Completion ===\n")

with llm.trace_inference(
    model="gpt-4o",
    system="openai",
    operation="chat",
    temperature=0.7,
    max_tokens=1000,
) as span:
    # Simulate LLM call
    span.set_prompt("Explain the AITF framework in 3 sentences.")
    # ... actual API call would go here ...
    span.set_completion(
        "AITF is a security-first telemetry framework for AI systems. "
        "It extends OpenTelemetry GenAI with native MCP, agent, and skills support. "
        "Every event is mapped to OCSF for SIEM/XDR integration."
    )
    span.set_response(
        response_id="chatcmpl-abc123",
        model="gpt-4o-2024-08-06",
        finish_reasons=["stop"],
    )
    span.set_usage(input_tokens=25, output_tokens=45)
    span.set_cost(input_cost=0.0000625, output_cost=0.00045)
    span.set_latency(total_ms=850.0, tokens_per_second=52.9)

# Example 2: Streaming with tool calls
print("\n=== Example 2: Streaming with Tool Calls ===\n")

with llm.trace_inference(
    model="claude-sonnet-4-5-20250929",
    system="anthropic",
    operation="chat",
    stream=True,
    max_tokens=4096,
    tools=[{"name": "search", "description": "Search the web"}],
) as span:
    span.set_prompt("Search for the latest AI telemetry standards.")
    span.mark_first_token()  # First token received
    span.set_tool_call(
        name="search",
        call_id="call_001",
        arguments='{"query": "AI telemetry standards 2026"}',
    )
    span.set_tool_result(
        name="search",
        call_id="call_001",
        result='[{"title": "AITF v1.0", "url": "..."}]',
    )
    span.set_completion("Based on my search, the latest standards include...")
    span.set_response(
        response_id="msg_xyz789",
        model="claude-sonnet-4-5-20250929",
        finish_reasons=["end_turn"],
    )
    span.set_usage(input_tokens=150, output_tokens=300)
    span.set_cost(input_cost=0.00045, output_cost=0.0045)

# Example 3: Embeddings
print("\n=== Example 3: Embeddings ===\n")

with llm.trace_inference(
    model="text-embedding-3-small",
    system="openai",
    operation="embeddings",
) as span:
    span.set_prompt("AITF AI Telemetry Framework")
    span.set_usage(input_tokens=5)
    span.set_cost(input_cost=0.0000001)
    span.set_latency(total_ms=120.0)

# Print summary
print(f"\nOCSF events exported: {ocsf_exporter.event_count}")
print("Events written to: /tmp/aitf_ocsf_events.jsonl")
