"""AITF Example: MCP Protocol Tracing.

Demonstrates how to trace MCP (Model Context Protocol) operations
including server connections, tool discovery, tool invocation,
resource access, and sampling.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from aitf.instrumentation.mcp import MCPInstrumentor
from aitf.instrumentation.llm import LLMInstrumentor
from aitf.exporters.ocsf_exporter import OCSFExporter

# --- Setup ---

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(OCSFExporter(
    output_file="/tmp/aitf_mcp_events.jsonl",
    compliance_frameworks=["nist_ai_rmf", "mitre_atlas", "eu_ai_act"],
)))
trace.set_tracer_provider(provider)

mcp = MCPInstrumentor(tracer_provider=provider)
llm = LLMInstrumentor(tracer_provider=provider)

# --- MCP Server Connection and Tool Use ---

print("=== Example 1: MCP Filesystem Server ===\n")

with mcp.trace_server_connect(
    server_name="filesystem",
    transport="stdio",
    protocol_version="2025-03-26",
) as conn:
    # Set server capabilities
    conn.set_capabilities(
        tools=True,
        resources=True,
        prompts=False,
        sampling=False,
    )

    # Discover available tools
    with conn.discover_tools() as discovery:
        discovery.set_tools(["read_file", "write_file", "list_directory", "search_files"])

    # Read a file via MCP
    with mcp.trace_tool_invoke(
        tool_name="read_file",
        server_name="filesystem",
        tool_input='{"path": "/data/config.yaml"}',
    ) as invocation:
        # Simulate tool execution
        invocation.set_output("server:\n  port: 8080\n  host: localhost", "text")

    # Write a file via MCP (requires approval)
    with mcp.trace_tool_invoke(
        tool_name="write_file",
        server_name="filesystem",
        tool_input='{"path": "/data/output.txt", "content": "Analysis results..."}',
        approval_required=True,
    ) as invocation:
        invocation.set_approved(approved=True, approver="user@example.com")
        invocation.set_output("File written successfully", "text")

# --- MCP Resource Access ---

print("\n=== Example 2: MCP Resource Access ===\n")

with mcp.trace_resource_read(
    resource_uri="file:///data/training_data.csv",
    server_name="filesystem",
    resource_name="training_data.csv",
    mime_type="text/csv",
) as span:
    span.set_attribute("aitf.mcp.resource.size_bytes", 1048576)

# --- MCP Prompt Retrieval ---

print("\n=== Example 3: MCP Prompt Template ===\n")

with mcp.trace_prompt_get(
    prompt_name="summarize",
    server_name="prompts-server",
    arguments='{"style": "executive", "max_length": 500}',
) as span:
    pass  # Prompt retrieved

# --- MCP Sampling (Server-Initiated LLM Request) ---

print("\n=== Example 4: MCP Sampling ===\n")

with mcp.trace_sampling_request(
    server_name="code-analyzer",
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
) as span:
    # The MCP server requests an LLM completion
    with llm.trace_inference(
        model="claude-sonnet-4-5-20250929",
        system="anthropic",
        operation="chat",
    ) as llm_span:
        llm_span.set_prompt("Analyze this code for security issues...")
        llm_span.set_completion("Found 2 potential issues: ...")
        llm_span.set_usage(input_tokens=200, output_tokens=150)

# --- Multiple MCP Servers ---

print("\n=== Example 5: Multiple MCP Servers ===\n")

# Connect to database MCP server
with mcp.trace_server_connect(
    server_name="postgres",
    transport="streamable_http",
    server_url="http://localhost:3001/mcp",
) as conn:
    conn.set_capabilities(tools=True, resources=True)

    with mcp.trace_tool_invoke(
        tool_name="query",
        server_name="postgres",
        tool_input='{"sql": "SELECT count(*) FROM users"}',
    ) as invocation:
        invocation.set_output('{"count": 1523}', "application/json")

# Connect to web search MCP server
with mcp.trace_server_connect(
    server_name="brave-search",
    transport="sse",
    server_url="http://localhost:3002/mcp",
) as conn:
    conn.set_capabilities(tools=True)

    with mcp.trace_tool_invoke(
        tool_name="web_search",
        server_name="brave-search",
        tool_input='{"query": "AITF framework 2026"}',
    ) as invocation:
        invocation.set_output('[{"title": "AITF v1.0", "snippet": "..."}]', "application/json")

print("\nMCP tracing complete. Events at /tmp/aitf_mcp_events.jsonl")
