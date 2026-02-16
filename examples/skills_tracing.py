"""AITF Example: Skills Tracing.

Demonstrates how to trace skill operations with AITF, including
skill discovery, invocation, error handling, and composition
workflows.

Usage:
    pip install opentelemetry-sdk aitf
    python skills_tracing.py
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from aitf.instrumentation.skills import SkillInstrumentor
from aitf.exporters.ocsf_exporter import OCSFExporter

# --- Setup ---

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(OCSFExporter(
    output_file="/tmp/aitf_skills_events.jsonl",
    compliance_frameworks=["nist_ai_rmf", "eu_ai_act"],
)))
trace.set_tracer_provider(provider)

skill_instr = SkillInstrumentor(tracer_provider=provider)
skill_instr.instrument()

# --- Example 1: Skill Discovery ---

print("=== Example 1: Skill Discovery ===\n")

with skill_instr.trace_discover(
    source="mcp://tools-registry:3001",
    filter_category="search",
) as discovery:
    # Simulate discovering available skills from a registry
    discovered = ["web-search", "code-search", "document-search"]
    discovery.set_skills(discovered)

print(f"  Discovered {len(discovered)} skills from registry.\n")

# --- Example 2: Basic Skill Invocation ---

print("=== Example 2: Basic Skill Invocation ===\n")

with skill_instr.trace_invoke(
    skill_name="web-search",
    version="2.1.0",
    skill_id="skill-ws-001",
    provider="builtin",
    category="search",
    description="Search the web for relevant information",
    skill_input='{"query": "AI telemetry standards 2026"}',
    source="mcp://tools-registry:3001",
    permissions=["network:read"],
) as invocation:
    # Simulate skill execution
    result = '[{"title": "AITF v1.0 Released", "url": "https://example.com/aitf"}]'
    invocation.set_output(result)

print("  web-search invocation complete.\n")

# --- Example 3: Skill Invocation with Error Handling ---

print("=== Example 3: Skill Invocation with Error and Retry ===\n")

with skill_instr.trace_invoke(
    skill_name="database-query",
    version="1.3.0",
    provider="custom",
    category="data",
    description="Query internal analytics database",
    skill_input='{"sql": "SELECT count(*) FROM events WHERE date > now() - interval 7 day"}',
    permissions=["database:read"],
) as invocation:
    # Simulate a transient error followed by success after retry
    invocation.set_retry_count(1)
    invocation.set_error(
        error_type="ConnectionTimeout",
        message="Database connection timed out on first attempt",
        retryable=True,
    )
    # After retry, set successful output
    invocation.set_output('{"count": 14523}')
    invocation.set_status("success")

print("  database-query invocation with retry complete.\n")

# --- Example 4: Skill Composition (Sequential Workflow) ---

print("=== Example 4: Sequential Skill Composition ===\n")

with skill_instr.trace_compose(
    workflow_name="research-and-summarize",
    skills=["web-search", "document-parse", "text-summarize"],
    pattern="sequential",
) as composition:

    # Step 1: Search the web
    with skill_instr.trace_invoke(
        skill_name="web-search",
        version="2.1.0",
        provider="builtin",
        category="search",
        skill_input='{"query": "OCSF AI telemetry category 7"}',
    ) as inv:
        inv.set_output('[{"title": "OCSF Cat 7 Spec", "url": "https://example.com/ocsf"}]')
    composition.mark_completed()

    # Step 2: Parse the document
    with skill_instr.trace_invoke(
        skill_name="document-parse",
        version="1.0.0",
        provider="builtin",
        category="parsing",
        skill_input='{"url": "https://example.com/ocsf"}',
    ) as inv:
        inv.set_output("OCSF Category 7 defines event classes for AI system activity...")
    composition.mark_completed()

    # Step 3: Summarize
    with skill_instr.trace_invoke(
        skill_name="text-summarize",
        version="1.2.0",
        provider="builtin",
        category="nlp",
        skill_input='{"text": "OCSF Category 7 defines event classes for AI system activity..."}',
    ) as inv:
        inv.set_output("OCSF Cat 7 covers AI inference, agent, tool, and security events.")
    composition.mark_completed()

print(f"  Workflow completed: {composition.completed_count}/3 skills.\n")

# --- Example 5: Parallel Skill Composition ---

print("=== Example 5: Parallel Skill Composition ===\n")

with skill_instr.trace_compose(
    workflow_name="multi-source-search",
    skills=["web-search", "code-search", "document-search"],
    pattern="parallel",
) as composition:

    # In a real application these would run concurrently;
    # here we trace them sequentially for demonstration.
    with skill_instr.trace_invoke(
        skill_name="web-search",
        version="2.1.0",
        provider="builtin",
        category="search",
        skill_input='{"query": "AITF framework"}',
    ) as inv:
        inv.set_output('[{"title": "AITF Docs"}]')
    composition.mark_completed()

    with skill_instr.trace_invoke(
        skill_name="code-search",
        version="1.0.0",
        provider="builtin",
        category="search",
        skill_input='{"query": "SkillInstrumentor", "repo": "aitf-sdk"}',
    ) as inv:
        inv.set_output('[{"file": "skills.py", "line": 22}]')
    composition.mark_completed()

    with skill_instr.trace_invoke(
        skill_name="document-search",
        version="1.1.0",
        provider="builtin",
        category="search",
        skill_input='{"query": "skill tracing", "index": "docs"}',
    ) as inv:
        inv.set_output('[{"doc": "skills-guide.md", "section": "Tracing"}]')
    composition.mark_completed()

print(f"  Parallel workflow completed: {composition.completed_count}/3 skills.\n")

# --- Summary ---

print("Skills tracing complete. Events at /tmp/aitf_skills_events.jsonl")
