"""AITF Example: Vendor Mapping Pipeline.

Demonstrates how vendor-supplied JSON mapping files normalize
framework-native telemetry (LangChain, CrewAI, etc.) into AITF
semantic conventions before OCSF event generation.

This is the recommended pattern for integrating any agentic framework
into the AITF pipeline: the framework vendor provides a JSON mapping
file, and the VendorMapper handles the translation automatically.

Usage:
    pip install opentelemetry-sdk aitf
    python vendor_mapping_tracing.py
"""

from unittest.mock import MagicMock

from aitf.ocsf.vendor_mapper import VendorMapper
from aitf.ocsf.mapper import OCSFMapper
from aitf.ocsf.compliance_mapper import ComplianceMapper


# ---------------------------------------------------------------------------
# Helper: create mock spans (simulates what OTel would produce)
# ---------------------------------------------------------------------------

def make_span(name: str, attributes: dict) -> MagicMock:
    """Create a mock ReadableSpan for demonstration."""
    span = MagicMock()
    span.name = name
    span.attributes = attributes
    span.start_time = 1700000000_000000000
    return span


# ---------------------------------------------------------------------------
# Setup: load mappers
# ---------------------------------------------------------------------------

vendor_mapper = VendorMapper()   # loads all built-in vendor mappings
ocsf_mapper = OCSFMapper()
compliance_mapper = ComplianceMapper(frameworks=["nist_ai_rmf", "eu_ai_act"])


# ---------------------------------------------------------------------------
# List loaded vendors
# ---------------------------------------------------------------------------

print("=" * 70)
print("AITF Vendor Mapping Pipeline Demo")
print("=" * 70)

print("\n--- Loaded Vendor Mappings ---\n")
for info in vendor_mapper.list_supported_vendors():
    print(f"  Vendor:      {info['vendor']}")
    print(f"  Version:     {info['version']}")
    print(f"  Event Types: {info['event_types']}")
    print(f"  Homepage:    {info['homepage']}")
    print()


# ---------------------------------------------------------------------------
# Example 1: LangChain LLM Inference
# ---------------------------------------------------------------------------

print("=" * 70)
print("Example 1: LangChain ChatOpenAI Inference")
print("=" * 70)

langchain_llm_span = make_span("ChatOpenAI", {
    "ls_provider": "openai",
    "ls_model_name": "gpt-4o",
    "ls_temperature": 0.7,
    "llm.token_count.prompt": 250,
    "llm.token_count.completion": 180,
    "llm.token_count.total": 430,
    "langchain.run.response_id": "chatcmpl-abc123",
    "langchain.run.finish_reason": "stop",
})

result = vendor_mapper.normalize_span(langchain_llm_span)
if result:
    vendor, event_type, aitf_attrs = result
    print(f"\n  Vendor:     {vendor}")
    print(f"  Event Type: {event_type}")
    print(f"  OCSF Class: {vendor_mapper.get_ocsf_class_uid(vendor, event_type)}")
    print(f"\n  Translated Attributes:")
    for key, value in sorted(aitf_attrs.items()):
        print(f"    {key}: {value}")


# ---------------------------------------------------------------------------
# Example 2: LangChain RAG Retrieval
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("Example 2: LangChain Vector Store Retrieval")
print("=" * 70)

langchain_rag_span = make_span("VectorStoreRetriever", {
    "langchain.retriever.name": "pinecone-knowledge-base",
    "langchain.retriever.type": "pinecone",
    "langchain.retriever.k": 10,
    "langchain.retriever.query": "What are best practices for AI security?",
    "langchain.retriever.documents": 8,
    "retriever.embeddings.model": "text-embedding-3-large",
})

result = vendor_mapper.normalize_span(langchain_rag_span)
if result:
    vendor, event_type, aitf_attrs = result
    print(f"\n  Vendor:     {vendor}")
    print(f"  Event Type: {event_type}")
    print(f"  OCSF Class: {vendor_mapper.get_ocsf_class_uid(vendor, event_type)}")
    print(f"\n  Translated Attributes:")
    for key, value in sorted(aitf_attrs.items()):
        print(f"    {key}: {value}")


# ---------------------------------------------------------------------------
# Example 3: CrewAI Agent Execution
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("Example 3: CrewAI Agent Execution")
print("=" * 70)

crewai_agent_span = make_span("Agent Execution", {
    "crewai.agent.role": "security-researcher",
    "crewai.agent.goal": "Analyze threat intelligence reports",
    "crewai.agent.backstory": "Expert in cyber threat analysis",
    "crewai.agent.id": "agent-sec-001",
    "crewai.agent.llm": "claude-sonnet-4-5-20250929",
    "crewai.agent.tools": "web_search,file_read,code_analysis",
    "crewai.crew.name": "threat-intel-crew",
    "crewai.crew.process": "hierarchical",
})

result = vendor_mapper.normalize_span(crewai_agent_span)
if result:
    vendor, event_type, aitf_attrs = result
    print(f"\n  Vendor:     {vendor}")
    print(f"  Event Type: {event_type}")
    print(f"  OCSF Class: {vendor_mapper.get_ocsf_class_uid(vendor, event_type)}")
    print(f"  Activity:   {vendor_mapper.get_ocsf_activity_id(vendor, event_type, 'agent_execution')}")
    print(f"\n  Translated Attributes:")
    for key, value in sorted(aitf_attrs.items()):
        print(f"    {key}: {value}")


# ---------------------------------------------------------------------------
# Example 4: CrewAI LLM Call (via LiteLLM)
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("Example 4: CrewAI LLM Call (via LiteLLM)")
print("=" * 70)

crewai_llm_span = make_span("LLM Call claude-sonnet-4-5-20250929", {
    "crewai.llm.model": "claude-sonnet-4-5-20250929",
    "crewai.llm.provider": "anthropic",
    "crewai.llm.temperature": 0.3,
    "crewai.llm.input_tokens": 1200,
    "crewai.llm.output_tokens": 800,
    "crewai.llm.total_tokens": 2000,
    "crewai.llm.response_time_ms": 2450.5,
    "crewai.llm.finish_reason": "end_turn",
})

result = vendor_mapper.normalize_span(crewai_llm_span)
if result:
    vendor, event_type, aitf_attrs = result
    print(f"\n  Vendor:     {vendor}")
    print(f"  Event Type: {event_type}")
    print(f"  OCSF Class: {vendor_mapper.get_ocsf_class_uid(vendor, event_type)}")
    print(f"\n  Translated Attributes:")
    for key, value in sorted(aitf_attrs.items()):
        print(f"    {key}: {value}")


# ---------------------------------------------------------------------------
# Example 5: CrewAI Task Delegation
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("Example 5: CrewAI Task Delegation")
print("=" * 70)

crewai_delegation_span = make_span("Task Delegation", {
    "crewai.delegation.from_agent": "team-lead",
    "crewai.delegation.to_agent": "security-researcher",
    "crewai.delegation.task": "Investigate CVE-2024-1234 impact assessment",
    "crewai.delegation.reason": "Requires specialized security knowledge",
    "crewai.delegation.strategy": "capability",
})

result = vendor_mapper.normalize_span(crewai_delegation_span)
if result:
    vendor, event_type, aitf_attrs = result
    print(f"\n  Vendor:     {vendor}")
    print(f"  Event Type: {event_type}")
    print(f"  OCSF Class: {vendor_mapper.get_ocsf_class_uid(vendor, event_type)}")
    print(f"  Activity:   {vendor_mapper.get_ocsf_activity_id(vendor, event_type)}")
    print(f"\n  Translated Attributes:")
    for key, value in sorted(aitf_attrs.items()):
        print(f"    {key}: {value}")


# ---------------------------------------------------------------------------
# Example 6: CrewAI Tool Usage
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("Example 6: CrewAI Tool Usage")
print("=" * 70)

crewai_tool_span = make_span("Tool Execution web_search", {
    "crewai.tool.name": "web_search",
    "crewai.tool.description": "Search the web for information",
    "crewai.tool.input": '{"query": "CVE-2024-1234 exploit details"}',
    "crewai.tool.output": "Found 5 results...",
    "crewai.tool.agent": "security-researcher",
    "crewai.tool.duration_ms": 1250.3,
    "crewai.tool.cache_hit": False,
})

result = vendor_mapper.normalize_span(crewai_tool_span)
if result:
    vendor, event_type, aitf_attrs = result
    print(f"\n  Vendor:     {vendor}")
    print(f"  Event Type: {event_type}")
    print(f"  OCSF Class: {vendor_mapper.get_ocsf_class_uid(vendor, event_type)}")
    print(f"\n  Translated Attributes:")
    for key, value in sorted(aitf_attrs.items()):
        print(f"    {key}: {value}")


# ---------------------------------------------------------------------------
# Example 7: Loading a Custom Vendor Mapping
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("Example 7: Loading a Custom Vendor Mapping at Runtime")
print("=" * 70)

import json
import tempfile
from pathlib import Path

custom_mapping = {
    "vendor": "autogen",
    "version": "0.4",
    "description": "Maps AutoGen telemetry to AITF conventions",
    "homepage": "https://microsoft.github.io/autogen/",
    "span_name_patterns": {
        "inference": ["^AutoGen\\.LLM"],
        "agent": ["^AutoGen\\.Agent", "^AutoGen\\.GroupChat"],
    },
    "attribute_mappings": {
        "inference": {
            "vendor_to_aitf": {
                "autogen.llm.model": "gen_ai.request.model",
                "autogen.llm.provider": "gen_ai.system",
            },
            "ocsf_class_uid": 7001,
            "ocsf_activity_id_map": {"chat": 1, "default": 1},
            "defaults": {"gen_ai.operation.name": "chat"},
        },
        "agent": {
            "vendor_to_aitf": {
                "autogen.agent.name": "aitf.agent.name",
                "autogen.agent.type": "aitf.agent.type",
            },
            "ocsf_class_uid": 7002,
            "ocsf_activity_id_map": {"default": 3},
            "defaults": {"aitf.agent.framework": "autogen"},
        },
    },
    "provider_detection": {
        "attribute_keys": ["autogen.llm.provider"],
        "model_prefix_to_provider": {"gpt-": "openai", "claude-": "anthropic"},
    },
    "severity_mapping": {},
    "metadata": {
        "ocsf_product": {
            "name": "AutoGen",
            "vendor_name": "Microsoft",
            "version": "0.4",
        }
    },
}

# Write to a temp file and load it
with tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False
) as f:
    json.dump(custom_mapping, f)
    custom_path = f.name

vendor_mapper.load_file(custom_path)

print(f"\n  Loaded custom mapping: autogen")
print(f"  Total vendors: {len(vendor_mapper.vendors)}")
print(f"  Vendors: {', '.join(vendor_mapper.vendors)}")

# Test it
autogen_span = make_span("AutoGen.Agent.execute", {
    "autogen.agent.name": "code-reviewer",
    "autogen.agent.type": "assistant",
})

result = vendor_mapper.normalize_span(autogen_span)
if result:
    vendor, event_type, aitf_attrs = result
    print(f"\n  Detected: {vendor}/{event_type}")
    print(f"  Translated:")
    for key, value in sorted(aitf_attrs.items()):
        print(f"    {key}: {value}")


# ---------------------------------------------------------------------------
# Example 8: Full Pipeline – Vendor Mapping → OCSF → Compliance
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("Example 8: Full Pipeline (Vendor → AITF → OCSF → Compliance)")
print("=" * 70)

# Simulate a LangChain inference span
span = make_span("ChatAnthropic", {
    "ls_provider": "anthropic",
    "ls_model_name": "claude-sonnet-4-5-20250929",
    "ls_temperature": 0.5,
    "llm.token_count.prompt": 500,
    "llm.token_count.completion": 300,
    "llm.token_count.total": 800,
})

# Step 1: Vendor normalization
norm = vendor_mapper.normalize_span(span)
if norm:
    vendor, event_type, aitf_attrs = norm

    print(f"\n  Step 1 – Vendor Detection:")
    print(f"    Vendor: {vendor}, Event Type: {event_type}")
    print(f"    AITF Attributes: {len(aitf_attrs)} keys")

    # Step 2: Create a normalized span for OCSF mapping
    #   (In production, you'd use a SpanProcessor to mutate the span)
    normalized_span = make_span("chat claude-sonnet-4-5-20250929", {
        **aitf_attrs,
    })

    ocsf_event = ocsf_mapper.map_span(normalized_span)
    if ocsf_event:
        print(f"\n  Step 2 – OCSF Mapping:")
        print(f"    Class UID: {ocsf_event.class_uid}")
        print(f"    Activity ID: {ocsf_event.activity_id}")
        print(f"    Message: {ocsf_event.message}")

        # Step 3: Compliance enrichment
        enriched = compliance_mapper.enrich_event(ocsf_event, "model_inference")
        print(f"\n  Step 3 – Compliance Enrichment:")
        if enriched.compliance:
            if enriched.compliance.nist_ai_rmf:
                print(f"    NIST AI RMF: {enriched.compliance.nist_ai_rmf}")
            if enriched.compliance.eu_ai_act:
                print(f"    EU AI Act:   {enriched.compliance.eu_ai_act}")

        # Final OCSF event
        event_dict = enriched.model_dump(exclude_none=True)
        print(f"\n  Final OCSF Event Keys: {list(event_dict.keys())}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("Pipeline Summary")
print("=" * 70)
print(f"""
  Vendor mapping files provide a declarative way for agentic framework
  vendors to integrate with AITF. Each JSON file defines:

    1. span_name_patterns    – Regex patterns to classify spans
    2. attribute_mappings    – Vendor → AITF attribute translations
    3. provider_detection    – LLM provider inference from model names
    4. severity_mapping      – Vendor status → OCSF severity
    5. metadata              – Vendor product info for OCSF events

  Pipeline: Vendor Telemetry → VendorMapper → OCSFMapper → ComplianceMapper → SIEM

  Built-in vendors: {', '.join(vendor_mapper.vendors)}
""")
