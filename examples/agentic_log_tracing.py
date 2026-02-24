"""AITF Example: Agentic Log Tracing (Table 10.1 Minimal Fields).

Demonstrates how to create structured agentic log entries that capture
security-relevant context for every action taken by an AI agent.

Based on Table 10.1: Agentic log with minimum fields, using the
Logi-Agent example from the specification.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from aitf.instrumentation.agentic_log import AgenticLogInstrumentor
from aitf.instrumentation.agent import AgentInstrumentor
from aitf.exporters.ocsf_exporter import OCSFExporter

# --- Setup ---

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(OCSFExporter(
    output_file="/tmp/aitf_agentic_log_events.jsonl",
    compliance_frameworks=["nist_ai_rmf", "eu_ai_act"],
)))
trace.set_tracer_provider(provider)

agentic_log = AgenticLogInstrumentor(tracer_provider=provider)
agent_instr = AgentInstrumentor(tracer_provider=provider)

# --- Example 1: Basic Agentic Log Entry (Logi-Agent from Table 10.1) ---

print("=== Example 1: Logi-Agent Log Entry (Table 10.1) ===\n")

with agentic_log.log_action(
    agent_id="agent-innovacorp-logicore-prod-042",
    session_id="sess-f0a1b2",
    event_id="e-44b1c8f0",
) as entry:
    # GoalID: The single most important field for security context
    entry.set_goal_id("goal-resolve-port-congestion-sg")

    # SubTaskID: Granular context for the specific action
    entry.set_sub_task_id("task-find-all-trucking-vendor")

    # ToolUsed: Pinpoints the exact capability being used
    entry.set_tool_used("mcp.server.github.list_tools")

    # ToolParameters: Sanitized (PII/credentials redacted)
    entry.set_tool_parameters({"repo": "innovacorp logistics-tools"})

    # Outcome: Basic operational monitoring
    entry.set_outcome("SUCCESS")

    # ConfidenceScore: A sudden drop can indicate a poisoned environment
    entry.set_confidence_score(0.92)

    # AnomalyScore: The primary input for automated alerting
    entry.set_anomaly_score(0.15)

    # PolicyEvaluation: Proactive compliance and guardrail enforcement
    entry.set_policy_evaluation({
        "policy": "max_spend",
        "shipment": True,
        "result": "PASS",
    })

print(f"  Event ID: {entry.event_id}")
print(f"  Timestamp: {entry.timestamp}")

# --- Example 2: Agentic Log within Agent Session ---

print("\n=== Example 2: Agentic Log within Agent Session ===\n")

with agent_instr.trace_session(
    agent_name="supply-chain-optimizer",
    agent_id="agent-innovacorp-sco-prod-007",
    framework="custom",
    description="Optimizes supply chain routes and vendor selection",
) as session:

    # Step 1: Planning - logged with agentic log
    with session.step("planning") as step:
        step.set_thought("Need to analyze port congestion data")

        with agentic_log.log_action(
            agent_id="agent-innovacorp-sco-prod-007",
            session_id="sess-d3e4f5",
            goal_id="goal-optimize-asia-pacific-routes",
            sub_task_id="task-analyze-port-congestion",
            tool_used="internal.analytics.port_status",
            tool_parameters={"region": "asia-pacific", "ports": ["SGSIN", "CNSHA"]},
            confidence_score=0.88,
        ) as log_entry:
            log_entry.set_outcome("SUCCESS")
            log_entry.set_anomaly_score(0.05)
            log_entry.set_policy_evaluation({
                "policy": "data_access_scope",
                "result": "PASS",
            })

    # Step 2: Tool Use - with higher anomaly score
    with session.step("tool_use") as step:
        step.set_action("query vendor database")

        with agentic_log.log_action(
            agent_id="agent-innovacorp-sco-prod-007",
            session_id="sess-d3e4f5",
            goal_id="goal-optimize-asia-pacific-routes",
            sub_task_id="task-query-vendor-pricing",
            tool_used="mcp.server.vendor-db.query",
            tool_parameters={
                "query_type": "pricing",
                "vendor_category": "trucking",
                "region": "singapore",
            },
            confidence_score=0.75,
        ) as log_entry:
            log_entry.set_outcome("SUCCESS")
            # Higher anomaly: this agent doesn't usually query pricing
            log_entry.set_anomaly_score(0.45)
            log_entry.set_policy_evaluation({
                "policy": "vendor_data_access",
                "result": "WARN",
                "reason": "Agent accessing pricing data outside normal pattern",
            })

        step.set_observation("Retrieved pricing for 12 trucking vendors")

    # Step 3: Denied action - policy blocks
    with session.step("tool_use") as step:
        step.set_action("attempt to modify vendor contract")

        with agentic_log.log_action(
            agent_id="agent-innovacorp-sco-prod-007",
            session_id="sess-d3e4f5",
            goal_id="goal-optimize-asia-pacific-routes",
            sub_task_id="task-update-vendor-contract",
            tool_used="mcp.server.vendor-db.update_contract",
            tool_parameters={
                "vendor_id": "VENDOR-TRK-042",
                "field": "rate_per_km",
            },
        ) as log_entry:
            log_entry.set_outcome("DENIED")
            log_entry.set_confidence_score(0.30)
            log_entry.set_anomaly_score(0.85)
            log_entry.set_policy_evaluation({
                "policy": "write_access_contract",
                "result": "FAIL",
                "reason": "Agent lacks write permissions to vendor contracts",
            })

        step.set_observation("Action denied by policy engine")
        step.set_status("denied")

# --- Example 3: Using kwargs for concise logging ---

print("\n=== Example 3: Concise Log Entry with kwargs ===\n")

with agentic_log.log_action(
    agent_id="agent-monitoring-sentinel-001",
    session_id="sess-g6h7i8",
    goal_id="goal-continuous-security-scan",
    sub_task_id="task-scan-api-endpoints",
    tool_used="security.scanner.api_audit",
    tool_parameters={"target": "api.internal.innovacorp.com", "scan_type": "full"},
    confidence_score=0.99,
    anomaly_score=0.02,
) as entry:
    entry.set_outcome("SUCCESS")
    entry.set_policy_evaluation({
        "policy": "scheduled_scan",
        "result": "PASS",
    })

print(f"  Event ID: {entry.event_id}")
print(f"  Timestamp: {entry.timestamp}")

print("\nAgentic log tracing complete. Events at /tmp/aitf_agentic_log_events.jsonl")
