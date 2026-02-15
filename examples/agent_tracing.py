"""AITF Example: Agent Tracing.

Demonstrates how to trace AI agent sessions with steps,
tool use, delegation, and memory access.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from aitf.instrumentation.agent import AgentInstrumentor
from aitf.instrumentation.llm import LLMInstrumentor
from aitf.instrumentation.skills import SkillInstrumentor
from aitf.exporters.ocsf_exporter import OCSFExporter

# --- Setup ---

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(OCSFExporter(
    output_file="/tmp/aitf_agent_events.jsonl",
    compliance_frameworks=["nist_ai_rmf", "eu_ai_act"],
)))
trace.set_tracer_provider(provider)

agent_instr = AgentInstrumentor(tracer_provider=provider)
llm_instr = LLMInstrumentor(tracer_provider=provider)
skill_instr = SkillInstrumentor(tracer_provider=provider)

# --- Single Agent Session ---

print("=== Example 1: Single Agent Session ===\n")

with agent_instr.trace_session(
    agent_name="research-agent",
    agent_type="autonomous",
    framework="langchain",
    description="Researches technical topics and produces summaries",
) as session:

    # Step 1: Planning
    with session.step("planning") as step:
        step.set_thought("User wants to know about AI telemetry. I should search first.")
        step.set_action("call_tool:web-search")

        # LLM call within agent step
        with llm_instr.trace_inference(
            model="gpt-4o", system="openai", operation="chat"
        ) as llm_span:
            llm_span.set_prompt("Plan research on AI telemetry frameworks")
            llm_span.set_completion("I'll search the web, then summarize findings.")
            llm_span.set_usage(input_tokens=20, output_tokens=30)

    # Step 2: Tool Use
    with session.step("tool_use") as step:
        step.set_action("web-search: AI telemetry frameworks 2026")

        # Skill invocation within agent step
        with skill_instr.trace_invoke(
            skill_name="web-search",
            version="2.1.0",
            category="search",
            provider="builtin",
            skill_input='{"query": "AI telemetry frameworks 2026"}',
        ) as skill:
            # Simulate search
            skill.set_output('[{"title": "AITF v1.0", "url": "https://example.com"}]')

        step.set_observation("Found 5 relevant results about AI telemetry.")

    # Step 3: Memory Store
    with session.memory_access(
        operation="store",
        store="short_term",
        key="search_results",
    ) as mem_span:
        mem_span.set_attribute("aitf.memory.provenance", "tool_result")

    # Step 4: Reasoning
    with session.step("reasoning") as step:
        step.set_thought("I have search results. Let me synthesize a summary.")

        with llm_instr.trace_inference(
            model="gpt-4o", system="openai", operation="chat"
        ) as llm_span:
            llm_span.set_usage(input_tokens=500, output_tokens=200)

    # Step 5: Response
    with session.step("response") as step:
        step.set_observation("Generated comprehensive summary of AI telemetry frameworks.")
        step.set_status("success")


# --- Multi-Agent Team ---

print("\n=== Example 2: Multi-Agent Team ===\n")

with agent_instr.trace_team(
    team_name="research-team",
    topology="hierarchical",
    members=["manager", "researcher", "writer"],
    coordinator="manager",
) as team_span:

    # Manager agent session
    with agent_instr.trace_session(
        agent_name="manager",
        agent_type="autonomous",
        framework="crewai",
        team_name="research-team",
    ) as manager:

        # Manager plans and delegates
        with manager.step("planning") as step:
            step.set_thought("Need to delegate research and writing tasks.")

        # Delegate to researcher
        with manager.delegate(
            target_agent="researcher",
            reason="Research expertise needed",
            strategy="capability",
            task="Research AI telemetry standards",
        ):
            # Researcher agent session
            with agent_instr.trace_session(
                agent_name="researcher",
                framework="crewai",
                team_name="research-team",
            ) as researcher:
                with researcher.step("tool_use") as step:
                    step.set_action("search AI telemetry")
                    step.set_observation("Found relevant papers and specs.")

        # Delegate to writer
        with manager.delegate(
            target_agent="writer",
            reason="Writing expertise needed",
            strategy="capability",
            task="Write summary document",
        ):
            with agent_instr.trace_session(
                agent_name="writer",
                framework="crewai",
                team_name="research-team",
            ) as writer:
                with writer.step("response") as step:
                    with llm_instr.trace_inference(
                        model="gpt-4o", system="openai", operation="chat"
                    ) as llm_span:
                        llm_span.set_usage(input_tokens=800, output_tokens=500)
                    step.set_observation("Summary document written.")

print("\nAgent tracing complete. Events at /tmp/aitf_agent_events.jsonl")
