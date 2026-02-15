"""AITF OCSF Mapper.

Maps OpenTelemetry spans to OCSF Category 7 AI events.
Based on the OCSF mapper from the AITelemetry project, enhanced
for AITF with MCP, Skills, and extended agent support.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan

from aitf.ocsf.event_classes import (
    AIAgentActivityEvent,
    AIDataRetrievalEvent,
    AIModelInferenceEvent,
    AISecurityFindingEvent,
    AIToolExecutionEvent,
)
from aitf.ocsf.schema import (
    AIBaseEvent,
    AICostInfo,
    AILatencyMetrics,
    AIModelInfo,
    AISecurityFinding,
    AITokenUsage,
    OCSFActor,
    OCSFMetadata,
    OCSFSeverity,
    OCSFStatus,
)
from aitf.semantic_conventions.attributes import (
    AgentAttributes,
    CostAttributes,
    GenAIAttributes,
    LatencyAttributes,
    MCPAttributes,
    RAGAttributes,
    SecurityAttributes,
    SkillAttributes,
)


class OCSFMapper:
    """Maps OTel spans to OCSF Category 7 AI events.

    Usage:
        mapper = OCSFMapper()
        ocsf_event = mapper.map_span(span)
        if ocsf_event:
            export_to_siem(ocsf_event.model_dump())
    """

    def map_span(self, span: ReadableSpan) -> AIBaseEvent | None:
        """Map an OTel span to an OCSF event.

        Returns the appropriate OCSF event class or None if the span
        is not an AI-related span.
        """
        name = span.name or ""
        attrs = dict(span.attributes or {})

        # Determine event type and map accordingly
        if self._is_inference_span(name, attrs):
            return self._map_inference(span, attrs)
        elif self._is_agent_span(name, attrs):
            return self._map_agent_activity(span, attrs)
        elif self._is_tool_span(name, attrs):
            return self._map_tool_execution(span, attrs)
        elif self._is_rag_span(name, attrs):
            return self._map_data_retrieval(span, attrs)
        elif self._is_security_span(name, attrs):
            return self._map_security_finding(span, attrs)

        return None

    def _is_inference_span(self, name: str, attrs: dict) -> bool:
        return (
            name.startswith("chat ")
            or name.startswith("embeddings ")
            or name.startswith("text_completion ")
            or GenAIAttributes.SYSTEM in attrs
        )

    def _is_agent_span(self, name: str, attrs: dict) -> bool:
        return name.startswith("agent.") or AgentAttributes.NAME in attrs

    def _is_tool_span(self, name: str, attrs: dict) -> bool:
        return (
            name.startswith("mcp.tool.")
            or name.startswith("skill.invoke")
            or MCPAttributes.TOOL_NAME in attrs
            or SkillAttributes.NAME in attrs
        )

    def _is_rag_span(self, name: str, attrs: dict) -> bool:
        return (
            name.startswith("rag.")
            or RAGAttributes.RETRIEVE_DATABASE in attrs
        )

    def _is_security_span(self, name: str, attrs: dict) -> bool:
        return SecurityAttributes.THREAT_DETECTED in attrs

    def _map_inference(self, span: ReadableSpan, attrs: dict) -> AIModelInferenceEvent:
        """Map inference span to OCSF 7001."""
        model_id = str(attrs.get(GenAIAttributes.REQUEST_MODEL, "unknown"))
        system = str(attrs.get(GenAIAttributes.SYSTEM, "unknown"))
        operation = str(attrs.get(GenAIAttributes.OPERATION_NAME, "chat"))

        # Activity: 1=chat, 2=text_completion, 3=embeddings
        activity_map = {"chat": 1, "text_completion": 2, "embeddings": 3}
        activity_id = activity_map.get(operation, 99)

        model = AIModelInfo(
            model_id=model_id,
            name=model_id,
            provider=system,
            type="llm" if operation != "embeddings" else "embedding",
            parameters={
                k.replace("gen_ai.request.", ""): v
                for k, v in attrs.items()
                if k.startswith("gen_ai.request.") and k != GenAIAttributes.REQUEST_MODEL
            } or None,
        )

        token_usage = AITokenUsage(
            input_tokens=int(attrs.get(GenAIAttributes.USAGE_INPUT_TOKENS, 0)),
            output_tokens=int(attrs.get(GenAIAttributes.USAGE_OUTPUT_TOKENS, 0)),
            cached_tokens=int(attrs.get(GenAIAttributes.USAGE_CACHED_TOKENS, 0)),
            reasoning_tokens=int(attrs.get(GenAIAttributes.USAGE_REASONING_TOKENS, 0)),
        )

        latency = None
        if LatencyAttributes.TOTAL_MS in attrs:
            latency = AILatencyMetrics(
                total_ms=float(attrs.get(LatencyAttributes.TOTAL_MS, 0)),
                time_to_first_token_ms=_opt_float(attrs.get(LatencyAttributes.TIME_TO_FIRST_TOKEN_MS)),
                tokens_per_second=_opt_float(attrs.get(LatencyAttributes.TOKENS_PER_SECOND)),
            )

        cost = None
        if CostAttributes.TOTAL_COST in attrs:
            cost = AICostInfo(
                input_cost_usd=float(attrs.get(CostAttributes.INPUT_COST, 0)),
                output_cost_usd=float(attrs.get(CostAttributes.OUTPUT_COST, 0)),
                total_cost_usd=float(attrs.get(CostAttributes.TOTAL_COST, 0)),
            )

        finish_reasons = attrs.get(GenAIAttributes.RESPONSE_FINISH_REASONS, ["stop"])
        finish_reason = finish_reasons[0] if isinstance(finish_reasons, (list, tuple)) else str(finish_reasons)

        return AIModelInferenceEvent(
            activity_id=activity_id,
            model=model,
            token_usage=token_usage,
            latency=latency,
            cost=cost,
            finish_reason=finish_reason,
            streaming=bool(attrs.get(GenAIAttributes.REQUEST_STREAM, False)),
            message=f"{operation} {model_id}",
            time=_span_time(span),
        )

    def _map_agent_activity(self, span: ReadableSpan, attrs: dict) -> AIAgentActivityEvent:
        """Map agent span to OCSF 7002."""
        name = span.name or ""
        agent_name = str(attrs.get(AgentAttributes.NAME, "unknown"))
        agent_id = str(attrs.get(AgentAttributes.ID, "unknown"))
        session_id = str(attrs.get(AgentAttributes.SESSION_ID, "unknown"))

        # Determine activity from span name
        if "session" in name:
            activity_id = 1  # Session Start
        elif "delegation" in name or "delegate" in name:
            activity_id = 4  # Delegation
        elif "memory" in name:
            activity_id = 5  # Memory Access
        else:
            activity_id = 3  # Step Execute

        return AIAgentActivityEvent(
            activity_id=activity_id,
            agent_name=agent_name,
            agent_id=agent_id,
            agent_type=str(attrs.get(AgentAttributes.TYPE, "autonomous")),
            framework=_opt_str(attrs.get(AgentAttributes.FRAMEWORK)),
            session_id=session_id,
            step_type=_opt_str(attrs.get(AgentAttributes.STEP_TYPE)),
            step_index=_opt_int(attrs.get(AgentAttributes.STEP_INDEX)),
            thought=_opt_str(attrs.get(AgentAttributes.STEP_THOUGHT)),
            action=_opt_str(attrs.get(AgentAttributes.STEP_ACTION)),
            observation=_opt_str(attrs.get(AgentAttributes.STEP_OBSERVATION)),
            delegation_target=_opt_str(attrs.get(AgentAttributes.DELEGATION_TARGET_AGENT)),
            message=name,
            time=_span_time(span),
        )

    def _map_tool_execution(self, span: ReadableSpan, attrs: dict) -> AIToolExecutionEvent:
        """Map tool/MCP/skill span to OCSF 7003."""
        # Determine tool type
        if MCPAttributes.TOOL_NAME in attrs:
            tool_name = str(attrs[MCPAttributes.TOOL_NAME])
            tool_type = "mcp_tool"
            activity_id = 2  # MCP Tool Invoke
        elif SkillAttributes.NAME in attrs:
            tool_name = str(attrs[SkillAttributes.NAME])
            tool_type = "skill"
            activity_id = 3  # Skill Invoke
        else:
            tool_name = str(attrs.get("gen_ai.tool.name", "unknown"))
            tool_type = "function"
            activity_id = 1  # Function Call

        return AIToolExecutionEvent(
            activity_id=activity_id,
            tool_name=tool_name,
            tool_type=tool_type,
            tool_input=_opt_str(attrs.get(MCPAttributes.TOOL_INPUT) or attrs.get(SkillAttributes.INPUT)),
            tool_output=_opt_str(attrs.get(MCPAttributes.TOOL_OUTPUT) or attrs.get(SkillAttributes.OUTPUT)),
            is_error=bool(attrs.get(MCPAttributes.TOOL_IS_ERROR, False)),
            duration_ms=_opt_float(attrs.get(MCPAttributes.TOOL_DURATION_MS) or attrs.get(SkillAttributes.DURATION_MS)),
            mcp_server=_opt_str(attrs.get(MCPAttributes.TOOL_SERVER)),
            mcp_transport=_opt_str(attrs.get(MCPAttributes.SERVER_TRANSPORT)),
            skill_category=_opt_str(attrs.get(SkillAttributes.CATEGORY)),
            skill_version=_opt_str(attrs.get(SkillAttributes.VERSION)),
            approval_required=bool(attrs.get(MCPAttributes.TOOL_APPROVAL_REQUIRED, False)),
            approved=_opt_bool(attrs.get(MCPAttributes.TOOL_APPROVED)),
            message=span.name or f"tool.execute {tool_name}",
            time=_span_time(span),
        )

    def _map_data_retrieval(self, span: ReadableSpan, attrs: dict) -> AIDataRetrievalEvent:
        """Map RAG/retrieval span to OCSF 7004."""
        database = str(attrs.get(RAGAttributes.RETRIEVE_DATABASE, "unknown"))
        stage = str(attrs.get(RAGAttributes.PIPELINE_STAGE, "retrieve"))

        # Activity: 1=vector_search, 2=document_retrieval, 5=reranking
        activity_map = {"retrieve": 1, "rerank": 5, "generate": 99, "evaluate": 99}
        activity_id = activity_map.get(stage, 99)

        return AIDataRetrievalEvent(
            activity_id=activity_id,
            database_name=database,
            database_type=database,
            query=_opt_str(attrs.get(RAGAttributes.QUERY)),
            top_k=_opt_int(attrs.get(RAGAttributes.RETRIEVE_TOP_K)),
            results_count=int(attrs.get(RAGAttributes.RETRIEVE_RESULTS_COUNT, 0)),
            min_score=_opt_float(attrs.get(RAGAttributes.RETRIEVE_MIN_SCORE)),
            max_score=_opt_float(attrs.get(RAGAttributes.RETRIEVE_MAX_SCORE)),
            filter=_opt_str(attrs.get(RAGAttributes.RETRIEVE_FILTER)),
            embedding_model=_opt_str(attrs.get(RAGAttributes.QUERY_EMBEDDING_MODEL)),
            pipeline_name=_opt_str(attrs.get(RAGAttributes.PIPELINE_NAME)),
            pipeline_stage=stage,
            message=span.name or f"rag.{stage} {database}",
            time=_span_time(span),
        )

    def _map_security_finding(self, span: ReadableSpan, attrs: dict) -> AISecurityFindingEvent:
        """Map security span to OCSF 7005."""
        finding = AISecurityFinding(
            finding_type=str(attrs.get(SecurityAttributes.THREAT_TYPE, "unknown")),
            owasp_category=_opt_str(attrs.get(SecurityAttributes.OWASP_CATEGORY)),
            risk_level=str(attrs.get(SecurityAttributes.RISK_LEVEL, "medium")),
            risk_score=float(attrs.get(SecurityAttributes.RISK_SCORE, 50.0)),
            confidence=float(attrs.get(SecurityAttributes.CONFIDENCE, 0.5)),
            detection_method=str(attrs.get(SecurityAttributes.DETECTION_METHOD, "pattern")),
            blocked=bool(attrs.get(SecurityAttributes.BLOCKED, False)),
        )

        severity_map = {
            "critical": OCSFSeverity.CRITICAL,
            "high": OCSFSeverity.HIGH,
            "medium": OCSFSeverity.MEDIUM,
            "low": OCSFSeverity.LOW,
            "info": OCSFSeverity.INFORMATIONAL,
        }

        return AISecurityFindingEvent(
            activity_id=1,  # Threat Detection
            finding=finding,
            severity_id=severity_map.get(finding.risk_level, OCSFSeverity.MEDIUM),
            message=span.name or f"security.{finding.finding_type}",
            time=_span_time(span),
        )


# --- Utility Functions ---

def _span_time(span: ReadableSpan) -> str:
    if span.start_time:
        return datetime.fromtimestamp(
            span.start_time / 1e9, tz=timezone.utc
        ).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _opt_str(val: Any) -> str | None:
    return str(val) if val is not None else None


def _opt_int(val: Any) -> int | None:
    return int(val) if val is not None else None


def _opt_float(val: Any) -> float | None:
    return float(val) if val is not None else None


def _opt_bool(val: Any) -> bool | None:
    return bool(val) if val is not None else None
