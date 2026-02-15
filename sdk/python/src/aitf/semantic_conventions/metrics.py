"""AITF Metric name constants."""


class AITFMetrics:
    """All AITF metric names."""

    # OTel GenAI metrics (preserved)
    GEN_AI_TOKEN_USAGE = "gen_ai.client.token.usage"
    GEN_AI_OPERATION_DURATION = "gen_ai.client.operation.duration"

    # Inference metrics
    INFERENCE_REQUESTS = "aitf.inference.requests"
    INFERENCE_ERRORS = "aitf.inference.errors"
    INFERENCE_TTFT = "aitf.inference.time_to_first_token"
    INFERENCE_TPS = "aitf.inference.tokens_per_second"

    # Agent metrics
    AGENT_SESSIONS = "aitf.agent.sessions"
    AGENT_STEPS = "aitf.agent.steps"
    AGENT_SESSION_DURATION = "aitf.agent.session.duration"
    AGENT_STEPS_PER_SESSION = "aitf.agent.steps_per_session"
    AGENT_DELEGATIONS = "aitf.agent.delegations"

    # MCP metrics
    MCP_TOOL_INVOCATIONS = "aitf.mcp.tool.invocations"
    MCP_TOOL_DURATION = "aitf.mcp.tool.duration"
    MCP_SERVER_CONNECTIONS = "aitf.mcp.server.connections"
    MCP_TOOL_APPROVALS = "aitf.mcp.tool.approvals"

    # Skill metrics
    SKILL_INVOCATIONS = "aitf.skill.invocations"
    SKILL_DURATION = "aitf.skill.duration"

    # Cost metrics
    COST_TOTAL = "aitf.cost.total"
    COST_BUDGET_UTILIZATION = "aitf.cost.budget.utilization"

    # Security metrics
    SECURITY_THREATS_DETECTED = "aitf.security.threats_detected"
    SECURITY_REQUESTS_BLOCKED = "aitf.security.requests_blocked"
    SECURITY_PII_DETECTED = "aitf.security.pii_detected"
    SECURITY_GUARDRAIL_CHECKS = "aitf.security.guardrail.checks"

    # RAG metrics
    RAG_RETRIEVALS = "aitf.rag.retrievals"
    RAG_RETRIEVAL_DURATION = "aitf.rag.retrieval.duration"
    RAG_RETRIEVAL_RESULTS = "aitf.rag.retrieval.results"

    # Quality metrics
    QUALITY_HALLUCINATION = "aitf.quality.hallucination"
    QUALITY_USER_RATING = "aitf.quality.user_rating"
