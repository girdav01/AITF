// Package semconv provides AITF metric name constants.
package semconv

// Metric names for AITF telemetry.
const (
	// OTel GenAI metrics (preserved)
	MetricGenAITokenUsage       = "gen_ai.client.token.usage"
	MetricGenAIOperationDuration = "gen_ai.client.operation.duration"

	// Inference metrics
	MetricInferenceRequests = "aitf.inference.requests"
	MetricInferenceErrors   = "aitf.inference.errors"
	MetricInferenceTTFT     = "aitf.inference.time_to_first_token"
	MetricInferenceTPS      = "aitf.inference.tokens_per_second"

	// Agent metrics
	MetricAgentSessions        = "aitf.agent.sessions"
	MetricAgentSteps           = "aitf.agent.steps"
	MetricAgentSessionDuration = "aitf.agent.session.duration"
	MetricAgentDelegations     = "aitf.agent.delegations"

	// MCP metrics
	MetricMCPToolInvocations   = "aitf.mcp.tool.invocations"
	MetricMCPToolDuration      = "aitf.mcp.tool.duration"
	MetricMCPServerConnections = "aitf.mcp.server.connections"
	MetricMCPToolApprovals     = "aitf.mcp.tool.approvals"

	// Skill metrics
	MetricSkillInvocations = "aitf.skill.invocations"
	MetricSkillDuration    = "aitf.skill.duration"

	// Cost metrics
	MetricCostTotal             = "aitf.cost.total"
	MetricCostBudgetUtilization = "aitf.cost.budget.utilization"

	// Security metrics
	MetricSecurityThreats    = "aitf.security.threats_detected"
	MetricSecurityBlocked    = "aitf.security.requests_blocked"
	MetricSecurityPII        = "aitf.security.pii_detected"
	MetricSecurityGuardrails = "aitf.security.guardrail.checks"

	// RAG metrics
	MetricRAGRetrievals       = "aitf.rag.retrievals"
	MetricRAGRetrievalDuration = "aitf.rag.retrieval.duration"

	// Quality metrics
	MetricQualityHallucination = "aitf.quality.hallucination"
	MetricQualityUserRating    = "aitf.quality.user_rating"
)
