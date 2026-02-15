// Package semconv provides AITF semantic convention constants for Go.
//
// All attribute keys used across AITF instrumentation, processors, and exporters.
// OTel GenAI attributes (gen_ai.*) are preserved for compatibility.
// AITF extensions use the aitf.* namespace.
package semconv

import "go.opentelemetry.io/otel/attribute"

// --- OTel GenAI Attributes (Preserved) ---

const (
	GenAISystemKey        = attribute.Key("gen_ai.system")
	GenAIOperationNameKey = attribute.Key("gen_ai.operation.name")

	// Request
	GenAIRequestModelKey            = attribute.Key("gen_ai.request.model")
	GenAIRequestMaxTokensKey        = attribute.Key("gen_ai.request.max_tokens")
	GenAIRequestTemperatureKey      = attribute.Key("gen_ai.request.temperature")
	GenAIRequestTopPKey             = attribute.Key("gen_ai.request.top_p")
	GenAIRequestTopKKey             = attribute.Key("gen_ai.request.top_k")
	GenAIRequestStreamKey           = attribute.Key("gen_ai.request.stream")
	GenAIRequestToolsKey            = attribute.Key("gen_ai.request.tools")
	GenAIRequestToolChoiceKey       = attribute.Key("gen_ai.request.tool_choice")
	GenAIRequestResponseFormatKey   = attribute.Key("gen_ai.request.response_format")
	GenAIRequestFrequencyPenaltyKey = attribute.Key("gen_ai.request.frequency_penalty")
	GenAIRequestPresencePenaltyKey  = attribute.Key("gen_ai.request.presence_penalty")
	GenAIRequestSeedKey             = attribute.Key("gen_ai.request.seed")

	// Response
	GenAIResponseIDKey            = attribute.Key("gen_ai.response.id")
	GenAIResponseModelKey         = attribute.Key("gen_ai.response.model")
	GenAIResponseFinishReasonsKey = attribute.Key("gen_ai.response.finish_reasons")

	// Usage
	GenAIUsageInputTokensKey     = attribute.Key("gen_ai.usage.input_tokens")
	GenAIUsageOutputTokensKey    = attribute.Key("gen_ai.usage.output_tokens")
	GenAIUsageCachedTokensKey    = attribute.Key("gen_ai.usage.cached_tokens")
	GenAIUsageReasoningTokensKey = attribute.Key("gen_ai.usage.reasoning_tokens")

	// Events
	GenAIPromptKey        = attribute.Key("gen_ai.prompt")
	GenAICompletionKey    = attribute.Key("gen_ai.completion")
	GenAIToolNameKey      = attribute.Key("gen_ai.tool.name")
	GenAIToolCallIDKey    = attribute.Key("gen_ai.tool.call_id")
	GenAIToolArgumentsKey = attribute.Key("gen_ai.tool.arguments")
	GenAIToolResultKey    = attribute.Key("gen_ai.tool.result")
)

// GenAI system values.
const (
	GenAISystemOpenAI    = "openai"
	GenAISystemAnthropic = "anthropic"
	GenAISystemBedrock   = "bedrock"
	GenAISystemAzure     = "azure"
	GenAISystemGCPVertex = "gcp_vertex"
	GenAISystemCohere    = "cohere"
	GenAISystemMistral   = "mistral"
	GenAISystemMeta      = "meta"
	GenAISystemGoogle    = "google"
)

// GenAI operation values.
const (
	GenAIOperationChat           = "chat"
	GenAIOperationTextCompletion = "text_completion"
	GenAIOperationEmbeddings     = "embeddings"
	GenAIOperationImageGen       = "image_generation"
	GenAIOperationAudio          = "audio"
)

// --- AITF Agent Attributes ---

const (
	AgentNameKey        = attribute.Key("aitf.agent.name")
	AgentIDKey          = attribute.Key("aitf.agent.id")
	AgentTypeKey        = attribute.Key("aitf.agent.type")
	AgentFrameworkKey   = attribute.Key("aitf.agent.framework")
	AgentVersionKey     = attribute.Key("aitf.agent.version")
	AgentDescriptionKey = attribute.Key("aitf.agent.description")

	AgentSessionIDKey        = attribute.Key("aitf.agent.session.id")
	AgentSessionTurnCountKey = attribute.Key("aitf.agent.session.turn_count")

	AgentStepTypeKey        = attribute.Key("aitf.agent.step.type")
	AgentStepIndexKey       = attribute.Key("aitf.agent.step.index")
	AgentStepThoughtKey     = attribute.Key("aitf.agent.step.thought")
	AgentStepActionKey      = attribute.Key("aitf.agent.step.action")
	AgentStepObservationKey = attribute.Key("aitf.agent.step.observation")
	AgentStepStatusKey      = attribute.Key("aitf.agent.step.status")

	AgentDelegationTargetAgentKey   = attribute.Key("aitf.agent.delegation.target_agent")
	AgentDelegationTargetAgentIDKey = attribute.Key("aitf.agent.delegation.target_agent_id")
	AgentDelegationReasonKey        = attribute.Key("aitf.agent.delegation.reason")
	AgentDelegationStrategyKey      = attribute.Key("aitf.agent.delegation.strategy")
	AgentDelegationTaskKey          = attribute.Key("aitf.agent.delegation.task")

	AgentTeamNameKey            = attribute.Key("aitf.agent.team.name")
	AgentTeamIDKey              = attribute.Key("aitf.agent.team.id")
	AgentTeamTopologyKey        = attribute.Key("aitf.agent.team.topology")
	AgentTeamCoordinatorKey     = attribute.Key("aitf.agent.team.coordinator")
	AgentTeamConsensusMethodKey = attribute.Key("aitf.agent.team.consensus_method")
)

// Agent type values.
const (
	AgentTypeConversational = "conversational"
	AgentTypeAutonomous     = "autonomous"
	AgentTypeReactive       = "reactive"
	AgentTypeProactive      = "proactive"
)

// Agent framework values.
const (
	AgentFrameworkLangChain      = "langchain"
	AgentFrameworkLangGraph      = "langgraph"
	AgentFrameworkCrewAI         = "crewai"
	AgentFrameworkAutoGen        = "autogen"
	AgentFrameworkSemanticKernel = "semantic_kernel"
	AgentFrameworkCustom         = "custom"
)

// Agent step type values.
const (
	AgentStepPlanning      = "planning"
	AgentStepReasoning     = "reasoning"
	AgentStepToolUse       = "tool_use"
	AgentStepDelegation    = "delegation"
	AgentStepResponse      = "response"
	AgentStepReflection    = "reflection"
	AgentStepMemoryAccess  = "memory_access"
	AgentStepGuardrail     = "guardrail_check"
	AgentStepHumanInLoop   = "human_in_loop"
	AgentStepErrorRecovery = "error_recovery"
)

// Agent team topology values.
const (
	TeamTopologyHierarchical = "hierarchical"
	TeamTopologyPeer         = "peer"
	TeamTopologyPipeline     = "pipeline"
	TeamTopologyConsensus    = "consensus"
	TeamTopologyDebate       = "debate"
	TeamTopologySwarm        = "swarm"
)

// --- AITF MCP Attributes ---

const (
	MCPServerNameKey      = attribute.Key("aitf.mcp.server.name")
	MCPServerVersionKey   = attribute.Key("aitf.mcp.server.version")
	MCPServerTransportKey = attribute.Key("aitf.mcp.server.transport")
	MCPServerURLKey       = attribute.Key("aitf.mcp.server.url")
	MCPProtocolVersionKey = attribute.Key("aitf.mcp.protocol.version")

	MCPToolNameKey             = attribute.Key("aitf.mcp.tool.name")
	MCPToolServerKey           = attribute.Key("aitf.mcp.tool.server")
	MCPToolInputKey            = attribute.Key("aitf.mcp.tool.input")
	MCPToolOutputKey           = attribute.Key("aitf.mcp.tool.output")
	MCPToolIsErrorKey          = attribute.Key("aitf.mcp.tool.is_error")
	MCPToolDurationMsKey       = attribute.Key("aitf.mcp.tool.duration_ms")
	MCPToolApprovalRequiredKey = attribute.Key("aitf.mcp.tool.approval_required")
	MCPToolApprovedKey         = attribute.Key("aitf.mcp.tool.approved")
	MCPToolCountKey            = attribute.Key("aitf.mcp.tool.count")

	MCPResourceURIKey      = attribute.Key("aitf.mcp.resource.uri")
	MCPResourceNameKey     = attribute.Key("aitf.mcp.resource.name")
	MCPResourceMimeTypeKey = attribute.Key("aitf.mcp.resource.mime_type")
	MCPResourceSizeBytesKey = attribute.Key("aitf.mcp.resource.size_bytes")

	MCPPromptNameKey      = attribute.Key("aitf.mcp.prompt.name")
	MCPPromptArgumentsKey = attribute.Key("aitf.mcp.prompt.arguments")

	MCPSamplingModelKey          = attribute.Key("aitf.mcp.sampling.model")
	MCPSamplingMaxTokensKey      = attribute.Key("aitf.mcp.sampling.max_tokens")
	MCPSamplingIncludeContextKey = attribute.Key("aitf.mcp.sampling.include_context")
)

// MCP transport values.
const (
	MCPTransportStdio          = "stdio"
	MCPTransportSSE            = "sse"
	MCPTransportStreamableHTTP = "streamable_http"
)

// --- AITF Skill Attributes ---

const (
	SkillNameKey        = attribute.Key("aitf.skill.name")
	SkillIDKey          = attribute.Key("aitf.skill.id")
	SkillVersionKey     = attribute.Key("aitf.skill.version")
	SkillProviderKey    = attribute.Key("aitf.skill.provider")
	SkillCategoryKey    = attribute.Key("aitf.skill.category")
	SkillDescriptionKey = attribute.Key("aitf.skill.description")
	SkillInputKey       = attribute.Key("aitf.skill.input")
	SkillOutputKey      = attribute.Key("aitf.skill.output")
	SkillStatusKey      = attribute.Key("aitf.skill.status")
	SkillDurationMsKey  = attribute.Key("aitf.skill.duration_ms")
	SkillRetryCountKey  = attribute.Key("aitf.skill.retry_count")
	SkillSourceKey      = attribute.Key("aitf.skill.source")
	SkillCountKey       = attribute.Key("aitf.skill.count")

	SkillComposeNameKey    = attribute.Key("aitf.skill.compose.name")
	SkillComposePatternKey = attribute.Key("aitf.skill.compose.pattern")
	SkillComposeTotalKey   = attribute.Key("aitf.skill.compose.total_skills")
)

// Skill status values.
const (
	SkillStatusSuccess = "success"
	SkillStatusError   = "error"
	SkillStatusTimeout = "timeout"
	SkillStatusDenied  = "denied"
	SkillStatusRetry   = "retry"
)

// Skill category values.
const (
	SkillCategorySearch        = "search"
	SkillCategoryCode          = "code"
	SkillCategoryData          = "data"
	SkillCategoryCommunication = "communication"
	SkillCategoryAnalysis      = "analysis"
	SkillCategoryGeneration    = "generation"
	SkillCategoryKnowledge     = "knowledge"
	SkillCategorySecurity      = "security"
	SkillCategoryIntegration   = "integration"
	SkillCategoryWorkflow      = "workflow"
)

// --- AITF RAG Attributes ---

const (
	RAGPipelineNameKey  = attribute.Key("aitf.rag.pipeline.name")
	RAGPipelineStageKey = attribute.Key("aitf.rag.pipeline.stage")
	RAGQueryKey         = attribute.Key("aitf.rag.query")

	RAGRetrieveDatabaseKey     = attribute.Key("aitf.rag.retrieve.database")
	RAGRetrieveIndexKey        = attribute.Key("aitf.rag.retrieve.index")
	RAGRetrieveTopKKey         = attribute.Key("aitf.rag.retrieve.top_k")
	RAGRetrieveResultsCountKey = attribute.Key("aitf.rag.retrieve.results_count")
	RAGRetrieveMinScoreKey     = attribute.Key("aitf.rag.retrieve.min_score")
	RAGRetrieveMaxScoreKey     = attribute.Key("aitf.rag.retrieve.max_score")
	RAGRetrieveFilterKey       = attribute.Key("aitf.rag.retrieve.filter")

	RAGRerankModelKey       = attribute.Key("aitf.rag.rerank.model")
	RAGRerankInputCountKey  = attribute.Key("aitf.rag.rerank.input_count")
	RAGRerankOutputCountKey = attribute.Key("aitf.rag.rerank.output_count")

	RAGQualityContextRelevanceKey = attribute.Key("aitf.rag.quality.context_relevance")
	RAGQualityFaithfulnessKey     = attribute.Key("aitf.rag.quality.faithfulness")
	RAGQualityGroundednessKey     = attribute.Key("aitf.rag.quality.groundedness")
)

// RAG stage values.
const (
	RAGStageRetrieve = "retrieve"
	RAGStageRerank   = "rerank"
	RAGStageGenerate = "generate"
	RAGStageEvaluate = "evaluate"
)

// --- AITF Security Attributes ---

const (
	SecurityRiskScoreKey       = attribute.Key("aitf.security.risk_score")
	SecurityRiskLevelKey       = attribute.Key("aitf.security.risk_level")
	SecurityThreatDetectedKey  = attribute.Key("aitf.security.threat_detected")
	SecurityThreatTypeKey      = attribute.Key("aitf.security.threat_type")
	SecurityOWASPCategoryKey   = attribute.Key("aitf.security.owasp_category")
	SecurityBlockedKey         = attribute.Key("aitf.security.blocked")
	SecurityDetectionMethodKey = attribute.Key("aitf.security.detection_method")
	SecurityConfidenceKey      = attribute.Key("aitf.security.confidence")

	SecurityGuardrailNameKey     = attribute.Key("aitf.security.guardrail.name")
	SecurityGuardrailTypeKey     = attribute.Key("aitf.security.guardrail.type")
	SecurityGuardrailResultKey   = attribute.Key("aitf.security.guardrail.result")
	SecurityGuardrailProviderKey = attribute.Key("aitf.security.guardrail.provider")

	SecurityPIIDetectedKey = attribute.Key("aitf.security.pii.detected")
	SecurityPIICountKey    = attribute.Key("aitf.security.pii.count")
	SecurityPIIActionKey   = attribute.Key("aitf.security.pii.action")
)

// OWASP LLM Top 10 categories.
const (
	OWASPLICM01 = "LLM01" // Prompt Injection
	OWASPLICM02 = "LLM02" // Sensitive Information Disclosure
	OWASPLICM03 = "LLM03" // Supply Chain
	OWASPLICM04 = "LLM04" // Data and Model Poisoning
	OWASPLICM05 = "LLM05" // Improper Output Handling
	OWASPLICM06 = "LLM06" // Excessive Agency
	OWASPLICM07 = "LLM07" // System Prompt Leakage
	OWASPLICM08 = "LLM08" // Vector and Embedding Weaknesses
	OWASPLICM09 = "LLM09" // Misinformation
	OWASPLICM10 = "LLM10" // Unbounded Consumption
)

// Security risk level values.
const (
	RiskLevelCritical = "critical"
	RiskLevelHigh     = "high"
	RiskLevelMedium   = "medium"
	RiskLevelLow      = "low"
	RiskLevelInfo     = "info"
)

// Threat type values.
const (
	ThreatTypePromptInjection = "prompt_injection"
	ThreatTypeJailbreak       = "jailbreak"
	ThreatTypeDataExfil       = "data_exfiltration"
	ThreatTypeSystemPromptLeak = "system_prompt_leak"
	ThreatTypeModelTheft      = "model_theft"
)

// --- AITF Cost Attributes ---

const (
	CostInputCostKey  = attribute.Key("aitf.cost.input_cost")
	CostOutputCostKey = attribute.Key("aitf.cost.output_cost")
	CostTotalCostKey  = attribute.Key("aitf.cost.total_cost")
	CostCurrencyKey   = attribute.Key("aitf.cost.currency")

	CostPricingInputPer1MKey  = attribute.Key("aitf.cost.model_pricing.input_per_1m")
	CostPricingOutputPer1MKey = attribute.Key("aitf.cost.model_pricing.output_per_1m")

	CostBudgetLimitKey     = attribute.Key("aitf.cost.budget.limit")
	CostBudgetUsedKey      = attribute.Key("aitf.cost.budget.used")
	CostBudgetRemainingKey = attribute.Key("aitf.cost.budget.remaining")

	CostAttributionUserKey    = attribute.Key("aitf.cost.attribution.user")
	CostAttributionTeamKey    = attribute.Key("aitf.cost.attribution.team")
	CostAttributionProjectKey = attribute.Key("aitf.cost.attribution.project")
)

// --- AITF Compliance Attributes ---

const (
	ComplianceNISTControlsKey  = attribute.Key("aitf.compliance.nist_ai_rmf.controls")
	ComplianceMITRETechniquesKey = attribute.Key("aitf.compliance.mitre_atlas.techniques")
	ComplianceISOControlsKey   = attribute.Key("aitf.compliance.iso_42001.controls")
	ComplianceEUArticlesKey    = attribute.Key("aitf.compliance.eu_ai_act.articles")
	ComplianceSOC2ControlsKey  = attribute.Key("aitf.compliance.soc2.controls")
	ComplianceGDPRArticlesKey  = attribute.Key("aitf.compliance.gdpr.articles")
	ComplianceCCPASectionsKey  = attribute.Key("aitf.compliance.ccpa.sections")
)

// --- AITF Latency Attributes ---

const (
	LatencyTotalMsKey            = attribute.Key("aitf.latency.total_ms")
	LatencyTimeToFirstTokenMsKey = attribute.Key("aitf.latency.time_to_first_token_ms")
	LatencyTokensPerSecondKey    = attribute.Key("aitf.latency.tokens_per_second")
	LatencyQueueTimeMsKey        = attribute.Key("aitf.latency.queue_time_ms")
	LatencyInferenceTimeMsKey    = attribute.Key("aitf.latency.inference_time_ms")
)

// --- AITF Memory Attributes ---

const (
	MemoryOperationKey  = attribute.Key("aitf.memory.operation")
	MemoryStoreKey      = attribute.Key("aitf.memory.store")
	MemoryKeyKey        = attribute.Key("aitf.memory.key")
	MemoryTTLSecondsKey = attribute.Key("aitf.memory.ttl_seconds")
	MemoryHitKey        = attribute.Key("aitf.memory.hit")
	MemoryProvenanceKey = attribute.Key("aitf.memory.provenance")
)

// --- AITF Quality Attributes ---

const (
	QualityHallucinationScoreKey = attribute.Key("aitf.quality.hallucination_score")
	QualityConfidenceKey         = attribute.Key("aitf.quality.confidence")
	QualityFactualityKey         = attribute.Key("aitf.quality.factuality")
	QualityToxicityScoreKey      = attribute.Key("aitf.quality.toxicity_score")
	QualityFeedbackRatingKey     = attribute.Key("aitf.quality.feedback.rating")
	QualityFeedbackThumbsKey     = attribute.Key("aitf.quality.feedback.thumbs")
)
