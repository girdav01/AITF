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

	// System prompt hash (CoSAI WS2: AI_INTERACTION)
	GenAISystemPromptHashKey = attribute.Key("gen_ai.system_prompt.hash")

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

	// CoSAI WS2: AGENT_TRACE fields
	AgentWorkflowIDKey = attribute.Key("aitf.agent.workflow_id")
	AgentStateKey      = attribute.Key("aitf.agent.state")
	AgentScratchpadKey = attribute.Key("aitf.agent.scratchpad")
	AgentNextActionKey = attribute.Key("aitf.agent.next_action")

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

	// CoSAI WS2: MCP_ACTIVITY fields
	MCPToolResponseErrorKey = attribute.Key("aitf.mcp.tool.response_error")
	MCPConnectionIDKey      = attribute.Key("aitf.mcp.connection.id")

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

	// CoSAI WS2: RAG_CONTEXT document-level fields
	RAGDocIDKey         = attribute.Key("aitf.rag.doc.id")
	RAGDocScoreKey      = attribute.Key("aitf.rag.doc.score")
	RAGDocProvenanceKey = attribute.Key("aitf.rag.doc.provenance")
	RAGRetrievalDocsKey = attribute.Key("aitf.rag.retrieval.docs")

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

// --- AITF Model Operations Attributes ---

const (
	// Training attributes
	ModelOpsTrainingRunIDKey          = attribute.Key("aitf.model_ops.training.run_id")
	ModelOpsTrainingTypeKey           = attribute.Key("aitf.model_ops.training.type")
	ModelOpsTrainingBaseModelKey      = attribute.Key("aitf.model_ops.training.base_model")
	ModelOpsTrainingFrameworkKey      = attribute.Key("aitf.model_ops.training.framework")
	ModelOpsTrainingDatasetIDKey      = attribute.Key("aitf.model_ops.training.dataset.id")
	ModelOpsTrainingDatasetVersionKey = attribute.Key("aitf.model_ops.training.dataset.version")
	ModelOpsTrainingDatasetSizeKey    = attribute.Key("aitf.model_ops.training.dataset.size")
	ModelOpsTrainingHyperparamsKey    = attribute.Key("aitf.model_ops.training.hyperparameters")
	ModelOpsTrainingEpochsKey         = attribute.Key("aitf.model_ops.training.epochs")
	ModelOpsTrainingLossFinalKey      = attribute.Key("aitf.model_ops.training.loss_final")
	ModelOpsTrainingValLossFinalKey   = attribute.Key("aitf.model_ops.training.val_loss_final")
	ModelOpsTrainingGPUTypeKey        = attribute.Key("aitf.model_ops.training.compute.gpu_type")
	ModelOpsTrainingGPUCountKey       = attribute.Key("aitf.model_ops.training.compute.gpu_count")
	ModelOpsTrainingGPUHoursKey       = attribute.Key("aitf.model_ops.training.compute.gpu_hours")
	ModelOpsTrainingOutputModelIDKey  = attribute.Key("aitf.model_ops.training.output_model.id")
	ModelOpsTrainingOutputModelHashKey = attribute.Key("aitf.model_ops.training.output_model.hash")
	ModelOpsTrainingCodeCommitKey     = attribute.Key("aitf.model_ops.training.code_commit")
	ModelOpsTrainingExperimentIDKey   = attribute.Key("aitf.model_ops.training.experiment.id")
	ModelOpsTrainingExperimentNameKey = attribute.Key("aitf.model_ops.training.experiment.name")
	ModelOpsTrainingStatusKey         = attribute.Key("aitf.model_ops.training.status")

	// Evaluation attributes
	ModelOpsEvalRunIDKey              = attribute.Key("aitf.model_ops.evaluation.run_id")
	ModelOpsEvalModelIDKey            = attribute.Key("aitf.model_ops.evaluation.model_id")
	ModelOpsEvalTypeKey               = attribute.Key("aitf.model_ops.evaluation.type")
	ModelOpsEvalDatasetIDKey          = attribute.Key("aitf.model_ops.evaluation.dataset.id")
	ModelOpsEvalDatasetSizeKey        = attribute.Key("aitf.model_ops.evaluation.dataset.size")
	ModelOpsEvalMetricsKey            = attribute.Key("aitf.model_ops.evaluation.metrics")
	ModelOpsEvalJudgeModelKey         = attribute.Key("aitf.model_ops.evaluation.judge_model")
	ModelOpsEvalBaselineModelKey      = attribute.Key("aitf.model_ops.evaluation.baseline_model")
	ModelOpsEvalRegressionDetectedKey = attribute.Key("aitf.model_ops.evaluation.regression_detected")
	ModelOpsEvalPassKey               = attribute.Key("aitf.model_ops.evaluation.pass")

	// Registry attributes
	ModelOpsRegistryOperationKey        = attribute.Key("aitf.model_ops.registry.operation")
	ModelOpsRegistryModelIDKey          = attribute.Key("aitf.model_ops.registry.model_id")
	ModelOpsRegistryModelVersionKey     = attribute.Key("aitf.model_ops.registry.model_version")
	ModelOpsRegistryModelAliasKey       = attribute.Key("aitf.model_ops.registry.model_alias")
	ModelOpsRegistryStageKey            = attribute.Key("aitf.model_ops.registry.stage")
	ModelOpsRegistryOwnerKey            = attribute.Key("aitf.model_ops.registry.owner")
	ModelOpsRegistryTrainingRunIDKey    = attribute.Key("aitf.model_ops.registry.lineage.training_run_id")
	ModelOpsRegistryParentModelIDKey    = attribute.Key("aitf.model_ops.registry.lineage.parent_model_id")

	// Deployment attributes
	ModelOpsDeploymentIDKey            = attribute.Key("aitf.model_ops.deployment.id")
	ModelOpsDeploymentModelIDKey       = attribute.Key("aitf.model_ops.deployment.model_id")
	ModelOpsDeploymentStrategyKey      = attribute.Key("aitf.model_ops.deployment.strategy")
	ModelOpsDeploymentEnvironmentKey   = attribute.Key("aitf.model_ops.deployment.environment")
	ModelOpsDeploymentEndpointKey      = attribute.Key("aitf.model_ops.deployment.endpoint")
	ModelOpsDeploymentInfraProviderKey = attribute.Key("aitf.model_ops.deployment.infrastructure.provider")
	ModelOpsDeploymentInfraGPUTypeKey  = attribute.Key("aitf.model_ops.deployment.infrastructure.gpu_type")
	ModelOpsDeploymentInfraReplicasKey = attribute.Key("aitf.model_ops.deployment.infrastructure.replicas")
	ModelOpsDeploymentCanaryPctKey     = attribute.Key("aitf.model_ops.deployment.canary_percent")
	ModelOpsDeploymentStatusKey        = attribute.Key("aitf.model_ops.deployment.status")
	ModelOpsDeploymentHealthStatusKey  = attribute.Key("aitf.model_ops.deployment.health_check.status")
	ModelOpsDeploymentHealthLatencyKey = attribute.Key("aitf.model_ops.deployment.health_check.latency_ms")

	// Serving attributes
	ModelOpsServingOperationKey         = attribute.Key("aitf.model_ops.serving.operation")
	ModelOpsServingRouteSelectedKey     = attribute.Key("aitf.model_ops.serving.route.selected_model")
	ModelOpsServingRouteReasonKey       = attribute.Key("aitf.model_ops.serving.route.reason")
	ModelOpsServingRouteCandidatesKey   = attribute.Key("aitf.model_ops.serving.route.candidates")
	ModelOpsServingFallbackChainKey     = attribute.Key("aitf.model_ops.serving.fallback.chain")
	ModelOpsServingFallbackDepthKey     = attribute.Key("aitf.model_ops.serving.fallback.depth")
	ModelOpsServingFallbackTriggerKey   = attribute.Key("aitf.model_ops.serving.fallback.trigger")
	ModelOpsServingFallbackOriginalKey  = attribute.Key("aitf.model_ops.serving.fallback.original_model")
	ModelOpsServingFallbackFinalKey     = attribute.Key("aitf.model_ops.serving.fallback.final_model")
	ModelOpsServingCacheHitKey          = attribute.Key("aitf.model_ops.serving.cache.hit")
	ModelOpsServingCacheTypeKey         = attribute.Key("aitf.model_ops.serving.cache.type")
	ModelOpsServingCacheSimilarityKey   = attribute.Key("aitf.model_ops.serving.cache.similarity_score")
	ModelOpsServingCacheCostSavedKey    = attribute.Key("aitf.model_ops.serving.cache.cost_saved_usd")

	// Monitoring attributes
	ModelOpsMonitorCheckTypeKey      = attribute.Key("aitf.model_ops.monitoring.check_type")
	ModelOpsMonitorModelIDKey        = attribute.Key("aitf.model_ops.monitoring.model_id")
	ModelOpsMonitorResultKey         = attribute.Key("aitf.model_ops.monitoring.result")
	ModelOpsMonitorMetricNameKey     = attribute.Key("aitf.model_ops.monitoring.metric_name")
	ModelOpsMonitorMetricValueKey    = attribute.Key("aitf.model_ops.monitoring.metric_value")
	ModelOpsMonitorBaselineValueKey  = attribute.Key("aitf.model_ops.monitoring.baseline_value")
	ModelOpsMonitorDriftScoreKey     = attribute.Key("aitf.model_ops.monitoring.drift_score")
	ModelOpsMonitorDriftTypeKey      = attribute.Key("aitf.model_ops.monitoring.drift_type")
	ModelOpsMonitorActionTriggeredKey = attribute.Key("aitf.model_ops.monitoring.action_triggered")

	// Prompt lifecycle attributes
	ModelOpsPromptNameKey        = attribute.Key("aitf.model_ops.prompt.name")
	ModelOpsPromptOperationKey   = attribute.Key("aitf.model_ops.prompt.operation")
	ModelOpsPromptVersionKey     = attribute.Key("aitf.model_ops.prompt.version")
	ModelOpsPromptContentHashKey = attribute.Key("aitf.model_ops.prompt.content_hash")
	ModelOpsPromptLabelKey       = attribute.Key("aitf.model_ops.prompt.label")
	ModelOpsPromptModelTargetKey = attribute.Key("aitf.model_ops.prompt.model_target")
	ModelOpsPromptEvalScoreKey   = attribute.Key("aitf.model_ops.prompt.evaluation.score")
	ModelOpsPromptEvalPassKey    = attribute.Key("aitf.model_ops.prompt.evaluation.pass")
	ModelOpsPromptABTestIDKey    = attribute.Key("aitf.model_ops.prompt.a_b_test.id")
	ModelOpsPromptABTestVariantKey = attribute.Key("aitf.model_ops.prompt.a_b_test.variant")
)

// --- AITF Drift Detection Attributes ---

const (
	DriftModelIDKey          = attribute.Key("aitf.drift.model_id")
	DriftTypeKey             = attribute.Key("aitf.drift.type")
	DriftScoreKey            = attribute.Key("aitf.drift.score")
	DriftResultKey           = attribute.Key("aitf.drift.result")
	DriftDetectionMethodKey  = attribute.Key("aitf.drift.detection_method")
	DriftBaselineMetricKey   = attribute.Key("aitf.drift.baseline_metric")
	DriftCurrentMetricKey    = attribute.Key("aitf.drift.current_metric")
	DriftMetricNameKey       = attribute.Key("aitf.drift.metric_name")
	DriftThresholdKey        = attribute.Key("aitf.drift.threshold")
	DriftPValueKey           = attribute.Key("aitf.drift.p_value")
	DriftRefDatasetKey       = attribute.Key("aitf.drift.reference_dataset")
	DriftRefPeriodKey        = attribute.Key("aitf.drift.reference_period")
	DriftSampleSizeKey       = attribute.Key("aitf.drift.sample_size")
	DriftAffectedSegmentsKey = attribute.Key("aitf.drift.affected_segments")
	DriftFeatureNameKey      = attribute.Key("aitf.drift.feature_name")
	DriftFeatureImportanceKey = attribute.Key("aitf.drift.feature_importance")
	DriftActionTriggeredKey  = attribute.Key("aitf.drift.action_triggered")

	// Baseline attributes
	DriftBaselineOperationKey  = attribute.Key("aitf.drift.baseline.operation")
	DriftBaselineIDKey         = attribute.Key("aitf.drift.baseline.id")
	DriftBaselineDatasetKey    = attribute.Key("aitf.drift.baseline.dataset")
	DriftBaselineSampleSizeKey = attribute.Key("aitf.drift.baseline.sample_size")
	DriftBaselinePeriodKey     = attribute.Key("aitf.drift.baseline.period")
	DriftBaselineMetricsKey    = attribute.Key("aitf.drift.baseline.metrics")
	DriftBaselineFeaturesKey   = attribute.Key("aitf.drift.baseline.features")
	DriftBaselinePreviousIDKey = attribute.Key("aitf.drift.baseline.previous_id")

	// Investigation attributes
	DriftInvestTriggerIDKey       = attribute.Key("aitf.drift.investigation.trigger_id")
	DriftInvestRootCauseKey       = attribute.Key("aitf.drift.investigation.root_cause")
	DriftInvestRootCauseCatKey    = attribute.Key("aitf.drift.investigation.root_cause_category")
	DriftInvestAffectedSegmentsKey = attribute.Key("aitf.drift.investigation.affected_segments")
	DriftInvestAffectedUsersKey   = attribute.Key("aitf.drift.investigation.affected_users_estimate")
	DriftInvestBlastRadiusKey     = attribute.Key("aitf.drift.investigation.blast_radius")
	DriftInvestSeverityKey        = attribute.Key("aitf.drift.investigation.severity")
	DriftInvestRecommendationKey  = attribute.Key("aitf.drift.investigation.recommendation")

	// Remediation attributes
	DriftRemediationActionKey      = attribute.Key("aitf.drift.remediation.action")
	DriftRemediationTriggerIDKey   = attribute.Key("aitf.drift.remediation.trigger_id")
	DriftRemediationAutomatedKey   = attribute.Key("aitf.drift.remediation.automated")
	DriftRemediationInitiatedByKey = attribute.Key("aitf.drift.remediation.initiated_by")
	DriftRemediationStatusKey      = attribute.Key("aitf.drift.remediation.status")
	DriftRemediationRollbackToKey  = attribute.Key("aitf.drift.remediation.rollback_to")
	DriftRemediationRetrainKey     = attribute.Key("aitf.drift.remediation.retrain_dataset")
	DriftRemediationValidPassedKey = attribute.Key("aitf.drift.remediation.validation_passed")
)

// --- AITF Identity Attributes ---

const (
	// Core identity attributes
	IdentityAgentIDKey       = attribute.Key("aitf.identity.agent_id")
	IdentityAgentNameKey     = attribute.Key("aitf.identity.agent_name")
	IdentityTypeKey          = attribute.Key("aitf.identity.type")
	IdentityProviderKey      = attribute.Key("aitf.identity.provider")
	IdentityOwnerKey         = attribute.Key("aitf.identity.owner")
	IdentityOwnerTypeKey     = attribute.Key("aitf.identity.owner_type")
	IdentityCredentialTypeKey = attribute.Key("aitf.identity.credential_type")
	IdentityCredentialIDKey  = attribute.Key("aitf.identity.credential_id")
	IdentityStatusKey        = attribute.Key("aitf.identity.status")
	IdentityPreviousStatusKey = attribute.Key("aitf.identity.previous_status")
	IdentityScopeKey         = attribute.Key("aitf.identity.scope")
	IdentityExpiresAtKey     = attribute.Key("aitf.identity.expires_at")
	IdentityTTLSecondsKey    = attribute.Key("aitf.identity.ttl_seconds")
	IdentityAutoRotateKey    = attribute.Key("aitf.identity.auto_rotate")
	IdentityRotationIntervalKey = attribute.Key("aitf.identity.rotation_interval_seconds")

	// Lifecycle
	IdentityLifecycleOpKey = attribute.Key("aitf.identity.lifecycle.operation")

	// Authentication attributes
	IdentityAuthMethodKey         = attribute.Key("aitf.identity.auth.method")
	IdentityAuthResultKey         = attribute.Key("aitf.identity.auth.result")
	IdentityAuthProviderKey       = attribute.Key("aitf.identity.auth.provider")
	IdentityAuthTargetServiceKey  = attribute.Key("aitf.identity.auth.target_service")
	IdentityAuthFailureReasonKey  = attribute.Key("aitf.identity.auth.failure_reason")
	IdentityAuthTokenTypeKey      = attribute.Key("aitf.identity.auth.token_type")
	IdentityAuthScopeRequestedKey = attribute.Key("aitf.identity.auth.scope_requested")
	IdentityAuthScopeGrantedKey   = attribute.Key("aitf.identity.auth.scope_granted")
	IdentityAuthContinuousKey     = attribute.Key("aitf.identity.auth.continuous")
	IdentityAuthPKCEUsedKey       = attribute.Key("aitf.identity.auth.pkce_used")
	IdentityAuthDPoPUsedKey       = attribute.Key("aitf.identity.auth.dpop_used")

	// Authorization attributes
	IdentityAuthzDecisionKey     = attribute.Key("aitf.identity.authz.decision")
	IdentityAuthzResourceKey     = attribute.Key("aitf.identity.authz.resource")
	IdentityAuthzActionKey       = attribute.Key("aitf.identity.authz.action")
	IdentityAuthzPolicyEngineKey = attribute.Key("aitf.identity.authz.policy_engine")
	IdentityAuthzPolicyIDKey     = attribute.Key("aitf.identity.authz.policy_id")
	IdentityAuthzDenyReasonKey   = attribute.Key("aitf.identity.authz.deny_reason")
	IdentityAuthzRiskScoreKey    = attribute.Key("aitf.identity.authz.risk_score")
	IdentityAuthzJEAKey          = attribute.Key("aitf.identity.authz.jea")
	IdentityAuthzTimeLimitedKey  = attribute.Key("aitf.identity.authz.time_limited")
	IdentityAuthzExpiresAtKey    = attribute.Key("aitf.identity.authz.expires_at")

	// Delegation attributes
	IdentityDelegDelegatorKey      = attribute.Key("aitf.identity.delegation.delegator")
	IdentityDelegDelegatorIDKey    = attribute.Key("aitf.identity.delegation.delegator_id")
	IdentityDelegDelegateeKey      = attribute.Key("aitf.identity.delegation.delegatee")
	IdentityDelegDelegateeIDKey    = attribute.Key("aitf.identity.delegation.delegatee_id")
	IdentityDelegTypeKey           = attribute.Key("aitf.identity.delegation.type")
	IdentityDelegChainKey          = attribute.Key("aitf.identity.delegation.chain")
	IdentityDelegChainDepthKey     = attribute.Key("aitf.identity.delegation.chain_depth")
	IdentityDelegScopeDelegatedKey = attribute.Key("aitf.identity.delegation.scope_delegated")
	IdentityDelegScopeAttenuatedKey = attribute.Key("aitf.identity.delegation.scope_attenuated")
	IdentityDelegResultKey         = attribute.Key("aitf.identity.delegation.result")
	IdentityDelegProofTypeKey      = attribute.Key("aitf.identity.delegation.proof_type")
	IdentityDelegTTLSecondsKey     = attribute.Key("aitf.identity.delegation.ttl_seconds")

	// Trust attributes
	IdentityTrustOperationKey  = attribute.Key("aitf.identity.trust.operation")
	IdentityTrustPeerAgentKey  = attribute.Key("aitf.identity.trust.peer_agent")
	IdentityTrustPeerAgentIDKey = attribute.Key("aitf.identity.trust.peer_agent_id")
	IdentityTrustResultKey     = attribute.Key("aitf.identity.trust.result")
	IdentityTrustMethodKey     = attribute.Key("aitf.identity.trust.method")
	IdentityTrustDomainKey     = attribute.Key("aitf.identity.trust.trust_domain")
	IdentityTrustPeerDomainKey = attribute.Key("aitf.identity.trust.peer_trust_domain")
	IdentityTrustCrossDomainKey = attribute.Key("aitf.identity.trust.cross_domain")
	IdentityTrustLevelKey      = attribute.Key("aitf.identity.trust.trust_level")
	IdentityTrustProtocolKey   = attribute.Key("aitf.identity.trust.protocol")

	// Session attributes
	IdentitySessionIDKey              = attribute.Key("aitf.identity.session.id")
	IdentitySessionOperationKey       = attribute.Key("aitf.identity.session.operation")
	IdentitySessionScopeKey           = attribute.Key("aitf.identity.session.scope")
	IdentitySessionExpiresAtKey       = attribute.Key("aitf.identity.session.expires_at")
	IdentitySessionActionsCountKey    = attribute.Key("aitf.identity.session.actions_count")
	IdentitySessionTerminationReasonKey = attribute.Key("aitf.identity.session.termination_reason")
)

// --- AITF Asset Inventory Attributes ---

const (
	// Core asset attributes
	AssetIDKey                = attribute.Key("aitf.asset.id")
	AssetNameKey              = attribute.Key("aitf.asset.name")
	AssetTypeKey              = attribute.Key("aitf.asset.type")
	AssetVersionKey           = attribute.Key("aitf.asset.version")
	AssetHashKey              = attribute.Key("aitf.asset.hash")
	AssetOwnerKey             = attribute.Key("aitf.asset.owner")
	AssetOwnerTypeKey         = attribute.Key("aitf.asset.owner_type")
	AssetDeployEnvKey         = attribute.Key("aitf.asset.deployment_environment")
	AssetRiskClassificationKey = attribute.Key("aitf.asset.risk_classification")
	AssetSourceRepoKey        = attribute.Key("aitf.asset.source_repository")
	AssetTagsKey              = attribute.Key("aitf.asset.tags")

	// Discovery attributes
	AssetDiscoveryScopeKey       = attribute.Key("aitf.asset.discovery.scope")
	AssetDiscoveryMethodKey      = attribute.Key("aitf.asset.discovery.method")
	AssetDiscoveryAssetsFoundKey = attribute.Key("aitf.asset.discovery.assets_found")
	AssetDiscoveryNewAssetsKey   = attribute.Key("aitf.asset.discovery.new_assets")
	AssetDiscoveryShadowKey      = attribute.Key("aitf.asset.discovery.shadow_assets")
	AssetDiscoveryStatusKey      = attribute.Key("aitf.asset.discovery.status")

	// Audit attributes
	AssetAuditTypeKey            = attribute.Key("aitf.asset.audit.type")
	AssetAuditResultKey          = attribute.Key("aitf.asset.audit.result")
	AssetAuditAuditorKey         = attribute.Key("aitf.asset.audit.auditor")
	AssetAuditFrameworkKey       = attribute.Key("aitf.asset.audit.framework")
	AssetAuditFindingsKey        = attribute.Key("aitf.asset.audit.findings")
	AssetAuditNextDueKey         = attribute.Key("aitf.asset.audit.next_audit_due")
	AssetAuditRiskScoreKey       = attribute.Key("aitf.asset.audit.risk_score")
	AssetAuditIntegrityKey       = attribute.Key("aitf.asset.audit.integrity_verified")
	AssetAuditComplianceKey      = attribute.Key("aitf.asset.audit.compliance_status")

	// Classification attributes
	AssetClassFrameworkKey        = attribute.Key("aitf.asset.classification.framework")
	AssetClassPreviousKey         = attribute.Key("aitf.asset.classification.previous")
	AssetClassReasonKey           = attribute.Key("aitf.asset.classification.reason")
	AssetClassAssessorKey         = attribute.Key("aitf.asset.classification.assessor")
	AssetClassUseCaseKey          = attribute.Key("aitf.asset.classification.use_case")
	AssetClassAutonomousDecisionKey = attribute.Key("aitf.asset.classification.autonomous_decision")

	// Decommission attributes
	AssetDecommissionReasonKey      = attribute.Key("aitf.asset.decommission.reason")
	AssetDecommissionReplacementKey = attribute.Key("aitf.asset.decommission.replacement_id")
	AssetDecommissionApprovedByKey  = attribute.Key("aitf.asset.decommission.approved_by")
)

// --- AITF Memory Security Attributes ---

const (
	MemorySecurityContentHashKey    = attribute.Key("aitf.memory.security.content_hash")
	MemorySecurityContentSizeKey    = attribute.Key("aitf.memory.security.content_size")
	MemorySecurityIntegrityHashKey  = attribute.Key("aitf.memory.security.integrity_hash")
	MemorySecurityPoisoningScoreKey = attribute.Key("aitf.memory.security.poisoning_score")
	MemorySecurityCrossSessionKey   = attribute.Key("aitf.memory.security.cross_session")
)

// --- AITF A2A (Agent-to-Agent Protocol) Attributes ---

const (
	A2AAgentNameKey            = attribute.Key("aitf.a2a.agent.name")
	A2AAgentURLKey             = attribute.Key("aitf.a2a.agent.url")
	A2AAgentVersionKey         = attribute.Key("aitf.a2a.agent.version")
	A2AAgentProviderOrgKey     = attribute.Key("aitf.a2a.agent.provider.organization")
	A2AAgentSkillsKey          = attribute.Key("aitf.a2a.agent.skills")
	A2AAgentCapStreamingKey    = attribute.Key("aitf.a2a.agent.capabilities.streaming")
	A2AAgentCapPushKey         = attribute.Key("aitf.a2a.agent.capabilities.push_notifications")
	A2AProtocolVersionKey      = attribute.Key("aitf.a2a.protocol.version")
	A2ATransportKey            = attribute.Key("aitf.a2a.transport")

	A2ATaskIDKey               = attribute.Key("aitf.a2a.task.id")
	A2ATaskContextIDKey        = attribute.Key("aitf.a2a.task.context_id")
	A2ATaskStateKey            = attribute.Key("aitf.a2a.task.state")
	A2ATaskPreviousStateKey    = attribute.Key("aitf.a2a.task.previous_state")
	A2ATaskArtifactsCountKey   = attribute.Key("aitf.a2a.task.artifacts_count")
	A2ATaskHistoryLengthKey    = attribute.Key("aitf.a2a.task.history_length")

	A2AMessageIDKey            = attribute.Key("aitf.a2a.message.id")
	A2AMessageRoleKey          = attribute.Key("aitf.a2a.message.role")
	A2AMessagePartsCountKey    = attribute.Key("aitf.a2a.message.parts_count")
	A2AMessagePartTypesKey     = attribute.Key("aitf.a2a.message.part_types")

	A2AMethodKey               = attribute.Key("aitf.a2a.method")
	A2AInteractionModeKey      = attribute.Key("aitf.a2a.interaction_mode")
	A2AJSONRPCRequestIDKey     = attribute.Key("aitf.a2a.jsonrpc.request_id")
	A2AJSONRPCErrorCodeKey     = attribute.Key("aitf.a2a.jsonrpc.error_code")
	A2AJSONRPCErrorMessageKey  = attribute.Key("aitf.a2a.jsonrpc.error_message")

	A2AArtifactIDKey           = attribute.Key("aitf.a2a.artifact.id")
	A2AArtifactNameKey         = attribute.Key("aitf.a2a.artifact.name")
	A2AArtifactPartsCountKey   = attribute.Key("aitf.a2a.artifact.parts_count")

	A2AStreamEventTypeKey      = attribute.Key("aitf.a2a.stream.event_type")
	A2AStreamIsFinalKey        = attribute.Key("aitf.a2a.stream.is_final")
	A2AStreamEventsCountKey    = attribute.Key("aitf.a2a.stream.events_count")

	A2APushURLKey              = attribute.Key("aitf.a2a.push.url")
)

// A2A task state values.
const (
	A2ATaskStateSubmitted     = "submitted"
	A2ATaskStateWorking       = "working"
	A2ATaskStateInputRequired = "input-required"
	A2ATaskStateCompleted     = "completed"
	A2ATaskStateCanceled      = "canceled"
	A2ATaskStateFailed        = "failed"
	A2ATaskStateRejected      = "rejected"
	A2ATaskStateAuthRequired  = "auth-required"
)

// A2A transport values.
const (
	A2ATransportJSONRPC  = "jsonrpc"
	A2ATransportGRPC     = "grpc"
	A2ATransportHTTPJSON = "http_json"
)

// --- AITF ACP (Agent Communication Protocol) Attributes ---

const (
	ACPAgentNameKey              = attribute.Key("aitf.acp.agent.name")
	ACPAgentDescriptionKey       = attribute.Key("aitf.acp.agent.description")
	ACPAgentInputContentTypesKey = attribute.Key("aitf.acp.agent.input_content_types")
	ACPAgentOutputContentTypesKey = attribute.Key("aitf.acp.agent.output_content_types")
	ACPAgentFrameworkKey         = attribute.Key("aitf.acp.agent.framework")
	ACPAgentSuccessRateKey       = attribute.Key("aitf.acp.agent.status.success_rate")
	ACPAgentAvgRunTimeKey        = attribute.Key("aitf.acp.agent.status.avg_run_time_seconds")

	ACPRunIDKey                  = attribute.Key("aitf.acp.run.id")
	ACPRunAgentNameKey           = attribute.Key("aitf.acp.run.agent_name")
	ACPRunSessionIDKey           = attribute.Key("aitf.acp.run.session_id")
	ACPRunModeKey                = attribute.Key("aitf.acp.run.mode")
	ACPRunStatusKey              = attribute.Key("aitf.acp.run.status")
	ACPRunPreviousStatusKey      = attribute.Key("aitf.acp.run.previous_status")
	ACPRunErrorCodeKey           = attribute.Key("aitf.acp.run.error.code")
	ACPRunErrorMessageKey        = attribute.Key("aitf.acp.run.error.message")
	ACPRunCreatedAtKey           = attribute.Key("aitf.acp.run.created_at")
	ACPRunFinishedAtKey          = attribute.Key("aitf.acp.run.finished_at")
	ACPRunDurationMsKey          = attribute.Key("aitf.acp.run.duration_ms")

	ACPMessageRoleKey            = attribute.Key("aitf.acp.message.role")
	ACPMessagePartsCountKey      = attribute.Key("aitf.acp.message.parts_count")
	ACPMessageContentTypesKey    = attribute.Key("aitf.acp.message.content_types")
	ACPMessageHasCitationsKey    = attribute.Key("aitf.acp.message.has_citations")
	ACPMessageHasTrajectoryKey   = attribute.Key("aitf.acp.message.has_trajectory")

	ACPAwaitActiveKey            = attribute.Key("aitf.acp.await.active")
	ACPAwaitCountKey             = attribute.Key("aitf.acp.await.count")
	ACPAwaitDurationMsKey        = attribute.Key("aitf.acp.await.duration_ms")

	ACPInputMessageCountKey      = attribute.Key("aitf.acp.input.message_count")
	ACPOutputMessageCountKey     = attribute.Key("aitf.acp.output.message_count")

	ACPOperationKey              = attribute.Key("aitf.acp.operation")
	ACPHTTPMethodKey             = attribute.Key("aitf.acp.http.method")
	ACPHTTPStatusCodeKey         = attribute.Key("aitf.acp.http.status_code")
	ACPHTTPURLKey                = attribute.Key("aitf.acp.http.url")

	ACPTrajectoryToolNameKey     = attribute.Key("aitf.acp.trajectory.tool_name")
	ACPTrajectoryMessageKey      = attribute.Key("aitf.acp.trajectory.message")
)

// ACP run status values.
const (
	ACPRunStatusCreated    = "created"
	ACPRunStatusInProgress = "in-progress"
	ACPRunStatusAwaiting   = "awaiting"
	ACPRunStatusCancelling = "cancelling"
	ACPRunStatusCancelled  = "cancelled"
	ACPRunStatusCompleted  = "completed"
	ACPRunStatusFailed     = "failed"
)

// ACP run mode values.
const (
	ACPRunModeSync  = "sync"
	ACPRunModeAsync = "async"
	ACPRunModeStream = "stream"
)
