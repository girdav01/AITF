"""AITF Semantic Convention Attribute Constants.

All attribute keys used across AITF instrumentation, processors, and exporters.
OTel GenAI attributes (gen_ai.*) are preserved for compatibility.
AITF extensions use the aitf.* namespace.
"""


class GenAIAttributes:
    """OpenTelemetry GenAI semantic convention attributes (preserved)."""

    SYSTEM = "gen_ai.system"
    OPERATION_NAME = "gen_ai.operation.name"

    # Request attributes
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    REQUEST_SEED = "gen_ai.request.seed"
    REQUEST_STREAM = "gen_ai.request.stream"
    REQUEST_TOOLS = "gen_ai.request.tools"
    REQUEST_TOOL_CHOICE = "gen_ai.request.tool_choice"
    REQUEST_RESPONSE_FORMAT = "gen_ai.request.response_format"

    # Response attributes
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Usage attributes
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_CACHED_TOKENS = "gen_ai.usage.cached_tokens"
    USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"

    # Token attributes
    TOKEN_TYPE = "gen_ai.token.type"

    # System prompt hash (CoSAI WS2: AI_INTERACTION)
    SYSTEM_PROMPT_HASH = "gen_ai.system_prompt.hash"

    # Event attributes
    PROMPT = "gen_ai.prompt"
    COMPLETION = "gen_ai.completion"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ID = "gen_ai.tool.call_id"
    TOOL_ARGUMENTS = "gen_ai.tool.arguments"
    TOOL_RESULT = "gen_ai.tool.result"

    # System values
    class System:
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        BEDROCK = "bedrock"
        AZURE = "azure"
        GCP_VERTEX = "gcp_vertex"
        COHERE = "cohere"
        MISTRAL = "mistral"
        META = "meta"
        GOOGLE = "google"

    # Operation values
    class Operation:
        CHAT = "chat"
        TEXT_COMPLETION = "text_completion"
        EMBEDDINGS = "embeddings"
        IMAGE_GENERATION = "image_generation"
        AUDIO = "audio"


class AgentAttributes:
    """AITF Agent semantic convention attributes."""

    # Core agent attributes
    NAME = "aitf.agent.name"
    ID = "aitf.agent.id"
    TYPE = "aitf.agent.type"
    FRAMEWORK = "aitf.agent.framework"
    VERSION = "aitf.agent.version"
    DESCRIPTION = "aitf.agent.description"

    # Session attributes
    SESSION_ID = "aitf.agent.session.id"
    SESSION_TURN_COUNT = "aitf.agent.session.turn_count"
    SESSION_START_TIME = "aitf.agent.session.start_time"

    # CoSAI WS2: AGENT_TRACE fields
    WORKFLOW_ID = "aitf.agent.workflow_id"
    STATE = "aitf.agent.state"
    SCRATCHPAD = "aitf.agent.scratchpad"
    NEXT_ACTION = "aitf.agent.next_action"

    # Step attributes
    STEP_TYPE = "aitf.agent.step.type"
    STEP_INDEX = "aitf.agent.step.index"
    STEP_THOUGHT = "aitf.agent.step.thought"
    STEP_ACTION = "aitf.agent.step.action"
    STEP_OBSERVATION = "aitf.agent.step.observation"
    STEP_STATUS = "aitf.agent.step.status"

    # Delegation attributes
    DELEGATION_TARGET_AGENT = "aitf.agent.delegation.target_agent"
    DELEGATION_TARGET_AGENT_ID = "aitf.agent.delegation.target_agent_id"
    DELEGATION_REASON = "aitf.agent.delegation.reason"
    DELEGATION_STRATEGY = "aitf.agent.delegation.strategy"
    DELEGATION_TASK = "aitf.agent.delegation.task"
    DELEGATION_RESULT = "aitf.agent.delegation.result"

    # Team attributes
    TEAM_NAME = "aitf.agent.team.name"
    TEAM_ID = "aitf.agent.team.id"
    TEAM_TOPOLOGY = "aitf.agent.team.topology"
    TEAM_MEMBERS = "aitf.agent.team.members"
    TEAM_COORDINATOR = "aitf.agent.team.coordinator"
    TEAM_CONSENSUS_METHOD = "aitf.agent.team.consensus_method"

    # Type values
    class Type:
        CONVERSATIONAL = "conversational"
        AUTONOMOUS = "autonomous"
        REACTIVE = "reactive"
        PROACTIVE = "proactive"

    # Framework values
    class Framework:
        LANGCHAIN = "langchain"
        LANGGRAPH = "langgraph"
        CREWAI = "crewai"
        AUTOGEN = "autogen"
        SEMANTIC_KERNEL = "semantic_kernel"
        CUSTOM = "custom"

    # Step type values
    class StepType:
        PLANNING = "planning"
        REASONING = "reasoning"
        TOOL_USE = "tool_use"
        DELEGATION = "delegation"
        RESPONSE = "response"
        REFLECTION = "reflection"
        MEMORY_ACCESS = "memory_access"
        GUARDRAIL_CHECK = "guardrail_check"
        HUMAN_IN_LOOP = "human_in_loop"
        ERROR_RECOVERY = "error_recovery"

    # Team topology values
    class TeamTopology:
        HIERARCHICAL = "hierarchical"
        PEER = "peer"
        PIPELINE = "pipeline"
        CONSENSUS = "consensus"
        DEBATE = "debate"
        SWARM = "swarm"


class MCPAttributes:
    """AITF MCP (Model Context Protocol) semantic convention attributes."""

    # Server attributes
    SERVER_NAME = "aitf.mcp.server.name"
    SERVER_VERSION = "aitf.mcp.server.version"
    SERVER_TRANSPORT = "aitf.mcp.server.transport"
    SERVER_URL = "aitf.mcp.server.url"
    PROTOCOL_VERSION = "aitf.mcp.protocol.version"

    # Tool attributes
    TOOL_NAME = "aitf.mcp.tool.name"
    TOOL_SERVER = "aitf.mcp.tool.server"
    TOOL_INPUT = "aitf.mcp.tool.input"
    TOOL_OUTPUT = "aitf.mcp.tool.output"
    TOOL_IS_ERROR = "aitf.mcp.tool.is_error"
    TOOL_DURATION_MS = "aitf.mcp.tool.duration_ms"
    TOOL_APPROVAL_REQUIRED = "aitf.mcp.tool.approval_required"
    TOOL_APPROVED = "aitf.mcp.tool.approved"
    TOOL_COUNT = "aitf.mcp.tool.count"
    TOOL_NAMES = "aitf.mcp.tool.names"

    # CoSAI WS2: MCP_ACTIVITY fields
    RESPONSE_ERROR = "aitf.mcp.tool.response_error"
    CONNECTION_ID = "aitf.mcp.connection.id"

    # Resource attributes
    RESOURCE_URI = "aitf.mcp.resource.uri"
    RESOURCE_NAME = "aitf.mcp.resource.name"
    RESOURCE_MIME_TYPE = "aitf.mcp.resource.mime_type"
    RESOURCE_SIZE_BYTES = "aitf.mcp.resource.size_bytes"

    # Prompt attributes
    PROMPT_NAME = "aitf.mcp.prompt.name"
    PROMPT_ARGUMENTS = "aitf.mcp.prompt.arguments"
    PROMPT_DESCRIPTION = "aitf.mcp.prompt.description"

    # Sampling attributes
    SAMPLING_MODEL = "aitf.mcp.sampling.model"
    SAMPLING_MAX_TOKENS = "aitf.mcp.sampling.max_tokens"
    SAMPLING_INCLUDE_CONTEXT = "aitf.mcp.sampling.include_context"

    # Transport values
    class Transport:
        STDIO = "stdio"
        SSE = "sse"
        STREAMABLE_HTTP = "streamable_http"


class SkillAttributes:
    """AITF Skills semantic convention attributes."""

    NAME = "aitf.skill.name"
    ID = "aitf.skill.id"
    VERSION = "aitf.skill.version"
    PROVIDER = "aitf.skill.provider"
    CATEGORY = "aitf.skill.category"
    DESCRIPTION = "aitf.skill.description"
    INPUT = "aitf.skill.input"
    OUTPUT = "aitf.skill.output"
    STATUS = "aitf.skill.status"
    DURATION_MS = "aitf.skill.duration_ms"
    RETRY_COUNT = "aitf.skill.retry_count"
    SOURCE = "aitf.skill.source"
    PERMISSIONS = "aitf.skill.permissions"
    COUNT = "aitf.skill.count"
    NAMES = "aitf.skill.names"

    # Composition attributes
    COMPOSE_NAME = "aitf.skill.compose.name"
    COMPOSE_SKILLS = "aitf.skill.compose.skills"
    COMPOSE_PATTERN = "aitf.skill.compose.pattern"
    COMPOSE_TOTAL = "aitf.skill.compose.total_skills"
    COMPOSE_COMPLETED = "aitf.skill.compose.completed_skills"

    # Resolution attributes
    RESOLVE_CAPABILITY = "aitf.skill.resolve.capability"
    RESOLVE_CANDIDATES = "aitf.skill.resolve.candidates"
    RESOLVE_SELECTED = "aitf.skill.resolve.selected"
    RESOLVE_REASON = "aitf.skill.resolve.reason"

    # Provider values
    class Provider:
        BUILTIN = "builtin"
        MARKETPLACE = "marketplace"
        CUSTOM = "custom"
        MCP = "mcp"

    # Category values
    class Category:
        SEARCH = "search"
        CODE = "code"
        DATA = "data"
        COMMUNICATION = "communication"
        ANALYSIS = "analysis"
        GENERATION = "generation"
        KNOWLEDGE = "knowledge"
        SECURITY = "security"
        INTEGRATION = "integration"
        WORKFLOW = "workflow"

    # Status values
    class Status:
        SUCCESS = "success"
        ERROR = "error"
        TIMEOUT = "timeout"
        DENIED = "denied"
        RETRY = "retry"


class RAGAttributes:
    """AITF RAG (Retrieval-Augmented Generation) semantic convention attributes."""

    # Pipeline attributes
    PIPELINE_NAME = "aitf.rag.pipeline.name"
    PIPELINE_STAGE = "aitf.rag.pipeline.stage"
    QUERY = "aitf.rag.query"
    QUERY_EMBEDDING_MODEL = "aitf.rag.query.embedding_model"
    QUERY_EMBEDDING_DIMENSIONS = "aitf.rag.query.embedding_dimensions"

    # Retrieval attributes
    RETRIEVE_DATABASE = "aitf.rag.retrieve.database"
    RETRIEVE_INDEX = "aitf.rag.retrieve.index"
    RETRIEVE_TOP_K = "aitf.rag.retrieve.top_k"
    RETRIEVE_RESULTS_COUNT = "aitf.rag.retrieve.results_count"
    RETRIEVE_MIN_SCORE = "aitf.rag.retrieve.min_score"
    RETRIEVE_MAX_SCORE = "aitf.rag.retrieve.max_score"
    RETRIEVE_FILTER = "aitf.rag.retrieve.filter"

    # CoSAI WS2: RAG_CONTEXT document-level fields
    DOC_ID = "aitf.rag.doc.id"
    DOC_SCORE = "aitf.rag.doc.score"
    DOC_PROVENANCE = "aitf.rag.doc.provenance"
    RETRIEVAL_DOCS = "aitf.rag.retrieval.docs"

    # Reranking attributes
    RERANK_MODEL = "aitf.rag.rerank.model"
    RERANK_INPUT_COUNT = "aitf.rag.rerank.input_count"
    RERANK_OUTPUT_COUNT = "aitf.rag.rerank.output_count"

    # Quality attributes
    QUALITY_CONTEXT_RELEVANCE = "aitf.rag.quality.context_relevance"
    QUALITY_ANSWER_RELEVANCE = "aitf.rag.quality.answer_relevance"
    QUALITY_FAITHFULNESS = "aitf.rag.quality.faithfulness"
    QUALITY_GROUNDEDNESS = "aitf.rag.quality.groundedness"

    # Stage values
    class Stage:
        RETRIEVE = "retrieve"
        RERANK = "rerank"
        GENERATE = "generate"
        EVALUATE = "evaluate"


class SecurityAttributes:
    """AITF Security semantic convention attributes."""

    RISK_SCORE = "aitf.security.risk_score"
    RISK_LEVEL = "aitf.security.risk_level"
    THREAT_DETECTED = "aitf.security.threat_detected"
    THREAT_TYPE = "aitf.security.threat_type"
    OWASP_CATEGORY = "aitf.security.owasp_category"
    BLOCKED = "aitf.security.blocked"
    DETECTION_METHOD = "aitf.security.detection_method"
    CONFIDENCE = "aitf.security.confidence"

    # Guardrail attributes
    GUARDRAIL_NAME = "aitf.security.guardrail.name"
    GUARDRAIL_TYPE = "aitf.security.guardrail.type"
    GUARDRAIL_RESULT = "aitf.security.guardrail.result"
    GUARDRAIL_PROVIDER = "aitf.security.guardrail.provider"
    GUARDRAIL_POLICY = "aitf.security.guardrail.policy"

    # PII attributes
    PII_DETECTED = "aitf.security.pii.detected"
    PII_TYPES = "aitf.security.pii.types"
    PII_COUNT = "aitf.security.pii.count"
    PII_ACTION = "aitf.security.pii.action"

    # Risk level values
    class RiskLevel:
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        INFO = "info"

    # OWASP LLM Top 10
    class OWASP:
        LLM01 = "LLM01"  # Prompt Injection
        LLM02 = "LLM02"  # Sensitive Information Disclosure
        LLM03 = "LLM03"  # Supply Chain Vulnerabilities
        LLM04 = "LLM04"  # Data and Model Poisoning
        LLM05 = "LLM05"  # Improper Output Handling
        LLM06 = "LLM06"  # Excessive Agency
        LLM07 = "LLM07"  # System Prompt Leakage
        LLM08 = "LLM08"  # Vector and Embedding Weaknesses
        LLM09 = "LLM09"  # Misinformation
        LLM10 = "LLM10"  # Unbounded Consumption

    # Threat type values
    class ThreatType:
        PROMPT_INJECTION = "prompt_injection"
        SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
        SUPPLY_CHAIN = "supply_chain"
        DATA_POISONING = "data_poisoning"
        IMPROPER_OUTPUT = "improper_output"
        EXCESSIVE_AGENCY = "excessive_agency"
        SYSTEM_PROMPT_LEAK = "system_prompt_leak"
        VECTOR_DATA_WEAKNESS = "vector_data_weakness"
        MISINFORMATION = "misinformation"
        UNBOUNDED_CONSUMPTION = "unbounded_consumption"
        JAILBREAK = "jailbreak"
        DATA_EXFILTRATION = "data_exfiltration"
        MODEL_THEFT = "model_theft"


class ComplianceAttributes:
    """AITF Compliance semantic convention attributes."""

    FRAMEWORKS = "aitf.compliance.frameworks"
    NIST_AI_RMF_CONTROLS = "aitf.compliance.nist_ai_rmf.controls"
    MITRE_ATLAS_TECHNIQUES = "aitf.compliance.mitre_atlas.techniques"
    ISO_42001_CONTROLS = "aitf.compliance.iso_42001.controls"
    EU_AI_ACT_ARTICLES = "aitf.compliance.eu_ai_act.articles"
    SOC2_CONTROLS = "aitf.compliance.soc2.controls"
    GDPR_ARTICLES = "aitf.compliance.gdpr.articles"
    CCPA_SECTIONS = "aitf.compliance.ccpa.sections"


class CostAttributes:
    """AITF Cost semantic convention attributes."""

    INPUT_COST = "aitf.cost.input_cost"
    OUTPUT_COST = "aitf.cost.output_cost"
    TOTAL_COST = "aitf.cost.total_cost"
    CURRENCY = "aitf.cost.currency"

    # Model pricing
    PRICING_INPUT_PER_1M = "aitf.cost.model_pricing.input_per_1m"
    PRICING_OUTPUT_PER_1M = "aitf.cost.model_pricing.output_per_1m"

    # Budget
    BUDGET_LIMIT = "aitf.cost.budget.limit"
    BUDGET_USED = "aitf.cost.budget.used"
    BUDGET_REMAINING = "aitf.cost.budget.remaining"

    # Attribution
    ATTRIBUTION_USER = "aitf.cost.attribution.user"
    ATTRIBUTION_TEAM = "aitf.cost.attribution.team"
    ATTRIBUTION_PROJECT = "aitf.cost.attribution.project"


class QualityAttributes:
    """AITF Quality semantic convention attributes."""

    HALLUCINATION_SCORE = "aitf.quality.hallucination_score"
    CONFIDENCE = "aitf.quality.confidence"
    FACTUALITY = "aitf.quality.factuality"
    COHERENCE = "aitf.quality.coherence"
    TOXICITY_SCORE = "aitf.quality.toxicity_score"
    BIAS_SCORE = "aitf.quality.bias_score"
    FEEDBACK_RATING = "aitf.quality.feedback.rating"
    FEEDBACK_THUMBS = "aitf.quality.feedback.thumbs"
    FEEDBACK_COMMENT = "aitf.quality.feedback.comment"


class SupplyChainAttributes:
    """AITF Supply Chain semantic convention attributes."""

    MODEL_SOURCE = "aitf.supply_chain.model.source"
    MODEL_HASH = "aitf.supply_chain.model.hash"
    MODEL_LICENSE = "aitf.supply_chain.model.license"
    MODEL_TRAINING_DATA = "aitf.supply_chain.model.training_data"
    MODEL_SIGNED = "aitf.supply_chain.model.signed"
    MODEL_SIGNER = "aitf.supply_chain.model.signer"
    AI_BOM_ID = "aitf.supply_chain.ai_bom.id"
    AI_BOM_COMPONENTS = "aitf.supply_chain.ai_bom.components"


class MemoryAttributes:
    """AITF Memory semantic convention attributes."""

    OPERATION = "aitf.memory.operation"
    STORE = "aitf.memory.store"
    KEY = "aitf.memory.key"
    TTL_SECONDS = "aitf.memory.ttl_seconds"
    HIT = "aitf.memory.hit"
    PROVENANCE = "aitf.memory.provenance"

    class Operation:
        STORE = "store"
        RETRIEVE = "retrieve"
        UPDATE = "update"
        DELETE = "delete"
        SEARCH = "search"

    class Store:
        SHORT_TERM = "short_term"
        LONG_TERM = "long_term"
        EPISODIC = "episodic"
        SEMANTIC = "semantic"
        PROCEDURAL = "procedural"


class ModelOpsAttributes:
    """AITF Model Operations (LLMOps/MLOps) semantic convention attributes."""

    # Training attributes
    TRAINING_RUN_ID = "aitf.model_ops.training.run_id"
    TRAINING_TYPE = "aitf.model_ops.training.type"
    TRAINING_BASE_MODEL = "aitf.model_ops.training.base_model"
    TRAINING_FRAMEWORK = "aitf.model_ops.training.framework"
    TRAINING_DATASET_ID = "aitf.model_ops.training.dataset.id"
    TRAINING_DATASET_VERSION = "aitf.model_ops.training.dataset.version"
    TRAINING_DATASET_SIZE = "aitf.model_ops.training.dataset.size"
    TRAINING_HYPERPARAMETERS = "aitf.model_ops.training.hyperparameters"
    TRAINING_EPOCHS = "aitf.model_ops.training.epochs"
    TRAINING_BATCH_SIZE = "aitf.model_ops.training.batch_size"
    TRAINING_LEARNING_RATE = "aitf.model_ops.training.learning_rate"
    TRAINING_LOSS_FINAL = "aitf.model_ops.training.loss_final"
    TRAINING_VAL_LOSS_FINAL = "aitf.model_ops.training.val_loss_final"
    TRAINING_COMPUTE_GPU_TYPE = "aitf.model_ops.training.compute.gpu_type"
    TRAINING_COMPUTE_GPU_COUNT = "aitf.model_ops.training.compute.gpu_count"
    TRAINING_COMPUTE_GPU_HOURS = "aitf.model_ops.training.compute.gpu_hours"
    TRAINING_OUTPUT_MODEL_ID = "aitf.model_ops.training.output_model.id"
    TRAINING_OUTPUT_MODEL_HASH = "aitf.model_ops.training.output_model.hash"
    TRAINING_CODE_COMMIT = "aitf.model_ops.training.code_commit"
    TRAINING_EXPERIMENT_ID = "aitf.model_ops.training.experiment.id"
    TRAINING_EXPERIMENT_NAME = "aitf.model_ops.training.experiment.name"
    TRAINING_STATUS = "aitf.model_ops.training.status"

    # Evaluation attributes
    EVALUATION_RUN_ID = "aitf.model_ops.evaluation.run_id"
    EVALUATION_MODEL_ID = "aitf.model_ops.evaluation.model_id"
    EVALUATION_TYPE = "aitf.model_ops.evaluation.type"
    EVALUATION_DATASET_ID = "aitf.model_ops.evaluation.dataset.id"
    EVALUATION_DATASET_VERSION = "aitf.model_ops.evaluation.dataset.version"
    EVALUATION_DATASET_SIZE = "aitf.model_ops.evaluation.dataset.size"
    EVALUATION_METRICS = "aitf.model_ops.evaluation.metrics"
    EVALUATION_JUDGE_MODEL = "aitf.model_ops.evaluation.judge_model"
    EVALUATION_BASELINE_MODEL = "aitf.model_ops.evaluation.baseline_model"
    EVALUATION_REGRESSION_DETECTED = "aitf.model_ops.evaluation.regression_detected"
    EVALUATION_PASS = "aitf.model_ops.evaluation.pass"

    # Registry attributes
    REGISTRY_OPERATION = "aitf.model_ops.registry.operation"
    REGISTRY_MODEL_ID = "aitf.model_ops.registry.model_id"
    REGISTRY_MODEL_VERSION = "aitf.model_ops.registry.model_version"
    REGISTRY_MODEL_ALIAS = "aitf.model_ops.registry.model_alias"
    REGISTRY_STAGE = "aitf.model_ops.registry.stage"
    REGISTRY_PREVIOUS_STAGE = "aitf.model_ops.registry.previous_stage"
    REGISTRY_OWNER = "aitf.model_ops.registry.owner"
    REGISTRY_LINEAGE_TRAINING_RUN_ID = "aitf.model_ops.registry.lineage.training_run_id"
    REGISTRY_LINEAGE_PARENT_MODEL_ID = "aitf.model_ops.registry.lineage.parent_model_id"

    # Deployment attributes
    DEPLOYMENT_ID = "aitf.model_ops.deployment.id"
    DEPLOYMENT_MODEL_ID = "aitf.model_ops.deployment.model_id"
    DEPLOYMENT_STRATEGY = "aitf.model_ops.deployment.strategy"
    DEPLOYMENT_MODEL_VERSION = "aitf.model_ops.deployment.model_version"
    DEPLOYMENT_ENVIRONMENT = "aitf.model_ops.deployment.environment"
    DEPLOYMENT_ENDPOINT = "aitf.model_ops.deployment.endpoint"
    DEPLOYMENT_INFRA_PROVIDER = "aitf.model_ops.deployment.infrastructure.provider"
    DEPLOYMENT_INFRA_GPU_TYPE = "aitf.model_ops.deployment.infrastructure.gpu_type"
    DEPLOYMENT_INFRA_REPLICAS = "aitf.model_ops.deployment.infrastructure.replicas"
    DEPLOYMENT_CANARY_PERCENT = "aitf.model_ops.deployment.canary_percent"
    DEPLOYMENT_STATUS = "aitf.model_ops.deployment.status"
    DEPLOYMENT_HEALTH_STATUS = "aitf.model_ops.deployment.health_check.status"
    DEPLOYMENT_HEALTH_LATENCY = "aitf.model_ops.deployment.health_check.latency_ms"

    # Serving attributes
    SERVING_OPERATION = "aitf.model_ops.serving.operation"
    SERVING_ROUTE_SELECTED_MODEL = "aitf.model_ops.serving.route.selected_model"
    SERVING_ROUTE_REASON = "aitf.model_ops.serving.route.reason"
    SERVING_ROUTE_CANDIDATES = "aitf.model_ops.serving.route.candidates"
    SERVING_FALLBACK_CHAIN = "aitf.model_ops.serving.fallback.chain"
    SERVING_FALLBACK_DEPTH = "aitf.model_ops.serving.fallback.depth"
    SERVING_FALLBACK_TRIGGER = "aitf.model_ops.serving.fallback.trigger"
    SERVING_FALLBACK_ORIGINAL_MODEL = "aitf.model_ops.serving.fallback.original_model"
    SERVING_FALLBACK_FINAL_MODEL = "aitf.model_ops.serving.fallback.final_model"
    SERVING_CACHE_HIT = "aitf.model_ops.serving.cache.hit"
    SERVING_CACHE_TYPE = "aitf.model_ops.serving.cache.type"
    SERVING_CACHE_SIMILARITY_SCORE = "aitf.model_ops.serving.cache.similarity_score"
    SERVING_CACHE_COST_SAVED = "aitf.model_ops.serving.cache.cost_saved_usd"
    SERVING_CIRCUIT_BREAKER_STATE = "aitf.model_ops.serving.circuit_breaker.state"
    SERVING_CIRCUIT_BREAKER_MODEL = "aitf.model_ops.serving.circuit_breaker.model"

    # Monitoring attributes
    MONITORING_CHECK_TYPE = "aitf.model_ops.monitoring.check_type"
    MONITORING_MODEL_ID = "aitf.model_ops.monitoring.model_id"
    MONITORING_RESULT = "aitf.model_ops.monitoring.result"
    MONITORING_METRIC_NAME = "aitf.model_ops.monitoring.metric_name"
    MONITORING_METRIC_VALUE = "aitf.model_ops.monitoring.metric_value"
    MONITORING_BASELINE_VALUE = "aitf.model_ops.monitoring.baseline_value"
    MONITORING_DRIFT_SCORE = "aitf.model_ops.monitoring.drift_score"
    MONITORING_DRIFT_TYPE = "aitf.model_ops.monitoring.drift_type"
    MONITORING_ACTION_TRIGGERED = "aitf.model_ops.monitoring.action_triggered"

    # Prompt attributes
    PROMPT_NAME = "aitf.model_ops.prompt.name"
    PROMPT_OPERATION = "aitf.model_ops.prompt.operation"
    PROMPT_VERSION = "aitf.model_ops.prompt.version"
    PROMPT_CONTENT_HASH = "aitf.model_ops.prompt.content_hash"
    PROMPT_LABEL = "aitf.model_ops.prompt.label"
    PROMPT_MODEL_TARGET = "aitf.model_ops.prompt.model_target"
    PROMPT_EVAL_SCORE = "aitf.model_ops.prompt.evaluation.score"
    PROMPT_EVAL_PASS = "aitf.model_ops.prompt.evaluation.pass"
    PROMPT_AB_TEST_ID = "aitf.model_ops.prompt.a_b_test.id"
    PROMPT_AB_TEST_VARIANT = "aitf.model_ops.prompt.a_b_test.variant"

    # Training type values
    class TrainingType:
        PRE_TRAINING = "pre_training"
        FINE_TUNING = "fine_tuning"
        RLHF = "rlhf"
        DPO = "dpo"
        LORA = "lora"
        QLORA = "qlora"
        DISTILLATION = "distillation"
        CONTINUED_PRE_TRAINING = "continued_pre_training"

    # Deployment strategy values
    class DeploymentStrategy:
        ROLLING = "rolling"
        CANARY = "canary"
        BLUE_GREEN = "blue_green"
        SHADOW = "shadow"
        AB_TEST = "a_b_test"
        IMMEDIATE = "immediate"

    # Drift type values
    class DriftType:
        DATA = "data"
        PREDICTION = "prediction"
        CONCEPT = "concept"
        EMBEDDING = "embedding"
        FEATURE = "feature"


class IdentityAttributes:
    """AITF Agentic Identity semantic convention attributes."""

    # Core identity attributes
    AGENT_ID = "aitf.identity.agent_id"
    AGENT_NAME = "aitf.identity.agent_name"
    TYPE = "aitf.identity.type"
    PROVIDER = "aitf.identity.provider"
    OWNER = "aitf.identity.owner"
    OWNER_TYPE = "aitf.identity.owner_type"
    CREDENTIAL_TYPE = "aitf.identity.credential_type"
    CREDENTIAL_ID = "aitf.identity.credential_id"
    STATUS = "aitf.identity.status"
    PREVIOUS_STATUS = "aitf.identity.previous_status"
    SCOPE = "aitf.identity.scope"
    EXPIRES_AT = "aitf.identity.expires_at"
    TTL_SECONDS = "aitf.identity.ttl_seconds"
    AUTO_ROTATE = "aitf.identity.auto_rotate"
    ROTATION_INTERVAL = "aitf.identity.rotation_interval_seconds"

    # Lifecycle attributes
    LIFECYCLE_OPERATION = "aitf.identity.lifecycle.operation"

    # Authentication attributes
    AUTH_METHOD = "aitf.identity.auth.method"
    AUTH_RESULT = "aitf.identity.auth.result"
    AUTH_PROVIDER = "aitf.identity.auth.provider"
    AUTH_TARGET_SERVICE = "aitf.identity.auth.target_service"
    AUTH_FAILURE_REASON = "aitf.identity.auth.failure_reason"
    AUTH_TOKEN_TYPE = "aitf.identity.auth.token_type"
    AUTH_SCOPE_REQUESTED = "aitf.identity.auth.scope_requested"
    AUTH_SCOPE_GRANTED = "aitf.identity.auth.scope_granted"
    AUTH_CONTINUOUS = "aitf.identity.auth.continuous"
    AUTH_PKCE_USED = "aitf.identity.auth.pkce_used"
    AUTH_DPOP_USED = "aitf.identity.auth.dpop_used"

    # Authorization attributes
    AUTHZ_DECISION = "aitf.identity.authz.decision"
    AUTHZ_RESOURCE = "aitf.identity.authz.resource"
    AUTHZ_ACTION = "aitf.identity.authz.action"
    AUTHZ_POLICY_ENGINE = "aitf.identity.authz.policy_engine"
    AUTHZ_POLICY_ID = "aitf.identity.authz.policy_id"
    AUTHZ_DENY_REASON = "aitf.identity.authz.deny_reason"
    AUTHZ_RISK_SCORE = "aitf.identity.authz.risk_score"
    AUTHZ_PRIVILEGE_LEVEL = "aitf.identity.authz.privilege_level"
    AUTHZ_JEA = "aitf.identity.authz.jea"
    AUTHZ_TIME_LIMITED = "aitf.identity.authz.time_limited"
    AUTHZ_EXPIRES_AT = "aitf.identity.authz.expires_at"

    # Delegation attributes
    DELEGATION_DELEGATOR = "aitf.identity.delegation.delegator"
    DELEGATION_DELEGATOR_ID = "aitf.identity.delegation.delegator_id"
    DELEGATION_DELEGATEE = "aitf.identity.delegation.delegatee"
    DELEGATION_DELEGATEE_ID = "aitf.identity.delegation.delegatee_id"
    DELEGATION_TYPE = "aitf.identity.delegation.type"
    DELEGATION_CHAIN = "aitf.identity.delegation.chain"
    DELEGATION_CHAIN_DEPTH = "aitf.identity.delegation.chain_depth"
    DELEGATION_SCOPE_DELEGATED = "aitf.identity.delegation.scope_delegated"
    DELEGATION_SCOPE_ATTENUATED = "aitf.identity.delegation.scope_attenuated"
    DELEGATION_RESULT = "aitf.identity.delegation.result"
    DELEGATION_PROOF_TYPE = "aitf.identity.delegation.proof_type"
    DELEGATION_TTL_SECONDS = "aitf.identity.delegation.ttl_seconds"

    # Trust attributes
    TRUST_OPERATION = "aitf.identity.trust.operation"
    TRUST_PEER_AGENT = "aitf.identity.trust.peer_agent"
    TRUST_PEER_AGENT_ID = "aitf.identity.trust.peer_agent_id"
    TRUST_RESULT = "aitf.identity.trust.result"
    TRUST_METHOD = "aitf.identity.trust.method"
    TRUST_DOMAIN = "aitf.identity.trust.trust_domain"
    TRUST_PEER_DOMAIN = "aitf.identity.trust.peer_trust_domain"
    TRUST_CROSS_DOMAIN = "aitf.identity.trust.cross_domain"
    TRUST_LEVEL = "aitf.identity.trust.trust_level"
    TRUST_PROTOCOL = "aitf.identity.trust.protocol"

    # Session attributes
    SESSION_ID = "aitf.identity.session.id"
    SESSION_OPERATION = "aitf.identity.session.operation"
    SESSION_SCOPE = "aitf.identity.session.scope"
    SESSION_EXPIRES_AT = "aitf.identity.session.expires_at"
    SESSION_ACTIONS_COUNT = "aitf.identity.session.actions_count"
    SESSION_DELEGATIONS_COUNT = "aitf.identity.session.delegations_count"
    SESSION_TERMINATION_REASON = "aitf.identity.session.termination_reason"

    # Identity type values
    class IdentityType:
        PERSISTENT = "persistent"
        EPHEMERAL = "ephemeral"
        DELEGATED = "delegated"
        FEDERATED = "federated"
        WORKLOAD = "workload"

    # Auth method values
    class AuthMethod:
        API_KEY = "api_key"
        OAUTH2 = "oauth2"
        OAUTH2_PKCE = "oauth2_pkce"
        JWT_BEARER = "jwt_bearer"
        MTLS = "mtls"
        SPIFFE_SVID = "spiffe_svid"
        DID_VC = "did_vc"
        HTTP_SIGNATURE = "http_signature"
        TOKEN_EXCHANGE = "token_exchange"

    # Delegation type values
    class DelegationType:
        ON_BEHALF_OF = "on_behalf_of"
        TOKEN_EXCHANGE = "token_exchange"
        CREDENTIAL_FORWARDING = "credential_forwarding"
        IMPERSONATION = "impersonation"
        CAPABILITY_GRANT = "capability_grant"
        SCOPED_PROXY = "scoped_proxy"


class AssetInventoryAttributes:
    """AITF AI Asset Inventory semantic convention attributes."""

    # Core asset attributes
    ID = "aitf.asset.id"
    NAME = "aitf.asset.name"
    TYPE = "aitf.asset.type"
    VERSION = "aitf.asset.version"
    HASH = "aitf.asset.hash"
    OWNER = "aitf.asset.owner"
    OWNER_TYPE = "aitf.asset.owner_type"
    DEPLOYMENT_ENVIRONMENT = "aitf.asset.deployment_environment"
    RISK_CLASSIFICATION = "aitf.asset.risk_classification"
    DESCRIPTION = "aitf.asset.description"
    TAGS = "aitf.asset.tags"
    SOURCE_REPOSITORY = "aitf.asset.source_repository"
    CREATED_AT = "aitf.asset.created_at"

    # Discovery attributes
    DISCOVERY_SCOPE = "aitf.asset.discovery.scope"
    DISCOVERY_METHOD = "aitf.asset.discovery.method"
    DISCOVERY_ASSETS_FOUND = "aitf.asset.discovery.assets_found"
    DISCOVERY_NEW_ASSETS = "aitf.asset.discovery.new_assets"
    DISCOVERY_SHADOW_ASSETS = "aitf.asset.discovery.shadow_assets"
    DISCOVERY_STATUS = "aitf.asset.discovery.status"

    # Audit attributes
    AUDIT_TYPE = "aitf.asset.audit.type"
    AUDIT_RESULT = "aitf.asset.audit.result"
    AUDIT_AUDITOR = "aitf.asset.audit.auditor"
    AUDIT_FRAMEWORK = "aitf.asset.audit.framework"
    AUDIT_FINDINGS = "aitf.asset.audit.findings"
    AUDIT_LAST_AUDIT_TIME = "aitf.asset.audit.last_audit_time"
    AUDIT_NEXT_AUDIT_DUE = "aitf.asset.audit.next_audit_due"
    AUDIT_RISK_SCORE = "aitf.asset.audit.risk_score"
    AUDIT_INTEGRITY_VERIFIED = "aitf.asset.audit.integrity_verified"
    AUDIT_COMPLIANCE_STATUS = "aitf.asset.audit.compliance_status"

    # Classification attributes
    CLASSIFICATION_FRAMEWORK = "aitf.asset.classification.framework"
    CLASSIFICATION_PREVIOUS = "aitf.asset.classification.previous"
    CLASSIFICATION_REASON = "aitf.asset.classification.reason"
    CLASSIFICATION_ASSESSOR = "aitf.asset.classification.assessor"
    CLASSIFICATION_USE_CASE = "aitf.asset.classification.use_case"
    CLASSIFICATION_AFFECTED_PERSONS = "aitf.asset.classification.affected_persons"
    CLASSIFICATION_SECTOR = "aitf.asset.classification.sector"
    CLASSIFICATION_BIOMETRIC = "aitf.asset.classification.biometric"
    CLASSIFICATION_AUTONOMOUS_DECISION = "aitf.asset.classification.autonomous_decision"

    # Dependency attributes
    DEPENDENCY_OPERATION = "aitf.asset.dependency.operation"
    DEPENDENCY_COUNT = "aitf.asset.dependency.count"
    DEPENDENCY_VULNERABLE_COUNT = "aitf.asset.dependency.vulnerable_count"

    # Decommission attributes
    DECOMMISSION_REASON = "aitf.asset.decommission.reason"
    DECOMMISSION_REPLACEMENT_ID = "aitf.asset.decommission.replacement_id"
    DECOMMISSION_DATA_RETENTION = "aitf.asset.decommission.data_retention"
    DECOMMISSION_APPROVED_BY = "aitf.asset.decommission.approved_by"

    class AssetType:
        MODEL = "model"
        DATASET = "dataset"
        PROMPT_TEMPLATE = "prompt_template"
        VECTOR_DB = "vector_db"
        MCP_SERVER = "mcp_server"
        AGENT = "agent"
        PIPELINE = "pipeline"
        GUARDRAIL = "guardrail"
        EMBEDDING_MODEL = "embedding_model"
        KNOWLEDGE_BASE = "knowledge_base"

    class RiskClassification:
        UNACCEPTABLE = "unacceptable"
        HIGH_RISK = "high_risk"
        LIMITED_RISK = "limited_risk"
        MINIMAL_RISK = "minimal_risk"
        SYSTEMIC = "systemic"
        NOT_CLASSIFIED = "not_classified"

    class DeploymentEnvironment:
        PRODUCTION = "production"
        STAGING = "staging"
        DEVELOPMENT = "development"
        SHADOW = "shadow"


class DriftDetectionAttributes:
    """AITF Model Drift Detection semantic convention attributes."""

    MODEL_ID = "aitf.drift.model_id"
    TYPE = "aitf.drift.type"
    SCORE = "aitf.drift.score"
    RESULT = "aitf.drift.result"
    DETECTION_METHOD = "aitf.drift.detection_method"
    BASELINE_METRIC = "aitf.drift.baseline_metric"
    CURRENT_METRIC = "aitf.drift.current_metric"
    METRIC_NAME = "aitf.drift.metric_name"
    THRESHOLD = "aitf.drift.threshold"
    P_VALUE = "aitf.drift.p_value"
    REFERENCE_DATASET = "aitf.drift.reference_dataset"
    REFERENCE_PERIOD = "aitf.drift.reference_period"
    EVALUATION_WINDOW = "aitf.drift.evaluation_window"
    SAMPLE_SIZE = "aitf.drift.sample_size"
    AFFECTED_SEGMENTS = "aitf.drift.affected_segments"
    FEATURE_NAME = "aitf.drift.feature_name"
    FEATURE_IMPORTANCE = "aitf.drift.feature_importance"
    ACTION_TRIGGERED = "aitf.drift.action_triggered"

    BASELINE_OPERATION = "aitf.drift.baseline.operation"
    BASELINE_ID = "aitf.drift.baseline.id"
    BASELINE_DATASET = "aitf.drift.baseline.dataset"
    BASELINE_SAMPLE_SIZE = "aitf.drift.baseline.sample_size"
    BASELINE_PERIOD = "aitf.drift.baseline.period"
    BASELINE_METRICS = "aitf.drift.baseline.metrics"
    BASELINE_FEATURES = "aitf.drift.baseline.features"
    BASELINE_PREVIOUS_ID = "aitf.drift.baseline.previous_id"

    INVESTIGATION_TRIGGER_ID = "aitf.drift.investigation.trigger_id"
    INVESTIGATION_ROOT_CAUSE = "aitf.drift.investigation.root_cause"
    INVESTIGATION_ROOT_CAUSE_CATEGORY = "aitf.drift.investigation.root_cause_category"
    INVESTIGATION_AFFECTED_SEGMENTS = "aitf.drift.investigation.affected_segments"
    INVESTIGATION_AFFECTED_USERS = "aitf.drift.investigation.affected_users_estimate"
    INVESTIGATION_BLAST_RADIUS = "aitf.drift.investigation.blast_radius"
    INVESTIGATION_SEVERITY = "aitf.drift.investigation.severity"
    INVESTIGATION_RECOMMENDATION = "aitf.drift.investigation.recommendation"

    REMEDIATION_ACTION = "aitf.drift.remediation.action"
    REMEDIATION_TRIGGER_ID = "aitf.drift.remediation.trigger_id"
    REMEDIATION_AUTOMATED = "aitf.drift.remediation.automated"
    REMEDIATION_INITIATED_BY = "aitf.drift.remediation.initiated_by"
    REMEDIATION_STATUS = "aitf.drift.remediation.status"
    REMEDIATION_ROLLBACK_TO = "aitf.drift.remediation.rollback_to"
    REMEDIATION_RETRAIN_DATASET = "aitf.drift.remediation.retrain_dataset"
    REMEDIATION_VALIDATION_PASSED = "aitf.drift.remediation.validation_passed"

    class DriftType:
        DATA_DISTRIBUTION = "data_distribution"
        CONCEPT = "concept"
        PERFORMANCE = "performance"
        CALIBRATION = "calibration"
        EMBEDDING = "embedding"
        FEATURE = "feature"
        PREDICTION = "prediction"
        LABEL = "label"

    class DetectionMethod:
        PSI = "psi"
        KS_TEST = "ks_test"
        CHI_SQUARED = "chi_squared"
        JS_DIVERGENCE = "js_divergence"
        KL_DIVERGENCE = "kl_divergence"
        WASSERSTEIN = "wasserstein"
        MMD = "mmd"
        ADWIN = "adwin"
        DDM = "ddm"
        PAGE_HINKLEY = "page_hinkley"
        CUSTOM = "custom"

    class RemediationAction:
        RETRAIN = "retrain"
        ROLLBACK = "rollback"
        RECALIBRATE = "recalibrate"
        FEATURE_GATE = "feature_gate"
        TRAFFIC_SHIFT = "traffic_shift"
        ALERT_ONLY = "alert_only"
        QUARANTINE = "quarantine"


class MemorySecurityAttributes:
    """AITF Memory Security semantic convention attributes (extends aitf.memory.*)."""

    CONTENT_HASH = "aitf.memory.security.content_hash"
    CONTENT_SIZE = "aitf.memory.security.content_size"
    INTEGRITY_HASH = "aitf.memory.security.integrity_hash"
    PROVENANCE_VERIFIED = "aitf.memory.security.provenance_verified"
    POISONING_SCORE = "aitf.memory.security.poisoning_score"
    CROSS_SESSION = "aitf.memory.security.cross_session"
    ISOLATION_VERIFIED = "aitf.memory.security.isolation_verified"
    MUTATION_COUNT = "aitf.memory.security.mutation_count"
    SNAPSHOT_BEFORE = "aitf.memory.security.snapshot_before"
    SNAPSHOT_AFTER = "aitf.memory.security.snapshot_after"


# --- A2A (Agent-to-Agent) Protocol Attributes ---


class A2AAttributes:
    """Google A2A (Agent-to-Agent) protocol semantic convention attributes."""

    # Agent Card / Discovery
    AGENT_NAME = "aitf.a2a.agent.name"
    AGENT_URL = "aitf.a2a.agent.url"
    AGENT_VERSION = "aitf.a2a.agent.version"
    AGENT_PROVIDER_ORG = "aitf.a2a.agent.provider.organization"
    AGENT_SKILLS = "aitf.a2a.agent.skills"
    AGENT_CAPABILITIES_STREAMING = "aitf.a2a.agent.capabilities.streaming"
    AGENT_CAPABILITIES_PUSH = "aitf.a2a.agent.capabilities.push_notifications"
    PROTOCOL_VERSION = "aitf.a2a.protocol.version"
    TRANSPORT = "aitf.a2a.transport"

    # Task
    TASK_ID = "aitf.a2a.task.id"
    TASK_CONTEXT_ID = "aitf.a2a.task.context_id"
    TASK_STATE = "aitf.a2a.task.state"
    TASK_PREVIOUS_STATE = "aitf.a2a.task.previous_state"
    TASK_ARTIFACTS_COUNT = "aitf.a2a.task.artifacts_count"
    TASK_HISTORY_LENGTH = "aitf.a2a.task.history_length"

    # Message
    MESSAGE_ID = "aitf.a2a.message.id"
    MESSAGE_ROLE = "aitf.a2a.message.role"
    MESSAGE_PARTS_COUNT = "aitf.a2a.message.parts_count"
    MESSAGE_PART_TYPES = "aitf.a2a.message.part_types"

    # Operation
    METHOD = "aitf.a2a.method"
    INTERACTION_MODE = "aitf.a2a.interaction_mode"
    JSONRPC_REQUEST_ID = "aitf.a2a.jsonrpc.request_id"
    JSONRPC_ERROR_CODE = "aitf.a2a.jsonrpc.error_code"
    JSONRPC_ERROR_MESSAGE = "aitf.a2a.jsonrpc.error_message"

    # Artifact
    ARTIFACT_ID = "aitf.a2a.artifact.id"
    ARTIFACT_NAME = "aitf.a2a.artifact.name"
    ARTIFACT_PARTS_COUNT = "aitf.a2a.artifact.parts_count"

    # Streaming
    STREAM_EVENT_TYPE = "aitf.a2a.stream.event_type"
    STREAM_IS_FINAL = "aitf.a2a.stream.is_final"
    STREAM_EVENTS_COUNT = "aitf.a2a.stream.events_count"

    # Push notifications
    PUSH_URL = "aitf.a2a.push.url"

    # Task state values
    class TaskState:
        SUBMITTED = "submitted"
        WORKING = "working"
        INPUT_REQUIRED = "input-required"
        COMPLETED = "completed"
        CANCELED = "canceled"
        FAILED = "failed"
        REJECTED = "rejected"
        AUTH_REQUIRED = "auth-required"

    # Interaction mode values
    class InteractionMode:
        SYNC = "sync"
        STREAM = "stream"
        PUSH = "push"

    # Transport values
    class Transport:
        JSONRPC = "jsonrpc"
        GRPC = "grpc"
        HTTP_JSON = "http_json"


# --- ACP (Agent Communication Protocol) Attributes ---


class ACPAttributes:
    """ACP (Agent Communication Protocol) semantic convention attributes."""

    # Agent discovery
    AGENT_NAME = "aitf.acp.agent.name"
    AGENT_DESCRIPTION = "aitf.acp.agent.description"
    AGENT_INPUT_CONTENT_TYPES = "aitf.acp.agent.input_content_types"
    AGENT_OUTPUT_CONTENT_TYPES = "aitf.acp.agent.output_content_types"
    AGENT_FRAMEWORK = "aitf.acp.agent.framework"
    AGENT_SUCCESS_RATE = "aitf.acp.agent.status.success_rate"
    AGENT_AVG_RUN_TIME = "aitf.acp.agent.status.avg_run_time_seconds"

    # Run
    RUN_ID = "aitf.acp.run.id"
    RUN_AGENT_NAME = "aitf.acp.run.agent_name"
    RUN_SESSION_ID = "aitf.acp.run.session_id"
    RUN_MODE = "aitf.acp.run.mode"
    RUN_STATUS = "aitf.acp.run.status"
    RUN_PREVIOUS_STATUS = "aitf.acp.run.previous_status"
    RUN_ERROR_CODE = "aitf.acp.run.error.code"
    RUN_ERROR_MESSAGE = "aitf.acp.run.error.message"
    RUN_CREATED_AT = "aitf.acp.run.created_at"
    RUN_FINISHED_AT = "aitf.acp.run.finished_at"
    RUN_DURATION_MS = "aitf.acp.run.duration_ms"

    # Message
    MESSAGE_ROLE = "aitf.acp.message.role"
    MESSAGE_PARTS_COUNT = "aitf.acp.message.parts_count"
    MESSAGE_CONTENT_TYPES = "aitf.acp.message.content_types"
    MESSAGE_HAS_CITATIONS = "aitf.acp.message.has_citations"
    MESSAGE_HAS_TRAJECTORY = "aitf.acp.message.has_trajectory"

    # Await/Resume
    AWAIT_ACTIVE = "aitf.acp.await.active"
    AWAIT_COUNT = "aitf.acp.await.count"
    AWAIT_DURATION_MS = "aitf.acp.await.duration_ms"

    # I/O counts
    INPUT_MESSAGE_COUNT = "aitf.acp.input.message_count"
    OUTPUT_MESSAGE_COUNT = "aitf.acp.output.message_count"

    # Operation
    OPERATION = "aitf.acp.operation"
    HTTP_METHOD = "aitf.acp.http.method"
    HTTP_STATUS_CODE = "aitf.acp.http.status_code"
    HTTP_URL = "aitf.acp.http.url"

    # Trajectory metadata
    TRAJECTORY_TOOL_NAME = "aitf.acp.trajectory.tool_name"
    TRAJECTORY_MESSAGE = "aitf.acp.trajectory.message"

    # Run status values
    class RunStatus:
        CREATED = "created"
        IN_PROGRESS = "in-progress"
        AWAITING = "awaiting"
        CANCELLING = "cancelling"
        CANCELLED = "cancelled"
        COMPLETED = "completed"
        FAILED = "failed"

    # Run mode values
    class RunMode:
        SYNC = "sync"
        ASYNC = "async"
        STREAM = "stream"


# Latency attributes (used across multiple span types)
class LatencyAttributes:
    """AITF Latency attributes for performance tracking."""

    TOTAL_MS = "aitf.latency.total_ms"
    TIME_TO_FIRST_TOKEN_MS = "aitf.latency.time_to_first_token_ms"
    TOKENS_PER_SECOND = "aitf.latency.tokens_per_second"
    QUEUE_TIME_MS = "aitf.latency.queue_time_ms"
    INFERENCE_TIME_MS = "aitf.latency.inference_time_ms"
