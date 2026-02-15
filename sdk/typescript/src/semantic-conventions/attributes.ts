/**
 * AITF Semantic Convention Attribute Constants.
 *
 * All attribute keys used across AITF instrumentation, processors, and exporters.
 * OTel GenAI attributes (gen_ai.*) are preserved for compatibility.
 * AITF extensions use the aitf.* namespace.
 */

/** OpenTelemetry GenAI semantic convention attributes (preserved). */
export const GenAIAttributes = {
  SYSTEM: "gen_ai.system",
  OPERATION_NAME: "gen_ai.operation.name",

  // Request attributes
  REQUEST_MODEL: "gen_ai.request.model",
  REQUEST_MAX_TOKENS: "gen_ai.request.max_tokens",
  REQUEST_TEMPERATURE: "gen_ai.request.temperature",
  REQUEST_TOP_P: "gen_ai.request.top_p",
  REQUEST_TOP_K: "gen_ai.request.top_k",
  REQUEST_STOP_SEQUENCES: "gen_ai.request.stop_sequences",
  REQUEST_FREQUENCY_PENALTY: "gen_ai.request.frequency_penalty",
  REQUEST_PRESENCE_PENALTY: "gen_ai.request.presence_penalty",
  REQUEST_SEED: "gen_ai.request.seed",
  REQUEST_STREAM: "gen_ai.request.stream",
  REQUEST_TOOLS: "gen_ai.request.tools",
  REQUEST_TOOL_CHOICE: "gen_ai.request.tool_choice",
  REQUEST_RESPONSE_FORMAT: "gen_ai.request.response_format",

  // Response attributes
  RESPONSE_ID: "gen_ai.response.id",
  RESPONSE_MODEL: "gen_ai.response.model",
  RESPONSE_FINISH_REASONS: "gen_ai.response.finish_reasons",

  // Usage attributes
  USAGE_INPUT_TOKENS: "gen_ai.usage.input_tokens",
  USAGE_OUTPUT_TOKENS: "gen_ai.usage.output_tokens",
  USAGE_CACHED_TOKENS: "gen_ai.usage.cached_tokens",
  USAGE_REASONING_TOKENS: "gen_ai.usage.reasoning_tokens",

  // Token attributes
  TOKEN_TYPE: "gen_ai.token.type",

  // Event attributes
  PROMPT: "gen_ai.prompt",
  COMPLETION: "gen_ai.completion",
  TOOL_NAME: "gen_ai.tool.name",
  TOOL_CALL_ID: "gen_ai.tool.call_id",
  TOOL_ARGUMENTS: "gen_ai.tool.arguments",
  TOOL_RESULT: "gen_ai.tool.result",

  /** System values */
  System: {
    OPENAI: "openai",
    ANTHROPIC: "anthropic",
    BEDROCK: "bedrock",
    AZURE: "azure",
    GCP_VERTEX: "gcp_vertex",
    COHERE: "cohere",
    MISTRAL: "mistral",
    META: "meta",
    GOOGLE: "google",
  },

  /** Operation values */
  Operation: {
    CHAT: "chat",
    TEXT_COMPLETION: "text_completion",
    EMBEDDINGS: "embeddings",
    IMAGE_GENERATION: "image_generation",
    AUDIO: "audio",
  },
} as const;

/** AITF Agent semantic convention attributes. */
export const AgentAttributes = {
  // Core agent attributes
  NAME: "aitf.agent.name",
  ID: "aitf.agent.id",
  TYPE: "aitf.agent.type",
  FRAMEWORK: "aitf.agent.framework",
  VERSION: "aitf.agent.version",
  DESCRIPTION: "aitf.agent.description",

  // Session attributes
  SESSION_ID: "aitf.agent.session.id",
  SESSION_TURN_COUNT: "aitf.agent.session.turn_count",
  SESSION_START_TIME: "aitf.agent.session.start_time",

  // Step attributes
  STEP_TYPE: "aitf.agent.step.type",
  STEP_INDEX: "aitf.agent.step.index",
  STEP_THOUGHT: "aitf.agent.step.thought",
  STEP_ACTION: "aitf.agent.step.action",
  STEP_OBSERVATION: "aitf.agent.step.observation",
  STEP_STATUS: "aitf.agent.step.status",

  // Delegation attributes
  DELEGATION_TARGET_AGENT: "aitf.agent.delegation.target_agent",
  DELEGATION_TARGET_AGENT_ID: "aitf.agent.delegation.target_agent_id",
  DELEGATION_REASON: "aitf.agent.delegation.reason",
  DELEGATION_STRATEGY: "aitf.agent.delegation.strategy",
  DELEGATION_TASK: "aitf.agent.delegation.task",
  DELEGATION_RESULT: "aitf.agent.delegation.result",

  // Team attributes
  TEAM_NAME: "aitf.agent.team.name",
  TEAM_ID: "aitf.agent.team.id",
  TEAM_TOPOLOGY: "aitf.agent.team.topology",
  TEAM_MEMBERS: "aitf.agent.team.members",
  TEAM_COORDINATOR: "aitf.agent.team.coordinator",
  TEAM_CONSENSUS_METHOD: "aitf.agent.team.consensus_method",

  /** Type values */
  Type: {
    CONVERSATIONAL: "conversational",
    AUTONOMOUS: "autonomous",
    REACTIVE: "reactive",
    PROACTIVE: "proactive",
  },

  /** Framework values */
  Framework: {
    LANGCHAIN: "langchain",
    LANGGRAPH: "langgraph",
    CREWAI: "crewai",
    AUTOGEN: "autogen",
    SEMANTIC_KERNEL: "semantic_kernel",
    CUSTOM: "custom",
  },

  /** Step type values */
  StepType: {
    PLANNING: "planning",
    REASONING: "reasoning",
    TOOL_USE: "tool_use",
    DELEGATION: "delegation",
    RESPONSE: "response",
    REFLECTION: "reflection",
    MEMORY_ACCESS: "memory_access",
    GUARDRAIL_CHECK: "guardrail_check",
    HUMAN_IN_LOOP: "human_in_loop",
    ERROR_RECOVERY: "error_recovery",
  },

  /** Team topology values */
  TeamTopology: {
    HIERARCHICAL: "hierarchical",
    PEER: "peer",
    PIPELINE: "pipeline",
    CONSENSUS: "consensus",
    DEBATE: "debate",
    SWARM: "swarm",
  },
} as const;

/** AITF MCP (Model Context Protocol) semantic convention attributes. */
export const MCPAttributes = {
  // Server attributes
  SERVER_NAME: "aitf.mcp.server.name",
  SERVER_VERSION: "aitf.mcp.server.version",
  SERVER_TRANSPORT: "aitf.mcp.server.transport",
  SERVER_URL: "aitf.mcp.server.url",
  PROTOCOL_VERSION: "aitf.mcp.protocol.version",

  // Tool attributes
  TOOL_NAME: "aitf.mcp.tool.name",
  TOOL_SERVER: "aitf.mcp.tool.server",
  TOOL_INPUT: "aitf.mcp.tool.input",
  TOOL_OUTPUT: "aitf.mcp.tool.output",
  TOOL_IS_ERROR: "aitf.mcp.tool.is_error",
  TOOL_DURATION_MS: "aitf.mcp.tool.duration_ms",
  TOOL_APPROVAL_REQUIRED: "aitf.mcp.tool.approval_required",
  TOOL_APPROVED: "aitf.mcp.tool.approved",
  TOOL_COUNT: "aitf.mcp.tool.count",
  TOOL_NAMES: "aitf.mcp.tool.names",

  // Resource attributes
  RESOURCE_URI: "aitf.mcp.resource.uri",
  RESOURCE_NAME: "aitf.mcp.resource.name",
  RESOURCE_MIME_TYPE: "aitf.mcp.resource.mime_type",
  RESOURCE_SIZE_BYTES: "aitf.mcp.resource.size_bytes",

  // Prompt attributes
  PROMPT_NAME: "aitf.mcp.prompt.name",
  PROMPT_ARGUMENTS: "aitf.mcp.prompt.arguments",
  PROMPT_DESCRIPTION: "aitf.mcp.prompt.description",

  // Sampling attributes
  SAMPLING_MODEL: "aitf.mcp.sampling.model",
  SAMPLING_MAX_TOKENS: "aitf.mcp.sampling.max_tokens",
  SAMPLING_INCLUDE_CONTEXT: "aitf.mcp.sampling.include_context",

  /** Transport values */
  Transport: {
    STDIO: "stdio",
    SSE: "sse",
    STREAMABLE_HTTP: "streamable_http",
  },
} as const;

/** AITF Skills semantic convention attributes. */
export const SkillAttributes = {
  NAME: "aitf.skill.name",
  ID: "aitf.skill.id",
  VERSION: "aitf.skill.version",
  PROVIDER: "aitf.skill.provider",
  CATEGORY: "aitf.skill.category",
  DESCRIPTION: "aitf.skill.description",
  INPUT: "aitf.skill.input",
  OUTPUT: "aitf.skill.output",
  STATUS: "aitf.skill.status",
  DURATION_MS: "aitf.skill.duration_ms",
  RETRY_COUNT: "aitf.skill.retry_count",
  SOURCE: "aitf.skill.source",
  PERMISSIONS: "aitf.skill.permissions",
  COUNT: "aitf.skill.count",
  NAMES: "aitf.skill.names",

  // Composition attributes
  COMPOSE_NAME: "aitf.skill.compose.name",
  COMPOSE_SKILLS: "aitf.skill.compose.skills",
  COMPOSE_PATTERN: "aitf.skill.compose.pattern",
  COMPOSE_TOTAL: "aitf.skill.compose.total_skills",
  COMPOSE_COMPLETED: "aitf.skill.compose.completed_skills",

  // Resolution attributes
  RESOLVE_CAPABILITY: "aitf.skill.resolve.capability",
  RESOLVE_CANDIDATES: "aitf.skill.resolve.candidates",
  RESOLVE_SELECTED: "aitf.skill.resolve.selected",
  RESOLVE_REASON: "aitf.skill.resolve.reason",

  /** Provider values */
  Provider: {
    BUILTIN: "builtin",
    MARKETPLACE: "marketplace",
    CUSTOM: "custom",
    MCP: "mcp",
  },

  /** Category values */
  Category: {
    SEARCH: "search",
    CODE: "code",
    DATA: "data",
    COMMUNICATION: "communication",
    ANALYSIS: "analysis",
    GENERATION: "generation",
    KNOWLEDGE: "knowledge",
    SECURITY: "security",
    INTEGRATION: "integration",
    WORKFLOW: "workflow",
  },

  /** Status values */
  Status: {
    SUCCESS: "success",
    ERROR: "error",
    TIMEOUT: "timeout",
    DENIED: "denied",
    RETRY: "retry",
  },
} as const;

/** AITF RAG (Retrieval-Augmented Generation) semantic convention attributes. */
export const RAGAttributes = {
  // Pipeline attributes
  PIPELINE_NAME: "aitf.rag.pipeline.name",
  PIPELINE_STAGE: "aitf.rag.pipeline.stage",
  QUERY: "aitf.rag.query",
  QUERY_EMBEDDING_MODEL: "aitf.rag.query.embedding_model",
  QUERY_EMBEDDING_DIMENSIONS: "aitf.rag.query.embedding_dimensions",

  // Retrieval attributes
  RETRIEVE_DATABASE: "aitf.rag.retrieve.database",
  RETRIEVE_INDEX: "aitf.rag.retrieve.index",
  RETRIEVE_TOP_K: "aitf.rag.retrieve.top_k",
  RETRIEVE_RESULTS_COUNT: "aitf.rag.retrieve.results_count",
  RETRIEVE_MIN_SCORE: "aitf.rag.retrieve.min_score",
  RETRIEVE_MAX_SCORE: "aitf.rag.retrieve.max_score",
  RETRIEVE_FILTER: "aitf.rag.retrieve.filter",

  // Reranking attributes
  RERANK_MODEL: "aitf.rag.rerank.model",
  RERANK_INPUT_COUNT: "aitf.rag.rerank.input_count",
  RERANK_OUTPUT_COUNT: "aitf.rag.rerank.output_count",

  // Quality attributes
  QUALITY_CONTEXT_RELEVANCE: "aitf.rag.quality.context_relevance",
  QUALITY_ANSWER_RELEVANCE: "aitf.rag.quality.answer_relevance",
  QUALITY_FAITHFULNESS: "aitf.rag.quality.faithfulness",
  QUALITY_GROUNDEDNESS: "aitf.rag.quality.groundedness",

  /** Stage values */
  Stage: {
    RETRIEVE: "retrieve",
    RERANK: "rerank",
    GENERATE: "generate",
    EVALUATE: "evaluate",
  },
} as const;

/** AITF Security semantic convention attributes. */
export const SecurityAttributes = {
  RISK_SCORE: "aitf.security.risk_score",
  RISK_LEVEL: "aitf.security.risk_level",
  THREAT_DETECTED: "aitf.security.threat_detected",
  THREAT_TYPE: "aitf.security.threat_type",
  OWASP_CATEGORY: "aitf.security.owasp_category",
  BLOCKED: "aitf.security.blocked",
  DETECTION_METHOD: "aitf.security.detection_method",
  CONFIDENCE: "aitf.security.confidence",

  // Guardrail attributes
  GUARDRAIL_NAME: "aitf.security.guardrail.name",
  GUARDRAIL_TYPE: "aitf.security.guardrail.type",
  GUARDRAIL_RESULT: "aitf.security.guardrail.result",
  GUARDRAIL_PROVIDER: "aitf.security.guardrail.provider",
  GUARDRAIL_POLICY: "aitf.security.guardrail.policy",

  // PII attributes
  PII_DETECTED: "aitf.security.pii.detected",
  PII_TYPES: "aitf.security.pii.types",
  PII_COUNT: "aitf.security.pii.count",
  PII_ACTION: "aitf.security.pii.action",

  /** Risk level values */
  RiskLevel: {
    CRITICAL: "critical",
    HIGH: "high",
    MEDIUM: "medium",
    LOW: "low",
    INFO: "info",
  },

  /** OWASP LLM Top 10 */
  OWASP: {
    LLM01: "LLM01", // Prompt Injection
    LLM02: "LLM02", // Sensitive Information Disclosure
    LLM03: "LLM03", // Supply Chain Vulnerabilities
    LLM04: "LLM04", // Data and Model Poisoning
    LLM05: "LLM05", // Improper Output Handling
    LLM06: "LLM06", // Excessive Agency
    LLM07: "LLM07", // System Prompt Leakage
    LLM08: "LLM08", // Vector and Embedding Weaknesses
    LLM09: "LLM09", // Misinformation
    LLM10: "LLM10", // Unbounded Consumption
  },

  /** Threat type values */
  ThreatType: {
    PROMPT_INJECTION: "prompt_injection",
    SENSITIVE_DATA_EXPOSURE: "sensitive_data_exposure",
    SUPPLY_CHAIN: "supply_chain",
    DATA_POISONING: "data_poisoning",
    IMPROPER_OUTPUT: "improper_output",
    EXCESSIVE_AGENCY: "excessive_agency",
    SYSTEM_PROMPT_LEAK: "system_prompt_leak",
    VECTOR_DATA_WEAKNESS: "vector_data_weakness",
    MISINFORMATION: "misinformation",
    UNBOUNDED_CONSUMPTION: "unbounded_consumption",
    JAILBREAK: "jailbreak",
    DATA_EXFILTRATION: "data_exfiltration",
    MODEL_THEFT: "model_theft",
  },
} as const;

/** AITF Compliance semantic convention attributes. */
export const ComplianceAttributes = {
  FRAMEWORKS: "aitf.compliance.frameworks",
  NIST_AI_RMF_CONTROLS: "aitf.compliance.nist_ai_rmf.controls",
  MITRE_ATLAS_TECHNIQUES: "aitf.compliance.mitre_atlas.techniques",
  ISO_42001_CONTROLS: "aitf.compliance.iso_42001.controls",
  EU_AI_ACT_ARTICLES: "aitf.compliance.eu_ai_act.articles",
  SOC2_CONTROLS: "aitf.compliance.soc2.controls",
  GDPR_ARTICLES: "aitf.compliance.gdpr.articles",
  CCPA_SECTIONS: "aitf.compliance.ccpa.sections",
} as const;

/** AITF Cost semantic convention attributes. */
export const CostAttributes = {
  INPUT_COST: "aitf.cost.input_cost",
  OUTPUT_COST: "aitf.cost.output_cost",
  TOTAL_COST: "aitf.cost.total_cost",
  CURRENCY: "aitf.cost.currency",

  // Model pricing
  PRICING_INPUT_PER_1M: "aitf.cost.model_pricing.input_per_1m",
  PRICING_OUTPUT_PER_1M: "aitf.cost.model_pricing.output_per_1m",

  // Budget
  BUDGET_LIMIT: "aitf.cost.budget.limit",
  BUDGET_USED: "aitf.cost.budget.used",
  BUDGET_REMAINING: "aitf.cost.budget.remaining",

  // Attribution
  ATTRIBUTION_USER: "aitf.cost.attribution.user",
  ATTRIBUTION_TEAM: "aitf.cost.attribution.team",
  ATTRIBUTION_PROJECT: "aitf.cost.attribution.project",
} as const;

/** AITF Quality semantic convention attributes. */
export const QualityAttributes = {
  HALLUCINATION_SCORE: "aitf.quality.hallucination_score",
  CONFIDENCE: "aitf.quality.confidence",
  FACTUALITY: "aitf.quality.factuality",
  COHERENCE: "aitf.quality.coherence",
  TOXICITY_SCORE: "aitf.quality.toxicity_score",
  BIAS_SCORE: "aitf.quality.bias_score",
  FEEDBACK_RATING: "aitf.quality.feedback.rating",
  FEEDBACK_THUMBS: "aitf.quality.feedback.thumbs",
  FEEDBACK_COMMENT: "aitf.quality.feedback.comment",
} as const;

/** AITF Supply Chain semantic convention attributes. */
export const SupplyChainAttributes = {
  MODEL_SOURCE: "aitf.supply_chain.model.source",
  MODEL_HASH: "aitf.supply_chain.model.hash",
  MODEL_LICENSE: "aitf.supply_chain.model.license",
  MODEL_TRAINING_DATA: "aitf.supply_chain.model.training_data",
  MODEL_SIGNED: "aitf.supply_chain.model.signed",
  MODEL_SIGNER: "aitf.supply_chain.model.signer",
  AI_BOM_ID: "aitf.supply_chain.ai_bom.id",
  AI_BOM_COMPONENTS: "aitf.supply_chain.ai_bom.components",
} as const;

/** AITF Memory semantic convention attributes. */
export const MemoryAttributes = {
  OPERATION: "aitf.memory.operation",
  STORE: "aitf.memory.store",
  KEY: "aitf.memory.key",
  TTL_SECONDS: "aitf.memory.ttl_seconds",
  HIT: "aitf.memory.hit",
  PROVENANCE: "aitf.memory.provenance",

  Operation: {
    STORE: "store",
    RETRIEVE: "retrieve",
    UPDATE: "update",
    DELETE: "delete",
    SEARCH: "search",
  },

  Store: {
    SHORT_TERM: "short_term",
    LONG_TERM: "long_term",
    EPISODIC: "episodic",
    SEMANTIC: "semantic",
    PROCEDURAL: "procedural",
  },
} as const;

/** AITF Latency attributes for performance tracking. */
export const LatencyAttributes = {
  TOTAL_MS: "aitf.latency.total_ms",
  TIME_TO_FIRST_TOKEN_MS: "aitf.latency.time_to_first_token_ms",
  TOKENS_PER_SECOND: "aitf.latency.tokens_per_second",
  QUEUE_TIME_MS: "aitf.latency.queue_time_ms",
  INFERENCE_TIME_MS: "aitf.latency.inference_time_ms",
} as const;
