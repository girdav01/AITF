# AITF Attributes Registry

Complete registry of all AITF semantic convention attributes. Organized by namespace.

## OTel GenAI Attributes (Preserved)

These attributes follow the OpenTelemetry GenAI semantic conventions exactly.

### `gen_ai.system`

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `gen_ai.system` | string | The AI system provider | `"openai"`, `"anthropic"`, `"bedrock"` |
| `gen_ai.operation.name` | string | The operation being performed | `"chat"`, `"text_completion"`, `"embeddings"` |

### `gen_ai.request.*`

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `gen_ai.request.model` | string | Model identifier | `"gpt-4o"`, `"claude-sonnet-4-5-20250929"` |
| `gen_ai.request.max_tokens` | int | Max tokens to generate | `4096` |
| `gen_ai.request.temperature` | double | Sampling temperature | `0.7` |
| `gen_ai.request.top_p` | double | Nucleus sampling parameter | `0.9` |
| `gen_ai.request.top_k` | int | Top-k sampling parameter | `40` |
| `gen_ai.request.stop_sequences` | string[] | Stop sequences | `["\n\n"]` |
| `gen_ai.request.frequency_penalty` | double | Frequency penalty | `0.5` |
| `gen_ai.request.presence_penalty` | double | Presence penalty | `0.5` |
| `gen_ai.request.seed` | int | Random seed for reproducibility | `42` |

### `gen_ai.response.*`

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `gen_ai.response.id` | string | Provider response ID | `"chatcmpl-abc123"` |
| `gen_ai.response.model` | string | Actual model used | `"gpt-4o-2024-08-06"` |
| `gen_ai.response.finish_reasons` | string[] | Finish reasons | `["stop"]`, `["tool_calls"]` |

### `gen_ai.usage.*`

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `gen_ai.usage.input_tokens` | int | Input/prompt tokens | `150` |
| `gen_ai.usage.output_tokens` | int | Output/completion tokens | `500` |

### `gen_ai.token.*`

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `gen_ai.token.type` | string | Token type | `"input"`, `"output"` |

---

## AITF Extended GenAI Attributes

Additional attributes for enhanced LLM observability.

### `gen_ai.request.*` (Extended)

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `gen_ai.request.tools` | string | JSON-encoded tools/functions | `"[{\"name\":\"search\"}]"` | Stable |
| `gen_ai.request.tool_choice` | string | Tool choice mode | `"auto"`, `"required"`, `"none"` | Stable |
| `gen_ai.request.response_format` | string | Response format | `"json_object"`, `"text"` | Stable |
| `gen_ai.request.stream` | boolean | Whether streaming | `true` | Stable |

### `gen_ai.usage.*` (Extended)

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `gen_ai.usage.cached_tokens` | int | Cached/prefix tokens | `50` | Stable |
| `gen_ai.usage.reasoning_tokens` | int | Reasoning/thinking tokens | `200` | Stable |

---

## AITF Agent Attributes

### `aitf.agent.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.agent.name` | string | Agent name | `"research-agent"` | Stable |
| `aitf.agent.id` | string | Unique agent instance ID | `"agent-abc123"` | Stable |
| `aitf.agent.type` | string | Agent type | `"conversational"`, `"autonomous"`, `"reactive"` | Stable |
| `aitf.agent.framework` | string | Agent framework | `"langchain"`, `"crewai"`, `"autogen"`, `"semantic_kernel"` | Stable |
| `aitf.agent.version` | string | Agent version | `"1.2.0"` | Stable |
| `aitf.agent.description` | string | Agent description/role | `"Researches technical topics"` | Stable |

### `aitf.agent.session.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.agent.session.id` | string | Agent session ID | `"sess-xyz789"` | Stable |
| `aitf.agent.session.turn_count` | int | Number of turns in session | `5` | Stable |
| `aitf.agent.session.start_time` | string | Session start ISO timestamp | `"2026-02-15T10:00:00Z"` | Stable |

### `aitf.agent.step.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.agent.step.type` | string | Step type | `"planning"`, `"reasoning"`, `"tool_use"`, `"delegation"`, `"response"` | Stable |
| `aitf.agent.step.index` | int | Step index in sequence | `3` | Stable |
| `aitf.agent.step.thought` | string | Agent's reasoning | `"I need to search for..."` | Stable |
| `aitf.agent.step.action` | string | Planned action | `"call_tool:search"` | Stable |
| `aitf.agent.step.observation` | string | Observation from action | `"Found 3 results..."` | Stable |

### `aitf.agent.delegation.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.agent.delegation.target_agent` | string | Delegated-to agent name | `"code-writer"` | Stable |
| `aitf.agent.delegation.target_agent_id` | string | Delegated-to agent ID | `"agent-def456"` | Stable |
| `aitf.agent.delegation.reason` | string | Why delegation occurred | `"Requires coding expertise"` | Stable |
| `aitf.agent.delegation.strategy` | string | Delegation strategy | `"round_robin"`, `"capability"`, `"hierarchical"` | Stable |

### `aitf.agent.team.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.agent.team.name` | string | Team name | `"research-team"` | Stable |
| `aitf.agent.team.id` | string | Team ID | `"team-abc"` | Stable |
| `aitf.agent.team.topology` | string | Team topology | `"hierarchical"`, `"peer"`, `"pipeline"`, `"consensus"` | Stable |
| `aitf.agent.team.members` | string[] | Member agent names | `["researcher","writer"]` | Stable |
| `aitf.agent.team.coordinator` | string | Coordinator agent name | `"manager"` | Stable |

---

## AITF MCP Attributes

### `aitf.mcp.server.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.mcp.server.name` | string | MCP server name | `"filesystem"` | Stable |
| `aitf.mcp.server.version` | string | MCP server version | `"1.0.0"` | Stable |
| `aitf.mcp.server.transport` | string | Transport type | `"stdio"`, `"sse"`, `"streamable_http"` | Stable |
| `aitf.mcp.server.url` | string | Server URL (if network) | `"http://localhost:3000/mcp"` | Stable |
| `aitf.mcp.protocol.version` | string | MCP protocol version | `"2025-03-26"` | Stable |

### `aitf.mcp.tool.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.mcp.tool.name` | string | Tool name | `"read_file"` | Stable |
| `aitf.mcp.tool.server` | string | Source MCP server | `"filesystem"` | Stable |
| `aitf.mcp.tool.input` | string | JSON input parameters | `"{\"path\":\"/tmp/f.txt\"}"` | Stable |
| `aitf.mcp.tool.output` | string | Tool output (may be redacted) | `"File contents..."` | Stable |
| `aitf.mcp.tool.is_error` | boolean | Whether tool returned error | `false` | Stable |
| `aitf.mcp.tool.duration_ms` | double | Tool execution time ms | `150.5` | Stable |
| `aitf.mcp.tool.approval_required` | boolean | Whether human approval needed | `true` | Stable |
| `aitf.mcp.tool.approved` | boolean | Whether approved (if required) | `true` | Stable |

### `aitf.mcp.resource.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.mcp.resource.uri` | string | Resource URI | `"file:///tmp/data.csv"` | Stable |
| `aitf.mcp.resource.name` | string | Resource name | `"data.csv"` | Stable |
| `aitf.mcp.resource.mime_type` | string | MIME type | `"text/csv"` | Stable |
| `aitf.mcp.resource.size_bytes` | int | Size in bytes | `1024` | Stable |

### `aitf.mcp.prompt.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.mcp.prompt.name` | string | Prompt template name | `"summarize"` | Stable |
| `aitf.mcp.prompt.arguments` | string | JSON prompt arguments | `"{\"style\":\"brief\"}"` | Stable |
| `aitf.mcp.prompt.description` | string | Prompt description | `"Summarize text"` | Stable |

### `aitf.mcp.sampling.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.mcp.sampling.model` | string | Requested model | `"claude-sonnet-4-5-20250929"` | Stable |
| `aitf.mcp.sampling.max_tokens` | int | Max tokens requested | `1024` | Stable |
| `aitf.mcp.sampling.include_context` | string | Context inclusion | `"thisServer"`, `"allServers"` | Stable |

---

## AITF Skills Attributes

### `aitf.skill.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.skill.name` | string | Skill name | `"web-search"` | Stable |
| `aitf.skill.id` | string | Unique skill ID | `"skill-search-001"` | Stable |
| `aitf.skill.version` | string | Skill version | `"2.1.0"` | Stable |
| `aitf.skill.provider` | string | Skill provider | `"builtin"`, `"marketplace"`, `"custom"` | Stable |
| `aitf.skill.category` | string | Skill category | `"search"`, `"code"`, `"data"`, `"communication"` | Stable |
| `aitf.skill.description` | string | Skill description | `"Search the web"` | Stable |
| `aitf.skill.input` | string | Skill input (JSON) | `"{\"query\":\"AI news\"}"` | Stable |
| `aitf.skill.output` | string | Skill output (may be redacted) | `"[{\"title\":...}]"` | Stable |
| `aitf.skill.status` | string | Execution status | `"success"`, `"error"`, `"timeout"`, `"denied"` | Stable |
| `aitf.skill.duration_ms` | double | Execution time ms | `250.0` | Stable |
| `aitf.skill.retry_count` | int | Number of retries | `0` | Stable |
| `aitf.skill.source` | string | Where skill was sourced | `"mcp:filesystem"`, `"api:openai"`, `"local"` | Stable |
| `aitf.skill.permissions` | string[] | Required permissions | `["file_read","network"]` | Stable |

---

## AITF RAG Attributes

### `aitf.rag.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.rag.pipeline.name` | string | Pipeline name | `"knowledge-base"` | Stable |
| `aitf.rag.pipeline.stage` | string | Current stage | `"retrieve"`, `"rerank"`, `"generate"`, `"evaluate"` | Stable |
| `aitf.rag.query` | string | User query | `"What is AITF?"` | Stable |
| `aitf.rag.query.embedding_model` | string | Embedding model | `"text-embedding-3-small"` | Stable |
| `aitf.rag.query.embedding_dimensions` | int | Embedding dimensions | `1536` | Stable |

### `aitf.rag.retrieve.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.rag.retrieve.database` | string | Vector DB name | `"pinecone"`, `"chromadb"`, `"weaviate"` | Stable |
| `aitf.rag.retrieve.index` | string | Index/collection name | `"documents"` | Stable |
| `aitf.rag.retrieve.top_k` | int | Number of results requested | `10` | Stable |
| `aitf.rag.retrieve.results_count` | int | Actual results returned | `8` | Stable |
| `aitf.rag.retrieve.min_score` | double | Minimum similarity score | `0.7` | Stable |
| `aitf.rag.retrieve.max_score` | double | Maximum similarity score | `0.95` | Stable |
| `aitf.rag.retrieve.filter` | string | Metadata filter (JSON) | `"{\"source\":\"docs\"}"` | Stable |

### `aitf.rag.rerank.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.rag.rerank.model` | string | Reranking model | `"cross-encoder/ms-marco"` | Stable |
| `aitf.rag.rerank.input_count` | int | Documents before rerank | `10` | Stable |
| `aitf.rag.rerank.output_count` | int | Documents after rerank | `5` | Stable |

### `aitf.rag.quality.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.rag.quality.context_relevance` | double | Context relevance score (0-1) | `0.85` | Experimental |
| `aitf.rag.quality.answer_relevance` | double | Answer relevance score (0-1) | `0.90` | Experimental |
| `aitf.rag.quality.faithfulness` | double | Answer faithfulness to context (0-1) | `0.88` | Experimental |
| `aitf.rag.quality.groundedness` | double | How grounded in sources (0-1) | `0.92` | Experimental |

---

## AITF Security Attributes

### `aitf.security.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.security.risk_score` | double | Overall risk score (0-100) | `75.5` | Stable |
| `aitf.security.risk_level` | string | Risk level | `"critical"`, `"high"`, `"medium"`, `"low"`, `"info"` | Stable |
| `aitf.security.threat_detected` | boolean | Whether threat detected | `true` | Stable |
| `aitf.security.threat_type` | string | Type of threat | `"prompt_injection"`, `"data_exfiltration"`, `"jailbreak"` | Stable |
| `aitf.security.owasp_category` | string | OWASP LLM category | `"LLM01"` through `"LLM10"` | Stable |
| `aitf.security.blocked` | boolean | Whether request blocked | `false` | Stable |
| `aitf.security.detection_method` | string | How threat was detected | `"pattern"`, `"ml_model"`, `"guardrail"`, `"policy"` | Stable |
| `aitf.security.confidence` | double | Detection confidence (0-1) | `0.95` | Stable |

### `aitf.security.guardrail.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.security.guardrail.name` | string | Guardrail name | `"content-filter"` | Stable |
| `aitf.security.guardrail.type` | string | Guardrail type | `"input"`, `"output"`, `"both"` | Stable |
| `aitf.security.guardrail.result` | string | Guardrail result | `"pass"`, `"fail"`, `"warn"` | Stable |
| `aitf.security.guardrail.provider` | string | Guardrail provider | `"nemo"`, `"guardrails_ai"`, `"llm_guard"`, `"bedrock"` | Stable |
| `aitf.security.guardrail.policy` | string | Policy violated (if any) | `"no-pii-output"` | Stable |

### `aitf.security.pii.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.security.pii.detected` | boolean | Whether PII found | `true` | Stable |
| `aitf.security.pii.types` | string[] | Types of PII found | `["email","phone","ssn"]` | Stable |
| `aitf.security.pii.count` | int | Number of PII instances | `3` | Stable |
| `aitf.security.pii.action` | string | Action taken | `"redacted"`, `"flagged"`, `"hashed"`, `"allowed"` | Stable |

---

## AITF Compliance Attributes

### `aitf.compliance.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.compliance.frameworks` | string[] | Mapped frameworks | `["nist_ai_rmf","eu_ai_act"]` | Stable |
| `aitf.compliance.nist_ai_rmf.controls` | string[] | NIST AI RMF controls | `["MAP-1.1","MEASURE-2.5"]` | Stable |
| `aitf.compliance.mitre_atlas.techniques` | string[] | MITRE ATLAS techniques | `["AML.T0051"]` | Stable |
| `aitf.compliance.iso_42001.controls` | string[] | ISO 42001 controls | `["6.1.4","8.4"]` | Stable |
| `aitf.compliance.eu_ai_act.articles` | string[] | EU AI Act articles | `["Article 9","Article 13"]` | Stable |
| `aitf.compliance.soc2.controls` | string[] | SOC 2 controls | `["CC6.1","CC7.2"]` | Stable |
| `aitf.compliance.gdpr.articles` | string[] | GDPR articles | `["Article 5","Article 22"]` | Stable |
| `aitf.compliance.ccpa.sections` | string[] | CCPA sections | `["1798.100"]` | Stable |

---

## AITF Cost Attributes

### `aitf.cost.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.cost.input_cost` | double | Cost for input tokens (USD) | `0.0015` | Stable |
| `aitf.cost.output_cost` | double | Cost for output tokens (USD) | `0.006` | Stable |
| `aitf.cost.total_cost` | double | Total cost (USD) | `0.0075` | Stable |
| `aitf.cost.currency` | string | Currency code | `"USD"` | Stable |
| `aitf.cost.model_pricing.input_per_1m` | double | Input price per 1M tokens | `3.00` | Stable |
| `aitf.cost.model_pricing.output_per_1m` | double | Output price per 1M tokens | `15.00` | Stable |
| `aitf.cost.budget.limit` | double | Budget limit (USD) | `100.00` | Stable |
| `aitf.cost.budget.used` | double | Budget used (USD) | `45.50` | Stable |
| `aitf.cost.budget.remaining` | double | Budget remaining (USD) | `54.50` | Stable |
| `aitf.cost.attribution.user` | string | User for cost attribution | `"user-123"` | Stable |
| `aitf.cost.attribution.team` | string | Team for cost attribution | `"engineering"` | Stable |
| `aitf.cost.attribution.project` | string | Project for cost attribution | `"chatbot-v2"` | Stable |

---

## AITF Quality Attributes

### `aitf.quality.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.quality.hallucination_score` | double | Hallucination score (0-1, lower=better) | `0.15` | Experimental |
| `aitf.quality.confidence` | double | Confidence score (0-1) | `0.85` | Experimental |
| `aitf.quality.factuality` | double | Factual accuracy (0-1) | `0.90` | Experimental |
| `aitf.quality.coherence` | double | Response coherence (0-1) | `0.88` | Experimental |
| `aitf.quality.toxicity_score` | double | Toxicity score (0-1, lower=better) | `0.02` | Experimental |
| `aitf.quality.bias_score` | double | Bias score (0-1, lower=better) | `0.05` | Experimental |
| `aitf.quality.feedback.rating` | double | User rating (1-5) | `4.5` | Experimental |
| `aitf.quality.feedback.thumbs` | string | User feedback | `"up"`, `"down"` | Experimental |

---

## AITF Supply Chain Attributes

### `aitf.supply_chain.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.supply_chain.model.source` | string | Model source | `"huggingface"`, `"openai"`, `"custom"` | Experimental |
| `aitf.supply_chain.model.hash` | string | Model file hash | `"sha256:abc123..."` | Experimental |
| `aitf.supply_chain.model.license` | string | Model license | `"apache-2.0"`, `"proprietary"` | Experimental |
| `aitf.supply_chain.model.training_data` | string | Training data description | `"CommonCrawl, Wikipedia"` | Experimental |
| `aitf.supply_chain.model.signed` | boolean | Whether model is signed | `true` | Experimental |
| `aitf.supply_chain.model.signer` | string | Model signer | `"anthropic"` | Experimental |
| `aitf.supply_chain.ai_bom.id` | string | AI Bill of Materials ID | `"bom-abc123"` | Experimental |
| `aitf.supply_chain.ai_bom.components` | string | JSON list of components | `"[{\"name\":\"gpt-4o\"}]"` | Experimental |

---

## AITF Memory Attributes

### `aitf.memory.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.memory.operation` | string | Memory operation | `"store"`, `"retrieve"`, `"update"`, `"delete"` | Experimental |
| `aitf.memory.store` | string | Memory store type | `"short_term"`, `"long_term"`, `"episodic"`, `"semantic"` | Experimental |
| `aitf.memory.key` | string | Memory key | `"user_preferences"` | Experimental |
| `aitf.memory.ttl_seconds` | int | Time to live in seconds | `3600` | Experimental |
| `aitf.memory.hit` | boolean | Whether memory was found | `true` | Experimental |
| `aitf.memory.provenance` | string | Origin of memory entry | `"conversation"`, `"tool_result"`, `"imported"` | Experimental |
