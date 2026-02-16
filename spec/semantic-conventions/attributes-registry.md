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

---

## AITF Model Operations (LLMOps/MLOps) Attributes

### `aitf.model_ops.training.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.model_ops.training.run_id` | string | Training run identifier | `"run-ft-20260215"` | Stable |
| `aitf.model_ops.training.type` | string | Training type | `"fine_tuning"`, `"lora"`, `"rlhf"`, `"dpo"` | Stable |
| `aitf.model_ops.training.base_model` | string | Base/foundation model | `"meta-llama/Llama-3.1-70B"` | Stable |
| `aitf.model_ops.training.framework` | string | Training framework | `"pytorch"`, `"transformers"`, `"jax"` | Stable |
| `aitf.model_ops.training.dataset.id` | string | Training dataset ID | `"customer-support-v3"` | Stable |
| `aitf.model_ops.training.dataset.version` | string | Dataset version hash | `"sha256:abc123"` | Stable |
| `aitf.model_ops.training.dataset.size` | int | Training examples count | `50000` | Stable |
| `aitf.model_ops.training.hyperparameters` | string | JSON hyperparameters | `"{\"lr\":0.0001}"` | Stable |
| `aitf.model_ops.training.epochs` | int | Training epochs | `3` | Stable |
| `aitf.model_ops.training.batch_size` | int | Batch size | `32` | Stable |
| `aitf.model_ops.training.learning_rate` | double | Learning rate | `0.0001` | Stable |
| `aitf.model_ops.training.loss_final` | double | Final training loss | `0.42` | Stable |
| `aitf.model_ops.training.val_loss_final` | double | Final validation loss | `0.48` | Stable |
| `aitf.model_ops.training.compute.gpu_type` | string | GPU type | `"H100"`, `"A100"` | Stable |
| `aitf.model_ops.training.compute.gpu_count` | int | GPU count | `8` | Stable |
| `aitf.model_ops.training.compute.gpu_hours` | double | Total GPU hours | `24.5` | Stable |
| `aitf.model_ops.training.output_model.id` | string | Output model ID | `"cs-llama-70b-lora-v3"` | Stable |
| `aitf.model_ops.training.output_model.hash` | string | Output model hash | `"sha256:def456"` | Stable |
| `aitf.model_ops.training.code_commit` | string | Code commit SHA | `"a1b2c3d"` | Stable |
| `aitf.model_ops.training.experiment.id` | string | Experiment tracker ID | `"exp-123"` | Stable |
| `aitf.model_ops.training.experiment.name` | string | Experiment name | `"cs-finetune-v3"` | Stable |
| `aitf.model_ops.training.status` | string | Run status | `"running"`, `"completed"`, `"failed"` | Stable |

### `aitf.model_ops.evaluation.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.model_ops.evaluation.run_id` | string | Evaluation run ID | `"eval-20260215"` | Stable |
| `aitf.model_ops.evaluation.model_id` | string | Model being evaluated | `"cs-llama-70b-lora-v3"` | Stable |
| `aitf.model_ops.evaluation.type` | string | Evaluation type | `"benchmark"`, `"llm_judge"`, `"safety"` | Stable |
| `aitf.model_ops.evaluation.dataset.id` | string | Eval dataset ID | `"eval-cs-v2"` | Stable |
| `aitf.model_ops.evaluation.dataset.version` | string | Dataset version | `"v2.1"` | Stable |
| `aitf.model_ops.evaluation.dataset.size` | int | Eval examples count | `1000` | Stable |
| `aitf.model_ops.evaluation.metrics` | string | JSON metric results | `"{\"accuracy\":0.94}"` | Stable |
| `aitf.model_ops.evaluation.judge_model` | string | LLM-as-judge model | `"gpt-4o"` | Stable |
| `aitf.model_ops.evaluation.baseline_model` | string | Baseline model for comparison | `"cs-llama-70b-lora-v2"` | Stable |
| `aitf.model_ops.evaluation.regression_detected` | boolean | Regression detected | `false` | Stable |
| `aitf.model_ops.evaluation.pass` | boolean | Passed quality gates | `true` | Stable |

### `aitf.model_ops.registry.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.model_ops.registry.operation` | string | Registry operation | `"register"`, `"promote"`, `"rollback"` | Stable |
| `aitf.model_ops.registry.model_id` | string | Model identifier | `"cs-llama-70b-lora-v3"` | Stable |
| `aitf.model_ops.registry.model_version` | string | Model version | `"3.0.0"` | Stable |
| `aitf.model_ops.registry.model_alias` | string | Model alias | `"@champion"` | Stable |
| `aitf.model_ops.registry.stage` | string | Lifecycle stage | `"staging"`, `"production"`, `"archived"` | Stable |
| `aitf.model_ops.registry.previous_stage` | string | Previous stage | `"staging"` | Stable |
| `aitf.model_ops.registry.owner` | string | Model owner | `"ml-team"` | Stable |
| `aitf.model_ops.registry.lineage.training_run_id` | string | Producing training run | `"run-ft-20260215"` | Stable |
| `aitf.model_ops.registry.lineage.parent_model_id` | string | Parent model | `"meta-llama/Llama-3.1-70B"` | Stable |

### `aitf.model_ops.deployment.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.model_ops.deployment.id` | string | Deployment identifier | `"deploy-cs-canary-001"` | Stable |
| `aitf.model_ops.deployment.model_id` | string | Model being deployed | `"cs-llama-70b-lora-v3"` | Stable |
| `aitf.model_ops.deployment.strategy` | string | Deployment strategy | `"canary"`, `"blue_green"`, `"rolling"` | Stable |
| `aitf.model_ops.deployment.model_version` | string | Model version | `"3.0.0"` | Stable |
| `aitf.model_ops.deployment.environment` | string | Target environment | `"production"` | Stable |
| `aitf.model_ops.deployment.endpoint` | string | Serving endpoint | `"https://api.example.com/v1/chat"` | Stable |
| `aitf.model_ops.deployment.canary_percent` | double | Canary traffic % | `10.0` | Stable |
| `aitf.model_ops.deployment.infrastructure.provider` | string | Infra provider | `"aws"`, `"gcp"` | Stable |
| `aitf.model_ops.deployment.infrastructure.gpu_type` | string | GPU type | `"H100"` | Stable |
| `aitf.model_ops.deployment.infrastructure.replicas` | int | Replica count | `4` | Stable |
| `aitf.model_ops.deployment.status` | string | Deployment status | `"completed"`, `"failed"`, `"rolled_back"` | Stable |
| `aitf.model_ops.deployment.health_check.status` | string | Health status | `"healthy"`, `"degraded"` | Stable |

### `aitf.model_ops.serving.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.model_ops.serving.operation` | string | Serving operation | `"route"`, `"fallback"`, `"cache_lookup"` | Stable |
| `aitf.model_ops.serving.route.selected_model` | string | Model selected by router | `"claude-opus-4-6"` | Stable |
| `aitf.model_ops.serving.route.reason` | string | Routing reason | `"cost"`, `"capability"`, `"latency"` | Stable |
| `aitf.model_ops.serving.route.candidates` | string[] | Candidate models | `["claude-opus-4-6","gpt-4o"]` | Stable |
| `aitf.model_ops.serving.fallback.chain` | string[] | Fallback chain | `["claude-opus-4-6","gpt-4o","llama-70b"]` | Stable |
| `aitf.model_ops.serving.fallback.depth` | int | Fallback depth | `1` | Stable |
| `aitf.model_ops.serving.fallback.trigger` | string | Fallback trigger | `"timeout"`, `"error"`, `"rate_limit"` | Stable |
| `aitf.model_ops.serving.cache.hit` | boolean | Cache hit | `true` | Stable |
| `aitf.model_ops.serving.cache.type` | string | Cache type | `"exact"`, `"semantic"` | Stable |
| `aitf.model_ops.serving.cache.similarity_score` | double | Semantic similarity | `0.95` | Stable |
| `aitf.model_ops.serving.cache.cost_saved_usd` | double | Cost saved | `0.003` | Stable |
| `aitf.model_ops.serving.circuit_breaker.state` | string | Circuit breaker state | `"closed"`, `"open"`, `"half_open"` | Stable |

### `aitf.model_ops.monitoring.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.model_ops.monitoring.check_type` | string | Monitoring check type | `"data_drift"`, `"embedding_drift"` | Stable |
| `aitf.model_ops.monitoring.model_id` | string | Monitored model | `"cs-llama-70b-lora-v3"` | Stable |
| `aitf.model_ops.monitoring.result` | string | Check result | `"normal"`, `"warning"`, `"alert"` | Stable |
| `aitf.model_ops.monitoring.metric_name` | string | Metric being checked | `"accuracy"` | Stable |
| `aitf.model_ops.monitoring.metric_value` | double | Current value | `0.94` | Stable |
| `aitf.model_ops.monitoring.baseline_value` | double | Baseline value | `0.96` | Stable |
| `aitf.model_ops.monitoring.drift_score` | double | Drift magnitude (0-1) | `0.15` | Stable |
| `aitf.model_ops.monitoring.drift_type` | string | Drift type | `"data"`, `"embedding"`, `"concept"` | Stable |
| `aitf.model_ops.monitoring.action_triggered` | string | Action taken | `"alert"`, `"retrain"`, `"rollback"` | Stable |

### `aitf.model_ops.prompt.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.model_ops.prompt.name` | string | Prompt template name | `"customer-greeting"` | Stable |
| `aitf.model_ops.prompt.operation` | string | Prompt operation | `"create"`, `"promote"`, `"rollback"` | Stable |
| `aitf.model_ops.prompt.version` | string | Prompt version (SemVer) | `"2.1.0"` | Stable |
| `aitf.model_ops.prompt.content_hash` | string | Template content hash | `"sha256:abc123"` | Stable |
| `aitf.model_ops.prompt.label` | string | Deployment label | `"production"`, `"staging"` | Stable |
| `aitf.model_ops.prompt.model_target` | string | Target model | `"claude-sonnet-4-5-20250929"` | Stable |
| `aitf.model_ops.prompt.evaluation.score` | double | Evaluation score (0-1) | `0.92` | Stable |
| `aitf.model_ops.prompt.evaluation.pass` | boolean | Passed quality gate | `true` | Stable |
| `aitf.model_ops.prompt.a_b_test.id` | string | A/B experiment ID | `"exp-prompt-123"` | Stable |
| `aitf.model_ops.prompt.a_b_test.variant` | string | Variant name | `"treatment_a"` | Stable |

---

## AITF Identity Attributes

### `aitf.identity.*` (Core)

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.identity.agent_id` | string | Agent identity identifier | `"agent-orch-001"` | Stable |
| `aitf.identity.agent_name` | string | Agent name | `"orchestrator"` | Stable |
| `aitf.identity.type` | string | Identity type | `"persistent"`, `"ephemeral"`, `"delegated"`, `"workload"` | Stable |
| `aitf.identity.provider` | string | Identity provider | `"okta"`, `"entra_id"`, `"spiffe"`, `"auth0"` | Stable |
| `aitf.identity.owner` | string | Identity owner | `"platform-team"` | Stable |
| `aitf.identity.owner_type` | string | Owner type | `"human"`, `"service"`, `"organization"` | Stable |
| `aitf.identity.credential_type` | string | Credential type | `"oauth_token"`, `"spiffe_svid"`, `"jwt"`, `"mtls_cert"` | Stable |
| `aitf.identity.credential_id` | string | Credential identifier | `"cred-abc123"` | Stable |
| `aitf.identity.status` | string | Identity status | `"active"`, `"suspended"`, `"revoked"`, `"expired"` | Stable |
| `aitf.identity.scope` | string[] | Granted scopes | `["tools:*","data:read"]` | Stable |
| `aitf.identity.expires_at` | string | Expiration timestamp | `"2026-02-16T10:00:00Z"` | Stable |
| `aitf.identity.ttl_seconds` | int | Time to live | `3600` | Stable |

### `aitf.identity.lifecycle.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.identity.lifecycle.operation` | string | Lifecycle operation | `"create"`, `"rotate"`, `"revoke"`, `"suspend"` | Stable |
| `aitf.identity.previous_status` | string | Previous status | `"active"` | Stable |
| `aitf.identity.auto_rotate` | boolean | Auto-rotation enabled | `true` | Stable |
| `aitf.identity.rotation_interval_seconds` | int | Rotation interval | `86400` | Stable |

### `aitf.identity.auth.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.identity.auth.method` | string | Auth method | `"oauth2_pkce"`, `"spiffe_svid"`, `"mtls"`, `"jwt_bearer"` | Stable |
| `aitf.identity.auth.result` | string | Auth result | `"success"`, `"failure"`, `"denied"`, `"expired"` | Stable |
| `aitf.identity.auth.provider` | string | Auth provider service | `"auth0"` | Stable |
| `aitf.identity.auth.target_service` | string | Service authenticated to | `"customer-db"` | Stable |
| `aitf.identity.auth.failure_reason` | string | Failure reason | `"invalid_token"` | Stable |
| `aitf.identity.auth.token_type` | string | Token type | `"bearer"`, `"dpop"`, `"mtls_bound"` | Stable |
| `aitf.identity.auth.scope_requested` | string[] | Scopes requested | `["data:read","tools:search"]` | Stable |
| `aitf.identity.auth.scope_granted` | string[] | Scopes granted | `["data:read"]` | Stable |
| `aitf.identity.auth.continuous` | boolean | Continuous re-auth | `true` | Stable |
| `aitf.identity.auth.pkce_used` | boolean | PKCE was used | `true` | Stable |
| `aitf.identity.auth.dpop_used` | boolean | DPoP proof included | `false` | Stable |

### `aitf.identity.authz.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.identity.authz.decision` | string | Authorization decision | `"allow"`, `"deny"`, `"conditional"` | Stable |
| `aitf.identity.authz.resource` | string | Resource accessed | `"customer-db"` | Stable |
| `aitf.identity.authz.action` | string | Action performed | `"read"`, `"write"`, `"execute"` | Stable |
| `aitf.identity.authz.policy_engine` | string | Policy engine | `"opa"`, `"cedar"`, `"casbin"` | Stable |
| `aitf.identity.authz.policy_id` | string | Matched policy | `"pol-agent-db-read"` | Stable |
| `aitf.identity.authz.deny_reason` | string | Denial reason | `"insufficient_scope"` | Stable |
| `aitf.identity.authz.risk_score` | double | Risk-based score (0-100) | `25.0` | Stable |
| `aitf.identity.authz.privilege_level` | string | Privilege level | `"standard"`, `"elevated"`, `"admin"` | Stable |
| `aitf.identity.authz.jea` | boolean | Just-Enough-Access applied | `true` | Stable |
| `aitf.identity.authz.time_limited` | boolean | Time-limited permission | `true` | Stable |

### `aitf.identity.delegation.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.identity.delegation.delegator` | string | Agent delegating authority | `"agent-orchestrator"` | Stable |
| `aitf.identity.delegation.delegator_id` | string | Delegator identity ID | `"agent-orch-001"` | Stable |
| `aitf.identity.delegation.delegatee` | string | Agent receiving authority | `"agent-researcher"` | Stable |
| `aitf.identity.delegation.delegatee_id` | string | Delegatee identity ID | `"agent-res-002"` | Stable |
| `aitf.identity.delegation.type` | string | Delegation type | `"on_behalf_of"`, `"token_exchange"`, `"capability_grant"` | Stable |
| `aitf.identity.delegation.chain` | string[] | Full delegation chain | `["user-alice","agent-orch","agent-res"]` | Stable |
| `aitf.identity.delegation.chain_depth` | int | Delegation depth | `2` | Stable |
| `aitf.identity.delegation.scope_delegated` | string[] | Scopes delegated | `["data:read"]` | Stable |
| `aitf.identity.delegation.scope_attenuated` | boolean | Scope was reduced | `true` | Stable |
| `aitf.identity.delegation.result` | string | Delegation result | `"success"`, `"failure"`, `"denied"` | Stable |
| `aitf.identity.delegation.proof_type` | string | Proof type | `"dpop"`, `"mtls_binding"` | Stable |
| `aitf.identity.delegation.ttl_seconds` | int | Delegation TTL | `300` | Stable |

### `aitf.identity.trust.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.identity.trust.operation` | string | Trust operation | `"establish"`, `"verify"`, `"revoke_trust"` | Stable |
| `aitf.identity.trust.peer_agent` | string | Peer agent name | `"agent-writer"` | Stable |
| `aitf.identity.trust.peer_agent_id` | string | Peer agent ID | `"agent-wrt-003"` | Stable |
| `aitf.identity.trust.result` | string | Trust result | `"established"`, `"failed"`, `"rejected"` | Stable |
| `aitf.identity.trust.method` | string | Trust method | `"mtls"`, `"spiffe"`, `"did_vc"` | Stable |
| `aitf.identity.trust.trust_domain` | string | Trust domain | `"spiffe://company.com"` | Stable |
| `aitf.identity.trust.cross_domain` | boolean | Cross-domain operation | `false` | Stable |
| `aitf.identity.trust.trust_level` | string | Trust level | `"basic"`, `"verified"`, `"high"`, `"full"` | Stable |
| `aitf.identity.trust.protocol` | string | Trust protocol | `"mcp"`, `"a2a"`, `"custom"` | Stable |

### `aitf.identity.session.*`

| Attribute | Type | Description | Example | Status |
|-----------|------|-------------|---------|--------|
| `aitf.identity.session.id` | string | Identity session ID | `"isess-xyz789"` | Stable |
| `aitf.identity.session.operation` | string | Session operation | `"create"`, `"refresh"`, `"terminate"` | Stable |
| `aitf.identity.session.scope` | string[] | Active session scopes | `["tools:*","data:read"]` | Stable |
| `aitf.identity.session.expires_at` | string | Session expiration | `"2026-02-16T11:00:00Z"` | Stable |
| `aitf.identity.session.actions_count` | int | Actions in session | `42` | Stable |
| `aitf.identity.session.delegations_count` | int | Delegations from session | `3` | Stable |
| `aitf.identity.session.termination_reason` | string | Termination reason | `"completed"`, `"timeout"`, `"revoked"` | Stable |
