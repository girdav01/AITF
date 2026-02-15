# GenAI Span Conventions

AITF preserves and extends OpenTelemetry GenAI span conventions for LLM inference operations.

## Span: `gen_ai.inference`

Represents a single LLM inference request (chat completion, text completion, embedding).

### Span Name

Format: `{gen_ai.operation.name} {gen_ai.request.model}`

Examples:
- `chat gpt-4o`
- `chat claude-sonnet-4-5-20250929`
- `embeddings text-embedding-3-small`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider identifier |
| `gen_ai.operation.name` | string | `"chat"`, `"text_completion"`, `"embeddings"` |
| `gen_ai.request.model` | string | Requested model ID |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.request.max_tokens` | int | Max tokens to generate |
| `gen_ai.request.temperature` | double | Sampling temperature |
| `gen_ai.request.top_p` | double | Nucleus sampling |
| `gen_ai.response.id` | string | Provider response ID |
| `gen_ai.response.model` | string | Actual model used |
| `gen_ai.response.finish_reasons` | string[] | Finish reasons |
| `gen_ai.usage.input_tokens` | int | Input tokens used |
| `gen_ai.usage.output_tokens` | int | Output tokens used |
| `server.address` | string | API endpoint host |
| `server.port` | int | API endpoint port |

### AITF Extended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.request.stream` | boolean | Streaming mode |
| `gen_ai.request.tools` | string | Tool definitions (JSON) |
| `gen_ai.request.tool_choice` | string | Tool selection mode |
| `gen_ai.request.response_format` | string | Expected format |
| `gen_ai.usage.cached_tokens` | int | Cached/prefix tokens |
| `gen_ai.usage.reasoning_tokens` | int | Reasoning tokens |
| `aitf.cost.total_cost` | double | Total request cost (USD) |
| `aitf.cost.input_cost` | double | Input cost (USD) |
| `aitf.cost.output_cost` | double | Output cost (USD) |
| `aitf.quality.confidence` | double | Response confidence |
| `aitf.security.risk_score` | double | Security risk score |

### Events

#### `gen_ai.content.prompt`

Emitted for each message in the prompt.

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.prompt` | string | The prompt content |

#### `gen_ai.content.completion`

Emitted for each message in the completion.

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.completion` | string | The completion content |

#### `gen_ai.tool.call` (AITF extension)

Emitted when the model makes a tool/function call.

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.tool.name` | string | Tool/function name |
| `gen_ai.tool.call_id` | string | Tool call ID |
| `gen_ai.tool.arguments` | string | Tool arguments (JSON) |

#### `gen_ai.tool.result` (AITF extension)

Emitted when a tool/function returns a result.

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.tool.name` | string | Tool/function name |
| `gen_ai.tool.call_id` | string | Tool call ID |
| `gen_ai.tool.result` | string | Tool result content |

### Span Status

- `OK` — Inference completed successfully
- `ERROR` — Inference failed (with error description)

### Example

```
Span: chat claude-sonnet-4-5-20250929
  Kind: CLIENT
  Status: OK
  Attributes:
    gen_ai.system: "anthropic"
    gen_ai.operation.name: "chat"
    gen_ai.request.model: "claude-sonnet-4-5-20250929"
    gen_ai.request.max_tokens: 4096
    gen_ai.request.temperature: 0.7
    gen_ai.response.id: "msg_abc123"
    gen_ai.response.model: "claude-sonnet-4-5-20250929"
    gen_ai.response.finish_reasons: ["end_turn"]
    gen_ai.usage.input_tokens: 150
    gen_ai.usage.output_tokens: 500
    aitf.cost.total_cost: 0.0075
    aitf.cost.input_cost: 0.00045
    aitf.cost.output_cost: 0.0075
  Events:
    gen_ai.content.prompt: {gen_ai.prompt: "Explain AITF"}
    gen_ai.content.completion: {gen_ai.completion: "AITF is..."}
```

## Span: `gen_ai.embeddings`

Represents an embedding generation request.

### Span Name

Format: `embeddings {gen_ai.request.model}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.system` | string | Provider identifier |
| `gen_ai.operation.name` | string | `"embeddings"` |
| `gen_ai.request.model` | string | Embedding model ID |

### AITF Extended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `gen_ai.request.encoding_format` | string | `"float"`, `"base64"` |
| `gen_ai.request.dimensions` | int | Requested dimensions |
| `gen_ai.usage.input_tokens` | int | Tokens processed |
| `aitf.cost.total_cost` | double | Cost (USD) |

## Streaming Support

For streaming responses, AITF extends OTel GenAI with additional timing attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `aitf.latency.total_ms` | double | Total latency (ms) |
| `aitf.latency.time_to_first_token_ms` | double | Time to first token (ms) |
| `aitf.latency.tokens_per_second` | double | Token generation rate |
| `aitf.latency.queue_time_ms` | double | Time in queue (ms) |
| `aitf.latency.inference_time_ms` | double | Pure inference time (ms) |
