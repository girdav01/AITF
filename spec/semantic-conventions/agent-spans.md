# Agent Span Conventions

AITF defines comprehensive semantic conventions for AI agent telemetry, supporting single agents, multi-agent orchestration, delegation, and agent memory.

## Span: `aitf.agent.session`

Represents a complete agent session from start to finish.

### Span Name

Format: `agent.session {aitf.agent.name}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Agent name |
| `aitf.agent.id` | string | Agent instance ID |
| `aitf.agent.session.id` | string | Session ID |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.type` | string | `"conversational"`, `"autonomous"`, `"reactive"`, `"proactive"` |
| `aitf.agent.framework` | string | `"langchain"`, `"crewai"`, `"autogen"`, `"semantic_kernel"`, `"custom"` |
| `aitf.agent.version` | string | Agent version |
| `aitf.agent.description` | string | Agent role/purpose |
| `aitf.agent.session.turn_count` | int | Total turns in session |
| `aitf.agent.team.name` | string | Team name (if part of a team) |
| `aitf.agent.team.id` | string | Team ID (if part of a team) |

### Child Spans

An agent session contains one or more `aitf.agent.step` spans.

---

## Span: `aitf.agent.step`

Represents a single step in the agent's execution loop (think-act-observe).

### Span Name

Format: `agent.step.{aitf.agent.step.type} {aitf.agent.name}`

Examples:
- `agent.step.planning research-agent`
- `agent.step.tool_use research-agent`
- `agent.step.reasoning research-agent`
- `agent.step.delegation manager-agent`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Agent name |
| `aitf.agent.step.type` | string | Step type (see below) |
| `aitf.agent.step.index` | int | Step number in sequence |

### Step Types

| Value | Description |
|-------|-------------|
| `planning` | Agent is planning next actions |
| `reasoning` | Agent is reasoning about observations |
| `tool_use` | Agent is calling a tool/function |
| `delegation` | Agent is delegating to another agent |
| `response` | Agent is generating final response |
| `reflection` | Agent is reflecting on its performance |
| `memory_access` | Agent is accessing memory |
| `guardrail_check` | Agent is checking guardrails |
| `human_in_loop` | Agent is waiting for human input |
| `error_recovery` | Agent is recovering from an error |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.step.thought` | string | Agent's internal reasoning (ReAct thought) |
| `aitf.agent.step.action` | string | Planned action |
| `aitf.agent.step.observation` | string | Result/observation from action |
| `aitf.agent.step.status` | string | `"success"`, `"error"`, `"retry"`, `"skipped"` |

### Child Spans

An agent step may contain:
- `gen_ai.inference` spans (LLM calls)
- `aitf.mcp.tool.invoke` spans (MCP tool calls)
- `aitf.skill.invoke` spans (skill invocations)
- `aitf.rag.pipeline` spans (RAG operations)
- `aitf.agent.session` spans (delegated sub-agent sessions)

---

## Span: `aitf.agent.delegation`

Represents agent-to-agent delegation within a multi-agent system.

### Span Name

Format: `agent.delegate {aitf.agent.name} -> {aitf.agent.delegation.target_agent}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Delegating agent |
| `aitf.agent.delegation.target_agent` | string | Target agent name |
| `aitf.agent.delegation.target_agent_id` | string | Target agent ID |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.delegation.reason` | string | Why delegation occurred |
| `aitf.agent.delegation.strategy` | string | `"round_robin"`, `"capability"`, `"hierarchical"`, `"vote"` |
| `aitf.agent.delegation.task` | string | Delegated task description |
| `aitf.agent.delegation.result` | string | Result from delegated agent |
| `aitf.agent.delegation.timeout_ms` | double | Delegation timeout |

---

## Span: `aitf.agent.team.orchestrate`

Represents a multi-agent team orchestration operation.

### Span Name

Format: `agent.team.orchestrate {aitf.agent.team.name}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.team.name` | string | Team name |
| `aitf.agent.team.id` | string | Team ID |
| `aitf.agent.team.topology` | string | Team topology |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.team.members` | string[] | Member agent names |
| `aitf.agent.team.coordinator` | string | Coordinator agent |
| `aitf.agent.team.task` | string | Team task |
| `aitf.agent.team.consensus_method` | string | `"majority"`, `"unanimous"`, `"coordinator"` |
| `aitf.agent.team.rounds` | int | Number of interaction rounds |

### Team Topologies

| Value | Description |
|-------|-------------|
| `hierarchical` | Manager/supervisor delegates to workers |
| `peer` | Agents collaborate as equals |
| `pipeline` | Sequential processing chain |
| `consensus` | Agents vote or reach consensus |
| `debate` | Agents debate to reach conclusion |
| `swarm` | Dynamic self-organizing agents |

---

## Span: `aitf.agent.memory`

Represents an agent memory operation.

### Span Name

Format: `agent.memory.{aitf.memory.operation} {aitf.agent.name}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.agent.name` | string | Agent name |
| `aitf.memory.operation` | string | `"store"`, `"retrieve"`, `"update"`, `"delete"`, `"search"` |
| `aitf.memory.store` | string | `"short_term"`, `"long_term"`, `"episodic"`, `"semantic"`, `"procedural"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.memory.key` | string | Memory key/identifier |
| `aitf.memory.hit` | boolean | Whether memory found (for retrieve) |
| `aitf.memory.ttl_seconds` | int | Time to live |
| `aitf.memory.provenance` | string | Origin of memory entry |

---

## Example: Multi-Agent Research System

```
Span: agent.team.orchestrate research-team
  aitf.agent.team.topology: "hierarchical"
  aitf.agent.team.members: ["manager", "researcher", "writer"]
  │
  ├─ Span: agent.session manager
  │    aitf.agent.type: "autonomous"
  │    aitf.agent.framework: "crewai"
  │    │
  │    ├─ Span: agent.step.planning manager
  │    │    aitf.agent.step.thought: "Need to research AI telemetry"
  │    │    └─ Span: chat gpt-4o
  │    │
  │    ├─ Span: agent.step.delegation manager
  │    │    aitf.agent.delegation.target_agent: "researcher"
  │    │    aitf.agent.delegation.reason: "Research expertise needed"
  │    │    │
  │    │    └─ Span: agent.session researcher
  │    │         ├─ Span: agent.step.tool_use researcher
  │    │         │    └─ Span: mcp.tool.invoke read_file
  │    │         └─ Span: agent.step.reasoning researcher
  │    │              └─ Span: chat claude-sonnet-4-5-20250929
  │    │
  │    └─ Span: agent.step.delegation manager
  │         aitf.agent.delegation.target_agent: "writer"
  │         │
  │         └─ Span: agent.session writer
  │              ├─ Span: agent.step.tool_use writer
  │              │    └─ Span: skill.invoke web-search
  │              └─ Span: agent.step.response writer
  │                   └─ Span: chat gpt-4o
```
