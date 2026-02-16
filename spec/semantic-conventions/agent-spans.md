# Agent Span Conventions (AGENT_TRACE)

Status: **Normative** | CoSAI WS2 Alignment: **AGENT_TRACE** | OCSF Class: **7002 Agent Activity**

AITF defines comprehensive semantic conventions for AI agent telemetry, supporting single agents, multi-agent orchestration, delegation, and agent memory. This specification defines the normative field requirements aligned with CoSAI Working Stream 2 (Telemetry for AI) and mapped to applicable compliance and threat frameworks.

Key words "MUST", "SHOULD", "MAY" follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## Span: `aitf.agent.session`

Represents a complete agent session from start to finish.

### Span Name

Format: `agent.session {aitf.agent.name}`

### Span Kind

`INTERNAL`

### Normative Field Table

Instrumentors MUST emit all Required fields. Instrumentors SHOULD emit Recommended fields when the data is available. Optional fields MAY be emitted for enhanced observability.

#### Agent Identity

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.name` | string | **Required** | Human-readable agent name | OWASP LLM06 (Excessive Agency), MITRE ATLAS [AML.T0048](https://atlas.mitre.org/techniques/AML.T0048) |
| `aitf.agent.id` | string | **Required** | Unique agent instance identifier | OWASP LLM06, MITRE ATLAS [AML.T0048](https://atlas.mitre.org/techniques/AML.T0048), EU AI Act Art.12 |
| `aitf.agent.session.id` | string | **Required** | Session identifier | NIST AI RMF GOVERN-1.2, EU AI Act Art.12 |
| `aitf.agent.workflow_id` | string | **Recommended** | Workflow/DAG identifier linking related agent sessions | NIST AI RMF GOVERN-1.2, EU AI Act Art.12 |

#### Agent Configuration

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.type` | string | **Recommended** | Agent type: `"conversational"`, `"autonomous"`, `"reactive"`, `"proactive"` | EU AI Act Art.13 (Transparency) |
| `aitf.agent.framework` | string | **Recommended** | Agent framework: `"langchain"`, `"crewai"`, `"autogen"`, `"semantic_kernel"`, `"custom"` | NIST AI RMF MAP-1.1 |
| `aitf.agent.version` | string | **Optional** | Agent version string | NIST AI RMF MAP-1.1 |
| `aitf.agent.description` | string | **Optional** | Agent role/purpose description | EU AI Act Art.13 |

#### Session State (CoSAI WS2)

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.state` | string | **Recommended** | Current agent lifecycle state: `"initializing"`, `"planning"`, `"executing"`, `"waiting"`, `"completed"`, `"failed"`, `"suspended"` | OWASP LLM06 (Excessive Agency), MITRE ATLAS [AML.T0048](https://atlas.mitre.org/techniques/AML.T0048) |
| `aitf.agent.session.turn_count` | int | **Recommended** | Total turns completed in session | OWASP LLM10 (Unbounded Consumption) |
| `aitf.agent.session.start_time` | string | **Optional** | Session start timestamp (ISO 8601) | EU AI Act Art.12 |

#### Team Affiliation

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.team.name` | string | **Optional** | Team name (if part of a multi-agent team) | NIST AI RMF MAP-1.1 |
| `aitf.agent.team.id` | string | **Optional** | Team identifier | NIST AI RMF GOVERN-1.2 |

---

## Span: `aitf.agent.step`

Represents a single step in the agent's execution loop (think-act-observe).

### Span Name

Format: `agent.step.{aitf.agent.step.type} {aitf.agent.name}`

### Span Kind

`INTERNAL`

### Normative Field Table

#### Step Identification

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.name` | string | **Required** | Agent name | OWASP LLM06, MITRE ATLAS AML.T0048 |
| `aitf.agent.step.type` | string | **Required** | Step type (see Step Types table below) | OWASP LLM06, NIST AI RMF MEASURE-2.5 |
| `aitf.agent.step.index` | int | **Required** | Step sequence number (0-indexed) | NIST AI RMF GOVERN-1.2 |

#### ReAct / Chain-of-Thought (CoSAI WS2)

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.step.thought` | string | **Recommended** | Agent's internal reasoning (ReAct "thought") | NIST AI RMF MAP-1.1, EU AI Act Art.13 (Transparency), MITRE ATLAS [AML.T0048](https://atlas.mitre.org/techniques/AML.T0048) |
| `aitf.agent.step.action` | string | **Recommended** | Planned/executed action | OWASP LLM06 (Excessive Agency) |
| `aitf.agent.step.observation` | string | **Recommended** | Result/observation from action | NIST AI RMF MEASURE-2.5 |
| `aitf.agent.scratchpad` | string | **Optional** | Accumulated agent scratchpad / working memory state (JSON) | OWASP LLM02 (Sensitive Info Disclosure), MITRE ATLAS [AML.T0048](https://atlas.mitre.org/techniques/AML.T0048) |
| `aitf.agent.next_action` | string | **Recommended** | Next planned action (forward-looking intent) | OWASP LLM06 (Excessive Agency), MITRE ATLAS [AML.T0048](https://atlas.mitre.org/techniques/AML.T0048) |
| `aitf.agent.step.status` | string | **Recommended** | Step outcome: `"success"`, `"error"`, `"retry"`, `"skipped"` | NIST AI RMF MEASURE-2.5 |

### Step Types

| Value | Description | Compliance |
|---|---|---|
| `planning` | Agent is planning next actions | EU AI Act Art.13 |
| `reasoning` | Agent is reasoning about observations | EU AI Act Art.13, NIST AI RMF MAP-1.1 |
| `tool_use` | Agent is calling a tool/function | OWASP LLM06, MITRE ATLAS AML.T0048 |
| `delegation` | Agent is delegating to another agent | OWASP LLM06, NIST AI RMF GOVERN-1.7 |
| `response` | Agent is generating final response | OWASP LLM05 |
| `reflection` | Agent is reflecting on its performance | NIST AI RMF MEASURE-2.5 |
| `memory_access` | Agent is accessing memory | OWASP LLM02 |
| `guardrail_check` | Agent is checking guardrails | OWASP LLM01–LLM10 |
| `human_in_loop` | Agent is waiting for human input | EU AI Act Art.14 (Human Oversight) |
| `error_recovery` | Agent is recovering from an error | NIST AI RMF MEASURE-2.5 |

---

## Span: `aitf.agent.delegation`

Represents agent-to-agent delegation within a multi-agent system.

### Span Name

Format: `agent.delegate {aitf.agent.name} -> {aitf.agent.delegation.target_agent}`

### Span Kind

`INTERNAL`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.name` | string | **Required** | Delegating agent name | OWASP LLM06, MITRE ATLAS AML.T0048 |
| `aitf.agent.delegation.target_agent` | string | **Required** | Target agent name | OWASP LLM06, MITRE ATLAS AML.T0048 |
| `aitf.agent.delegation.target_agent_id` | string | **Required** | Target agent instance ID | NIST AI RMF GOVERN-1.2, EU AI Act Art.12 |
| `aitf.agent.delegation.reason` | string | **Recommended** | Why delegation occurred | EU AI Act Art.13 (Transparency) |
| `aitf.agent.delegation.strategy` | string | **Recommended** | Strategy: `"round_robin"`, `"capability"`, `"hierarchical"`, `"vote"` | NIST AI RMF GOVERN-1.7 |
| `aitf.agent.delegation.task` | string | **Recommended** | Delegated task description | OWASP LLM06 |
| `aitf.agent.delegation.result` | string | **Optional** | Result from delegated agent | NIST AI RMF MEASURE-2.5 |
| `aitf.agent.delegation.timeout_ms` | double | **Optional** | Delegation timeout in milliseconds | OWASP LLM10 |

---

## Span: `aitf.agent.team.orchestrate`

Represents a multi-agent team orchestration operation.

### Span Name

Format: `agent.team.orchestrate {aitf.agent.team.name}`

### Span Kind

`INTERNAL`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.team.name` | string | **Required** | Team name | NIST AI RMF MAP-1.1 |
| `aitf.agent.team.id` | string | **Required** | Team identifier | NIST AI RMF GOVERN-1.2 |
| `aitf.agent.team.topology` | string | **Required** | Team topology (see table below) | NIST AI RMF GOVERN-1.7 |
| `aitf.agent.team.members` | string[] | **Recommended** | Member agent names | OWASP LLM06 |
| `aitf.agent.team.coordinator` | string | **Recommended** | Coordinator agent name | OWASP LLM06 |
| `aitf.agent.team.task` | string | **Optional** | Team task description | EU AI Act Art.13 |
| `aitf.agent.team.consensus_method` | string | **Optional** | `"majority"`, `"unanimous"`, `"coordinator"` | NIST AI RMF GOVERN-1.7 |
| `aitf.agent.team.rounds` | int | **Optional** | Number of interaction rounds | OWASP LLM10 |

### Team Topologies

| Value | Description | Compliance |
|---|---|---|
| `hierarchical` | Manager/supervisor delegates to workers | NIST AI RMF GOVERN-1.7 |
| `peer` | Agents collaborate as equals | NIST AI RMF GOVERN-1.7 |
| `pipeline` | Sequential processing chain | — |
| `consensus` | Agents vote or reach consensus | EU AI Act Art.14 |
| `debate` | Agents debate to reach conclusion | NIST AI RMF MEASURE-2.5 |
| `swarm` | Dynamic self-organizing agents | OWASP LLM06 |

---

## Span: `aitf.agent.memory`

Represents an agent memory operation.

### Span Name

Format: `agent.memory.{aitf.memory.operation} {aitf.agent.name}`

### Span Kind

`INTERNAL`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.agent.name` | string | **Required** | Agent name | OWASP LLM06 |
| `aitf.memory.operation` | string | **Required** | Operation: `"store"`, `"retrieve"`, `"update"`, `"delete"`, `"search"` | OWASP LLM02 (Sensitive Info) |
| `aitf.memory.store` | string | **Required** | Store type: `"short_term"`, `"long_term"`, `"episodic"`, `"semantic"`, `"procedural"` | NIST AI RMF MAP-1.5 |
| `aitf.memory.key` | string | **Recommended** | Memory key/identifier | OWASP LLM02 |
| `aitf.memory.hit` | boolean | **Recommended** | Whether memory was found (for retrieve) | NIST AI RMF MEASURE-2.5 |
| `aitf.memory.ttl_seconds` | int | **Optional** | Time to live in seconds | — |
| `aitf.memory.provenance` | string | **Optional** | Origin of memory entry | NIST AI RMF MAP-1.5 |

---

## CoSAI WS2 Field Mapping

Cross-reference between CoSAI WS2 `AGENT_TRACE` field names and AITF attribute keys:

| CoSAI WS2 Field | AITF Attribute | Notes |
|---|---|---|
| `agent.id` | `aitf.agent.id` | Direct match |
| `agent.workflow_id` | `aitf.agent.workflow_id` | New in CoSAI WS2 alignment |
| `agent.state` | `aitf.agent.state` | New in CoSAI WS2 alignment |
| `agent.thought` | `aitf.agent.step.thought` | Set on step spans |
| `agent.scratchpad` | `aitf.agent.scratchpad` | New in CoSAI WS2 alignment |
| `agent.next_action` | `aitf.agent.next_action` | New in CoSAI WS2 alignment |

---

## Example: Multi-Agent Research System

```
Span: agent.team.orchestrate research-team
  aitf.agent.team.topology: "hierarchical"
  aitf.agent.team.members: ["manager", "researcher", "writer"]
  |
  +- Span: agent.session manager
  |    aitf.agent.id: "agent-mgr-001"
  |    aitf.agent.type: "autonomous"
  |    aitf.agent.framework: "crewai"
  |    aitf.agent.workflow_id: "wf-research-abc123"
  |    aitf.agent.state: "executing"
  |    |
  |    +- Span: agent.step.planning manager
  |    |    aitf.agent.step.thought: "Need to research AI telemetry"
  |    |    aitf.agent.next_action: "delegate to researcher"
  |    |    +- Span: chat gpt-4o
  |    |
  |    +- Span: agent.step.delegation manager
  |    |    aitf.agent.delegation.target_agent: "researcher"
  |    |    aitf.agent.delegation.reason: "Research expertise needed"
  |    |    |
  |    |    +- Span: agent.session researcher
  |    |         aitf.agent.workflow_id: "wf-research-abc123"
  |    |         +- Span: agent.step.tool_use researcher
  |    |         |    +- Span: mcp.tool.invoke read_file
  |    |         +- Span: agent.step.reasoning researcher
  |    |              aitf.agent.scratchpad: "{\"findings\": [...]}"
  |    |              +- Span: chat claude-sonnet-4-5-20250929
  |    |
  |    +- Span: agent.step.delegation manager
  |         aitf.agent.delegation.target_agent: "writer"
  |         |
  |         +- Span: agent.session writer
  |              aitf.agent.workflow_id: "wf-research-abc123"
  |              +- Span: agent.step.response writer
  |                   +- Span: chat gpt-4o
```
