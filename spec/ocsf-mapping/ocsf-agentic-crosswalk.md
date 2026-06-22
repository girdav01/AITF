# AITF ↔ OCSF Agentic AI Crosswalk

> **Status:** Alignment proposal — tracks OCSF
> [PR #1641](https://github.com/ocsf/ocsf-schema/pull/1641) and
> [issue #1640](https://github.com/ocsf/ocsf-schema/issues/1640).

This document reconciles AITF's established **OCSF Category 7** AI event classes
with the upstream OCSF direction for autonomous AI agents. It is the
*compromise* layer: AITF keeps its richer, AI-native class set (no breaking
change for existing SIEM/XDR consumers), while every event now also carries
**OCSF-conformant primitives** — the `ai_agent` object, the `ai_operation`
profile, and the `delegation` object/lineage — so telemetry maps cleanly onto
OCSF's native agentic schema as it lands.

## Why a crosswalk (and not a rename)

When AITF first defined its AI events it placed them in **Category 7**. Since
then OCSF has moved on two fronts:

1. **Category 7 is already taken.** In released OCSF, `uid: 7` is
   **Remediation** and `uid: 8` is **Unmanned Systems**. The agentic-AI
   proposal in issue #1640 reserves a **new `ai` category at `uid: 9`**.
2. **OCSF models agents as a reusable object + profile**, not only as
   standalone classes. PR #1641 adds the `ai_agent` object and extends the
   `ai_operation` profile so *any* event (system, network, web, email,
   process, …) can be attributed to the agent and delegation behind it.

A hard rename of Category 7 → 9 would break every deployed AITF detection rule,
dashboard, and Security Lake table today, for a category whose UID is not yet
finalized upstream. The chosen compromise is therefore:

- **Keep** AITF Category 7 classes (`7001`–`7010`) as the emitted schema.
- **Add** the OCSF `ai_agent` object + `delegation` object to every event via
  the `ai_operation` profile.
- **Publish** the activity/class crosswalk tables (below) so a consumer can
  translate Category 7 telemetry onto the proposed OCSF `ai` category (`uid 9`)
  deterministically once it is ratified.

## 1. The `ai_agent` object (OCSF PR #1641)

AITF carries the OCSF `ai_agent` object on the `ai_operation` profile of every
agent-attributable event. It is **distinct** from the OCSF `agent` object
(which models security sensors such as EDR/DLP).

| OCSF `ai_agent` field | Req. | AITF source attribute | Notes |
|---|---|---|---|
| `uid` | Required | `gen_ai.agent.id` ∥ `identity.agent_id` ∥ `agent.workflow_id` | Stable logical agent id |
| `instance_uid` | Recommended | `gen_ai.conversation.id` | Restart-sensitive running instance |
| `name` | Recommended | `gen_ai.agent.name` ∥ `identity.agent_name` | |
| `type` | Optional | caption of `type_id` | e.g. `LangChain` |
| `type_id` | Recommended | normalized from `agent.framework` | enum below |
| `ai_model` | Recommended | `gen_ai.request.model` ∥ `gen_ai.response.model` | Backing model |
| `version` | Recommended | `gen_ai.agent.version` | Agent code/config revision |
| `charter` | Optional | `gen_ai.agent.description` | Role / operating boundaries |

### `type_id` enum (framework normalization)

Matches the OCSF PR #1641 enum exactly. AITF frameworks without a dedicated
OCSF member normalize to `Other (99)`.

| `type_id` | Caption | AITF `agent.framework` values |
|---|---|---|
| 0 | Unknown | *(absent)* |
| 1 | Native | `native` |
| 2 | LangChain | `langchain`, `langgraph` |
| 3 | AutoGen | `autogen` |
| 4 | CrewAI | `crewai` |
| 99 | Other | `semantic_kernel`, `custom`, … |

## 2. The `ai_operation` profile (OCSF PR #1641)

PR #1641 adds `ai_agent` and `ai_model` to the `ai_operation` profile so that
existing data-plane events become agent-attributable. AITF applies the same
profile to **all** Category 7 events — and, for SIEM crosswalk, the same
profile is what a consumer would attach to base OCSF classes
(`system`, `network`, `web_resources_activity`, `email_activity`) when
translating AITF tool/inference activity into native OCSF categories.

Profile fields added to the AITF OCSF base event:

| Field | Type | Source |
|---|---|---|
| `ai_agent` | `ai_agent` object | §1 |
| `ai_model` | string | `gen_ai.request.model` |
| `delegation` | `delegation` object | §3 |
| `delegation_lineage` | `delegation_lineage` object | §3 |

## 3. The `delegation` object & lineage (OCSF issue #1640)

Issue #1640 introduces a **durable** `delegation` authorization context plus a
`delegation_lineage` / `delegation_node` directed graph for ancestry queries.
AITF maps its identity/delegation telemetry onto these:

| OCSF `delegation` field | AITF source attribute |
|---|---|
| `uid` (required) | `identity.delegation.delegatee_id` ∥ `agent.delegation.target_agent_id` ∥ last node of chain |
| `parent_uid` | `identity.delegation.delegator_id` |
| `issuer_uid` | `identity.provider` |
| `delegator` | `identity.delegation.delegator` |
| `delegatee` | `identity.delegation.delegatee` |
| `type` | `identity.delegation.type` |
| `scope` | `identity.delegation.scope_delegated` |
| `proof_type` | `identity.delegation.proof_type` |
| `ttl_seconds` | `identity.delegation.ttl_seconds` |

`delegation_lineage` is materialized from `identity.delegation.chain` (ordered
origin→current); each entry becomes a `delegation_node` with
`uid`, `parent_uid`, `agent_uid`, and `depth`.

## 4. Control-plane class crosswalk (OCSF issue #1640)

Issue #1640 proposes two control-plane classes in the future `ai` category:
`agent_activity` (agent lifecycle) and `delegation_activity` (authorization
lifecycle). UIDs are not yet finalized upstream, so AITF publishes only the
**stable activity-name mapping**.

### AITF `7002` AI Agent Activity → OCSF `agent_activity`

| AITF `activity_id` | AITF activity | OCSF `agent_activity` |
|---|---|---|
| 1 | Session Start | Spawn |
| 2 | Session End | Terminate |
| 3 | Step Execute | Update |
| 4 | Delegation | Register |
| 5 | Memory Access | Resume |
| 6 | Error Recovery | Resume |
| 7 | Human Approval | Suspend |
| 99 | Other | Unknown |

### AITF `7008` AI Identity (delegation) → OCSF `delegation_activity`

| AITF delegation op | OCSF `delegation_activity` |
|---|---|
| `create` / `grant` | Create |
| `revoke` | Revoke |
| `expire` | Expire |
| `complete` | Complete |

## 5. Class-level crosswalk to the proposed `ai` category (`uid 9`)

How each AITF Category 7 class would land in OCSF once the `ai` category is
ratified. Findings-shaped classes route to the existing **Findings (2)**
category; inventory routes to **Discovery (5)**.

| AITF class | OCSF target category | OCSF target class |
|---|---|---|
| 7001 Model Inference | 9 (AI) | `ai_inference_activity` |
| 7002 Agent Activity | 9 (AI) | `agent_activity` |
| 7003 Tool Execution | 9 (AI) | `ai_tool_activity` |
| 7004 Data Retrieval | 9 (AI) | `ai_retrieval_activity` |
| 7005 Security Finding | 2 (Findings) | `detection_finding` |
| 7006 Supply Chain | 2 (Findings) | `vulnerability_finding` |
| 7007 Governance | 2 (Findings) | `compliance_finding` |
| 7008 Identity / Delegation | 9 (AI) | `delegation_activity` |
| 7009 Model Operations | 9 (AI) | `ai_model_activity` |
| 7010 Asset Inventory | 5 (Discovery) | `inventory_info` |

> Class/category names for the `ai` category are AITF's proposed targets and
> will be reconciled with the final OCSF names once issue #1640 is merged.

## 6. `hosted_ai_agent_list` on `process` (OCSF PR #1641)

PR #1641 adds `hosted_ai_agent_list` to the OCSF `process` object (modeling
multiple agents hosted by one OS process, analogous to Windows hosted
services). When AITF emits an event with host/process context **and** an
`ai_agent`, exporters that populate the OCSF `process` object SHOULD append the
`ai_agent` to `process.hosted_ai_agent_list`.

## 7. Reference implementation

The crosswalk is implemented in each AITF SDK:

| SDK | Objects & profile | Builders & crosswalk tables |
|---|---|---|
| Python | `aitf/ocsf/schema.py` | `aitf/ocsf/crosswalk.py` |
| TypeScript | `src/ocsf/schema.ts` | `src/ocsf/crosswalk.ts` |
| Go | `ocsf/schema.go` | `ocsf/crosswalk.go` |

Each `OCSFMapper` enriches every mapped event with the `ai_operation` profile
(`ai_agent`, `ai_model`, `delegation`, `delegation_lineage`) automatically.
