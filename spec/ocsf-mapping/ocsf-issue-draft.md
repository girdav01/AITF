# OCSF Issue — ready to post

**Title:**

```
[Proposal] Extend the ai_operation profile with inference, tool, and finding telemetry
```

---

**Body:**

### Summary

Following the `ai_agent` object + `ai_operation` profile in #1641 and the
`delegation` / control-plane work in #1640, I'd like to propose adding the
**substance of an AI operation** to the `ai_operation` profile: token usage,
tool/MCP invocation, and AI-specific finding context. These are **optional
objects and profile attributes on existing classes** — no new category, no new
required fields.

Opening this as a discussion to agree on placement and naming before I submit
the schema PR.

### Gap

#1641 / #1640 capture *who* acted (`ai_agent`) and *under what authority*
(`delegation`), and route agent/delegation lifecycle into the `ai` category.
They do not yet capture what the operation consumed or did:

| Gap (not in OCSF today) | Why it matters |
|---|---|
| LLM **token usage** | quota/abuse detection, cost attribution |
| **Tool / function / MCP invocation** | the agent's *action* — where tool-abuse, RCE, and data-access detections attach |
| **AI finding context** (OWASP LLM Top 10, guardrail, PII) | triage of prompt injection / leakage, and distinguishing an AI Detection Finding from a non-AI one on the same class |

### Proposed (high level)

Riding on existing classes (API Activity `6003`, Datastore Activity `6005`,
Detection Finding `2004`):

1. **`ai_token_usage` object** — `input_tokens`, `output_tokens`,
   `cached_tokens`, `reasoning_tokens`, `total_tokens`.
2. **`ai_tool` object** — `name`, `type_id` (Function / MCP Tool / Skill / API
   / Other), `server`, `transport`, `approval_required`, `is_approved`.
3. **`ai_operation` profile additions** — `token_usage`, `ai_tool`,
   `finish_reason`, `is_streaming`.
4. **`ai_finding` profile** — `owasp_llm_id` (OWASP LLM Top 10 2025),
   `detection_method`, `guardrail_name`, `is_blocked`, `pii_types`.

### Out of scope (potential follow-ups)

Supply-chain/provenance (AI-BOM) on Vulnerability Finding; an `ai_model_activity`
MLOps class; agent-reasoning / multi-agent objects on `agent_activity`; AI
quality metrics.

### Backwards compatibility

All additions are optional; nothing existing changes; no category is added.

### Prior art

Drawn from the AI Telemetry Framework (AITF), which already implements these as
OCSF-mapped output. I have a complete schema PR (object JSON, profile diffs,
`dictionary.json` entries) ready to open once placement is agreed.

Would maintainers prefer these as `ai_operation` profile extensions, or as a
separate profile? Happy to align with whatever fits the roadmap from #1640.
