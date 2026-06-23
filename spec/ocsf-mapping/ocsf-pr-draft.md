# OCSF PR — ready to post

**Title:**

```
Add AI inference & tool telemetry to the ai_operation profile (ai_token_usage, ai_tool, ai_finding)
```

---

**Body:**

## Description

This PR extends the `ai_operation` profile introduced in #1641 so that AI
operations carried on existing event classes (e.g. API Activity `6003`,
Datastore Activity `6005`) can record the **substance of the operation** —
token usage, request/tool detail, and AI-specific finding context.

It complements #1641 (the `ai_agent` object — *who* acted) and #1640 (the
`delegation` object — *under what authority*) by adding *what the operation
consumed and did*. All additions are **optional objects and profile
attributes on existing classes** — consistent with OCSF's reuse model. **No
new category and no new required fields.**

Closes part of the gap discussed in #1640.

## Motivation

#1641 and #1640 establish agent identity, delegated authority, and
agent/delegation control-plane lifecycle. They do not yet capture what a SOC,
FinOps team, or AI-governance reviewer needs from the operation itself:

- **Token usage / cost** — there is no OCSF representation of LLM token
  consumption. Needed for quota/abuse detection and cost attribution.
- **Tool / function / MCP invocation** — the agent's *action* (the place
  RCE, data-access, and tool-abuse detections attach) has no object.
- **AI-specific finding context** — when an AI Detection Finding (`2004`) is
  raised for prompt injection or data leakage, there is no standard place to
  record the OWASP LLM category, the guardrail that fired, or detected PII —
  and no way to distinguish an AI finding from a non-AI one on the same class.

## Proposed changes

### New object — `objects/ai_token_usage.json`

```json
{
  "caption": "AI Token Usage",
  "name": "ai_token_usage",
  "description": "The token consumption of an AI model invocation.",
  "extends": "object",
  "attributes": {
    "input_tokens":     { "requirement": "recommended" },
    "output_tokens":    { "requirement": "recommended" },
    "cached_tokens":    { "requirement": "optional" },
    "reasoning_tokens": { "requirement": "optional" },
    "total_tokens":     { "requirement": "recommended" }
  }
}
```

### New object — `objects/ai_tool.json`

```json
{
  "caption": "AI Tool",
  "name": "ai_tool",
  "description": "A tool, function, skill, or MCP capability invoked by an AI agent. This is the agent's action and the primary attachment point for tool-use security controls.",
  "extends": "object",
  "attributes": {
    "name":     { "requirement": "required" },
    "type":     { "requirement": "optional" },
    "type_id": {
      "requirement": "recommended",
      "enum": {
        "0":  { "caption": "Unknown" },
        "1":  { "caption": "Function" },
        "2":  { "caption": "MCP Tool" },
        "3":  { "caption": "Skill" },
        "4":  { "caption": "API" },
        "99": { "caption": "Other" }
      }
    },
    "server":            { "requirement": "optional", "description": "MCP / remote server identifier." },
    "transport":         { "requirement": "optional", "description": "MCP transport, e.g. stdio, sse, http." },
    "approval_required": { "requirement": "optional" },
    "is_approved":       { "requirement": "optional" }
  }
}
```

### Profile update — `profiles/ai_operation.json`

Add to the existing profile (alongside `ai_agent` / `ai_model` from #1641):

```json
{
  "attributes": {
    "token_usage":   { "requirement": "optional", "description": "Token consumption of this AI operation." },
    "ai_tool":       { "requirement": "optional", "description": "The tool, function, or MCP capability invoked." },
    "finish_reason": { "requirement": "optional", "description": "The reason the model stopped generating." },
    "is_streaming":  { "requirement": "optional", "description": "Whether the response was streamed." }
  }
}
```

### New profile — `profiles/ai_finding.json`

Applied to Detection Finding (`2004`) / Compliance Finding (`2003`) so AI
findings are queryable and distinguishable from non-AI findings on the same
class.

```json
{
  "caption": "AI Finding",
  "name": "ai_finding",
  "meta": "profile",
  "description": "AI-specific risk context for findings raised on AI operations.",
  "attributes": {
    "owasp_llm_id": {
      "requirement": "recommended",
      "enum": {
        "0":  { "caption": "Unknown" },
        "1":  { "caption": "Prompt Injection" },
        "2":  { "caption": "Sensitive Information Disclosure" },
        "3":  { "caption": "Supply Chain" },
        "4":  { "caption": "Data and Model Poisoning" },
        "5":  { "caption": "Improper Output Handling" },
        "6":  { "caption": "Excessive Agency" },
        "7":  { "caption": "System Prompt Leakage" },
        "8":  { "caption": "Vector and Embedding Weaknesses" },
        "9":  { "caption": "Misinformation" },
        "10": { "caption": "Unbounded Consumption" },
        "99": { "caption": "Other" }
      },
      "description": "OWASP Top 10 for LLM Applications (2025) category."
    },
    "detection_method": { "requirement": "recommended", "description": "pattern, ml_model, guardrail, or policy." },
    "guardrail_name":   { "requirement": "optional", "description": "Name of the guardrail/control that fired." },
    "is_blocked":       { "requirement": "optional", "description": "Whether the offending action was blocked." },
    "pii_types":        { "requirement": "optional", "description": "Categories of PII detected." }
  }
}
```

### Dictionary additions — `dictionary.json`

```json
{
  "attributes": {
    "input_tokens":      { "caption": "Input Tokens",      "description": "Number of prompt/input tokens.",            "type": "long_t" },
    "output_tokens":     { "caption": "Output Tokens",     "description": "Number of completion/output tokens.",       "type": "long_t" },
    "cached_tokens":     { "caption": "Cached Tokens",     "description": "Prompt tokens served from cache.",          "type": "long_t" },
    "reasoning_tokens":  { "caption": "Reasoning Tokens",  "description": "Reasoning/thinking tokens.",                "type": "long_t" },
    "total_tokens":      { "caption": "Total Tokens",      "description": "Total tokens billed.",                      "type": "long_t" },
    "token_usage":       { "caption": "Token Usage",       "description": "AI model token consumption.",               "type": "ai_token_usage" },
    "ai_tool":           { "caption": "AI Tool",           "description": "A tool/function/skill/MCP capability.",     "type": "ai_tool" },
    "approval_required": { "caption": "Approval Required", "description": "Whether human approval was required.",      "type": "boolean_t" },
    "is_approved":       { "caption": "Approved",          "description": "Whether approval was granted.",             "type": "boolean_t" },
    "finish_reason":     { "caption": "Finish Reason",     "description": "Reason the model stopped generating.",      "type": "string_t" },
    "is_streaming":      { "caption": "Streaming",         "description": "Whether the response was streamed.",        "type": "boolean_t" },
    "owasp_llm_id":      { "caption": "OWASP LLM ID",      "description": "OWASP Top 10 for LLM Applications id.",      "type": "integer_t" },
    "guardrail_name":    { "caption": "Guardrail Name",    "description": "Guardrail/control that fired.",             "type": "string_t" },
    "is_blocked":        { "caption": "Blocked",           "description": "Whether the action was blocked.",           "type": "boolean_t" },
    "pii_types":         { "caption": "PII Types",         "description": "Categories of PII detected.",               "type": "string_t", "is_array": true }
  }
}
```

> `name`, `type`, `type_id`, `server`, `transport`, and `detection_method`
> reuse existing dictionary attributes where present; only add the entries
> above that are not already defined on `main`.

## Out of scope (proposed as follow-on PRs)

To keep this PR reviewable, the following are deliberately deferred:

1. **AI supply-chain / provenance** (model hash/signature/AI-BOM) for reuse on
   Vulnerability Finding `2002`.
2. **MLOps lifecycle** — a new `ai_model_activity` class in the `ai` category
   (training / evaluation / deployment / drift).
3. **Agent reasoning & multi-agent** objects (`ai_reasoning`, `ai_team`) on
   `agent_activity`.
4. **AI quality metrics** (hallucination, confidence, factuality) on
   `ai_operation`.

## Backwards compatibility

All additions are optional. No existing attribute is changed, no field becomes
required, and no category is added. Producers that do not emit these fields are
unaffected.

## Testing

- [ ] `make test` / schema validation passes with the new objects and profiles.
- [ ] Server renders the new objects, profiles, and enums.
- [ ] `dictionary.json` references resolve for all added attributes.

## Checklist

- [ ] New objects added under `objects/`.
- [ ] Profile updated / added under `profiles/`.
- [ ] `dictionary.json` updated.
- [ ] `CHANGELOG.md` updated.
- [ ] Linked to #1640 and #1641.
