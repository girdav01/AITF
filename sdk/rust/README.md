# AITF Rust SDK (`aitf`)

An idiomatic Rust port of the core of the AITF Go/Python SDKs. It implements the
OCSF **class-reuse** model (OCSF PR #1641 / issue #1640): AI telemetry is emitted
under existing OCSF classes enriched with the `ai_operation` profile, and only
the new agent / delegation control-plane lifecycle uses the proposed "ai"
category (uid 9).

What's included:

- **`semconv`** — attribute-key string constants (GenAI, Agent, MCP, RAG,
  Security, Identity, Cost, ModelOps, AssetInventory, Quality, A2A, ACP, ANP,
  canonical AgentComm, Claude Compliance), grouped by namespace module.
- **`ocsf::schema`** — OCSF severity/status/activity/category/class UIDs, the
  `ai_agent` / `delegation` / `delegation_lineage` / `agent_message` objects, the
  AI extension models, and `AIBaseEvent`.
- **`ocsf::crosswalk`** — `build_ai_agent`, `build_delegation`,
  `build_delegation_lineage`, `build_agent_message`, `canonical_comm_status`, the
  event-name → OCSF class table, and the control-plane activity crosswalks.
- **`ocsf::mapper`** — `OcsfMapper::map_span` mapping `SpanData` to OCSF events
  with the same dispatch order as the Go/Python SDKs, enriching every event with
  the `ai_operation` profile.
- **`ocsf::claude_compliance`** — the Anthropic Claude Compliance Activity Feed
  mapper (`classify`, `ClaudeComplianceMapper::map_activity`).

## Install

This is a standalone crate (no workspace). Add it as a path dependency:

```toml
[dependencies]
aitf = { path = "../AITF/sdk/rust" }
```

Runtime dependencies are limited to `serde` and `serde_json`.

## Usage

```rust
use aitf::ocsf::{OcsfMapper, SpanData};
use aitf::semconv;

fn main() {
    // Build a span (e.g. from an OTel ReadableSpan).
    let span = SpanData::new("chat gpt-4o")
        .with_attr(semconv::gen_ai::SYSTEM, "openai")
        .with_attr(semconv::gen_ai::REQUEST_MODEL, "gpt-4o")
        .with_attr(semconv::gen_ai::OPERATION_NAME, "chat")
        .with_attr(semconv::gen_ai::USAGE_INPUT_TOKENS, 100_i64)
        .with_attr(semconv::gen_ai::USAGE_OUTPUT_TOKENS, 50_i64)
        // agent identity attaches the ai_operation profile
        .with_attr(semconv::agent::ID, "agent-001")
        .with_attr(semconv::agent::NAME, "orchestrator")
        .with_attr(semconv::agent::FRAMEWORK, "crewai");

    let event = OcsfMapper::new().map_span(&span).expect("AI span");
    assert_eq!(event.class_uid, 6003); // OCSF API Activity (reuse model)
    assert_eq!(event.ai_agent.as_ref().unwrap().type_id, Some(4)); // CrewAI

    // Serialize to OCSF JSON for a SIEM/XDR.
    let json = event.to_json().unwrap();
    println!("{json}");
}
```

Mapping Claude Compliance Activity Feed records:

```rust
use aitf::ocsf::ClaudeComplianceMapper;
use serde_json::json;

let mapper = ClaudeComplianceMapper::new();
let event = mapper.map_activity(&json!({
    "type": "claude_chat_created",
    "actor": {"type": "user_actor", "user_id": "user_1"},
}));
assert_eq!(event.class_uid, 6001); // Web Resources Activity
```

## Scope / not yet ported

This is a v0 focused on the schema + mapping core. The following are present in
the Go/Python SDKs but **not** ported here yet:

- Exporters: OTLP, OCSF, and CEF.
- The vendor mapper (OpenAI/Anthropic/etc. response normalization).
- The compliance-framework mapper (NIST AI RMF / EU AI Act / CSA AICM / ...).
- Metrics instruments and auto-instrumentation.
- The Claude Compliance Activity Feed **HTTP poller** (the Rust standard library
  has no HTTP client; the mapper maps records you supply).

Timestamps and metadata UIDs use a dependency-free placeholder scheme in v0
(no `chrono`/`uuid`); mappers overwrite `time` from the span start time.
```
