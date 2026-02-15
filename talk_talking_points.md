# Defining AI Stack Telemetry to Achieve Detection, Incident Response, Compliance and Governance

## Technical Talk: Structured Talking Points
**Duration:** 30 minutes  
**Date:** February 2026

---

## Talk Outline with Timing

| Section | Duration | Timing |
|---------|----------|--------|
| **1. Introduction & Problem Statement** | 3-4 min | 0:00 - 4:00 |
| **2. Current State: Telemetry Gaps in AI Deployments** | 6-7 min | 4:00 - 11:00 |
| **3. Framework Analysis: What Each Framework Requires** | 10-12 min | 11:00 - 23:00 |
| **4. Proposed AI Telemetry Standard** | 6-7 min | 23:00 - 29:00 |
| **5. Actionable Recommendations & Conclusion** | 1-2 min | 29:00 - 30:00 |

---

## Section 1: Introduction & Problem Statement (3-4 minutes)

### Opening Hook (30 sec)
> "In January 2024, Air Canada was held legally liable when their chatbot invented a bereavement fare policy that didn't exist. The company had no telemetry to detect the hallucination, no confidence scores to flag uncertainty, and no audit trail to understand what went wrong. This is the AI observability crisis."

### The Core Problem (1.5 min)

**Key Talking Points:**
- AI systems are fundamentally different from traditional software—they're probabilistic, not deterministic
- Traditional APM (Application Performance Monitoring) was designed for request-response patterns, not for:
  - Multi-turn conversations with accumulated context
  - Autonomous agents making chains of decisions
  - Retrieval-augmented generation with multiple data sources
  - Token economics and cost attribution

**The Visibility Gap:**
- We can tell you a request took 2.3 seconds
- We cannot tell you *why* the model said what it said
- We cannot detect prompt injection until it's exploited
- We cannot prove compliance without comprehensive logs

### Why Now? (1 min)

**Convergence of Pressures:**
1. **Regulatory Mandates:** EU AI Act, NIST AI RMF, ISO 42001 all require audit trails
2. **Security Threats:** MITRE ATLAS has cataloged 66+ techniques targeting AI systems
3. **Production Scale:** AI moved from experiments to mission-critical systems
4. **Agentic Future:** Autonomous agents amplify risk exponentially

**Transition Statement:**
> "Today I'll show you exactly what telemetry you're missing, what frameworks require, and give you a concrete standard to implement."

---

## Section 2: Current State - Telemetry Gaps (6-7 minutes)

### The AI Stack Blind Spots (2 min)

**Visual: Show the "visibility pyramid"**

```
        ┌─────────────────┐
        │   Application   │  ← What we see: User requests
        ├─────────────────┤
        │   LLM APIs      │  ← Missing: Reasoning traces, confidence
        ├─────────────────┤
        │   RAG/Vectors   │  ← Missing: Retrieval scores, attribution
        ├─────────────────┤
        │   Agents/Tools  │  ← Missing: Decision chains, parameters
        ├─────────────────┤
        │   Model Serving │  ← Missing: Per-request cost, batching
        └─────────────────┘
```

**Key Points:**

| Component | What We Log | What We're Missing |
|-----------|-------------|-------------------|
| **LLM APIs** | Tokens, latency | Confidence scores, reasoning traces, PII flags |
| **Vector DBs** | Query latency | Similarity scores, cross-tenant access, retrieval provenance |
| **RAG Pipelines** | Document counts | Chunk attribution, re-ranking decisions, freshness |
| **Agents** | Task completion | Tool parameters, inter-agent comms, memory mutations |
| **Fine-tuning** | Loss curves | Data lineage, license compliance, poisoning indicators |

### Real-World Failures (2 min)

**Case Study Rapid-Fire:**

1. **Samsung ChatGPT Leak (2023)**
   - Employees uploaded proprietary code to ChatGPT
   - Zero corporate visibility into shadow AI usage
   - Missing: DLP integration, prompt classification, user correlation

2. **Air Canada Chatbot (2024)**
   - Invented a policy, company held liable
   - Missing: Hallucination detection, confidence scores, source attribution

3. **Chevrolet Dealer Bot (2023)**
   - Agreed to sell car for $1, wrote Python code
   - Missing: Prompt injection detection, output boundary monitoring

4. **ICE Facial Recognition**
   - Same woman identified as two different people
   - Missing: Confidence thresholds, alternative candidates logging

> "Every one of these incidents was preventable with proper telemetry."

### Why Organizations Fail to Log (2 min)

**The Five Barriers:**

1. **Cost Anxiety**
   - Full prompt logging = 10-100x storage increase
   - At 1M requests/day: $2,700-27,000/quarter just for storage
   - SIEM ingestion: $5-15/GB for verbose AI telemetry

2. **Privacy-Security Paradox**
   ```
   Security: "We need full logs to detect attacks"
   Privacy:  "Storing prompts violates GDPR minimization"
   Legal:    "Both of you are creating liability"
   ```

3. **Vendor Limitations**
   - OpenAI: No internal model states
   - Anthropic: Filtering decisions not exposed
   - Pinecone: Audit logging enterprise-only

4. **Lack of Standards**
   - No agreed schema for LLM telemetry
   - OpenTelemetry lacks AI semantic conventions
   - Each vendor uses proprietary formats

5. **Organizational Silos**
   - ML team uses MLflow/W&B
   - Security uses Splunk/SIEM
   - Platform uses Datadog
   - No single source of truth

---

## Section 3: Framework Analysis (10-12 minutes)

### Framework Overview (1 min)

**Key Point:** Six major frameworks now address AI telemetry—they overlap but each has unique requirements.

| Framework | Focus | Telemetry Emphasis |
|-----------|-------|-------------------|
| OWASP LLM 2025 | LLM vulnerabilities | Input/output, resource consumption |
| OWASP Agentic 2026 | Autonomous agents | Tool calls, decision chains |
| MITRE ATLAS | Adversarial threats | Attack detection, anomaly monitoring |
| CSA AICM | Cloud AI controls | 243 controls, audit logging |
| NIST AI RMF | Risk lifecycle | Continuous monitoring, drift |
| ISO 42001 | Management systems | Tamper-evident, lifecycle events |

---

### 3.1 OWASP LLM & Agentic (3 min)

**OWASP LLM Top 10 (2025) - Key Telemetry Requirements:**

| Vulnerability | Required Telemetry |
|--------------|-------------------|
| **LLM01: Prompt Injection** | All prompts (sanitized), external content sources, output deviations |
| **LLM02: Sensitive Disclosure** | Output DLP scanning, credential pattern matches, extraction attempts |
| **LLM04: Data Poisoning** | Training data provenance, fine-tuning events, quality scores |
| **LLM06: Excessive Agency** | Tool invocations, permission requests, human overrides |
| **LLM10: Unbounded Consumption** | Token usage, cost tracking, rate limits |

**OWASP Agentic Top 10 (2026) - December 2025 Release:**

Two foundational principles drive telemetry requirements:
1. **Least Agency:** Monitor that agents operate within minimum autonomy
2. **Strong Observability:** Track what, why, and with what tools/identities

**Critical Agent Telemetry:**

| Vulnerability | Required Telemetry |
|--------------|-------------------|
| **ASI01: Goal Hijack** | Goal state changes, content ingestion provenance, behavioral baselines |
| **ASI02: Tool Misuse** | Every tool call with full parameters, sandbox violations |
| **ASI03: Identity Abuse** | Credential usage/delegation, privilege escalation events |
| **ASI05: Code Execution** | All generated code, execution requests, sandbox logs |
| **ASI06: Memory Poisoning** | Memory writes/reads with provenance, cross-session persistence |
| **ASI07: Inter-Agent Comms** | All messages, mTLS verification, replay detection |
| **ASI10: Rogue Agents** | Behavioral baselines, self-replication events, kill switch response |

> "The agentic framework is the first to mandate inter-agent communication logging—this is entirely new territory."

---

### 3.2 MITRE ATLAS (2 min)

**Overview:** 15 tactics, 66 techniques, 46 sub-techniques for AI/ML threats. October 2025 added 14 new agentic techniques.

**Key Detection Points by Attack Phase:**

| Tactic | Technique | Required Telemetry |
|--------|-----------|-------------------|
| **Reconnaissance** | Discover ML Artifacts | API access logs, endpoint probing patterns |
| **Initial Access** | Prompt Injection (AML.T0051) | Complete prompt logging, output analysis |
| **ML Attack Staging** | Poison Training Data | Training data changes, quality drift |
| **ML Attack Staging** | Create Proxy Model | Query volume patterns (extraction detection) |
| **Exfiltration** | Model Extraction via API | Query patterns, response completeness |

**New Agentic Techniques (October 2025):**
- AI Agent Context Poisoning → Memory integrity monitoring
- RAG Credential Harvesting → Credential pattern detection in retrievals
- Exfiltration via AI Agent Tool Invocation → Tool call data flow analysis

> "ATLAS is your threat model for detection rules. Every technique maps to specific telemetry requirements."

---

### 3.3 CSA AI Controls Matrix (2 min)

**Scale:** 243 control objectives across 18 security domains

**Audit Logging Domain (LOG) Highlights:**

| Control | Requirement |
|---------|-------------|
| **LOG-01** | Define what AI events to log, review annually |
| **LOG-02** | Tamper-evident storage, retention policies |
| **LOG-05** | Generate records with security-relevant information |
| **LOG-07** | Log key lifecycle events |

**AI-Specific Security Domain Telemetry:**

| Domain | Telemetry Focus |
|--------|-----------------|
| Model Manipulation | Performance drift detection |
| Data Poisoning | Data lineage, quality monitoring |
| Sensitive Disclosure | DLP integration, redaction audit |
| Model Theft | Extraction pattern detection |
| Insecure Supply Chain | SBOM, integrity verification |

**Key Point:** CSA maps to ISO 42001, ISO 27001, and NIST AI RMF—implement once, satisfy multiple frameworks.

---

### 3.4 NIST AI RMF & ISO 42001 (3 min)

**NIST AI RMF - Telemetry by Function:**

| Function | Key Telemetry Requirements |
|----------|---------------------------|
| **GOVERN** | AI system inventory, policy compliance monitoring, third-party assessments |
| **MAP** | System context docs, risk tolerance thresholds, dependency mapping |
| **MEASURE** | Bias/fairness metrics, drift detection, security monitoring, privacy metrics |
| **MANAGE** | Risk prioritization, mitigation logs, incident response records |

**NIST Generative AI Profile (AI 600-1) - Additional Requirements:**

| GenAI Risk | Required Telemetry |
|------------|-------------------|
| Confabulation | Hallucination rate, factual verification |
| Dangerous Content | Content filtering logs, flagged outputs |
| Information Integrity | Source verification, misinformation detection |
| Human-AI Config | Decision attribution, interaction logs |

**ISO 42001 - The Audit Standard:**

**Control A.6.2.8 - AI System Event Logging** (the key control):
- **Tamper-evident:** Protected from modification (WORM storage or crypto signing)
- **Lifecycle-mapped:** Events across all stages (dev → deploy → retire)
- **Complete:** Every significant event logged
- **Auditable:** Supports external audit

**AI Lifecycle Event Logging:**

| Stage | Events to Log |
|-------|---------------|
| Preparation | Requirements, data collection plans |
| Development | Training runs, hyperparameters, datasets used |
| Evaluation | Test results, validation metrics |
| Deployment | Release events, configuration changes |
| Delivery | User interactions, inference requests |
| Retirement | Decommissioning, data disposal |

> "ISO 42001 will be the audit framework. If you can't produce tamper-evident lifecycle logs, you fail certification."

---

## Section 4: Proposed AI Telemetry Standard (6-7 minutes)

### Framework Architecture (1 min)

**Design Principles:**
1. **Defense in Depth:** Multiple telemetry layers
2. **Privacy by Design:** Redaction and encryption built-in
3. **Cross-Framework:** Satisfies OWASP, ATLAS, NIST, ISO simultaneously
4. **Implementation-Agnostic:** Works with any stack

---

### 4.1 Telemetry Categories (2 min)

#### Category 1: Input/Output Telemetry

| Data Point | Purpose | Priority |
|------------|---------|----------|
| Request ID, timestamp, session ID | Correlation | **Critical** |
| Sanitized prompt content (hashed) | Pattern analysis | **Critical** |
| Full prompt (encrypted, access-controlled) | Forensics | **High** |
| Output content hash | Integrity verification | **Critical** |
| Token counts (input/output) | Cost attribution | **Critical** |
| PII detection flags | Data protection | **Critical** |
| Injection pattern matches | Attack detection | **Critical** |

#### Category 2: Model Operations Telemetry

| Data Point | Purpose | Priority |
|------------|---------|----------|
| Model version, checkpoint | Reproducibility | **Critical** |
| Confidence/uncertainty scores | Quality gating | **High** |
| Latency (TTFT, total) | Performance | **High** |
| Finish reason | Error analysis | **Critical** |
| Token probabilities (top-k) | Explainability | **Medium** |
| Context window utilization | Capacity planning | **High** |

#### Category 3: Agent Actions Telemetry

| Data Point | Purpose | Priority |
|------------|---------|----------|
| Tool name, parameters, return value | Action audit | **Critical** |
| Permission checks | Authorization | **Critical** |
| Goal state changes | Hijack detection | **Critical** |
| Inter-agent message content | Communication audit | **High** |
| Memory read/write events | Poisoning detection | **High** |
| Credential usage events | Identity abuse | **Critical** |

#### Category 4: Data Pipeline Telemetry

| Data Point | Purpose | Priority |
|------------|---------|----------|
| Retrieved document IDs | Attribution | **High** |
| Similarity scores | Retrieval quality | **High** |
| Chunk provenance | Source tracking | **High** |
| Embedding version | Reproducibility | **Medium** |
| Training data lineage | Compliance | **Critical** |
| Data quality scores | Poisoning detection | **High** |

#### Category 5: Security Events

| Data Point | Purpose | Priority |
|------------|---------|----------|
| Prompt injection detected | Attack alerting | **Critical** |
| Jailbreak attempt patterns | Threat intel | **Critical** |
| Data exfiltration indicators | DLP | **Critical** |
| Extraction attack patterns | Model theft | **High** |
| Behavioral anomalies | Rogue detection | **High** |
| Circuit breaker activations | Failure isolation | **High** |

---

### 4.2 Threat Model Mapping (1.5 min)

**How Telemetry Maps to MITRE ATLAS & OWASP:**

| Telemetry Category | MITRE ATLAS Techniques Detected | OWASP Vulnerabilities Addressed |
|-------------------|--------------------------------|--------------------------------|
| **Input/Output** | Prompt Injection (T0051), Reconnaissance | LLM01, LLM02, LLM07 |
| **Model Operations** | Model Extraction, Adversarial Data | LLM09 (Misinformation), LLM10 |
| **Agent Actions** | Context Poisoning, Tool Invocation Exfil | ASI01-03, ASI05, ASI10 |
| **Data Pipeline** | Training Data Poisoning, RAG Credential Harvest | LLM04, LLM08, ASI06 |
| **Security Events** | All Initial Access, Exfiltration tactics | All LLM & ASI vulnerabilities |

---

### 4.3 Governance Framework Mapping (1 min)

| Telemetry Category | NIST AI RMF | ISO 42001 | CSA AICM |
|-------------------|-------------|-----------|----------|
| **Input/Output** | MEASURE 2.3, 2.7 | A.6.2.8 | LOG-05, LOG-07 |
| **Model Operations** | MEASURE 2.1-2.4 | A.7.1, A.10.3 | Model Manipulation |
| **Agent Actions** | MANAGE 3, 4 | A.5.1 | Service Failures |
| **Data Pipeline** | MAP 3, MEASURE 2.8 | A.6.2.8 | Data Poisoning |
| **Security Events** | MANAGE 4 | A.6.2.8 | All domains |

---

### 4.4 Recommended Schema (1 min)

```json
{
  "event_id": "uuid-v4",
  "timestamp": "2026-02-06T14:30:00.000Z",
  "event_type": "inference|tool_call|memory_write|security_alert",
  
  "ai_system": {
    "id": "system-123",
    "component_type": "llm|agent|rag|vector_db",
    "version": "1.2.3"
  },
  
  "actor": {
    "type": "user|agent|system",
    "id": "user-456",
    "session_id": "session-789",
    "credentials_used": ["api_key_hash"]
  },
  
  "input": {
    "hash": "sha256:abc123",
    "token_count": 150,
    "pii_detected": ["email", "phone"],
    "injection_score": 0.02,
    "content_encrypted": "base64:..."
  },
  
  "output": {
    "hash": "sha256:def456",
    "token_count": 500,
    "confidence_score": 0.87,
    "content_filtered": false
  },
  
  "agent_context": {
    "goal_state": "summarize_document",
    "tools_invoked": [
      {"name": "web_search", "params_hash": "sha256:...", "status": "success"}
    ],
    "memory_operations": ["read:context_123"]
  },
  
  "retrieval": {
    "documents": ["doc-1", "doc-2"],
    "scores": [0.92, 0.87],
    "chunks_used": 5
  },
  
  "security": {
    "threat_indicators": [],
    "risk_score": 0.1,
    "mitigations_applied": ["pii_redaction"]
  },
  
  "compliance": {
    "frameworks": ["ISO42001", "NIST_AI_RMF"],
    "controls_satisfied": ["A.6.2.8", "MEASURE-2.3"]
  },
  
  "performance": {
    "latency_ms": 1234,
    "time_to_first_token_ms": 89,
    "cost_estimate_usd": 0.0023
  }
}
```

---

### 4.5 Implementation Priorities (0.5 min)

| Priority | Category | Rationale |
|----------|----------|-----------|
| **Critical** | Input/Output basics, Security events | Foundation for all detection |
| **Critical** | Agent tool invocations | Agency control and audit |
| **Critical** | Authentication/authorization | Access verification |
| **High** | Model operations metrics | Quality and cost control |
| **High** | RAG retrieval attribution | Source accountability |
| **High** | Inter-agent communication | Multi-agent security |
| **Medium** | Token-level attribution | Explainability compliance |
| **Medium** | Bias/fairness metrics | Equity requirements |

---

## Section 5: Actionable Recommendations & Conclusion (1-2 minutes)

### Quick Wins (This Week)

1. **Enable what you have:** Turn on audit logging in your existing tools (even if limited)
2. **Deploy an AI gateway:** Portkey, Helicone, or custom proxy for centralized logging
3. **Hash and store prompts:** Even hashed prompts enable pattern detection
4. **Add token cost tracking:** Know who's spending what

### Medium-Term (30-90 Days)

1. **Implement tiered logging:** Metadata always, content with PII redaction
2. **Adopt OpenTelemetry:** Add AI semantic conventions as they stabilize
3. **Create unified schema:** Get ML, Security, and Platform on same format
4. **Set up SIEM integration:** Feed AI telemetry into security monitoring
5. **Deploy drift detection:** Statistical monitoring of model outputs

### Long-Term (6-12 Months)

1. **Build compliance evidence automation:** Map logs to control requirements
2. **Implement behavioral baselines:** Anomaly detection for agents
3. **Establish data lineage:** Full provenance for training data
4. **Prepare for ISO 42001 audit:** Tamper-evident, lifecycle-mapped logs

### Closing Statement

> "The AI observability gap is closing—not because we've solved it, but because regulators, frameworks, and real-world incidents are forcing our hand. The organizations that treat telemetry as a first-class concern—not an afterthought—will be the ones that can actually detect attacks, respond to incidents, prove compliance, and maintain governance. Start with the Critical tier today. The cost of comprehensive telemetry is high; the cost of flying blind is higher."

---

## Key Quotes & Statistics

### Memorable Statistics

| Statistic | Source | Use In Section |
|-----------|--------|---------------|
| "243 control objectives across 18 domains" | CSA AICM | Framework Analysis |
| "66 techniques, 46 sub-techniques" for AI threats | MITRE ATLAS | Framework Analysis |
| "14 new agentic AI techniques" added October 2025 | MITRE ATLAS | Framework Analysis |
| ">70% license omission rate" in training datasets | MIT Sloan Audit | Telemetry Gaps |
| "$2,700-27,000/quarter" for prompt storage at scale | Research analysis | Why Orgs Fail |
| "5-15% latency increase" from comprehensive tracing | Research analysis | Why Orgs Fail |
| "$5-15 per GB" SIEM ingestion cost | Industry average | Why Orgs Fail |

### Quotable Statements

1. **On the problem:**
   > "We can tell you a request took 2.3 seconds. We cannot tell you why the model said what it said."

2. **On the privacy paradox:**
   > "Security wants full logs to detect attacks. Privacy says storing prompts violates GDPR. Legal says both are creating liability."

3. **On vendor gaps:**
   > "Your observability is only as good as your least observable vendor."

4. **On agents:**
   > "When an agent can call tools, pass credentials, and communicate with other agents—your telemetry requirements just 10x'd."

5. **On ISO 42001:**
   > "If you can't produce tamper-evident lifecycle logs, you fail certification."

6. **On cost vs. risk:**
   > "The cost of comprehensive telemetry is high; the cost of flying blind is higher."

### Case Study One-Liners

- **Samsung:** "Zero corporate visibility into which employees used ChatGPT or what data was submitted."
- **Air Canada:** "No telemetry to detect the hallucination, no confidence scores, no audit trail."
- **Chevrolet:** "The bot agreed to sell a car for $1—and no one knew until it went viral."

---

## Appendix: Speaker Notes

### Timing Checkpoints
- At 4:00 → Should be entering "Telemetry Gaps"
- At 11:00 → Should be entering "Framework Analysis"
- At 23:00 → Should be entering "Proposed Standard"
- At 29:00 → Should be in "Recommendations"

### Audience Engagement Points
- **0:30** - Air Canada story (hooks attention)
- **8:00** - Case study rapid-fire (concrete examples)
- **15:00** - OWASP Agentic newness (cutting-edge info)
- **25:00** - Schema example (technical depth)

### If Running Short on Time
- Cut: NIST AI RMF function details (keep high-level)
- Cut: Schema JSON (reference slide instead)
- Keep: All case studies (memorable)
- Keep: Priority matrix (actionable)

### If Running Long
- Expand: Implementation priority rationale
- Expand: Q&A on specific framework requirements
- Add: Live demo of telemetry gap in common tool

---

*Document prepared: February 6, 2026*
