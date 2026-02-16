# AITF Compliance Framework Mapping

AITF automatically maps AI telemetry events to eight regulatory and security frameworks.

## Overview

Every OCSF event produced by AITF is enriched with compliance metadata, mapping the event to relevant controls across multiple frameworks. This enables automated compliance reporting and audit trail generation.

## Framework Coverage Matrix

| AI Event Type | NIST AI RMF | MITRE ATLAS | ISO 42001 | EU AI Act | SOC 2 | GDPR | CCPA | CSA AICM |
|---------------|-------------|-------------|-----------|-----------|-------|------|------|----------|
| Model Inference | MAP-1.1, MEASURE-2.5 | AML.T0040 | 6.1.4, 8.4 | Art. 13, 15 | CC6.1 | Art. 5, 22 | 1798.100 | AIS-04, MDS-01, LOG-07 |
| Agent Activity | GOVERN-1.2, MANAGE-3.1 | AML.T0048 | 8.2, A.6.2.5 | Art. 14, 52 | CC7.2 | Art. 22 | - | AIS-02, MDS-05, GRC-02 |
| Tool Execution | MAP-3.5, MANAGE-4.2 | AML.T0043 | A.6.2.7 | Art. 9 | CC6.3 | Art. 25 | - | AIS-01, AIS-04, LOG-05 |
| Data Retrieval | MAP-1.5, MEASURE-2.7 | AML.T0025 | A.7.4 | Art. 10 | CC6.1 | Art. 5, 6 | 1798.100 | DSP-01, DSP-04, CEK-03 |
| Security Finding | MANAGE-2.4, MANAGE-4.1 | AML.T0051 | 6.1.2, A.6.2.4 | Art. 9, 62 | CC7.2, CC7.3 | Art. 32, 33 | 1798.150 | SEF-03, TVM-01, LOG-04 |
| Supply Chain | MAP-5.2, GOVERN-6.1 | AML.T0010 | A.6.2.3 | Art. 15, 28 | CC9.2 | Art. 28 | - | STA-01, STA-03, CCC-01 |
| Governance | GOVERN-1.1, MANAGE-1.3 | - | 5.1, 9.1 | Art. 9, 61 | CC1.2 | Art. 5 | 1798.185 | GRC-01, A&A-01, LOG-01 |
| Identity | GOVERN-1.5, MANAGE-2.1 | AML.T0052 | A.6.2.6 | Art. 9 | CC6.1, CC6.2 | Art. 32 | 1798.140 | IAM-01, IAM-02, IAM-04 |

---

## Framework Details

### 1. NIST AI RMF (AI Risk Management Framework)

The NIST AI RMF provides a structured approach to managing AI risks.

#### GOVERN Function

| Control | Description | Triggered By |
|---------|-------------|-------------|
| GOVERN-1.1 | AI risk management policies established | Governance events |
| GOVERN-1.2 | Responsibility and accountability assigned | Agent activity, delegation |
| GOVERN-1.5 | Ongoing monitoring mechanisms | Identity, auth events |
| GOVERN-6.1 | Policies for third-party AI | Supply chain events |

#### MAP Function

| Control | Description | Triggered By |
|---------|-------------|-------------|
| MAP-1.1 | Intended purpose documented | Model inference events |
| MAP-1.5 | AI limitations identified | Data retrieval, quality events |
| MAP-3.5 | Scientific integrity maintained | Tool execution events |
| MAP-5.2 | Third-party components identified | Supply chain events |

#### MEASURE Function

| Control | Description | Triggered By |
|---------|-------------|-------------|
| MEASURE-2.5 | Performance monitoring | Model inference, quality metrics |
| MEASURE-2.7 | AI system fairness assessed | Data retrieval, bias detection |

#### MANAGE Function

| Control | Description | Triggered By |
|---------|-------------|-------------|
| MANAGE-1.3 | Risk treatment applied | Governance events |
| MANAGE-2.1 | Access controls implemented | Identity events |
| MANAGE-2.4 | Mechanisms for feedback | Security findings |
| MANAGE-3.1 | Incidents managed | Agent errors, security events |
| MANAGE-4.1 | Post-deployment monitoring | Security findings |
| MANAGE-4.2 | Post-deployment activity managed | Tool execution events |

---

### 2. MITRE ATLAS (Adversarial Threat Landscape for AI Systems)

MITRE ATLAS catalogs adversarial techniques against AI systems.

| Technique | Description | Detected From |
|-----------|-------------|---------------|
| AML.T0010 | ML Supply Chain Compromise | Supply chain events |
| AML.T0020 | Poison Training Data | Data poisoning detection |
| AML.T0025 | Exfiltration via ML Inference API | Data retrieval anomalies |
| AML.T0040 | ML Model Inference API Access | Inference monitoring |
| AML.T0043 | Craft Adversarial Data | Tool execution, input analysis |
| AML.T0048 | Command and Control via AI | Agent activity patterns |
| AML.T0051 | LLM Prompt Injection | Prompt injection detection |
| AML.T0052 | Phishing via AI | Identity, social engineering |
| AML.T0054 | LLM Jailbreak | Jailbreak attempt detection |

---

### 3. ISO/IEC 42001 (AI Management System)

| Control | Description | Triggered By |
|---------|-------------|-------------|
| 5.1 | Leadership commitment | Governance events |
| 6.1.2 | AI risk assessment | Security findings |
| 6.1.4 | AI risk treatment | Inference controls |
| 8.2 | AI system lifecycle | Agent activity |
| 8.4 | AI system operation | Inference events |
| 9.1 | Monitoring and measurement | All telemetry |
| A.6.2.3 | Third-party management | Supply chain events |
| A.6.2.4 | AI security controls | Security findings |
| A.6.2.5 | AI system transparency | Agent activity |
| A.6.2.6 | Access control for AI | Identity events |
| A.6.2.7 | AI tool management | Tool execution |
| A.7.4 | Data management for AI | Data retrieval |

---

### 4. EU AI Act

| Article | Description | Triggered By |
|---------|-------------|-------------|
| Art. 9 | Risk management system | Security findings, governance, identity |
| Art. 10 | Data governance | Data retrieval events |
| Art. 13 | Transparency | Inference events, model info |
| Art. 14 | Human oversight | Agent activity, human-in-loop |
| Art. 15 | Accuracy, robustness, security | Inference quality, supply chain |
| Art. 28 | Obligations of deployers | Supply chain events |
| Art. 52 | Transparency obligations | Agent interaction disclosure |
| Art. 61 | Post-market monitoring | Governance events |
| Art. 62 | Reporting serious incidents | Security findings |

---

### 5. SOC 2

| Control | Description | Triggered By |
|---------|-------------|-------------|
| CC1.2 | Board oversight | Governance events |
| CC6.1 | Logical access controls | Inference, identity, data retrieval |
| CC6.2 | User authentication | Identity events |
| CC6.3 | Access authorization | Tool execution, permissions |
| CC7.2 | Monitoring for anomalies | Security findings, agent activity |
| CC7.3 | Evaluation of security events | Security findings |
| CC9.2 | Vendor management | Supply chain events |

---

### 6. GDPR (General Data Protection Regulation)

| Article | Description | Triggered By |
|---------|-------------|-------------|
| Art. 5 | Principles of processing | Inference, data retrieval, governance |
| Art. 6 | Lawfulness of processing | Data retrieval events |
| Art. 22 | Automated decision-making | Inference, agent activity |
| Art. 25 | Data protection by design | Tool execution, PII detection |
| Art. 28 | Processor obligations | Supply chain events |
| Art. 32 | Security of processing | Security findings, identity |
| Art. 33 | Notification of breach | Security findings (critical) |

---

### 7. CCPA (California Consumer Privacy Act)

| Section | Description | Triggered By |
|---------|-------------|-------------|
| 1798.100 | Right to know | Inference, data retrieval |
| 1798.105 | Right to delete | Data management events |
| 1798.140 | Definition of personal info | Identity, PII detection |
| 1798.150 | Private right of action | Security findings (breach) |
| 1798.185 | Rulemaking authority | Governance events |

---

### 8. CSA AI Controls Matrix (AICM)

The Cloud Security Alliance AI Controls Matrix provides 243 control objectives across 18 security domains for cloud-based AI systems. It extends the Cloud Controls Matrix (CCM v4) with AI-specific controls and a shared responsibility model across Cloud Service Providers, Model Providers, Orchestrated Service Providers, and Application Providers.

#### 18 Control Domains

| Domain ID | Domain Name | AI Relevance |
|-----------|-------------|--------------|
| A&A | Audit & Assurance | AI system audit trails, compliance evidence |
| AIS | Application & Interface Security | LLM API security, prompt/completion interfaces |
| BCR | Business Continuity & Operational Resilience | AI service availability, failover |
| CCC | Change Control & Configuration Management | Model versioning, deployment changes |
| CEK | Cryptography, Encryption & Key Management | Data-at-rest/in-transit encryption for AI data |
| DCS | Datacenter Security | GPU cluster physical security |
| DSP | Data Security & Privacy | Training data protection, PII in prompts |
| GRC | Governance, Risk Management & Compliance | AI governance policies, risk treatment |
| HRS | Human Resources Security | AI ethics training, awareness |
| IAM | Identity & Access Management | Agent identity, API auth, delegation |
| IPY | Interoperability & Portability | Model format portability, vendor lock-in |
| IVS | Infrastructure & Virtualization Security | GPU/TPU infrastructure, container security |
| LOG | Logging & Monitoring | AI event logging, inference audit trails |
| MDS | Model Security | Model tampering, adversarial robustness, provenance |
| SEF | Security Incident Management & Forensics | AI incident response, breach notification |
| STA | Supply Chain Management, Transparency & Accountability | Model provenance, AI-BOM, third-party models |
| TVM | Threat & Vulnerability Management | AI-specific threat detection, vulnerability scanning |
| UEM | Universal Endpoint Management | Edge AI device management |

#### Control Mapping by Event Type

| Control | Description | Triggered By |
|---------|-------------|-------------|
| AIS-01 | Application security policy | Tool execution, API calls |
| AIS-02 | Application security standards | Agent activity, agent-to-agent |
| AIS-04 | Secure application interfaces | Model inference, tool execution |
| MDS-01 | Model tampering protection | Model inference, integrity checks |
| MDS-02 | Adversarial robustness testing | Security findings, prompt injection |
| MDS-03 | Model access controls | Model inference, identity events |
| MDS-04 | Model provenance tracking | Supply chain events |
| MDS-05 | Model behavior monitoring | Agent activity, drift detection |
| DSP-01 | Data security policy | Data retrieval, PII events |
| DSP-04 | Data classification | Data retrieval, training data |
| CEK-03 | Encryption of data at rest/transit | Data retrieval, inference payloads |
| IAM-01 | IAM policy and procedures | Identity events |
| IAM-02 | User access provisioning | Identity events, agent creation |
| IAM-04 | Least privilege enforcement | Identity events, tool permissions |
| LOG-01 | Audit logging policy | Governance events |
| LOG-04 | Audit log monitoring | Security findings, anomaly detection |
| LOG-05 | Log record generation | Tool execution, data retrieval |
| LOG-07 | Transaction/activity logging | Model inference, agent activity |
| GRC-01 | Governance program | Governance events |
| GRC-02 | Risk management framework | Agent activity, governance |
| SEF-03 | Incident response plan | Security findings, tool execution |
| TVM-01 | Threat management policy | Security findings |
| STA-01 | Supply chain policy | Supply chain events |
| STA-03 | Supply chain inventory | Supply chain events, AI-BOM |
| CCC-01 | Change control policy | Supply chain, model deployments |
| A&A-01 | Audit planning and execution | Governance events, compliance |

#### Five Critical Pillars

Each AICM control is analyzed through:

1. **Control Type Classification** — AI-specific, hybrid AI-cloud, or traditional cloud controls
2. **Control Applicability and Ownership** — Shared responsibility across CSP, Model Provider, Orchestrated Service Provider, Application Provider
3. **Architectural Relevance** — Physical, network, compute, storage, application, and data layers
4. **LLM Lifecycle Relevance** — Preparation, Development, Deployment, Operation, Retirement
5. **Threat Category** — Mapping to specific AI threat vectors

---

## Compliance Event Format

Each OCSF event includes a `compliance` field:

```json
{
  "compliance": {
    "nist_ai_rmf": {
      "controls": ["MAP-1.1", "MEASURE-2.5"],
      "function": "MAP"
    },
    "mitre_atlas": {
      "techniques": ["AML.T0040"],
      "tactic": "ML Attack Staging"
    },
    "iso_42001": {
      "controls": ["6.1.4", "8.4"],
      "clause": "Operation"
    },
    "eu_ai_act": {
      "articles": ["Article 13", "Article 15"],
      "risk_level": "high"
    },
    "soc2": {
      "controls": ["CC6.1"],
      "criteria": "Common Criteria"
    },
    "gdpr": {
      "articles": ["Article 5", "Article 22"],
      "lawful_basis": "legitimate_interest"
    },
    "ccpa": {
      "sections": ["1798.100"],
      "category": "personal_information"
    },
    "csa_aicm": {
      "controls": ["AIS-04", "MDS-01", "LOG-07"],
      "domain": "Model Security"
    }
  }
}
```

## Audit Record Generation

AITF can generate audit records from compliance events:

```json
{
  "audit_id": "aud-abc123",
  "timestamp": "2026-02-15T10:30:00Z",
  "event_type": "model_inference",
  "ocsf_class_uid": 7001,
  "frameworks_mapped": 8,
  "controls_mapped": 15,
  "violations_detected": 0,
  "risk_score": 15.0,
  "actor": {"user": "analyst@example.com"},
  "model": {"name": "gpt-4o", "provider": "openai"},
  "compliance_details": {
    "nist_ai_rmf": ["MAP-1.1", "MEASURE-2.5"],
    "eu_ai_act": ["Article 13"]
  }
}
```
