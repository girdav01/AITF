# AITF Compliance Framework Mapping

AITF automatically maps AI telemetry events to seven regulatory and security frameworks.

## Overview

Every OCSF event produced by AITF is enriched with compliance metadata, mapping the event to relevant controls across multiple frameworks. This enables automated compliance reporting and audit trail generation.

## Framework Coverage Matrix

| AI Event Type | NIST AI RMF | MITRE ATLAS | ISO 42001 | EU AI Act | SOC 2 | GDPR | CCPA |
|---------------|-------------|-------------|-----------|-----------|-------|------|------|
| Model Inference | MAP-1.1, MEASURE-2.5 | AML.T0040 | 6.1.4, 8.4 | Art. 13, 15 | CC6.1 | Art. 5, 22 | 1798.100 |
| Agent Activity | GOVERN-1.2, MANAGE-3.1 | AML.T0048 | 8.2, A.6.2.5 | Art. 14, 52 | CC7.2 | Art. 22 | - |
| Tool Execution | MAP-3.5, MANAGE-4.2 | AML.T0043 | A.6.2.7 | Art. 9 | CC6.3 | Art. 25 | - |
| Data Retrieval | MAP-1.5, MEASURE-2.7 | AML.T0025 | A.7.4 | Art. 10 | CC6.1 | Art. 5, 6 | 1798.100 |
| Security Finding | MANAGE-2.4, MANAGE-4.1 | AML.T0051 | 6.1.2, A.6.2.4 | Art. 9, 62 | CC7.2, CC7.3 | Art. 32, 33 | 1798.150 |
| Supply Chain | MAP-5.2, GOVERN-6.1 | AML.T0010 | A.6.2.3 | Art. 15, 28 | CC9.2 | Art. 28 | - |
| Governance | GOVERN-1.1, MANAGE-1.3 | - | 5.1, 9.1 | Art. 9, 61 | CC1.2 | Art. 5 | 1798.185 |
| Identity | GOVERN-1.5, MANAGE-2.1 | AML.T0052 | A.6.2.6 | Art. 9 | CC6.1, CC6.2 | Art. 32 | 1798.140 |

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
  "frameworks_mapped": 7,
  "controls_mapped": 12,
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
