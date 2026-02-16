"""AITF Compliance Processor.

OTel SpanProcessor that maps AI telemetry events to compliance framework controls.
Supports NIST AI RMF, MITRE ATLAS, ISO 42001, EU AI Act, SOC 2, GDPR, CCPA,
and CSA AI Controls Matrix (AICM).

Based on compliance mapping from AITelemetry project.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from aitf.semantic_conventions.attributes import ComplianceAttributes

# Compliance mappings by AI event type
COMPLIANCE_MAPPINGS: dict[str, dict[str, Any]] = {
    "model_inference": {
        "nist_ai_rmf": {"controls": ["MAP-1.1", "MEASURE-2.5"], "function": "MAP"},
        "mitre_atlas": {"techniques": ["AML.T0040"], "tactic": "ML Attack Staging"},
        "iso_42001": {"controls": ["6.1.4", "8.4"], "clause": "Operation"},
        "eu_ai_act": {"articles": ["Article 13", "Article 15"], "risk_level": "high"},
        "soc2": {"controls": ["CC6.1"], "criteria": "Common Criteria"},
        "gdpr": {"articles": ["Article 5", "Article 22"], "lawful_basis": "legitimate_interest"},
        "ccpa": {"sections": ["1798.100"], "category": "personal_information"},
        "csa_aicm": {"controls": ["AIS-04", "MDS-01", "LOG-07"], "domain": "Model Security"},
    },
    "agent_activity": {
        "nist_ai_rmf": {"controls": ["GOVERN-1.2", "MANAGE-3.1"], "function": "GOVERN"},
        "mitre_atlas": {"techniques": ["AML.T0048"], "tactic": "ML Attack Staging"},
        "iso_42001": {"controls": ["8.2", "A.6.2.5"], "clause": "Operation"},
        "eu_ai_act": {"articles": ["Article 14", "Article 52"], "risk_level": "high"},
        "soc2": {"controls": ["CC7.2"], "criteria": "Common Criteria"},
        "gdpr": {"articles": ["Article 22"], "lawful_basis": "legitimate_interest"},
        "csa_aicm": {"controls": ["AIS-02", "MDS-05", "GRC-02"], "domain": "Governance, Risk & Compliance"},
    },
    "tool_execution": {
        "nist_ai_rmf": {"controls": ["MAP-3.5", "MANAGE-4.2"], "function": "MANAGE"},
        "mitre_atlas": {"techniques": ["AML.T0043"], "tactic": "ML Attack Staging"},
        "iso_42001": {"controls": ["A.6.2.7"], "clause": "Annex A"},
        "eu_ai_act": {"articles": ["Article 9"], "risk_level": "high"},
        "soc2": {"controls": ["CC6.3"], "criteria": "Common Criteria"},
        "gdpr": {"articles": ["Article 25"], "lawful_basis": "legitimate_interest"},
        "csa_aicm": {"controls": ["AIS-01", "AIS-04", "LOG-05"], "domain": "Application & Interface Security"},
    },
    "data_retrieval": {
        "nist_ai_rmf": {"controls": ["MAP-1.5", "MEASURE-2.7"], "function": "MAP"},
        "mitre_atlas": {"techniques": ["AML.T0025"], "tactic": "Exfiltration"},
        "iso_42001": {"controls": ["A.7.4"], "clause": "Annex A"},
        "eu_ai_act": {"articles": ["Article 10"], "risk_level": "high"},
        "soc2": {"controls": ["CC6.1"], "criteria": "Common Criteria"},
        "gdpr": {"articles": ["Article 5", "Article 6"], "lawful_basis": "legitimate_interest"},
        "ccpa": {"sections": ["1798.100"], "category": "personal_information"},
        "csa_aicm": {"controls": ["DSP-01", "DSP-04", "CEK-03"], "domain": "Data Security & Privacy"},
    },
    "security_finding": {
        "nist_ai_rmf": {"controls": ["MANAGE-2.4", "MANAGE-4.1"], "function": "MANAGE"},
        "mitre_atlas": {"techniques": ["AML.T0051"], "tactic": "Initial Access"},
        "iso_42001": {"controls": ["6.1.2", "A.6.2.4"], "clause": "Planning"},
        "eu_ai_act": {"articles": ["Article 9", "Article 62"], "risk_level": "high"},
        "soc2": {"controls": ["CC7.2", "CC7.3"], "criteria": "Common Criteria"},
        "gdpr": {"articles": ["Article 32", "Article 33"], "lawful_basis": "legal_obligation"},
        "ccpa": {"sections": ["1798.150"], "category": "breach"},
        "csa_aicm": {"controls": ["SEF-03", "TVM-01", "LOG-04"], "domain": "Security Incident Management"},
    },
    "supply_chain": {
        "nist_ai_rmf": {"controls": ["MAP-5.2", "GOVERN-6.1"], "function": "GOVERN"},
        "mitre_atlas": {"techniques": ["AML.T0010"], "tactic": "Resource Development"},
        "iso_42001": {"controls": ["A.6.2.3"], "clause": "Annex A"},
        "eu_ai_act": {"articles": ["Article 15", "Article 28"], "risk_level": "high"},
        "soc2": {"controls": ["CC9.2"], "criteria": "Common Criteria"},
        "gdpr": {"articles": ["Article 28"], "lawful_basis": "contractual"},
        "csa_aicm": {"controls": ["STA-01", "STA-03", "CCC-01"], "domain": "Supply Chain Management"},
    },
    "governance": {
        "nist_ai_rmf": {"controls": ["GOVERN-1.1", "MANAGE-1.3"], "function": "GOVERN"},
        "iso_42001": {"controls": ["5.1", "9.1"], "clause": "Leadership"},
        "eu_ai_act": {"articles": ["Article 9", "Article 61"], "risk_level": "high"},
        "soc2": {"controls": ["CC1.2"], "criteria": "Common Criteria"},
        "gdpr": {"articles": ["Article 5"], "lawful_basis": "legal_obligation"},
        "ccpa": {"sections": ["1798.185"], "category": "rulemaking"},
        "csa_aicm": {"controls": ["GRC-01", "A&A-01", "LOG-01"], "domain": "Governance, Risk & Compliance"},
    },
    "identity": {
        "nist_ai_rmf": {"controls": ["GOVERN-1.5", "MANAGE-2.1"], "function": "GOVERN"},
        "mitre_atlas": {"techniques": ["AML.T0052"], "tactic": "Initial Access"},
        "iso_42001": {"controls": ["A.6.2.6"], "clause": "Annex A"},
        "eu_ai_act": {"articles": ["Article 9"], "risk_level": "high"},
        "soc2": {"controls": ["CC6.1", "CC6.2"], "criteria": "Common Criteria"},
        "gdpr": {"articles": ["Article 32"], "lawful_basis": "legal_obligation"},
        "ccpa": {"sections": ["1798.140"], "category": "personal_information"},
        "csa_aicm": {"controls": ["IAM-01", "IAM-02", "IAM-04"], "domain": "Identity & Access Management"},
    },
}


class ComplianceProcessor(SpanProcessor):
    """OTel SpanProcessor that adds compliance framework mappings to AI spans.

    Usage:
        provider.add_span_processor(ComplianceProcessor(
            frameworks=["nist_ai_rmf", "mitre_atlas", "eu_ai_act"],
        ))
    """

    def __init__(
        self,
        frameworks: list[str] | None = None,
    ):
        all_frameworks = [
            "nist_ai_rmf", "mitre_atlas", "iso_42001",
            "eu_ai_act", "soc2", "gdpr", "ccpa", "csa_aicm",
        ]
        self._frameworks = frameworks or all_frameworks

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Map span to compliance controls based on span type."""
        attrs = span.attributes or {}

        # Determine event type from span name and attributes
        event_type = self._classify_event(span)
        if not event_type:
            return

        # Get compliance mapping
        mapping = self.get_compliance_mapping(event_type)
        if mapping:
            # In practice, compliance data would be attached during on_start
            # or via a wrapping mechanism. ReadableSpan is immutable.
            pass

    def get_compliance_mapping(self, event_type: str) -> dict[str, Any]:
        """Get compliance mapping for a given event type.

        Returns dict of framework -> controls.
        """
        full_mapping = COMPLIANCE_MAPPINGS.get(event_type, {})
        if not full_mapping:
            return {}

        result: dict[str, Any] = {}
        active_frameworks: list[str] = []

        for framework in self._frameworks:
            if framework in full_mapping:
                result[framework] = full_mapping[framework]
                active_frameworks.append(framework)

        if active_frameworks:
            result["_frameworks"] = active_frameworks

        return result

    def get_compliance_attributes(self, event_type: str) -> dict[str, Any]:
        """Get compliance attributes suitable for span attributes."""
        mapping = self.get_compliance_mapping(event_type)
        if not mapping:
            return {}

        attributes: dict[str, Any] = {}
        frameworks = mapping.pop("_frameworks", [])
        if frameworks:
            attributes[ComplianceAttributes.FRAMEWORKS] = frameworks

        for framework, controls in mapping.items():
            if framework == "nist_ai_rmf" and "controls" in controls:
                attributes[ComplianceAttributes.NIST_AI_RMF_CONTROLS] = controls["controls"]
            elif framework == "mitre_atlas" and "techniques" in controls:
                attributes[ComplianceAttributes.MITRE_ATLAS_TECHNIQUES] = controls["techniques"]
            elif framework == "iso_42001" and "controls" in controls:
                attributes[ComplianceAttributes.ISO_42001_CONTROLS] = controls["controls"]
            elif framework == "eu_ai_act" and "articles" in controls:
                attributes[ComplianceAttributes.EU_AI_ACT_ARTICLES] = controls["articles"]
            elif framework == "soc2" and "controls" in controls:
                attributes[ComplianceAttributes.SOC2_CONTROLS] = controls["controls"]
            elif framework == "gdpr" and "articles" in controls:
                attributes[ComplianceAttributes.GDPR_ARTICLES] = controls["articles"]
            elif framework == "ccpa" and "sections" in controls:
                attributes[ComplianceAttributes.CCPA_SECTIONS] = controls["sections"]
            elif framework == "csa_aicm" and "controls" in controls:
                attributes[ComplianceAttributes.CSA_AICM_CONTROLS] = controls["controls"]

        return attributes

    def get_coverage_matrix(self) -> dict[str, dict[str, list[str]]]:
        """Generate a coverage matrix showing which controls are mapped per event type."""
        matrix: dict[str, dict[str, list[str]]] = {}
        for event_type, mapping in COMPLIANCE_MAPPINGS.items():
            matrix[event_type] = {}
            for framework in self._frameworks:
                if framework in mapping:
                    controls = mapping[framework]
                    key = "controls" if "controls" in controls else "techniques" if "techniques" in controls else "articles" if "articles" in controls else "sections"
                    matrix[event_type][framework] = controls.get(key, [])
        return matrix

    def _classify_event(self, span: ReadableSpan) -> str | None:
        """Classify a span into an AI event type."""
        name = span.name or ""
        attrs = span.attributes or {}

        if name.startswith("chat ") or name.startswith("embeddings "):
            return "model_inference"
        if "gen_ai.system" in attrs:
            return "model_inference"
        if name.startswith("agent."):
            return "agent_activity"
        if name.startswith("mcp.tool.") or name.startswith("skill.invoke"):
            return "tool_execution"
        if name.startswith("rag.") or name.startswith("mcp.resource."):
            return "data_retrieval"
        if "aitf.security." in " ".join(str(k) for k in attrs.keys()):
            return "security_finding"

        return None

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
