"""AITF Compliance Mapper.

Maps AI events to seven regulatory framework controls.
Based on the compliance mapper from the AITelemetry project.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from aitf.ocsf.schema import AIBaseEvent, ComplianceMetadata

# Full compliance mappings (reused from AITelemetry project)
FRAMEWORK_MAPPINGS: dict[str, dict[str, dict[str, Any]]] = {
    "nist_ai_rmf": {
        "model_inference": {"controls": ["MAP-1.1", "MEASURE-2.5"], "function": "MAP"},
        "agent_activity": {"controls": ["GOVERN-1.2", "MANAGE-3.1"], "function": "GOVERN"},
        "tool_execution": {"controls": ["MAP-3.5", "MANAGE-4.2"], "function": "MANAGE"},
        "data_retrieval": {"controls": ["MAP-1.5", "MEASURE-2.7"], "function": "MAP"},
        "security_finding": {"controls": ["MANAGE-2.4", "MANAGE-4.1"], "function": "MANAGE"},
        "supply_chain": {"controls": ["MAP-5.2", "GOVERN-6.1"], "function": "GOVERN"},
        "governance": {"controls": ["GOVERN-1.1", "MANAGE-1.3"], "function": "GOVERN"},
        "identity": {"controls": ["GOVERN-1.5", "MANAGE-2.1"], "function": "GOVERN"},
    },
    "mitre_atlas": {
        "model_inference": {"techniques": ["AML.T0040"], "tactic": "ML Attack Staging"},
        "agent_activity": {"techniques": ["AML.T0048"], "tactic": "ML Attack Staging"},
        "tool_execution": {"techniques": ["AML.T0043"], "tactic": "ML Attack Staging"},
        "data_retrieval": {"techniques": ["AML.T0025"], "tactic": "Exfiltration"},
        "security_finding": {"techniques": ["AML.T0051"], "tactic": "Initial Access"},
        "supply_chain": {"techniques": ["AML.T0010"], "tactic": "Resource Development"},
        "identity": {"techniques": ["AML.T0052"], "tactic": "Initial Access"},
    },
    "iso_42001": {
        "model_inference": {"controls": ["6.1.4", "8.4"], "clause": "Operation"},
        "agent_activity": {"controls": ["8.2", "A.6.2.5"], "clause": "Operation"},
        "tool_execution": {"controls": ["A.6.2.7"], "clause": "Annex A"},
        "data_retrieval": {"controls": ["A.7.4"], "clause": "Annex A"},
        "security_finding": {"controls": ["6.1.2", "A.6.2.4"], "clause": "Planning"},
        "supply_chain": {"controls": ["A.6.2.3"], "clause": "Annex A"},
        "governance": {"controls": ["5.1", "9.1"], "clause": "Leadership"},
        "identity": {"controls": ["A.6.2.6"], "clause": "Annex A"},
    },
    "eu_ai_act": {
        "model_inference": {"articles": ["Article 13", "Article 15"], "risk_level": "high"},
        "agent_activity": {"articles": ["Article 14", "Article 52"], "risk_level": "high"},
        "tool_execution": {"articles": ["Article 9"], "risk_level": "high"},
        "data_retrieval": {"articles": ["Article 10"], "risk_level": "high"},
        "security_finding": {"articles": ["Article 9", "Article 62"], "risk_level": "high"},
        "supply_chain": {"articles": ["Article 15", "Article 28"], "risk_level": "high"},
        "governance": {"articles": ["Article 9", "Article 61"], "risk_level": "high"},
        "identity": {"articles": ["Article 9"], "risk_level": "high"},
    },
    "soc2": {
        "model_inference": {"controls": ["CC6.1"], "criteria": "Common Criteria"},
        "agent_activity": {"controls": ["CC7.2"], "criteria": "Common Criteria"},
        "tool_execution": {"controls": ["CC6.3"], "criteria": "Common Criteria"},
        "data_retrieval": {"controls": ["CC6.1"], "criteria": "Common Criteria"},
        "security_finding": {"controls": ["CC7.2", "CC7.3"], "criteria": "Common Criteria"},
        "supply_chain": {"controls": ["CC9.2"], "criteria": "Common Criteria"},
        "governance": {"controls": ["CC1.2"], "criteria": "Common Criteria"},
        "identity": {"controls": ["CC6.1", "CC6.2"], "criteria": "Common Criteria"},
    },
    "gdpr": {
        "model_inference": {"articles": ["Article 5", "Article 22"], "lawful_basis": "legitimate_interest"},
        "agent_activity": {"articles": ["Article 22"], "lawful_basis": "legitimate_interest"},
        "tool_execution": {"articles": ["Article 25"], "lawful_basis": "legitimate_interest"},
        "data_retrieval": {"articles": ["Article 5", "Article 6"], "lawful_basis": "legitimate_interest"},
        "security_finding": {"articles": ["Article 32", "Article 33"], "lawful_basis": "legal_obligation"},
        "supply_chain": {"articles": ["Article 28"], "lawful_basis": "contractual"},
        "governance": {"articles": ["Article 5"], "lawful_basis": "legal_obligation"},
        "identity": {"articles": ["Article 32"], "lawful_basis": "legal_obligation"},
    },
    "ccpa": {
        "model_inference": {"sections": ["1798.100"], "category": "personal_information"},
        "data_retrieval": {"sections": ["1798.100"], "category": "personal_information"},
        "security_finding": {"sections": ["1798.150"], "category": "breach"},
        "governance": {"sections": ["1798.185"], "category": "rulemaking"},
        "identity": {"sections": ["1798.140"], "category": "personal_information"},
    },
}


class ComplianceMapper:
    """Maps AI events to compliance framework controls.

    Usage:
        mapper = ComplianceMapper(frameworks=["nist_ai_rmf", "eu_ai_act"])
        compliance = mapper.map_event("model_inference")
        # ComplianceMetadata with nist_ai_rmf and eu_ai_act populated
    """

    def __init__(
        self,
        frameworks: list[str] | None = None,
    ):
        all_frameworks = list(FRAMEWORK_MAPPINGS.keys())
        self._frameworks = frameworks or all_frameworks

    def map_event(self, event_type: str) -> ComplianceMetadata:
        """Map an event type to compliance frameworks.

        Args:
            event_type: One of 'model_inference', 'agent_activity',
                'tool_execution', 'data_retrieval', 'security_finding',
                'supply_chain', 'governance', 'identity'.

        Returns:
            ComplianceMetadata with active framework mappings.
        """
        result: dict[str, Any] = {}

        for framework in self._frameworks:
            framework_map = FRAMEWORK_MAPPINGS.get(framework, {})
            event_map = framework_map.get(event_type)
            if event_map:
                result[framework] = event_map

        return ComplianceMetadata(**result)

    def enrich_event(self, event: AIBaseEvent, event_type: str) -> AIBaseEvent:
        """Add compliance metadata to an OCSF event."""
        event.compliance = self.map_event(event_type)
        return event

    def get_coverage_matrix(self) -> dict[str, dict[str, list[str]]]:
        """Generate a coverage matrix showing which controls apply per event type."""
        event_types = [
            "model_inference", "agent_activity", "tool_execution",
            "data_retrieval", "security_finding", "supply_chain",
            "governance", "identity",
        ]
        matrix: dict[str, dict[str, list[str]]] = {}

        for event_type in event_types:
            matrix[event_type] = {}
            for framework in self._frameworks:
                framework_map = FRAMEWORK_MAPPINGS.get(framework, {})
                event_map = framework_map.get(event_type, {})
                # Extract the primary list (controls, techniques, articles, sections)
                for key in ("controls", "techniques", "articles", "sections"):
                    if key in event_map:
                        matrix[event_type][framework] = event_map[key]
                        break

        return matrix

    def generate_audit_record(
        self,
        event_type: str,
        actor: str | None = None,
        model: str | None = None,
        risk_score: float = 0.0,
    ) -> dict[str, Any]:
        """Generate an audit record from compliance mappings."""
        compliance = self.map_event(event_type)
        all_controls: list[str] = []

        compliance_details: dict[str, list[str]] = {}
        for framework in self._frameworks:
            framework_data = getattr(compliance, framework, None)
            if framework_data:
                for key in ("controls", "techniques", "articles", "sections"):
                    if key in framework_data:
                        controls = framework_data[key]
                        compliance_details[framework] = controls
                        all_controls.extend(controls)

        return {
            "audit_id": f"aud-{uuid.uuid4().hex[:12]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "frameworks_mapped": len(compliance_details),
            "controls_mapped": len(all_controls),
            "violations_detected": 0,
            "risk_score": risk_score,
            "actor": actor,
            "model": model,
            "compliance_details": compliance_details,
        }
