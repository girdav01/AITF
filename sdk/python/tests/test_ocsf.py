"""Tests for AITF OCSF schema and mapper."""

from aitf.ocsf.schema import (
    AIBaseEvent,
    AIClassUID,
    AIModelInfo,
    AITokenUsage,
    AILatencyMetrics,
    AICostInfo,
    ComplianceMetadata,
    OCSFMetadata,
    OCSFSeverity,
    OCSFStatus,
)
from aitf.ocsf.event_classes import (
    AIModelInferenceEvent,
    AIAgentActivityEvent,
    AIToolExecutionEvent,
    AIDataRetrievalEvent,
    AISecurityFindingEvent,
)
from aitf.ocsf.compliance_mapper import ComplianceMapper


class TestOCSFSchema:
    def test_class_uids(self):
        assert AIClassUID.MODEL_INFERENCE == 7001
        assert AIClassUID.AGENT_ACTIVITY == 7002
        assert AIClassUID.TOOL_EXECUTION == 7003
        assert AIClassUID.DATA_RETRIEVAL == 7004
        assert AIClassUID.SECURITY_FINDING == 7005
        assert AIClassUID.SUPPLY_CHAIN == 7006
        assert AIClassUID.GOVERNANCE == 7007
        assert AIClassUID.IDENTITY == 7008

    def test_token_usage_total(self):
        usage = AITokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_metadata_defaults(self):
        meta = OCSFMetadata()
        assert meta.version == "1.1.0"
        assert meta.product["name"] == "AITF"
        assert meta.uid is not None

    def test_base_event_type_uid(self):
        event = AIBaseEvent(class_uid=7001, activity_id=1)
        assert event.type_uid == 700101
        assert event.category_uid == 7


class TestEventClasses:
    def test_model_inference_event(self):
        event = AIModelInferenceEvent(
            activity_id=1,
            model=AIModelInfo(model_id="gpt-4o", provider="openai"),
            token_usage=AITokenUsage(input_tokens=100, output_tokens=50),
            finish_reason="stop",
        )
        assert event.class_uid == 7001
        assert event.type_uid == 700101
        assert event.model.model_id == "gpt-4o"
        assert event.token_usage.total_tokens == 150

    def test_agent_activity_event(self):
        event = AIAgentActivityEvent(
            activity_id=3,
            agent_name="research-agent",
            agent_id="agent-001",
            session_id="sess-001",
            step_type="planning",
            step_index=1,
        )
        assert event.class_uid == 7002
        assert event.agent_name == "research-agent"

    def test_tool_execution_event(self):
        event = AIToolExecutionEvent(
            activity_id=2,
            tool_name="read_file",
            tool_type="mcp_tool",
            mcp_server="filesystem",
            is_error=False,
        )
        assert event.class_uid == 7003
        assert event.tool_type == "mcp_tool"

    def test_data_retrieval_event(self):
        event = AIDataRetrievalEvent(
            activity_id=1,
            database_name="pinecone",
            database_type="pinecone",
            top_k=10,
            results_count=8,
        )
        assert event.class_uid == 7004
        assert event.results_count == 8

    def test_event_serialization(self):
        event = AIModelInferenceEvent(
            model=AIModelInfo(model_id="gpt-4o", provider="openai"),
            finish_reason="stop",
        )
        data = event.model_dump(exclude_none=True)
        assert data["class_uid"] == 7001
        assert data["category_uid"] == 7
        assert data["model"]["model_id"] == "gpt-4o"


class TestComplianceMapper:
    def setup_method(self):
        self.mapper = ComplianceMapper()

    def test_map_model_inference(self):
        compliance = self.mapper.map_event("model_inference")
        assert compliance.nist_ai_rmf is not None
        assert compliance.eu_ai_act is not None
        assert compliance.mitre_atlas is not None

    def test_map_security_finding(self):
        compliance = self.mapper.map_event("security_finding")
        assert compliance.nist_ai_rmf is not None
        assert "MANAGE-2.4" in compliance.nist_ai_rmf["controls"]

    def test_coverage_matrix(self):
        matrix = self.mapper.get_coverage_matrix()
        assert len(matrix) == 8  # All 8 event types
        assert "model_inference" in matrix
        assert "nist_ai_rmf" in matrix["model_inference"]

    def test_audit_record(self):
        record = self.mapper.generate_audit_record(
            event_type="model_inference",
            actor="analyst@example.com",
            model="gpt-4o",
        )
        assert record["event_type"] == "model_inference"
        assert record["frameworks_mapped"] > 0
        assert record["audit_id"].startswith("aud-")

    def test_filtered_frameworks(self):
        mapper = ComplianceMapper(frameworks=["eu_ai_act"])
        compliance = mapper.map_event("model_inference")
        assert compliance.eu_ai_act is not None
        assert compliance.nist_ai_rmf is None

    def test_enrich_event(self):
        event = AIModelInferenceEvent(
            model=AIModelInfo(model_id="gpt-4o", provider="openai"),
            finish_reason="stop",
        )
        enriched = self.mapper.enrich_event(event, "model_inference")
        assert enriched.compliance is not None
        assert enriched.compliance.nist_ai_rmf is not None
