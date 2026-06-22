"""AITF OCSF Base Schema.

OCSF v1.1.0 base objects and AI-specific extension models.
Based on the OCSF schema from the AITelemetry project, enhanced
for AITF Category 7 AI events.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# --- OCSF Enumerations ---

class OCSFSeverity(IntEnum):
    UNKNOWN = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5
    FATAL = 6


class OCSFStatus(IntEnum):
    UNKNOWN = 0
    SUCCESS = 1
    FAILURE = 2
    OTHER = 99


class OCSFActivity(IntEnum):
    UNKNOWN = 0
    CREATE = 1
    READ = 2
    UPDATE = 3
    DELETE = 4
    OTHER = 99


class AIClassUID(IntEnum):
    """AITF OCSF Category 7 class UIDs."""
    MODEL_INFERENCE = 7001
    AGENT_ACTIVITY = 7002
    TOOL_EXECUTION = 7003
    DATA_RETRIEVAL = 7004
    SECURITY_FINDING = 7005
    SUPPLY_CHAIN = 7006
    GOVERNANCE = 7007
    IDENTITY = 7008
    MODEL_OPS = 7009
    ASSET_INVENTORY = 7010


class AgentTypeID(IntEnum):
    """OCSF ``ai_agent.type_id`` — normalized agent framework.

    Mirrors the enum introduced by OCSF PR #1641 (``objects/ai_agent.json``)
    so AITF telemetry maps cleanly onto the upstream OCSF ``ai_agent`` object.
    """
    UNKNOWN = 0
    NATIVE = 1
    LANGCHAIN = 2
    AUTOGEN = 3
    CREWAI = 4
    OTHER = 99


# Caption labels for AgentTypeID, per OCSF PR #1641.
AGENT_TYPE_LABELS: dict[int, str] = {
    0: "Unknown",
    1: "Native",
    2: "LangChain",
    3: "AutoGen",
    4: "CrewAI",
    99: "Other",
}

# AITF framework value -> OCSF ai_agent.type_id. Frameworks without a
# dedicated OCSF enum member (langgraph, semantic_kernel, custom, ...)
# normalize to OTHER (99), matching OCSF's open-enum guidance.
_FRAMEWORK_TO_TYPE_ID: dict[str, int] = {
    "native": AgentTypeID.NATIVE,
    "langchain": AgentTypeID.LANGCHAIN,
    "langgraph": AgentTypeID.LANGCHAIN,
    "autogen": AgentTypeID.AUTOGEN,
    "crewai": AgentTypeID.CREWAI,
}


def normalize_agent_type_id(framework: str | None) -> int:
    """Map an AITF framework string to an OCSF ``ai_agent.type_id`` value."""
    if not framework:
        return AgentTypeID.UNKNOWN
    return int(_FRAMEWORK_TO_TYPE_ID.get(framework.strip().lower(), AgentTypeID.OTHER))


# OCSF AI category and control-plane classes proposed in OCSF issue #1640.
# AITF keeps its established Category 7 classes but records the upstream
# target so consumers can crosswalk to the future native ``ai`` category.
OCSF_AI_CATEGORY_UID = 9  # proposed "AI Activity" category (OCSF issue #1640)


# --- OCSF Base Objects ---

class OCSFMetadata(BaseModel):
    """OCSF event metadata."""
    version: str = "1.1.0"
    product: dict[str, str] = Field(
        default_factory=lambda: {
            "name": "AITF",
            "vendor_name": "AITF",
            "version": "1.0.0",
        }
    )
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_uid: str | None = None
    original_time: str | None = None
    logged_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class OCSFActor(BaseModel):
    """OCSF actor information."""
    user: dict[str, Any] | None = None
    session: dict[str, Any] | None = None
    app_name: str | None = None


class OCSFDevice(BaseModel):
    """OCSF device/host information."""
    hostname: str | None = None
    ip: str | None = None
    type: str | None = None
    os: dict[str, str] | None = None
    cloud: dict[str, str] | None = None
    container: dict[str, str] | None = None


class OCSFEnrichment(BaseModel):
    """OCSF enrichment data."""
    name: str
    value: str
    type: str | None = None
    provider: str | None = None


class OCSFObservable(BaseModel):
    """OCSF observable value."""
    name: str
    type: str
    value: str


# --- AI-Specific Extension Models ---

class AIModelInfo(BaseModel):
    """AI model information."""
    model_id: str
    name: str | None = None
    version: str | None = None
    provider: str | None = None
    type: str | None = None  # llm, embedding, image, audio, multimodal
    parameters: dict[str, Any] | None = None


class AITokenUsage(BaseModel):
    """AI token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float | None = None

    @model_validator(mode="after")
    def compute_total(self) -> "AITokenUsage":
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


class AILatencyMetrics(BaseModel):
    """AI operation latency metrics."""
    total_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    tokens_per_second: float | None = None
    queue_time_ms: float | None = None
    inference_time_ms: float | None = None


class AICostInfo(BaseModel):
    """AI operation cost information."""
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    currency: str = "USD"


class AITeamInfo(BaseModel):
    """Multi-agent team information."""
    team_name: str
    team_id: str | None = None
    topology: str | None = None
    members: list[str] = Field(default_factory=list)
    coordinator: str | None = None


class AISecurityFinding(BaseModel):
    """Security finding details."""
    finding_type: str
    owasp_category: str | None = None
    risk_level: str
    risk_score: float
    confidence: float
    detection_method: str = "pattern"
    blocked: bool = False
    details: str | None = None
    pii_types: list[str] = Field(default_factory=list)
    matched_patterns: list[str] = Field(default_factory=list)
    remediation: str | None = None


class OCSFAIAgent(BaseModel):
    """OCSF ``ai_agent`` object (OCSF PR #1641).

    An autonomous AI agent operating under delegated authority. Distinct from
    the OCSF ``agent`` object (which models security sensors such as EDR/DLP)
    and from human principals. Attached to events via the ``ai_operation``
    profile so any activity can be attributed to the agent that performed it.
    """
    uid: str  # required: stable logical identifier
    instance_uid: str | None = None  # restart-sensitive running instance id
    name: str | None = None
    type: str | None = None  # caption of type_id (Native, LangChain, ...)
    type_id: int = AgentTypeID.UNKNOWN
    ai_model: str | None = None  # model backing the agent at event time
    version: str | None = None  # agent code/configuration revision
    charter: str | None = None  # role / operating-boundary reference


class OCSFDelegation(BaseModel):
    """OCSF ``delegation`` object (OCSF issue #1640).

    A durable authorization context that persists independently of any single
    trace or session. ``uid``/``parent_uid``/``issuer_uid`` provide the OCSF
    core; the remaining fields preserve AITF's richer delegation telemetry.
    """
    uid: str  # required: stable delegation identifier
    parent_uid: str | None = None  # parent delegation (lineage)
    issuer_uid: str | None = None  # trusted issuer that minted the delegation
    delegator: str | None = None
    delegatee: str | None = None
    type: str | None = None  # on_behalf_of, token_exchange, capability_grant, ...
    scope: list[str] = Field(default_factory=list)
    proof_type: str | None = None  # dpop, mtls_binding, signed_assertion
    ttl_seconds: int | None = None


class OCSFDelegationNode(BaseModel):
    """A single node in an OCSF ``delegation_lineage`` graph (OCSF issue #1640)."""
    uid: str
    parent_uid: str | None = None
    agent_uid: str | None = None
    depth: int | None = None


class OCSFDelegationLineage(BaseModel):
    """OCSF ``delegation_lineage`` — directed graph for ancestry queries."""
    nodes: list[OCSFDelegationNode] = Field(default_factory=list)


class ComplianceMetadata(BaseModel):
    """Compliance framework mappings."""
    nist_ai_rmf: dict[str, Any] | None = None
    mitre_atlas: dict[str, Any] | None = None
    iso_42001: dict[str, Any] | None = None
    eu_ai_act: dict[str, Any] | None = None
    soc2: dict[str, Any] | None = None
    gdpr: dict[str, Any] | None = None
    ccpa: dict[str, Any] | None = None
    csa_aicm: dict[str, Any] | None = None


# --- OCSF Base Event ---

class AIBaseEvent(BaseModel):
    """Base OCSF event for all AITF Category 7 events."""
    activity_id: int = OCSFActivity.OTHER
    category_uid: int = 7  # AI System Activity
    class_uid: int
    type_uid: int = 0
    time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    severity_id: int = OCSFSeverity.INFORMATIONAL
    status_id: int = OCSFStatus.SUCCESS
    message: str = ""
    metadata: OCSFMetadata = Field(default_factory=OCSFMetadata)
    actor: OCSFActor | None = None
    device: OCSFDevice | None = None
    compliance: ComplianceMetadata | None = None
    observables: list[OCSFObservable] = Field(default_factory=list)
    enrichments: list[OCSFEnrichment] = Field(default_factory=list)

    # OCSF ``ai_operation`` profile (OCSF PR #1641) + delegation context
    # (OCSF issue #1640). Populated by the crosswalk so every AITF event can
    # be attributed to the AI agent and delegation that produced it.
    ai_agent: OCSFAIAgent | None = None
    ai_model: str | None = None
    delegation: OCSFDelegation | None = None
    delegation_lineage: OCSFDelegationLineage | None = None

    @model_validator(mode="after")
    def compute_type_uid(self) -> "AIBaseEvent":
        if self.type_uid == 0:
            self.type_uid = self.class_uid * 100 + self.activity_id
        return self
