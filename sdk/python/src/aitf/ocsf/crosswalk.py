"""AITF <-> OCSF agentic alignment helpers.

Implements OCSF's "reuse existing objects and profiles" direction:

  * OCSF PR #1641  -- ``objects/ai_agent.json`` + the ``ai_operation`` profile
  * OCSF issue #1640 -- the proposed ``ai`` category (uid 9), the
    ``delegation`` object, ``delegation_lineage``/``delegation_node`` graph,
    and the ``agent_activity`` / ``delegation_activity`` control-plane classes.

AITF emits AI telemetry under existing OCSF classes (API Activity, Datastore
Activity, Findings, IAM, Discovery, ...) enriched with the ``ai_operation``
profile (``ai_agent`` + ``ai_model``) and the ``delegation`` object; only
agent/delegation lifecycle use the proposed ``ai`` category. This module
provides the object builders and publishes ``OCSF_CLASS_CROSSWALK`` (the
authoritative AITF event -> OCSF class table) plus the control-plane activity
crosswalks.
"""

from __future__ import annotations

from typing import Any

from aitf.ocsf.schema import (
    AGENT_TYPE_LABELS,
    OCSF_AI_CATEGORY_UID,
    AgentTypeID,
    OCSFAIAgent,
    OCSFDelegation,
    OCSFDelegationLineage,
    OCSFDelegationNode,
    normalize_agent_type_id,
)
from aitf.semantic_conventions.attributes import (
    AgentAttributes,
    GenAIAttributes,
    IdentityAttributes,
)

__all__ = [
    "build_ai_agent",
    "build_delegation",
    "build_delegation_lineage",
    "OCSF_AGENT_ACTIVITY_CROSSWALK",
    "OCSF_DELEGATION_ACTIVITY_CROSSWALK",
    "OCSF_CLASS_CROSSWALK",
]


def _opt_str(val: Any) -> str | None:
    return str(val) if val is not None else None


def _as_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val]
    return [str(val)]


def build_ai_agent(attrs: dict[str, Any]) -> OCSFAIAgent | None:
    """Build an OCSF ``ai_agent`` object (PR #1641) from span attributes.

    Returns ``None`` when no agent identity is present, so non-agentic events
    are left untouched.
    """
    uid = (
        attrs.get(GenAIAttributes.AGENT_ID)
        or attrs.get(IdentityAttributes.AGENT_ID)
        or attrs.get(AgentAttributes.WORKFLOW_ID)
    )
    name = attrs.get(GenAIAttributes.AGENT_NAME) or attrs.get(IdentityAttributes.AGENT_NAME)
    if uid is None and name is None:
        return None

    framework = _opt_str(attrs.get(AgentAttributes.FRAMEWORK))
    type_id = normalize_agent_type_id(framework)

    return OCSFAIAgent(
        uid=str(uid if uid is not None else name),
        instance_uid=_opt_str(attrs.get(GenAIAttributes.CONVERSATION_ID)),
        name=_opt_str(name),
        type=AGENT_TYPE_LABELS.get(type_id, AGENT_TYPE_LABELS[AgentTypeID.OTHER])
        if type_id != AgentTypeID.UNKNOWN
        else None,
        type_id=type_id,
        ai_model=_opt_str(
            attrs.get(GenAIAttributes.REQUEST_MODEL)
            or attrs.get(GenAIAttributes.RESPONSE_MODEL)
        ),
        version=_opt_str(attrs.get(GenAIAttributes.AGENT_VERSION)),
        charter=_opt_str(attrs.get(GenAIAttributes.AGENT_DESCRIPTION)),
    )


def build_delegation(attrs: dict[str, Any]) -> OCSFDelegation | None:
    """Build an OCSF ``delegation`` object (issue #1640) from span attributes.

    Returns ``None`` when no delegation context is present.
    """
    delegatee_id = attrs.get(IdentityAttributes.DELEGATION_DELEGATEE_ID) or attrs.get(
        AgentAttributes.DELEGATION_TARGET_AGENT_ID
    )
    delegator_id = attrs.get(IdentityAttributes.DELEGATION_DELEGATOR_ID)
    delegatee = attrs.get(IdentityAttributes.DELEGATION_DELEGATEE) or attrs.get(
        AgentAttributes.DELEGATION_TARGET_AGENT
    )
    delegator = attrs.get(IdentityAttributes.DELEGATION_DELEGATOR)
    deleg_type = attrs.get(IdentityAttributes.DELEGATION_TYPE)
    chain = attrs.get(IdentityAttributes.DELEGATION_CHAIN)

    if not any([delegatee_id, delegator_id, delegatee, delegator, deleg_type, chain]):
        return None

    # OCSF delegation.uid is the stable identifier of the granted authority.
    uid = delegatee_id or delegatee or (chain[-1] if isinstance(chain, (list, tuple)) and chain else None)

    return OCSFDelegation(
        uid=str(uid) if uid is not None else "unknown",
        parent_uid=_opt_str(delegator_id),
        issuer_uid=_opt_str(attrs.get(IdentityAttributes.PROVIDER)),
        delegator=_opt_str(delegator),
        delegatee=_opt_str(delegatee),
        type=_opt_str(deleg_type),
        scope=_as_list(attrs.get(IdentityAttributes.DELEGATION_SCOPE_DELEGATED)),
        proof_type=_opt_str(attrs.get(IdentityAttributes.DELEGATION_PROOF_TYPE)),
        ttl_seconds=(
            int(attrs[IdentityAttributes.DELEGATION_TTL_SECONDS])
            if attrs.get(IdentityAttributes.DELEGATION_TTL_SECONDS) is not None
            else None
        ),
    )


def build_delegation_lineage(attrs: dict[str, Any]) -> OCSFDelegationLineage | None:
    """Build an OCSF ``delegation_lineage`` graph from a delegation chain.

    The AITF ``identity.delegation.chain`` (ordered from origin to current)
    is materialized into the directed ``delegation_node`` graph proposed in
    OCSF issue #1640.
    """
    chain = attrs.get(IdentityAttributes.DELEGATION_CHAIN)
    if not isinstance(chain, (list, tuple)) or not chain:
        return None

    nodes: list[OCSFDelegationNode] = []
    parent: str | None = None
    for depth, node_uid in enumerate(chain):
        nodes.append(
            OCSFDelegationNode(
                uid=str(node_uid),
                parent_uid=parent,
                agent_uid=str(node_uid),
                depth=depth,
            )
        )
        parent = str(node_uid)
    return OCSFDelegationLineage(nodes=nodes)


# --- Control-plane crosswalk tables (OCSF issue #1640) ---------------------
#
# OCSF issue #1640 proposes a native ``ai`` category (uid 9) with dedicated
# control-plane classes. These tables map AITF activities onto the proposed
# OCSF activities. UIDs for the proposed classes are not yet finalized
# upstream, so only the activity-name mapping (which is stable) is published.

# AITF agent activity_id -> OCSF agent_activity activity.
OCSF_AGENT_ACTIVITY_CROSSWALK: dict[int, str] = {
    1: "Spawn",       # AITF Session Start  -> agent spawned
    2: "Terminate",   # AITF Session End    -> agent terminated
    3: "Update",      # AITF Step Execute   -> agent state update
    4: "Register",    # AITF Delegation     -> registers delegated authority
    5: "Resume",      # AITF Memory Access  -> resume/continue
    6: "Resume",      # AITF Error Recovery -> resume after recovery
    7: "Suspend",     # AITF Human Approval -> suspended pending human input
    99: "Unknown",
}

# AITF identity delegation activity -> OCSF delegation_activity.
OCSF_DELEGATION_ACTIVITY_CROSSWALK: dict[str, str] = {
    "create": "Create",
    "grant": "Create",
    "revoke": "Revoke",
    "expire": "Expire",
    "complete": "Complete",
}

# Authoritative AITF -> OCSF class mapping (the classes AITF actually emits).
# Per OCSF's "reuse existing objects and profiles" model: data-plane AI
# activity reuses existing OCSF classes carrying the ``ai_operation`` profile;
# only agent / delegation lifecycle use the proposed ``ai`` category (uid 9).
OCSF_CLASS_CROSSWALK: dict[str, dict[str, Any]] = {
    "model_inference": {"ocsf_category_uid": 6, "ocsf_class_uid": 6003, "ocsf_class": "api_activity"},
    "tool_execution": {"ocsf_category_uid": 6, "ocsf_class_uid": 6003, "ocsf_class": "api_activity"},
    "data_retrieval": {"ocsf_category_uid": 6, "ocsf_class_uid": 6005, "ocsf_class": "datastore_activity"},
    "model_ops": {"ocsf_category_uid": 6, "ocsf_class_uid": 6002, "ocsf_class": "application_lifecycle"},
    "security_finding": {"ocsf_category_uid": 2, "ocsf_class_uid": 2004, "ocsf_class": "detection_finding"},
    "supply_chain": {"ocsf_category_uid": 2, "ocsf_class_uid": 2002, "ocsf_class": "vulnerability_finding"},
    "governance": {"ocsf_category_uid": 2, "ocsf_class_uid": 2003, "ocsf_class": "compliance_finding"},
    "identity": {"ocsf_category_uid": 3, "ocsf_class_uid": 3002, "ocsf_class": "authentication"},
    "asset_inventory": {"ocsf_category_uid": 5, "ocsf_class_uid": 5001, "ocsf_class": "inventory_info"},
    # New control-plane classes in the proposed ``ai`` category (provisional UIDs).
    "agent_activity": {"ocsf_category_uid": OCSF_AI_CATEGORY_UID, "ocsf_class_uid": 9001, "ocsf_class": "agent_activity"},
    "delegation_activity": {"ocsf_category_uid": OCSF_AI_CATEGORY_UID, "ocsf_class_uid": 9002, "ocsf_class": "delegation_activity"},
}
