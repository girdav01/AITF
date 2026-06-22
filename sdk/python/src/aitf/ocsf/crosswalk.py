"""AITF <-> OCSF agentic crosswalk.

Reconciles AITF's established OCSF Category 7 AI event classes with the
upstream OCSF agentic-AI direction defined in:

  * OCSF PR #1641  -- ``objects/ai_agent.json`` + the ``ai_operation`` profile
  * OCSF issue #1640 -- the proposed ``ai`` category (uid 9), the
    ``delegation`` object, ``delegation_lineage``/``delegation_node`` graph,
    and the ``agent_activity`` / ``delegation_activity`` control-plane classes.

AITF keeps emitting Category 7 events (no breaking change for existing
SIEM/XDR consumers), but every event is now enriched with the OCSF
``ai_agent`` and ``delegation`` objects via the ``ai_operation`` profile, and
this module publishes the activity/class crosswalk tables that let consumers
translate Category 7 telemetry onto OCSF's native ``ai`` category once it
lands. This is the "compromise" layer: conformant OCSF primitives carried on
top of AITF's richer class set.
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
# AITF retains Category 7 today; OCSF issue #1640 proposes a native ``ai``
# category (uid 9) with dedicated control-plane classes. These tables let a
# consumer translate AITF activities onto the proposed OCSF activities. UIDs
# for the proposed classes are not yet finalized upstream, so only the
# activity-name mapping (which is stable) is published here.

# AITF AIAgentActivityEvent (7002) activity_id -> OCSF agent_activity activity.
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

# AITF AIIdentityEvent (7008) delegation activity -> OCSF delegation_activity.
OCSF_DELEGATION_ACTIVITY_CROSSWALK: dict[str, str] = {
    "create": "Create",
    "grant": "Create",
    "revoke": "Revoke",
    "expire": "Expire",
    "complete": "Complete",
}

# AITF Category 7 class_uid -> proposed OCSF ``ai`` category (uid 9) target.
# ``None`` means the AITF class maps onto a base-class event carrying the
# ``ai_operation`` profile rather than a dedicated ``ai``-category class.
OCSF_CLASS_CROSSWALK: dict[int, dict[str, Any]] = {
    7001: {"ocsf_category_uid": OCSF_AI_CATEGORY_UID, "ocsf_class": "ai_inference_activity"},
    7002: {"ocsf_category_uid": OCSF_AI_CATEGORY_UID, "ocsf_class": "agent_activity"},
    7003: {"ocsf_category_uid": OCSF_AI_CATEGORY_UID, "ocsf_class": "ai_tool_activity"},
    7004: {"ocsf_category_uid": OCSF_AI_CATEGORY_UID, "ocsf_class": "ai_retrieval_activity"},
    7005: {"ocsf_category_uid": 2, "ocsf_class": "detection_finding"},
    7006: {"ocsf_category_uid": 2, "ocsf_class": "vulnerability_finding"},
    7007: {"ocsf_category_uid": 2, "ocsf_class": "compliance_finding"},
    7008: {"ocsf_category_uid": OCSF_AI_CATEGORY_UID, "ocsf_class": "delegation_activity"},
    7009: {"ocsf_category_uid": OCSF_AI_CATEGORY_UID, "ocsf_class": "ai_model_activity"},
    7010: {"ocsf_category_uid": 5, "ocsf_class": "inventory_info"},
}
