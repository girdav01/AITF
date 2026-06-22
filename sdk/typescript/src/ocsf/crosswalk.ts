/**
 * AITF <-> OCSF agentic crosswalk.
 *
 * Reconciles AITF's established OCSF Category 7 AI event classes with the
 * upstream OCSF agentic-AI direction defined in:
 *
 *   * OCSF PR #1641   -- `objects/ai_agent.json` + the `ai_operation` profile
 *   * OCSF issue #1640 -- the proposed `ai` category (uid 9), the
 *     `delegation` object, `delegation_lineage`/`delegation_node` graph,
 *     and the `agent_activity` / `delegation_activity` control-plane classes.
 *
 * AITF keeps emitting Category 7 events (no breaking change for existing
 * SIEM/XDR consumers), but every event is now enriched with the OCSF
 * `ai_agent` and `delegation` objects via the `ai_operation` profile, and
 * this module publishes the activity/class crosswalk tables that let consumers
 * translate Category 7 telemetry onto OCSF's native `ai` category once it
 * lands. This is the "compromise" layer: conformant OCSF primitives carried on
 * top of AITF's richer class set.
 */

import {
  AGENT_TYPE_LABELS,
  AgentTypeID,
  OCSF_AI_CATEGORY_UID,
  OCSFAIAgent,
  OCSFDelegation,
  OCSFDelegationLineage,
  OCSFDelegationNode,
  normalizeAgentTypeId,
} from "./schema";
import {
  AgentAttributes,
  GenAIAttributes,
  IdentityAttributes,
} from "../semantic-conventions/attributes";

type SpanAttributes = Record<string, unknown>;

function optStr(val: unknown): string | undefined {
  return val !== undefined && val !== null ? String(val) : undefined;
}

function asList(val: unknown): string[] {
  if (val === undefined || val === null) {
    return [];
  }
  if (Array.isArray(val)) {
    return val.map((v) => String(v));
  }
  return [String(val)];
}

/**
 * Build an OCSF `ai_agent` object (PR #1641) from span attributes.
 *
 * Returns `undefined` when no agent identity is present, so non-agentic
 * events are left untouched.
 */
export function buildAiAgent(attrs: SpanAttributes): OCSFAIAgent | undefined {
  const uid =
    attrs[AgentAttributes.ID] ??
    attrs[IdentityAttributes.AGENT_ID] ??
    attrs[AgentAttributes.WORKFLOW_ID];
  const name =
    attrs[AgentAttributes.NAME] ?? attrs[IdentityAttributes.AGENT_NAME];
  if (uid === undefined && name === undefined) {
    return undefined;
  }

  const framework = optStr(attrs[AgentAttributes.FRAMEWORK]);
  const typeId = normalizeAgentTypeId(framework);

  return {
    uid: String(uid !== undefined ? uid : name),
    instance_uid: optStr(attrs[AgentAttributes.SESSION_ID]),
    name: optStr(name),
    type:
      typeId !== AgentTypeID.UNKNOWN
        ? AGENT_TYPE_LABELS[typeId] ?? AGENT_TYPE_LABELS[AgentTypeID.OTHER]
        : undefined,
    type_id: typeId,
    ai_model: optStr(
      attrs[GenAIAttributes.REQUEST_MODEL] ??
        attrs[GenAIAttributes.RESPONSE_MODEL]
    ),
    version: optStr(attrs[AgentAttributes.VERSION]),
    charter: optStr(attrs[AgentAttributes.DESCRIPTION]),
  };
}

/**
 * Build an OCSF `delegation` object (issue #1640) from span attributes.
 *
 * Returns `undefined` when no delegation context is present.
 */
export function buildDelegation(
  attrs: SpanAttributes
): OCSFDelegation | undefined {
  const delegateeId =
    attrs[IdentityAttributes.DELEGATION_DELEGATEE_ID] ??
    attrs[AgentAttributes.DELEGATION_TARGET_AGENT_ID];
  const delegatorId = attrs[IdentityAttributes.DELEGATION_DELEGATOR_ID];
  const delegatee =
    attrs[IdentityAttributes.DELEGATION_DELEGATEE] ??
    attrs[AgentAttributes.DELEGATION_TARGET_AGENT];
  const delegator = attrs[IdentityAttributes.DELEGATION_DELEGATOR];
  const delegType = attrs[IdentityAttributes.DELEGATION_TYPE];
  const chain = attrs[IdentityAttributes.DELEGATION_CHAIN];

  if (
    delegateeId === undefined &&
    delegatorId === undefined &&
    delegatee === undefined &&
    delegator === undefined &&
    delegType === undefined &&
    (chain === undefined || (Array.isArray(chain) && chain.length === 0))
  ) {
    return undefined;
  }

  // OCSF delegation.uid is the stable identifier of the granted authority.
  let uid: unknown = delegateeId ?? delegatee;
  if (uid === undefined && Array.isArray(chain) && chain.length > 0) {
    uid = chain[chain.length - 1];
  }

  const ttlRaw = attrs[IdentityAttributes.DELEGATION_TTL_SECONDS];

  return {
    uid: uid !== undefined && uid !== null ? String(uid) : "unknown",
    parent_uid: optStr(delegatorId),
    issuer_uid: optStr(attrs[IdentityAttributes.PROVIDER]),
    delegator: optStr(delegator),
    delegatee: optStr(delegatee),
    type: optStr(delegType),
    scope: asList(attrs[IdentityAttributes.DELEGATION_SCOPE_DELEGATED]),
    proof_type: optStr(attrs[IdentityAttributes.DELEGATION_PROOF_TYPE]),
    ttl_seconds:
      ttlRaw !== undefined && ttlRaw !== null ? Number(ttlRaw) : undefined,
  };
}

/**
 * Build an OCSF `delegation_lineage` graph from a delegation chain.
 *
 * The AITF `identity.delegation.chain` (ordered from origin to current)
 * is materialized into the directed `delegation_node` graph proposed in
 * OCSF issue #1640.
 */
export function buildDelegationLineage(
  attrs: SpanAttributes
): OCSFDelegationLineage | undefined {
  const chain = attrs[IdentityAttributes.DELEGATION_CHAIN];
  if (!Array.isArray(chain) || chain.length === 0) {
    return undefined;
  }

  const nodes: OCSFDelegationNode[] = [];
  let parent: string | undefined;
  chain.forEach((nodeUidRaw, depth) => {
    const nodeUid = String(nodeUidRaw);
    nodes.push({
      uid: nodeUid,
      parent_uid: parent,
      agent_uid: nodeUid,
      depth,
    });
    parent = nodeUid;
  });
  return { nodes };
}

// --- Control-plane crosswalk tables (OCSF issue #1640) ---------------------
//
// AITF retains Category 7 today; OCSF issue #1640 proposes a native `ai`
// category (uid 9) with dedicated control-plane classes. These tables let a
// consumer translate AITF activities onto the proposed OCSF activities. UIDs
// for the proposed classes are not yet finalized upstream, so only the
// activity-name mapping (which is stable) is published here.

/** AITF AIAgentActivityEvent (7002) activity_id -> OCSF agent_activity activity. */
export const OCSF_AGENT_ACTIVITY_CROSSWALK: Record<number, string> = {
  1: "Spawn", // AITF Session Start  -> agent spawned
  2: "Terminate", // AITF Session End    -> agent terminated
  3: "Update", // AITF Step Execute   -> agent state update
  4: "Register", // AITF Delegation     -> registers delegated authority
  5: "Resume", // AITF Memory Access  -> resume/continue
  6: "Resume", // AITF Error Recovery -> resume after recovery
  7: "Suspend", // AITF Human Approval -> suspended pending human input
  99: "Unknown",
};

/** AITF AIIdentityEvent (7008) delegation activity -> OCSF delegation_activity. */
export const OCSF_DELEGATION_ACTIVITY_CROSSWALK: Record<string, string> = {
  create: "Create",
  grant: "Create",
  revoke: "Revoke",
  expire: "Expire",
  complete: "Complete",
};

/**
 * AITF Category 7 class_uid -> proposed OCSF `ai` category (uid 9) target.
 */
export interface OCSFClassTarget {
  ocsf_category_uid: number;
  ocsf_class: string;
}

export const OCSF_CLASS_CROSSWALK: Record<number, OCSFClassTarget> = {
  7001: { ocsf_category_uid: OCSF_AI_CATEGORY_UID, ocsf_class: "ai_inference_activity" },
  7002: { ocsf_category_uid: OCSF_AI_CATEGORY_UID, ocsf_class: "agent_activity" },
  7003: { ocsf_category_uid: OCSF_AI_CATEGORY_UID, ocsf_class: "ai_tool_activity" },
  7004: { ocsf_category_uid: OCSF_AI_CATEGORY_UID, ocsf_class: "ai_retrieval_activity" },
  7005: { ocsf_category_uid: 2, ocsf_class: "detection_finding" },
  7006: { ocsf_category_uid: 2, ocsf_class: "vulnerability_finding" },
  7007: { ocsf_category_uid: 2, ocsf_class: "compliance_finding" },
  7008: { ocsf_category_uid: OCSF_AI_CATEGORY_UID, ocsf_class: "delegation_activity" },
  7009: { ocsf_category_uid: OCSF_AI_CATEGORY_UID, ocsf_class: "ai_model_activity" },
  7010: { ocsf_category_uid: 5, ocsf_class: "inventory_info" },
};
