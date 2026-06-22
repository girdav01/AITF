/**
 * AITF OCSF Base Schema.
 *
 * OCSF v1.1.0 base objects and AI-specific extension models.
 * Based on the OCSF schema from the AITelemetry project, enhanced
 * for AITF Category 7 AI events.
 */

import { randomUUID } from "crypto";

// --- OCSF Enumerations ---

export enum OCSFSeverity {
  UNKNOWN = 0,
  INFORMATIONAL = 1,
  LOW = 2,
  MEDIUM = 3,
  HIGH = 4,
  CRITICAL = 5,
  FATAL = 6,
}

export enum OCSFStatus {
  UNKNOWN = 0,
  SUCCESS = 1,
  FAILURE = 2,
  OTHER = 99,
}

export enum OCSFActivity {
  UNKNOWN = 0,
  CREATE = 1,
  READ = 2,
  UPDATE = 3,
  DELETE = 4,
  OTHER = 99,
}

/** AITF OCSF Category 7 class UIDs. */
export enum AIClassUID {
  MODEL_INFERENCE = 7001,
  AGENT_ACTIVITY = 7002,
  TOOL_EXECUTION = 7003,
  DATA_RETRIEVAL = 7004,
  SECURITY_FINDING = 7005,
  SUPPLY_CHAIN = 7006,
  GOVERNANCE = 7007,
  IDENTITY = 7008,
  MODEL_OPS = 7009,
  ASSET_INVENTORY = 7010,
}

/**
 * OCSF `ai_agent.type_id` — normalized agent framework.
 *
 * Mirrors the enum introduced by OCSF PR #1641 (`objects/ai_agent.json`)
 * so AITF telemetry maps cleanly onto the upstream OCSF `ai_agent` object.
 */
export enum AgentTypeID {
  UNKNOWN = 0,
  NATIVE = 1,
  LANGCHAIN = 2,
  AUTOGEN = 3,
  CREWAI = 4,
  OTHER = 99,
}

/** Caption labels for AgentTypeID, per OCSF PR #1641. */
export const AGENT_TYPE_LABELS: Record<number, string> = {
  0: "Unknown",
  1: "Native",
  2: "LangChain",
  3: "AutoGen",
  4: "CrewAI",
  99: "Other",
};

/**
 * AITF framework value -> OCSF ai_agent.type_id. Frameworks without a
 * dedicated OCSF enum member (langgraph, semantic_kernel, custom, ...)
 * normalize to OTHER (99), matching OCSF's open-enum guidance.
 */
const FRAMEWORK_TO_TYPE_ID: Record<string, AgentTypeID> = {
  native: AgentTypeID.NATIVE,
  langchain: AgentTypeID.LANGCHAIN,
  langgraph: AgentTypeID.LANGCHAIN,
  autogen: AgentTypeID.AUTOGEN,
  crewai: AgentTypeID.CREWAI,
};

/** Map an AITF framework string to an OCSF `ai_agent.type_id` value. */
export function normalizeAgentTypeId(framework?: string | null): number {
  if (!framework) {
    return AgentTypeID.UNKNOWN;
  }
  const key = framework.trim().toLowerCase();
  return key in FRAMEWORK_TO_TYPE_ID
    ? FRAMEWORK_TO_TYPE_ID[key]
    : AgentTypeID.OTHER;
}

/**
 * OCSF AI category and control-plane classes proposed in OCSF issue #1640.
 * AITF keeps its established Category 7 classes but records the upstream
 * target so consumers can crosswalk to the future native `ai` category.
 */
export const OCSF_AI_CATEGORY_UID = 9; // proposed "AI Activity" category (OCSF issue #1640)

// --- OCSF Base Object Interfaces ---

/** OCSF event metadata. */
export interface OCSFMetadata {
  version: string;
  product: {
    name: string;
    vendor_name: string;
    version: string;
  };
  uid: string;
  correlation_uid?: string;
  original_time?: string;
  logged_time: string;
}

/** OCSF actor information. */
export interface OCSFActor {
  user?: Record<string, unknown>;
  session?: Record<string, unknown>;
  app_name?: string;
}

/** OCSF device/host information. */
export interface OCSFDevice {
  hostname?: string;
  ip?: string;
  type?: string;
  os?: Record<string, string>;
  cloud?: Record<string, string>;
  container?: Record<string, string>;
}

/** OCSF enrichment data. */
export interface OCSFEnrichment {
  name: string;
  value: string;
  type?: string;
  provider?: string;
}

/** OCSF observable value. */
export interface OCSFObservable {
  name: string;
  type: string;
  value: string;
}

// --- AI-Specific Extension Interfaces ---

/** AI model information. */
export interface AIModelInfo {
  model_id: string;
  name?: string;
  version?: string;
  provider?: string;
  type?: string; // llm, embedding, image, audio, multimodal
  parameters?: Record<string, unknown>;
}

/** AI token usage statistics. */
export interface AITokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  reasoning_tokens: number;
  estimated_cost_usd?: number;
}

/** AI operation latency metrics. */
export interface AILatencyMetrics {
  total_ms: number;
  time_to_first_token_ms?: number;
  tokens_per_second?: number;
  queue_time_ms?: number;
  inference_time_ms?: number;
}

/** AI operation cost information. */
export interface AICostInfo {
  input_cost_usd: number;
  output_cost_usd: number;
  total_cost_usd: number;
  currency: string;
}

/** Multi-agent team information. */
export interface AITeamInfo {
  team_name: string;
  team_id?: string;
  topology?: string;
  members: string[];
  coordinator?: string;
}

/** Security finding details. */
export interface AISecurityFinding {
  finding_type: string;
  owasp_category?: string;
  risk_level: string;
  risk_score: number;
  confidence: number;
  detection_method: string;
  blocked: boolean;
  details?: string;
  pii_types: string[];
  matched_patterns: string[];
  remediation?: string;
}

/**
 * OCSF `ai_agent` object (OCSF PR #1641).
 *
 * An autonomous AI agent operating under delegated authority. Distinct from
 * the OCSF `agent` object (which models security sensors such as EDR/DLP)
 * and from human principals. Attached to events via the `ai_operation`
 * profile so any activity can be attributed to the agent that performed it.
 */
export interface OCSFAIAgent {
  uid: string; // required: stable logical identifier
  instance_uid?: string; // restart-sensitive running instance id
  name?: string;
  type?: string; // caption of type_id (Native, LangChain, ...)
  type_id: number;
  ai_model?: string; // model backing the agent at event time
  version?: string; // agent code/configuration revision
  charter?: string; // role / operating-boundary reference
}

/**
 * OCSF `delegation` object (OCSF issue #1640).
 *
 * A durable authorization context that persists independently of any single
 * trace or session. `uid`/`parent_uid`/`issuer_uid` provide the OCSF core;
 * the remaining fields preserve AITF's richer delegation telemetry.
 */
export interface OCSFDelegation {
  uid: string; // required: stable delegation identifier
  parent_uid?: string; // parent delegation (lineage)
  issuer_uid?: string; // trusted issuer that minted the delegation
  delegator?: string;
  delegatee?: string;
  type?: string; // on_behalf_of, token_exchange, capability_grant, ...
  scope: string[];
  proof_type?: string; // dpop, mtls_binding, signed_assertion
  ttl_seconds?: number;
}

/** A single node in an OCSF `delegation_lineage` graph (OCSF issue #1640). */
export interface OCSFDelegationNode {
  uid: string;
  parent_uid?: string;
  agent_uid?: string;
  depth?: number;
}

/** OCSF `delegation_lineage` — directed graph for ancestry queries. */
export interface OCSFDelegationLineage {
  nodes: OCSFDelegationNode[];
}

/** Compliance framework mappings. */
export interface ComplianceMetadata {
  nist_ai_rmf?: Record<string, unknown>;
  mitre_atlas?: Record<string, unknown>;
  iso_42001?: Record<string, unknown>;
  eu_ai_act?: Record<string, unknown>;
  soc2?: Record<string, unknown>;
  gdpr?: Record<string, unknown>;
  ccpa?: Record<string, unknown>;
  csa_aicm?: Record<string, unknown>;
}

// --- OCSF Base Event ---

/** Base OCSF event for all AITF Category 7 events. */
export interface AIBaseEvent {
  activity_id: number;
  category_uid: number;
  class_uid: number;
  type_uid: number;
  time: string;
  severity_id: number;
  status_id: number;
  message: string;
  metadata: OCSFMetadata;
  actor?: OCSFActor;
  device?: OCSFDevice;
  compliance?: ComplianceMetadata;
  observables: OCSFObservable[];
  enrichments: OCSFEnrichment[];

  // OCSF `ai_operation` profile (OCSF PR #1641) + delegation context
  // (OCSF issue #1640). Populated by the crosswalk so every AITF event can
  // be attributed to the AI agent and delegation that produced it.
  ai_agent?: OCSFAIAgent;
  ai_model?: string;
  delegation?: OCSFDelegation;
  delegation_lineage?: OCSFDelegationLineage;
}

// --- Factory Functions ---

/** Create default OCSF metadata. */
export function createMetadata(
  correlationUid?: string
): OCSFMetadata {
  return {
    version: "1.1.0",
    product: {
      name: "AITF",
      vendor_name: "AITF",
      version: "1.0.0",
    },
    uid: randomUUID(),
    correlation_uid: correlationUid,
    logged_time: new Date().toISOString(),
  };
}

/** Create default token usage. */
export function createTokenUsage(
  options: Partial<AITokenUsage> = {}
): AITokenUsage {
  const usage: AITokenUsage = {
    input_tokens: options.input_tokens ?? 0,
    output_tokens: options.output_tokens ?? 0,
    total_tokens: options.total_tokens ?? 0,
    cached_tokens: options.cached_tokens ?? 0,
    reasoning_tokens: options.reasoning_tokens ?? 0,
    estimated_cost_usd: options.estimated_cost_usd,
  };
  if (usage.total_tokens === 0) {
    usage.total_tokens = usage.input_tokens + usage.output_tokens;
  }
  return usage;
}

/** Create a base OCSF AI event. */
export function createBaseEvent(
  classUid: number,
  options: Partial<AIBaseEvent> = {}
): AIBaseEvent {
  const activityId = options.activity_id ?? OCSFActivity.OTHER;
  return {
    activity_id: activityId,
    category_uid: 7, // AI System Activity
    class_uid: classUid,
    type_uid: options.type_uid ?? classUid * 100 + activityId,
    time: options.time ?? new Date().toISOString(),
    severity_id: options.severity_id ?? OCSFSeverity.INFORMATIONAL,
    status_id: options.status_id ?? OCSFStatus.SUCCESS,
    message: options.message ?? "",
    metadata: options.metadata ?? createMetadata(),
    actor: options.actor,
    device: options.device,
    compliance: options.compliance,
    observables: options.observables ?? [],
    enrichments: options.enrichments ?? [],
  };
}

/**
 * Remove undefined/null properties from an object (for JSON export).
 */
export function stripNulls<T extends Record<string, unknown>>(
  obj: T
): Partial<T> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value !== undefined && value !== null) {
      if (
        typeof value === "object" &&
        !Array.isArray(value) &&
        value !== null
      ) {
        const stripped = stripNulls(value as Record<string, unknown>);
        if (Object.keys(stripped).length > 0) {
          result[key] = stripped;
        }
      } else {
        result[key] = value;
      }
    }
  }
  return result as Partial<T>;
}
