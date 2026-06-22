package ocsf

// AITF <-> OCSF agentic crosswalk.
//
// Reconciles AITF's established OCSF Category 7 AI event classes with the
// upstream OCSF agentic-AI direction defined in:
//
//   - OCSF PR #1641   -- objects/ai_agent.json + the ai_operation profile
//   - OCSF issue #1640 -- the proposed "ai" category (uid 9), the delegation
//     object, delegation_lineage/delegation_node graph, and the agent_activity
//     / delegation_activity control-plane classes.
//
// AITF keeps emitting Category 7 events (no breaking change for existing
// SIEM/XDR consumers), but every event is now enriched with the OCSF ai_agent
// and delegation objects via the ai_operation profile, and this file publishes
// the activity/class crosswalk tables that let consumers translate Category 7
// telemetry onto OCSF's native "ai" category once it lands.

import (
	"strconv"

	"github.com/girdav01/AITF/sdk/go/semconv"
)

// BuildAIAgent builds an OCSF ai_agent object (PR #1641) from span attributes.
//
// Returns nil when no agent identity is present, so non-agentic events are left
// untouched.
func BuildAIAgent(attrs map[string]interface{}) *OCSFAIAgent {
	uid := firstAttrStr(attrs,
		string(semconv.AgentIDKey),
		string(semconv.IdentityAgentIDKey),
		string(semconv.AgentWorkflowIDKey),
	)
	name := firstAttrStr(attrs,
		string(semconv.AgentNameKey),
		string(semconv.IdentityAgentNameKey),
	)
	if uid == "" && name == "" {
		return nil
	}

	framework := attrStr(attrs, string(semconv.AgentFrameworkKey), "")
	typeID := NormalizeAgentTypeID(framework)

	resolvedUID := uid
	if resolvedUID == "" {
		resolvedUID = name
	}

	typeLabel := ""
	if typeID != AgentTypeIDUnknown {
		if label, ok := AgentTypeLabels[typeID]; ok {
			typeLabel = label
		} else {
			typeLabel = AgentTypeLabels[AgentTypeIDOther]
		}
	}

	return &OCSFAIAgent{
		UID:         resolvedUID,
		InstanceUID: attrStr(attrs, string(semconv.AgentSessionIDKey), ""),
		Name:        name,
		Type:        typeLabel,
		TypeID:      typeID,
		AIModel: firstAttrStr(attrs,
			string(semconv.GenAIRequestModelKey),
			string(semconv.GenAIResponseModelKey),
		),
		Version: attrStr(attrs, string(semconv.AgentVersionKey), ""),
		Charter: attrStr(attrs, string(semconv.AgentDescriptionKey), ""),
	}
}

// BuildDelegation builds an OCSF delegation object (issue #1640) from span
// attributes. Returns nil when no delegation context is present.
func BuildDelegation(attrs map[string]interface{}) *OCSFDelegation {
	delegateeID := firstAttrStr(attrs,
		string(semconv.IdentityDelegDelegateeIDKey),
		string(semconv.AgentDelegationTargetAgentIDKey),
	)
	delegatorID := attrStr(attrs, string(semconv.IdentityDelegDelegatorIDKey), "")
	delegatee := firstAttrStr(attrs,
		string(semconv.IdentityDelegDelegateeKey),
		string(semconv.AgentDelegationTargetAgentKey),
	)
	delegator := attrStr(attrs, string(semconv.IdentityDelegDelegatorKey), "")
	delegType := attrStr(attrs, string(semconv.IdentityDelegTypeKey), "")
	chain := attrStrList(attrs, string(semconv.IdentityDelegChainKey))

	if delegateeID == "" && delegatorID == "" && delegatee == "" &&
		delegator == "" && delegType == "" && len(chain) == 0 {
		return nil
	}

	// OCSF delegation.uid is the stable identifier of the granted authority.
	uid := delegateeID
	if uid == "" {
		uid = delegatee
	}
	if uid == "" && len(chain) > 0 {
		uid = chain[len(chain)-1]
	}
	if uid == "" {
		uid = "unknown"
	}

	d := &OCSFDelegation{
		UID:       uid,
		ParentUID: delegatorID,
		IssuerUID: attrStr(attrs, string(semconv.IdentityProviderKey), ""),
		Delegator: delegator,
		Delegatee: delegatee,
		Type:      delegType,
		Scope:     attrStrList(attrs, string(semconv.IdentityDelegScopeDelegatedKey)),
		ProofType: attrStr(attrs, string(semconv.IdentityDelegProofTypeKey), ""),
	}

	if v, ok := attrs[string(semconv.IdentityDelegTTLSecondsKey)]; ok && v != nil {
		ttl := toInt(v)
		d.TTLSeconds = &ttl
	}

	return d
}

// BuildDelegationLineage builds an OCSF delegation_lineage graph from a
// delegation chain.
//
// The AITF identity.delegation.chain (ordered from origin to current) is
// materialized into the directed delegation_node graph proposed in OCSF issue
// #1640. Returns nil when no chain is present.
func BuildDelegationLineage(attrs map[string]interface{}) *OCSFDelegationLineage {
	chain := attrStrList(attrs, string(semconv.IdentityDelegChainKey))
	if len(chain) == 0 {
		return nil
	}

	nodes := make([]OCSFDelegationNode, 0, len(chain))
	parent := ""
	for depth, nodeUID := range chain {
		nodes = append(nodes, OCSFDelegationNode{
			UID:       nodeUID,
			ParentUID: parent,
			AgentUID:  nodeUID,
			Depth:     depth,
		})
		parent = nodeUID
	}
	return &OCSFDelegationLineage{Nodes: nodes}
}

// --- Control-plane crosswalk tables (OCSF issue #1640) ---------------------
//
// AITF retains Category 7 today; OCSF issue #1640 proposes a native "ai"
// category (uid 9) with dedicated control-plane classes. These tables let a
// consumer translate AITF activities onto the proposed OCSF activities. UIDs
// for the proposed classes are not yet finalized upstream, so only the
// activity-name mapping (which is stable) is published here.

// OCSFAgentActivityCrosswalk maps AITF AIAgentActivityEvent (7002) activity_id
// to the OCSF agent_activity activity name.
var OCSFAgentActivityCrosswalk = map[int]string{
	1:  "Spawn",     // AITF Session Start  -> agent spawned
	2:  "Terminate", // AITF Session End    -> agent terminated
	3:  "Update",    // AITF Step Execute   -> agent state update
	4:  "Register",  // AITF Delegation     -> registers delegated authority
	5:  "Resume",    // AITF Memory Access  -> resume/continue
	6:  "Resume",    // AITF Error Recovery -> resume after recovery
	7:  "Suspend",   // AITF Human Approval -> suspended pending human input
	99: "Unknown",
}

// OCSFDelegationActivityCrosswalk maps AITF AIIdentityEvent (7008) delegation
// activity to the OCSF delegation_activity name.
var OCSFDelegationActivityCrosswalk = map[string]string{
	"create":   "Create",
	"grant":    "Create",
	"revoke":   "Revoke",
	"expire":   "Expire",
	"complete": "Complete",
}

// OCSFClassCrosswalkEntry describes the proposed OCSF target for an AITF class.
type OCSFClassCrosswalkEntry struct {
	OCSFCategoryUID int    `json:"ocsf_category_uid"`
	OCSFClass       string `json:"ocsf_class"`
}

// OCSFClassCrosswalk maps each AITF Category 7 class_uid onto its proposed OCSF
// target. Findings-shaped classes route to the existing Findings (2) category;
// inventory routes to Discovery (5); the rest route to the proposed "ai"
// category (uid 9).
var OCSFClassCrosswalk = map[int]OCSFClassCrosswalkEntry{
	7001: {OCSFCategoryUID: OCSFAICategoryUID, OCSFClass: "ai_inference_activity"},
	7002: {OCSFCategoryUID: OCSFAICategoryUID, OCSFClass: "agent_activity"},
	7003: {OCSFCategoryUID: OCSFAICategoryUID, OCSFClass: "ai_tool_activity"},
	7004: {OCSFCategoryUID: OCSFAICategoryUID, OCSFClass: "ai_retrieval_activity"},
	7005: {OCSFCategoryUID: 2, OCSFClass: "detection_finding"},
	7006: {OCSFCategoryUID: 2, OCSFClass: "vulnerability_finding"},
	7007: {OCSFCategoryUID: 2, OCSFClass: "compliance_finding"},
	7008: {OCSFCategoryUID: OCSFAICategoryUID, OCSFClass: "delegation_activity"},
	7009: {OCSFCategoryUID: OCSFAICategoryUID, OCSFClass: "ai_model_activity"},
	7010: {OCSFCategoryUID: 5, OCSFClass: "inventory_info"},
}

// --- crosswalk attribute helpers ------------------------------------------

// firstAttrStr returns the first non-empty string attribute among keys.
func firstAttrStr(attrs map[string]interface{}, keys ...string) string {
	for _, k := range keys {
		if v := attrStr(attrs, k, ""); v != "" {
			return v
		}
	}
	return ""
}

// attrStrList normalizes an attribute value into a []string. Handles []string,
// []interface{}, and scalar values; returns nil when absent.
func attrStrList(attrs map[string]interface{}, key string) []string {
	v, ok := attrs[key]
	if !ok || v == nil {
		return nil
	}
	switch val := v.(type) {
	case []string:
		out := make([]string, len(val))
		copy(out, val)
		return out
	case []interface{}:
		out := make([]string, 0, len(val))
		for _, e := range val {
			out = append(out, anyToStr(e))
		}
		return out
	default:
		return []string{anyToStr(val)}
	}
}

// anyToStr converts a scalar attribute value to its string form.
func anyToStr(v interface{}) string {
	switch val := v.(type) {
	case string:
		return val
	case int:
		return strconv.Itoa(val)
	case int64:
		return strconv.FormatInt(val, 10)
	case float64:
		return strconv.FormatFloat(val, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(val)
	default:
		return ""
	}
}
