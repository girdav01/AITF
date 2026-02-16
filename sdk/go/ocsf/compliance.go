package ocsf

import (
	"fmt"
	"time"
)

// FrameworkMappings maps compliance frameworks to event-type-level controls.
// Structure: FrameworkMappings[framework][eventType] -> map of control data.
var FrameworkMappings = map[string]map[string]map[string]interface{}{
	"nist_ai_rmf": {
		"model_inference":  {"controls": []string{"MAP-1.1", "MEASURE-2.5"}, "function": "MAP"},
		"agent_activity":   {"controls": []string{"GOVERN-1.2", "MANAGE-3.1"}, "function": "GOVERN"},
		"tool_execution":   {"controls": []string{"MAP-3.5", "MANAGE-4.2"}, "function": "MANAGE"},
		"data_retrieval":   {"controls": []string{"MAP-1.5", "MEASURE-2.7"}, "function": "MAP"},
		"security_finding": {"controls": []string{"MANAGE-2.4", "MANAGE-4.1"}, "function": "MANAGE"},
		"supply_chain":     {"controls": []string{"MAP-5.2", "GOVERN-6.1"}, "function": "GOVERN"},
		"governance":       {"controls": []string{"GOVERN-1.1", "MANAGE-1.3"}, "function": "GOVERN"},
		"identity":         {"controls": []string{"GOVERN-1.5", "MANAGE-2.1"}, "function": "GOVERN"},
	},
	"mitre_atlas": {
		"model_inference":  {"techniques": []string{"AML.T0040"}, "tactic": "ML Attack Staging"},
		"agent_activity":   {"techniques": []string{"AML.T0048"}, "tactic": "ML Attack Staging"},
		"tool_execution":   {"techniques": []string{"AML.T0043"}, "tactic": "ML Attack Staging"},
		"data_retrieval":   {"techniques": []string{"AML.T0025"}, "tactic": "Exfiltration"},
		"security_finding": {"techniques": []string{"AML.T0051"}, "tactic": "Initial Access"},
		"supply_chain":     {"techniques": []string{"AML.T0010"}, "tactic": "Resource Development"},
		"identity":         {"techniques": []string{"AML.T0052"}, "tactic": "Initial Access"},
	},
	"iso_42001": {
		"model_inference":  {"controls": []string{"6.1.4", "8.4"}, "clause": "Operation"},
		"agent_activity":   {"controls": []string{"8.2", "A.6.2.5"}, "clause": "Operation"},
		"tool_execution":   {"controls": []string{"A.6.2.7"}, "clause": "Annex A"},
		"data_retrieval":   {"controls": []string{"A.7.4"}, "clause": "Annex A"},
		"security_finding": {"controls": []string{"6.1.2", "A.6.2.4"}, "clause": "Planning"},
		"supply_chain":     {"controls": []string{"A.6.2.3"}, "clause": "Annex A"},
		"governance":       {"controls": []string{"5.1", "9.1"}, "clause": "Leadership"},
		"identity":         {"controls": []string{"A.6.2.6"}, "clause": "Annex A"},
	},
	"eu_ai_act": {
		"model_inference":  {"articles": []string{"Article 13", "Article 15"}, "risk_level": "high"},
		"agent_activity":   {"articles": []string{"Article 14", "Article 52"}, "risk_level": "high"},
		"tool_execution":   {"articles": []string{"Article 9"}, "risk_level": "high"},
		"data_retrieval":   {"articles": []string{"Article 10"}, "risk_level": "high"},
		"security_finding": {"articles": []string{"Article 9", "Article 62"}, "risk_level": "high"},
		"supply_chain":     {"articles": []string{"Article 15", "Article 28"}, "risk_level": "high"},
		"governance":       {"articles": []string{"Article 9", "Article 61"}, "risk_level": "high"},
		"identity":         {"articles": []string{"Article 9"}, "risk_level": "high"},
	},
	"soc2": {
		"model_inference":  {"controls": []string{"CC6.1"}, "criteria": "Common Criteria"},
		"agent_activity":   {"controls": []string{"CC7.2"}, "criteria": "Common Criteria"},
		"tool_execution":   {"controls": []string{"CC6.3"}, "criteria": "Common Criteria"},
		"data_retrieval":   {"controls": []string{"CC6.1"}, "criteria": "Common Criteria"},
		"security_finding": {"controls": []string{"CC7.2", "CC7.3"}, "criteria": "Common Criteria"},
		"supply_chain":     {"controls": []string{"CC9.2"}, "criteria": "Common Criteria"},
		"governance":       {"controls": []string{"CC1.2"}, "criteria": "Common Criteria"},
		"identity":         {"controls": []string{"CC6.1", "CC6.2"}, "criteria": "Common Criteria"},
	},
	"gdpr": {
		"model_inference":  {"articles": []string{"Article 5", "Article 22"}, "lawful_basis": "legitimate_interest"},
		"agent_activity":   {"articles": []string{"Article 22"}, "lawful_basis": "legitimate_interest"},
		"tool_execution":   {"articles": []string{"Article 25"}, "lawful_basis": "legitimate_interest"},
		"data_retrieval":   {"articles": []string{"Article 5", "Article 6"}, "lawful_basis": "legitimate_interest"},
		"security_finding": {"articles": []string{"Article 32", "Article 33"}, "lawful_basis": "legal_obligation"},
		"supply_chain":     {"articles": []string{"Article 28"}, "lawful_basis": "contractual"},
		"governance":       {"articles": []string{"Article 5"}, "lawful_basis": "legal_obligation"},
		"identity":         {"articles": []string{"Article 32"}, "lawful_basis": "legal_obligation"},
	},
	"ccpa": {
		"model_inference":  {"sections": []string{"1798.100"}, "category": "personal_information"},
		"data_retrieval":   {"sections": []string{"1798.100"}, "category": "personal_information"},
		"security_finding": {"sections": []string{"1798.150"}, "category": "breach"},
		"governance":       {"sections": []string{"1798.185"}, "category": "rulemaking"},
		"identity":         {"sections": []string{"1798.140"}, "category": "personal_information"},
	},
	"csa_aicm": {
		"model_inference":  {"controls": []string{"AIS-04", "MDS-01", "LOG-07"}, "domain": "Model Security"},
		"agent_activity":   {"controls": []string{"AIS-02", "MDS-05", "GRC-02"}, "domain": "Governance, Risk & Compliance"},
		"tool_execution":   {"controls": []string{"AIS-01", "AIS-04", "LOG-05"}, "domain": "Application & Interface Security"},
		"data_retrieval":   {"controls": []string{"DSP-01", "DSP-04", "CEK-03"}, "domain": "Data Security & Privacy"},
		"security_finding": {"controls": []string{"SEF-03", "TVM-01", "LOG-04"}, "domain": "Security Incident Management"},
		"supply_chain":     {"controls": []string{"STA-01", "STA-03", "CCC-01"}, "domain": "Supply Chain Management"},
		"governance":       {"controls": []string{"GRC-01", "A&A-01", "LOG-01"}, "domain": "Governance, Risk & Compliance"},
		"identity":         {"controls": []string{"IAM-01", "IAM-02", "IAM-04"}, "domain": "Identity & Access Management"},
	},
}

// allComplianceFrameworks lists all supported compliance frameworks.
var allComplianceFrameworks = []string{
	"nist_ai_rmf", "mitre_atlas", "iso_42001",
	"eu_ai_act", "soc2", "gdpr", "ccpa", "csa_aicm",
}

// ComplianceMapper maps AI events to compliance framework controls.
type ComplianceMapper struct {
	frameworks []string
}

// NewComplianceMapper creates a new compliance mapper.
// If frameworks is nil or empty, all 8 frameworks are enabled.
func NewComplianceMapper(frameworks []string) *ComplianceMapper {
	if len(frameworks) == 0 {
		frameworks = append([]string{}, allComplianceFrameworks...)
	}
	return &ComplianceMapper{frameworks: frameworks}
}

// MapEvent maps an event type to compliance frameworks, returning a ComplianceMetadata.
func (c *ComplianceMapper) MapEvent(eventType string) *ComplianceMetadata {
	result := &ComplianceMetadata{}

	for _, framework := range c.frameworks {
		frameworkMap, ok := FrameworkMappings[framework]
		if !ok {
			continue
		}
		eventMap, ok := frameworkMap[eventType]
		if !ok {
			continue
		}

		switch framework {
		case "nist_ai_rmf":
			result.NISTAIMRMF = eventMap
		case "mitre_atlas":
			result.MITREAtlas = eventMap
		case "iso_42001":
			result.ISO42001 = eventMap
		case "eu_ai_act":
			result.EUAIAct = eventMap
		case "soc2":
			result.SOC2 = eventMap
		case "gdpr":
			result.GDPR = eventMap
		case "ccpa":
			result.CCPA = eventMap
		case "csa_aicm":
			result.CSAAICM = eventMap
		}
	}

	return result
}

// EnrichEvent adds compliance metadata to an OCSF event and returns it.
func (c *ComplianceMapper) EnrichEvent(event *AIBaseEvent, eventType string) *AIBaseEvent {
	event.Compliance = c.MapEvent(eventType)
	return event
}

// GetCoverageMatrix generates a coverage matrix showing which controls apply
// per event type across the configured frameworks.
func (c *ComplianceMapper) GetCoverageMatrix() map[string]map[string][]string {
	eventTypes := []string{
		"model_inference", "agent_activity", "tool_execution",
		"data_retrieval", "security_finding", "supply_chain",
		"governance", "identity",
	}

	matrix := make(map[string]map[string][]string)
	for _, eventType := range eventTypes {
		matrix[eventType] = make(map[string][]string)
		for _, framework := range c.frameworks {
			frameworkMap, ok := FrameworkMappings[framework]
			if !ok {
				continue
			}
			eventMap, ok := frameworkMap[eventType]
			if !ok {
				continue
			}
			// Extract the primary list (controls, techniques, articles, sections)
			for _, key := range []string{"controls", "techniques", "articles", "sections"} {
				if v, ok := eventMap[key]; ok {
					if strs, ok := v.([]string); ok {
						matrix[eventType][framework] = strs
						break
					}
				}
			}
		}
	}

	return matrix
}

// AuditRecord represents a compliance audit record.
type AuditRecord struct {
	AuditID            string                       `json:"audit_id"`
	Timestamp          string                       `json:"timestamp"`
	EventType          string                       `json:"event_type"`
	FrameworksMapped   int                          `json:"frameworks_mapped"`
	ControlsMapped     int                          `json:"controls_mapped"`
	ViolationsDetected int                          `json:"violations_detected"`
	RiskScore          float64                      `json:"risk_score"`
	Actor              string                       `json:"actor,omitempty"`
	Model              string                       `json:"model,omitempty"`
	ComplianceDetails  map[string][]string          `json:"compliance_details"`
}

// GenerateAuditRecord creates an audit record from compliance mappings.
func (c *ComplianceMapper) GenerateAuditRecord(
	eventType string,
	actor string,
	model string,
	riskScore float64,
) *AuditRecord {
	compliance := c.MapEvent(eventType)
	complianceDetails := make(map[string][]string)
	var allControls []string

	// Extract controls from each framework
	frameworkData := map[string]map[string]interface{}{
		"nist_ai_rmf": compliance.NISTAIMRMF,
		"mitre_atlas": compliance.MITREAtlas,
		"iso_42001":   compliance.ISO42001,
		"eu_ai_act":   compliance.EUAIAct,
		"soc2":        compliance.SOC2,
		"gdpr":        compliance.GDPR,
		"ccpa":        compliance.CCPA,
		"csa_aicm":    compliance.CSAAICM,
	}

	for framework, data := range frameworkData {
		if data == nil {
			continue
		}
		for _, key := range []string{"controls", "techniques", "articles", "sections"} {
			if v, ok := data[key]; ok {
				if strs, ok := v.([]string); ok {
					complianceDetails[framework] = strs
					allControls = append(allControls, strs...)
					break
				}
			}
		}
	}

	return &AuditRecord{
		AuditID:            fmt.Sprintf("aud-%s", time.Now().UTC().Format("20060102150405")),
		Timestamp:          time.Now().UTC().Format(time.RFC3339),
		EventType:          eventType,
		FrameworksMapped:   len(complianceDetails),
		ControlsMapped:     len(allControls),
		ViolationsDetected: 0,
		RiskScore:          riskScore,
		Actor:              actor,
		Model:              model,
		ComplianceDetails:  complianceDetails,
	}
}
