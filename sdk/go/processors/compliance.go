// Package processors provides AITF span processors for Go.
package processors

import (
	"strings"

	"github.com/girdav01/AITF/sdk/go/semconv"
	"go.opentelemetry.io/otel/attribute"
)

// ComplianceFrameworkMapping holds the control mapping for a single framework
// applied to a single event type.
type ComplianceFrameworkMapping struct {
	Controls   []string // NIST, ISO, SOC2, CSA AICM
	Techniques []string // MITRE ATLAS
	Articles   []string // EU AI Act, GDPR
	Sections   []string // CCPA
	Function   string   // NIST function (MAP, GOVERN, MANAGE)
	Tactic     string   // MITRE tactic
	Clause     string   // ISO clause
	RiskLevel  string   // EU AI Act risk level
	Criteria   string   // SOC2 criteria
	LawfulBasis string  // GDPR lawful basis
	Category   string   // CCPA category
	Domain     string   // CSA AICM domain
}

// PrimaryControls returns the primary control list for this mapping.
func (m *ComplianceFrameworkMapping) PrimaryControls() []string {
	if len(m.Controls) > 0 {
		return m.Controls
	}
	if len(m.Techniques) > 0 {
		return m.Techniques
	}
	if len(m.Articles) > 0 {
		return m.Articles
	}
	if len(m.Sections) > 0 {
		return m.Sections
	}
	return nil
}

// ComplianceMappings maps event types to framework mappings.
// Key structure: ComplianceMappings[eventType][framework] -> ComplianceFrameworkMapping
var ComplianceMappings = map[string]map[string]ComplianceFrameworkMapping{
	"model_inference": {
		"nist_ai_rmf": {Controls: []string{"MAP-1.1", "MEASURE-2.5"}, Function: "MAP"},
		"mitre_atlas":  {Techniques: []string{"AML.T0040"}, Tactic: "ML Attack Staging"},
		"iso_42001":    {Controls: []string{"6.1.4", "8.4"}, Clause: "Operation"},
		"eu_ai_act":    {Articles: []string{"Article 13", "Article 15"}, RiskLevel: "high"},
		"soc2":         {Controls: []string{"CC6.1"}, Criteria: "Common Criteria"},
		"gdpr":         {Articles: []string{"Article 5", "Article 22"}, LawfulBasis: "legitimate_interest"},
		"ccpa":         {Sections: []string{"1798.100"}, Category: "personal_information"},
		"csa_aicm":     {Controls: []string{"AIS-04", "MDS-01", "LOG-07"}, Domain: "Model Security"},
	},
	"agent_activity": {
		"nist_ai_rmf": {Controls: []string{"GOVERN-1.2", "MANAGE-3.1"}, Function: "GOVERN"},
		"mitre_atlas":  {Techniques: []string{"AML.T0048"}, Tactic: "ML Attack Staging"},
		"iso_42001":    {Controls: []string{"8.2", "A.6.2.5"}, Clause: "Operation"},
		"eu_ai_act":    {Articles: []string{"Article 14", "Article 52"}, RiskLevel: "high"},
		"soc2":         {Controls: []string{"CC7.2"}, Criteria: "Common Criteria"},
		"gdpr":         {Articles: []string{"Article 22"}, LawfulBasis: "legitimate_interest"},
		"csa_aicm":     {Controls: []string{"AIS-02", "MDS-05", "GRC-02"}, Domain: "Governance, Risk & Compliance"},
	},
	"tool_execution": {
		"nist_ai_rmf": {Controls: []string{"MAP-3.5", "MANAGE-4.2"}, Function: "MANAGE"},
		"mitre_atlas":  {Techniques: []string{"AML.T0043"}, Tactic: "ML Attack Staging"},
		"iso_42001":    {Controls: []string{"A.6.2.7"}, Clause: "Annex A"},
		"eu_ai_act":    {Articles: []string{"Article 9"}, RiskLevel: "high"},
		"soc2":         {Controls: []string{"CC6.3"}, Criteria: "Common Criteria"},
		"gdpr":         {Articles: []string{"Article 25"}, LawfulBasis: "legitimate_interest"},
		"csa_aicm":     {Controls: []string{"AIS-01", "AIS-04", "LOG-05"}, Domain: "Application & Interface Security"},
	},
	"data_retrieval": {
		"nist_ai_rmf": {Controls: []string{"MAP-1.5", "MEASURE-2.7"}, Function: "MAP"},
		"mitre_atlas":  {Techniques: []string{"AML.T0025"}, Tactic: "Exfiltration"},
		"iso_42001":    {Controls: []string{"A.7.4"}, Clause: "Annex A"},
		"eu_ai_act":    {Articles: []string{"Article 10"}, RiskLevel: "high"},
		"soc2":         {Controls: []string{"CC6.1"}, Criteria: "Common Criteria"},
		"gdpr":         {Articles: []string{"Article 5", "Article 6"}, LawfulBasis: "legitimate_interest"},
		"ccpa":         {Sections: []string{"1798.100"}, Category: "personal_information"},
		"csa_aicm":     {Controls: []string{"DSP-01", "DSP-04", "CEK-03"}, Domain: "Data Security & Privacy"},
	},
	"security_finding": {
		"nist_ai_rmf": {Controls: []string{"MANAGE-2.4", "MANAGE-4.1"}, Function: "MANAGE"},
		"mitre_atlas":  {Techniques: []string{"AML.T0051"}, Tactic: "Initial Access"},
		"iso_42001":    {Controls: []string{"6.1.2", "A.6.2.4"}, Clause: "Planning"},
		"eu_ai_act":    {Articles: []string{"Article 9", "Article 62"}, RiskLevel: "high"},
		"soc2":         {Controls: []string{"CC7.2", "CC7.3"}, Criteria: "Common Criteria"},
		"gdpr":         {Articles: []string{"Article 32", "Article 33"}, LawfulBasis: "legal_obligation"},
		"ccpa":         {Sections: []string{"1798.150"}, Category: "breach"},
		"csa_aicm":     {Controls: []string{"SEF-03", "TVM-01", "LOG-04"}, Domain: "Security Incident Management"},
	},
	"supply_chain": {
		"nist_ai_rmf": {Controls: []string{"MAP-5.2", "GOVERN-6.1"}, Function: "GOVERN"},
		"mitre_atlas":  {Techniques: []string{"AML.T0010"}, Tactic: "Resource Development"},
		"iso_42001":    {Controls: []string{"A.6.2.3"}, Clause: "Annex A"},
		"eu_ai_act":    {Articles: []string{"Article 15", "Article 28"}, RiskLevel: "high"},
		"soc2":         {Controls: []string{"CC9.2"}, Criteria: "Common Criteria"},
		"gdpr":         {Articles: []string{"Article 28"}, LawfulBasis: "contractual"},
		"csa_aicm":     {Controls: []string{"STA-01", "STA-03", "CCC-01"}, Domain: "Supply Chain Management"},
	},
	"governance": {
		"nist_ai_rmf": {Controls: []string{"GOVERN-1.1", "MANAGE-1.3"}, Function: "GOVERN"},
		"iso_42001":    {Controls: []string{"5.1", "9.1"}, Clause: "Leadership"},
		"eu_ai_act":    {Articles: []string{"Article 9", "Article 61"}, RiskLevel: "high"},
		"soc2":         {Controls: []string{"CC1.2"}, Criteria: "Common Criteria"},
		"gdpr":         {Articles: []string{"Article 5"}, LawfulBasis: "legal_obligation"},
		"ccpa":         {Sections: []string{"1798.185"}, Category: "rulemaking"},
		"csa_aicm":     {Controls: []string{"GRC-01", "A&A-01", "LOG-01"}, Domain: "Governance, Risk & Compliance"},
	},
	"identity": {
		"nist_ai_rmf": {Controls: []string{"GOVERN-1.5", "MANAGE-2.1"}, Function: "GOVERN"},
		"mitre_atlas":  {Techniques: []string{"AML.T0052"}, Tactic: "Initial Access"},
		"iso_42001":    {Controls: []string{"A.6.2.6"}, Clause: "Annex A"},
		"eu_ai_act":    {Articles: []string{"Article 9"}, RiskLevel: "high"},
		"soc2":         {Controls: []string{"CC6.1", "CC6.2"}, Criteria: "Common Criteria"},
		"gdpr":         {Articles: []string{"Article 32"}, LawfulBasis: "legal_obligation"},
		"ccpa":         {Sections: []string{"1798.140"}, Category: "personal_information"},
		"csa_aicm":     {Controls: []string{"IAM-01", "IAM-02", "IAM-04"}, Domain: "Identity & Access Management"},
	},
}

// allFrameworks lists all supported compliance frameworks.
var allFrameworks = []string{
	"nist_ai_rmf", "mitre_atlas", "iso_42001",
	"eu_ai_act", "soc2", "gdpr", "ccpa", "csa_aicm",
}

// ComplianceProcessor maps AI telemetry events to compliance framework controls.
type ComplianceProcessor struct {
	frameworks []string
}

// NewComplianceProcessor creates a new compliance processor.
// If frameworks is nil or empty, all 8 frameworks are enabled.
func NewComplianceProcessor(frameworks []string) *ComplianceProcessor {
	if len(frameworks) == 0 {
		frameworks = append([]string{}, allFrameworks...)
	}
	return &ComplianceProcessor{frameworks: frameworks}
}

// GetComplianceMapping returns the compliance mapping for a given event type,
// filtered to only the configured frameworks.
func (c *ComplianceProcessor) GetComplianceMapping(eventType string) map[string]ComplianceFrameworkMapping {
	fullMapping, ok := ComplianceMappings[eventType]
	if !ok {
		return nil
	}

	result := make(map[string]ComplianceFrameworkMapping)
	for _, framework := range c.frameworks {
		if m, ok := fullMapping[framework]; ok {
			result[framework] = m
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// GetComplianceAttributes returns compliance data as OTel-compatible attributes
// suitable for span attributes.
func (c *ComplianceProcessor) GetComplianceAttributes(eventType string) []attribute.KeyValue {
	mapping := c.GetComplianceMapping(eventType)
	if mapping == nil {
		return nil
	}

	var attrs []attribute.KeyValue

	for framework, m := range mapping {
		switch framework {
		case "nist_ai_rmf":
			if len(m.Controls) > 0 {
				attrs = append(attrs, semconv.ComplianceNISTControlsKey.String(strings.Join(m.Controls, ",")))
			}
		case "mitre_atlas":
			if len(m.Techniques) > 0 {
				attrs = append(attrs, semconv.ComplianceMITRETechniquesKey.String(strings.Join(m.Techniques, ",")))
			}
		case "iso_42001":
			if len(m.Controls) > 0 {
				attrs = append(attrs, semconv.ComplianceISOControlsKey.String(strings.Join(m.Controls, ",")))
			}
		case "eu_ai_act":
			if len(m.Articles) > 0 {
				attrs = append(attrs, semconv.ComplianceEUArticlesKey.String(strings.Join(m.Articles, ",")))
			}
		case "soc2":
			if len(m.Controls) > 0 {
				attrs = append(attrs, semconv.ComplianceSOC2ControlsKey.String(strings.Join(m.Controls, ",")))
			}
		case "gdpr":
			if len(m.Articles) > 0 {
				attrs = append(attrs, semconv.ComplianceGDPRArticlesKey.String(strings.Join(m.Articles, ",")))
			}
		case "ccpa":
			if len(m.Sections) > 0 {
				attrs = append(attrs, semconv.ComplianceCCPASectionsKey.String(strings.Join(m.Sections, ",")))
			}
		case "csa_aicm":
			if len(m.Controls) > 0 {
				attrs = append(attrs, semconv.ComplianceCSAAICMControlsKey.String(strings.Join(m.Controls, ",")))
			}
		}
	}

	return attrs
}

// GetCoverageMatrix generates a coverage matrix showing which controls are
// mapped per event type for the configured frameworks.
func (c *ComplianceProcessor) GetCoverageMatrix() map[string]map[string][]string {
	matrix := make(map[string]map[string][]string)

	for eventType, frameworkMap := range ComplianceMappings {
		matrix[eventType] = make(map[string][]string)
		for _, framework := range c.frameworks {
			if m, ok := frameworkMap[framework]; ok {
				controls := m.PrimaryControls()
				if len(controls) > 0 {
					matrix[eventType][framework] = controls
				}
			}
		}
	}

	return matrix
}

// ClassifySpanName classifies a span name into an AI event type.
// Returns empty string if the span is not an AI-related span.
func ClassifySpanName(name string, attrKeys []string) string {
	if strings.HasPrefix(name, "chat ") || strings.HasPrefix(name, "embeddings ") {
		return "model_inference"
	}
	for _, k := range attrKeys {
		if k == string(semconv.GenAISystemKey) {
			return "model_inference"
		}
	}
	if strings.HasPrefix(name, "agent.") {
		return "agent_activity"
	}
	if strings.HasPrefix(name, "mcp.tool.") || strings.HasPrefix(name, "skill.invoke") {
		return "tool_execution"
	}
	if strings.HasPrefix(name, "rag.") || strings.HasPrefix(name, "mcp.resource.") {
		return "data_retrieval"
	}
	for _, k := range attrKeys {
		if strings.HasPrefix(k, "aitf.security.") {
			return "security_finding"
		}
	}
	return ""
}
