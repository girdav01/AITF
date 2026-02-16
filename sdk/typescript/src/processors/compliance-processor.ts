/**
 * AITF Compliance Processor.
 *
 * OTel SpanProcessor that maps AI telemetry events to compliance framework controls.
 * Supports NIST AI RMF, MITRE ATLAS, ISO 42001, EU AI Act, SOC 2, GDPR, CCPA,
 * and CSA AI Controls Matrix (AICM).
 *
 * Based on compliance mapping from AITelemetry project.
 */

import { Context, Span } from "@opentelemetry/api";
import { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { ComplianceAttributes } from "../semantic-conventions/attributes";

/** Compliance mapping data structure. */
export interface ComplianceMapping {
  controls?: string[];
  techniques?: string[];
  articles?: string[];
  sections?: string[];
  function?: string;
  tactic?: string;
  clause?: string;
  risk_level?: string;
  criteria?: string;
  lawful_basis?: string;
  category?: string;
  domain?: string;
}

/** Compliance mappings by AI event type. */
export const COMPLIANCE_MAPPINGS: Record<
  string,
  Record<string, ComplianceMapping>
> = {
  model_inference: {
    nist_ai_rmf: { controls: ["MAP-1.1", "MEASURE-2.5"], function: "MAP" },
    mitre_atlas: {
      techniques: ["AML.T0040"],
      tactic: "ML Attack Staging",
    },
    iso_42001: { controls: ["6.1.4", "8.4"], clause: "Operation" },
    eu_ai_act: {
      articles: ["Article 13", "Article 15"],
      risk_level: "high",
    },
    soc2: { controls: ["CC6.1"], criteria: "Common Criteria" },
    gdpr: {
      articles: ["Article 5", "Article 22"],
      lawful_basis: "legitimate_interest",
    },
    ccpa: { sections: ["1798.100"], category: "personal_information" },
    csa_aicm: { controls: ["AIS-04", "MDS-01", "LOG-07"], domain: "Model Security" },
  },
  agent_activity: {
    nist_ai_rmf: {
      controls: ["GOVERN-1.2", "MANAGE-3.1"],
      function: "GOVERN",
    },
    mitre_atlas: {
      techniques: ["AML.T0048"],
      tactic: "ML Attack Staging",
    },
    iso_42001: { controls: ["8.2", "A.6.2.5"], clause: "Operation" },
    eu_ai_act: {
      articles: ["Article 14", "Article 52"],
      risk_level: "high",
    },
    soc2: { controls: ["CC7.2"], criteria: "Common Criteria" },
    gdpr: {
      articles: ["Article 22"],
      lawful_basis: "legitimate_interest",
    },
    csa_aicm: { controls: ["AIS-02", "MDS-05", "GRC-02"], domain: "Governance, Risk & Compliance" },
  },
  tool_execution: {
    nist_ai_rmf: {
      controls: ["MAP-3.5", "MANAGE-4.2"],
      function: "MANAGE",
    },
    mitre_atlas: {
      techniques: ["AML.T0043"],
      tactic: "ML Attack Staging",
    },
    iso_42001: { controls: ["A.6.2.7"], clause: "Annex A" },
    eu_ai_act: { articles: ["Article 9"], risk_level: "high" },
    soc2: { controls: ["CC6.3"], criteria: "Common Criteria" },
    gdpr: {
      articles: ["Article 25"],
      lawful_basis: "legitimate_interest",
    },
    csa_aicm: { controls: ["AIS-01", "AIS-04", "LOG-05"], domain: "Application & Interface Security" },
  },
  data_retrieval: {
    nist_ai_rmf: {
      controls: ["MAP-1.5", "MEASURE-2.7"],
      function: "MAP",
    },
    mitre_atlas: { techniques: ["AML.T0025"], tactic: "Exfiltration" },
    iso_42001: { controls: ["A.7.4"], clause: "Annex A" },
    eu_ai_act: { articles: ["Article 10"], risk_level: "high" },
    soc2: { controls: ["CC6.1"], criteria: "Common Criteria" },
    gdpr: {
      articles: ["Article 5", "Article 6"],
      lawful_basis: "legitimate_interest",
    },
    ccpa: { sections: ["1798.100"], category: "personal_information" },
    csa_aicm: { controls: ["DSP-01", "DSP-04", "CEK-03"], domain: "Data Security & Privacy" },
  },
  security_finding: {
    nist_ai_rmf: {
      controls: ["MANAGE-2.4", "MANAGE-4.1"],
      function: "MANAGE",
    },
    mitre_atlas: { techniques: ["AML.T0051"], tactic: "Initial Access" },
    iso_42001: { controls: ["6.1.2", "A.6.2.4"], clause: "Planning" },
    eu_ai_act: {
      articles: ["Article 9", "Article 62"],
      risk_level: "high",
    },
    soc2: { controls: ["CC7.2", "CC7.3"], criteria: "Common Criteria" },
    gdpr: {
      articles: ["Article 32", "Article 33"],
      lawful_basis: "legal_obligation",
    },
    ccpa: { sections: ["1798.150"], category: "breach" },
    csa_aicm: { controls: ["SEF-03", "TVM-01", "LOG-04"], domain: "Security Incident Management" },
  },
  supply_chain: {
    nist_ai_rmf: {
      controls: ["MAP-5.2", "GOVERN-6.1"],
      function: "GOVERN",
    },
    mitre_atlas: {
      techniques: ["AML.T0010"],
      tactic: "Resource Development",
    },
    iso_42001: { controls: ["A.6.2.3"], clause: "Annex A" },
    eu_ai_act: {
      articles: ["Article 15", "Article 28"],
      risk_level: "high",
    },
    soc2: { controls: ["CC9.2"], criteria: "Common Criteria" },
    gdpr: { articles: ["Article 28"], lawful_basis: "contractual" },
    csa_aicm: { controls: ["STA-01", "STA-03", "CCC-01"], domain: "Supply Chain Management" },
  },
  governance: {
    nist_ai_rmf: {
      controls: ["GOVERN-1.1", "MANAGE-1.3"],
      function: "GOVERN",
    },
    iso_42001: { controls: ["5.1", "9.1"], clause: "Leadership" },
    eu_ai_act: {
      articles: ["Article 9", "Article 61"],
      risk_level: "high",
    },
    soc2: { controls: ["CC1.2"], criteria: "Common Criteria" },
    gdpr: { articles: ["Article 5"], lawful_basis: "legal_obligation" },
    ccpa: { sections: ["1798.185"], category: "rulemaking" },
    csa_aicm: { controls: ["GRC-01", "A&A-01", "LOG-01"], domain: "Governance, Risk & Compliance" },
  },
  identity: {
    nist_ai_rmf: {
      controls: ["GOVERN-1.5", "MANAGE-2.1"],
      function: "GOVERN",
    },
    mitre_atlas: { techniques: ["AML.T0052"], tactic: "Initial Access" },
    iso_42001: { controls: ["A.6.2.6"], clause: "Annex A" },
    eu_ai_act: { articles: ["Article 9"], risk_level: "high" },
    soc2: { controls: ["CC6.1", "CC6.2"], criteria: "Common Criteria" },
    gdpr: { articles: ["Article 32"], lawful_basis: "legal_obligation" },
    ccpa: { sections: ["1798.140"], category: "personal_information" },
    csa_aicm: { controls: ["IAM-01", "IAM-02", "IAM-04"], domain: "Identity & Access Management" },
  },
};

/** Options for configuring the ComplianceProcessor. */
export interface ComplianceProcessorOptions {
  frameworks?: string[];
}

/**
 * OTel SpanProcessor that adds compliance framework mappings to AI spans.
 *
 * Usage:
 *   provider.addSpanProcessor(new ComplianceProcessor({
 *     frameworks: ["nist_ai_rmf", "mitre_atlas", "eu_ai_act"],
 *   }));
 */
export class ComplianceProcessor implements SpanProcessor {
  private readonly _frameworks: string[];

  constructor(options: ComplianceProcessorOptions = {}) {
    const allFrameworks = [
      "nist_ai_rmf",
      "mitre_atlas",
      "iso_42001",
      "eu_ai_act",
      "soc2",
      "gdpr",
      "ccpa",
      "csa_aicm",
    ];
    this._frameworks = options.frameworks ?? allFrameworks;
  }

  onStart(_span: Span, _parentContext: Context): void {
    // No-op
  }

  onEnd(span: ReadableSpan): void {
    const eventType = this._classifyEvent(span);
    if (!eventType) {
      return;
    }

    // Get compliance mapping (for ReadableSpan, data is immutable;
    // in practice use wrapping mechanism or pre-on_start)
    this.getComplianceMapping(eventType);
  }

  /**
   * Get compliance mapping for a given event type.
   * Returns map of framework -> controls.
   */
  getComplianceMapping(
    eventType: string
  ): Record<string, ComplianceMapping> {
    const fullMapping = COMPLIANCE_MAPPINGS[eventType];
    if (!fullMapping) {
      return {};
    }

    const result: Record<string, ComplianceMapping> = {};
    const activeFrameworks: string[] = [];

    for (const framework of this._frameworks) {
      if (fullMapping[framework]) {
        result[framework] = fullMapping[framework];
        activeFrameworks.push(framework);
      }
    }

    if (activeFrameworks.length > 0) {
      result["_frameworks"] = {
        controls: activeFrameworks,
      };
    }

    return result;
  }

  /**
   * Get compliance attributes suitable for span attributes.
   */
  getComplianceAttributes(
    eventType: string
  ): Record<string, string | string[]> {
    const mapping = this.getComplianceMapping(eventType);
    if (Object.keys(mapping).length === 0) {
      return {};
    }

    const attributes: Record<string, string | string[]> = {};
    const frameworksMeta = mapping["_frameworks"];
    if (frameworksMeta?.controls) {
      attributes[ComplianceAttributes.FRAMEWORKS] = frameworksMeta.controls;
    }

    for (const [framework, controls] of Object.entries(mapping)) {
      if (framework === "_frameworks") continue;

      if (framework === "nist_ai_rmf" && controls.controls) {
        attributes[ComplianceAttributes.NIST_AI_RMF_CONTROLS] =
          controls.controls;
      } else if (framework === "mitre_atlas" && controls.techniques) {
        attributes[ComplianceAttributes.MITRE_ATLAS_TECHNIQUES] =
          controls.techniques;
      } else if (framework === "iso_42001" && controls.controls) {
        attributes[ComplianceAttributes.ISO_42001_CONTROLS] =
          controls.controls;
      } else if (framework === "eu_ai_act" && controls.articles) {
        attributes[ComplianceAttributes.EU_AI_ACT_ARTICLES] =
          controls.articles;
      } else if (framework === "soc2" && controls.controls) {
        attributes[ComplianceAttributes.SOC2_CONTROLS] = controls.controls;
      } else if (framework === "gdpr" && controls.articles) {
        attributes[ComplianceAttributes.GDPR_ARTICLES] = controls.articles;
      } else if (framework === "ccpa" && controls.sections) {
        attributes[ComplianceAttributes.CCPA_SECTIONS] = controls.sections;
      } else if (framework === "csa_aicm" && controls.controls) {
        attributes[ComplianceAttributes.CSA_AICM_CONTROLS] = controls.controls;
      }
    }

    return attributes;
  }

  /**
   * Generate a coverage matrix showing which controls are mapped per event type.
   */
  getCoverageMatrix(): Record<string, Record<string, string[]>> {
    const matrix: Record<string, Record<string, string[]>> = {};

    for (const [eventType, mapping] of Object.entries(COMPLIANCE_MAPPINGS)) {
      matrix[eventType] = {};
      for (const framework of this._frameworks) {
        if (mapping[framework]) {
          const controls = mapping[framework];
          const key =
            "controls" in controls
              ? "controls"
              : "techniques" in controls
                ? "techniques"
                : "articles" in controls
                  ? "articles"
                  : "sections";
          const values = controls[key as keyof ComplianceMapping];
          if (Array.isArray(values)) {
            matrix[eventType][framework] = values;
          }
        }
      }
    }

    return matrix;
  }

  private _classifyEvent(span: ReadableSpan): string | null {
    const name = span.name ?? "";
    const attrs = span.attributes ?? {};

    if (name.startsWith("chat ") || name.startsWith("embeddings ")) {
      return "model_inference";
    }
    if ("gen_ai.system" in attrs) {
      return "model_inference";
    }
    if (name.startsWith("agent.")) {
      return "agent_activity";
    }
    if (name.startsWith("mcp.tool.") || name.startsWith("skill.invoke")) {
      return "tool_execution";
    }
    if (name.startsWith("rag.") || name.startsWith("mcp.resource.")) {
      return "data_retrieval";
    }
    if (
      Object.keys(attrs).some((k) => String(k).includes("aitf.security."))
    ) {
      return "security_finding";
    }

    return null;
  }

  async shutdown(): Promise<void> {
    // No-op
  }

  async forceFlush(): Promise<void> {
    // No-op
  }
}
