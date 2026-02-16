/**
 * AITF Compliance Mapper.
 *
 * Maps AI events to eight regulatory framework controls.
 * Based on the compliance mapper from the AITelemetry project.
 */

import { randomUUID } from "crypto";
import { AIBaseEvent, ComplianceMetadata } from "./schema";

/** Full compliance mappings (reused from AITelemetry project). */
export const FRAMEWORK_MAPPINGS: Record<
  string,
  Record<string, Record<string, unknown>>
> = {
  nist_ai_rmf: {
    model_inference: {
      controls: ["MAP-1.1", "MEASURE-2.5"],
      function: "MAP",
    },
    agent_activity: {
      controls: ["GOVERN-1.2", "MANAGE-3.1"],
      function: "GOVERN",
    },
    tool_execution: {
      controls: ["MAP-3.5", "MANAGE-4.2"],
      function: "MANAGE",
    },
    data_retrieval: {
      controls: ["MAP-1.5", "MEASURE-2.7"],
      function: "MAP",
    },
    security_finding: {
      controls: ["MANAGE-2.4", "MANAGE-4.1"],
      function: "MANAGE",
    },
    supply_chain: {
      controls: ["MAP-5.2", "GOVERN-6.1"],
      function: "GOVERN",
    },
    governance: {
      controls: ["GOVERN-1.1", "MANAGE-1.3"],
      function: "GOVERN",
    },
    identity: {
      controls: ["GOVERN-1.5", "MANAGE-2.1"],
      function: "GOVERN",
    },
  },
  mitre_atlas: {
    model_inference: {
      techniques: ["AML.T0040"],
      tactic: "ML Attack Staging",
    },
    agent_activity: {
      techniques: ["AML.T0048"],
      tactic: "ML Attack Staging",
    },
    tool_execution: {
      techniques: ["AML.T0043"],
      tactic: "ML Attack Staging",
    },
    data_retrieval: {
      techniques: ["AML.T0025"],
      tactic: "Exfiltration",
    },
    security_finding: {
      techniques: ["AML.T0051"],
      tactic: "Initial Access",
    },
    supply_chain: {
      techniques: ["AML.T0010"],
      tactic: "Resource Development",
    },
    identity: {
      techniques: ["AML.T0052"],
      tactic: "Initial Access",
    },
  },
  iso_42001: {
    model_inference: { controls: ["6.1.4", "8.4"], clause: "Operation" },
    agent_activity: {
      controls: ["8.2", "A.6.2.5"],
      clause: "Operation",
    },
    tool_execution: { controls: ["A.6.2.7"], clause: "Annex A" },
    data_retrieval: { controls: ["A.7.4"], clause: "Annex A" },
    security_finding: {
      controls: ["6.1.2", "A.6.2.4"],
      clause: "Planning",
    },
    supply_chain: { controls: ["A.6.2.3"], clause: "Annex A" },
    governance: { controls: ["5.1", "9.1"], clause: "Leadership" },
    identity: { controls: ["A.6.2.6"], clause: "Annex A" },
  },
  eu_ai_act: {
    model_inference: {
      articles: ["Article 13", "Article 15"],
      risk_level: "high",
    },
    agent_activity: {
      articles: ["Article 14", "Article 52"],
      risk_level: "high",
    },
    tool_execution: { articles: ["Article 9"], risk_level: "high" },
    data_retrieval: { articles: ["Article 10"], risk_level: "high" },
    security_finding: {
      articles: ["Article 9", "Article 62"],
      risk_level: "high",
    },
    supply_chain: {
      articles: ["Article 15", "Article 28"],
      risk_level: "high",
    },
    governance: {
      articles: ["Article 9", "Article 61"],
      risk_level: "high",
    },
    identity: { articles: ["Article 9"], risk_level: "high" },
  },
  soc2: {
    model_inference: {
      controls: ["CC6.1"],
      criteria: "Common Criteria",
    },
    agent_activity: {
      controls: ["CC7.2"],
      criteria: "Common Criteria",
    },
    tool_execution: {
      controls: ["CC6.3"],
      criteria: "Common Criteria",
    },
    data_retrieval: {
      controls: ["CC6.1"],
      criteria: "Common Criteria",
    },
    security_finding: {
      controls: ["CC7.2", "CC7.3"],
      criteria: "Common Criteria",
    },
    supply_chain: {
      controls: ["CC9.2"],
      criteria: "Common Criteria",
    },
    governance: {
      controls: ["CC1.2"],
      criteria: "Common Criteria",
    },
    identity: {
      controls: ["CC6.1", "CC6.2"],
      criteria: "Common Criteria",
    },
  },
  gdpr: {
    model_inference: {
      articles: ["Article 5", "Article 22"],
      lawful_basis: "legitimate_interest",
    },
    agent_activity: {
      articles: ["Article 22"],
      lawful_basis: "legitimate_interest",
    },
    tool_execution: {
      articles: ["Article 25"],
      lawful_basis: "legitimate_interest",
    },
    data_retrieval: {
      articles: ["Article 5", "Article 6"],
      lawful_basis: "legitimate_interest",
    },
    security_finding: {
      articles: ["Article 32", "Article 33"],
      lawful_basis: "legal_obligation",
    },
    supply_chain: {
      articles: ["Article 28"],
      lawful_basis: "contractual",
    },
    governance: {
      articles: ["Article 5"],
      lawful_basis: "legal_obligation",
    },
    identity: {
      articles: ["Article 32"],
      lawful_basis: "legal_obligation",
    },
  },
  ccpa: {
    model_inference: {
      sections: ["1798.100"],
      category: "personal_information",
    },
    data_retrieval: {
      sections: ["1798.100"],
      category: "personal_information",
    },
    security_finding: { sections: ["1798.150"], category: "breach" },
    governance: { sections: ["1798.185"], category: "rulemaking" },
    identity: {
      sections: ["1798.140"],
      category: "personal_information",
    },
  },
  csa_aicm: {
    model_inference: {
      controls: ["AIS-04", "MDS-01", "LOG-07"],
      domain: "Model Security",
    },
    agent_activity: {
      controls: ["AIS-02", "MDS-05", "GRC-02"],
      domain: "Governance, Risk & Compliance",
    },
    tool_execution: {
      controls: ["AIS-01", "AIS-04", "LOG-05"],
      domain: "Application & Interface Security",
    },
    data_retrieval: {
      controls: ["DSP-01", "DSP-04", "CEK-03"],
      domain: "Data Security & Privacy",
    },
    security_finding: {
      controls: ["SEF-03", "TVM-01", "LOG-04"],
      domain: "Security Incident Management",
    },
    supply_chain: {
      controls: ["STA-01", "STA-03", "CCC-01"],
      domain: "Supply Chain Management",
    },
    governance: {
      controls: ["GRC-01", "A&A-01", "LOG-01"],
      domain: "Governance, Risk & Compliance",
    },
    identity: {
      controls: ["IAM-01", "IAM-02", "IAM-04"],
      domain: "Identity & Access Management",
    },
  },
};

/**
 * Maps AI events to compliance framework controls.
 *
 * Usage:
 *   const mapper = new ComplianceMapper({ frameworks: ["nist_ai_rmf", "eu_ai_act"] });
 *   const compliance = mapper.mapEvent("model_inference");
 */
export class ComplianceMapper {
  private readonly _frameworks: string[];

  constructor(options: { frameworks?: string[] } = {}) {
    const allFrameworks = Object.keys(FRAMEWORK_MAPPINGS);
    this._frameworks = options.frameworks ?? allFrameworks;
  }

  /**
   * Map an event type to compliance frameworks.
   *
   * @param eventType - One of 'model_inference', 'agent_activity',
   *   'tool_execution', 'data_retrieval', 'security_finding',
   *   'supply_chain', 'governance', 'identity'.
   * @returns ComplianceMetadata with active framework mappings.
   */
  mapEvent(eventType: string): ComplianceMetadata {
    const result: Record<string, unknown> = {};

    for (const framework of this._frameworks) {
      const frameworkMap = FRAMEWORK_MAPPINGS[framework];
      if (!frameworkMap) continue;
      const eventMap = frameworkMap[eventType];
      if (eventMap) {
        result[framework] = eventMap;
      }
    }

    return result as ComplianceMetadata;
  }

  /** Add compliance metadata to an OCSF event. */
  enrichEvent(event: AIBaseEvent, eventType: string): AIBaseEvent {
    event.compliance = this.mapEvent(eventType);
    return event;
  }

  /** Generate a coverage matrix showing which controls apply per event type. */
  getCoverageMatrix(): Record<string, Record<string, string[]>> {
    const eventTypes = [
      "model_inference",
      "agent_activity",
      "tool_execution",
      "data_retrieval",
      "security_finding",
      "supply_chain",
      "governance",
      "identity",
    ];

    const matrix: Record<string, Record<string, string[]>> = {};

    for (const eventType of eventTypes) {
      matrix[eventType] = {};
      for (const framework of this._frameworks) {
        const frameworkMap = FRAMEWORK_MAPPINGS[framework];
        if (!frameworkMap) continue;
        const eventMap = frameworkMap[eventType];
        if (!eventMap) continue;

        for (const key of [
          "controls",
          "techniques",
          "articles",
          "sections",
        ]) {
          if (key in eventMap) {
            matrix[eventType][framework] = eventMap[key] as string[];
            break;
          }
        }
      }
    }

    return matrix;
  }

  /**
   * Generate an audit record from compliance mappings.
   */
  generateAuditRecord(
    eventType: string,
    options: {
      actor?: string;
      model?: string;
      riskScore?: number;
    } = {}
  ): Record<string, unknown> {
    const compliance = this.mapEvent(eventType);
    const allControls: string[] = [];
    const complianceDetails: Record<string, string[]> = {};

    for (const framework of this._frameworks) {
      const frameworkData = (compliance as Record<string, unknown>)[
        framework
      ] as Record<string, unknown> | undefined;
      if (!frameworkData) continue;

      for (const key of [
        "controls",
        "techniques",
        "articles",
        "sections",
      ]) {
        if (key in frameworkData) {
          const controls = frameworkData[key] as string[];
          complianceDetails[framework] = controls;
          allControls.push(...controls);
          break;
        }
      }
    }

    return {
      audit_id: `aud-${randomUUID().replace(/-/g, "").slice(0, 12)}`,
      timestamp: new Date().toISOString(),
      event_type: eventType,
      frameworks_mapped: Object.keys(complianceDetails).length,
      controls_mapped: allControls.length,
      violations_detected: 0,
      risk_score: options.riskScore ?? 0.0,
      actor: options.actor ?? null,
      model: options.model ?? null,
      compliance_details: complianceDetails,
    };
  }
}
