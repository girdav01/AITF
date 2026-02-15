/**
 * AITF OCSF Exporter.
 *
 * OTel SpanExporter that converts AI spans to OCSF Category 7 events
 * and exports them to SIEM/XDR endpoints, S3, or local files.
 *
 * Based on forwarder architecture from the AITelemetry project.
 */

import * as fs from "fs";
import * as path from "path";
import * as http from "http";
import * as https from "https";
import { URL } from "url";
import { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import { ExportResult, ExportResultCode } from "@opentelemetry/core";
import { OCSFMapper } from "../ocsf/mapper";
import { ComplianceMapper } from "../ocsf/compliance-mapper";
import { AIBaseEvent, stripNulls } from "../ocsf/schema";

/** Options for configuring the OCSFExporter. */
export interface OCSFExporterOptions {
  endpoint?: string;
  outputFile?: string;
  complianceFrameworks?: string[];
  includeRawSpan?: boolean;
  apiKey?: string;
}

/**
 * Exports OTel spans as OCSF Category 7 AI events.
 *
 * Converts AI-related spans to OCSF events using OCSFMapper,
 * enriches with compliance metadata, and exports to configured
 * destinations.
 *
 * Usage:
 *   import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
 *   import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
 *   import { OCSFExporter } from "@aitf/sdk";
 *
 *   const exporter = new OCSFExporter({
 *     outputFile: "/var/log/aitf/ocsf_events.jsonl",
 *     complianceFrameworks: ["nist_ai_rmf", "eu_ai_act", "mitre_atlas"],
 *   });
 *   const provider = new NodeTracerProvider();
 *   provider.addSpanProcessor(new BatchSpanProcessor(exporter));
 */
export class OCSFExporter implements SpanExporter {
  private readonly _endpoint: string | null;
  private readonly _outputFile: string | null;
  private readonly _includeRawSpan: boolean;
  private readonly _apiKey: string | null;
  private readonly _mapper: OCSFMapper;
  private readonly _complianceMapper: ComplianceMapper;
  private _eventCount = 0;

  constructor(options: OCSFExporterOptions = {}) {
    this._endpoint = options.endpoint ?? null;
    this._outputFile = options.outputFile ?? null;
    this._includeRawSpan = options.includeRawSpan ?? false;
    this._apiKey = options.apiKey ?? null;
    this._mapper = new OCSFMapper();
    this._complianceMapper = new ComplianceMapper({
      frameworks: options.complianceFrameworks,
    });

    // Ensure output directory exists
    if (this._outputFile) {
      const dir = path.dirname(this._outputFile);
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  /**
   * Export spans as OCSF events.
   */
  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void
  ): void {
    const events: Record<string, unknown>[] = [];

    for (const span of spans) {
      const ocsfEvent = this._mapper.mapSpan(span);
      if (!ocsfEvent) {
        continue;
      }

      // Enrich with compliance metadata
      const eventType = this._classifyEvent(ocsfEvent);
      if (eventType) {
        this._complianceMapper.enrichEvent(ocsfEvent, eventType);
      }

      const eventDict = stripNulls(
        ocsfEvent as unknown as Record<string, unknown>
      );
      events.push(eventDict);
      this._eventCount++;
    }

    if (events.length === 0) {
      resultCallback({ code: ExportResultCode.SUCCESS });
      return;
    }

    try {
      if (this._outputFile) {
        this._exportToFile(events);
      }
      if (this._endpoint) {
        this._exportToEndpoint(events)
          .then(() => {
            resultCallback({ code: ExportResultCode.SUCCESS });
          })
          .catch((err) => {
            console.error("OCSF export to endpoint failed:", err);
            resultCallback({ code: ExportResultCode.FAILED });
          });
        return;
      }
      resultCallback({ code: ExportResultCode.SUCCESS });
    } catch (err) {
      console.error("OCSF export failed:", err);
      resultCallback({ code: ExportResultCode.FAILED });
    }
  }

  private _exportToFile(events: Record<string, unknown>[]): void {
    if (!this._outputFile) return;

    const lines = events
      .map((event) => JSON.stringify(event))
      .join("\n");
    fs.appendFileSync(this._outputFile, lines + "\n", "utf-8");
  }

  private async _exportToEndpoint(
    events: Record<string, unknown>[]
  ): Promise<void> {
    if (!this._endpoint) return;

    const url = new URL(this._endpoint);
    const payload = JSON.stringify(events);
    const isHttps = url.protocol === "https:";

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Content-Length": String(Buffer.byteLength(payload)),
    };
    if (this._apiKey) {
      headers["Authorization"] = `Bearer ${this._apiKey}`;
    }

    return new Promise<void>((resolve, reject) => {
      const requestOptions: http.RequestOptions = {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method: "POST",
        headers,
        timeout: 30000,
      };

      const transport = isHttps ? https : http;
      const req = transport.request(requestOptions, (res) => {
        if (res.statusCode && res.statusCode >= 400) {
          reject(
            new Error(
              `OCSF endpoint returned ${res.statusCode}`
            )
          );
        } else {
          resolve();
        }
        // Consume response data to free memory
        res.resume();
      });

      req.on("error", reject);
      req.on("timeout", () => {
        req.destroy();
        reject(new Error("OCSF endpoint request timed out"));
      });

      req.write(payload);
      req.end();
    });
  }

  private _classifyEvent(event: AIBaseEvent): string | null {
    const classUid = event.class_uid;
    const mapping: Record<number, string> = {
      7001: "model_inference",
      7002: "agent_activity",
      7003: "tool_execution",
      7004: "data_retrieval",
      7005: "security_finding",
      7006: "supply_chain",
      7007: "governance",
      7008: "identity",
    };
    return mapping[classUid] ?? null;
  }

  /** Number of OCSF events exported. */
  get eventCount(): number {
    return this._eventCount;
  }

  async shutdown(): Promise<void> {
    // No-op
  }

  async forceFlush(): Promise<void> {
    // No-op
  }
}
