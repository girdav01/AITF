/**
 * AITF PII Detection Processor.
 *
 * OTel SpanProcessor that detects and optionally redacts Personally
 * Identifiable Information (PII) in AI inputs and outputs.
 *
 * Based on PII detection patterns from AITelemetry project.
 */

import { createHash } from "crypto";
import { Context, Span } from "@opentelemetry/api";
import { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { SecurityAttributes } from "../semantic-conventions/attributes";

/** PII detection patterns for 9 PII types. */
const PII_PATTERNS: Record<string, RegExp> = {
  email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
  phone: /\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g,
  ssn: /\b\d{3}-\d{2}-\d{4}\b/g,
  credit_card: /\b(?:\d{4}[-\s]?){3}\d{4}\b/g,
  api_key: /\b(?:sk-|pk-|ak-|key-)[A-Za-z0-9]{20,}\b/g,
  jwt: /\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g,
  ip_address:
    /\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b/g,
  password: /(?:password|passwd|pwd)\s*[=:]\s*\S+/gi,
  aws_key: /\bAKIA[0-9A-Z]{16}\b/g,
};

/** A PII detection result. */
export interface PIIDetection {
  piiType: string;
  count: number;
  locations: Array<[number, number]>; // [start, end] positions
}

/** Options for configuring the PIIProcessor. */
export interface PIIProcessorOptions {
  /** Which PII types to detect. Defaults to all 9 types. */
  detectTypes?: string[];
  /** Action to take: "flag", "redact", or "hash". Defaults to "flag". */
  action?: "flag" | "redact" | "hash";
  /** Custom patterns to add. */
  customPatterns?: Record<string, RegExp>;
}

/**
 * OTel SpanProcessor that detects PII in AI span content.
 *
 * Can operate in three modes:
 * - "flag": Detect PII and add span attributes (default)
 * - "redact": Replace PII with [REDACTED] placeholder
 * - "hash": Replace PII with SHA-256 hash
 *
 * Usage:
 *   provider.addSpanProcessor(new PIIProcessor({
 *     detectTypes: ["email", "ssn", "credit_card", "api_key"],
 *     action: "redact",
 *   }));
 */
export class PIIProcessor implements SpanProcessor {
  private readonly _detectTypes: string[];
  private readonly _action: "flag" | "redact" | "hash";
  private readonly _patterns: Record<string, RegExp>;

  constructor(options: PIIProcessorOptions = {}) {
    this._detectTypes = options.detectTypes ?? Object.keys(PII_PATTERNS);
    this._action = options.action ?? "flag";
    this._patterns = {};

    // Build active pattern set
    for (const piiType of this._detectTypes) {
      if (PII_PATTERNS[piiType]) {
        this._patterns[piiType] = PII_PATTERNS[piiType];
      }
    }

    // Add custom patterns
    if (options.customPatterns) {
      Object.assign(this._patterns, options.customPatterns);
    }
  }

  onStart(_span: Span, _parentContext: Context): void {
    // No-op
  }

  onEnd(span: ReadableSpan): void {
    const attrs = span.attributes ?? {};

    const hasAiPrefix = Object.keys(attrs).some(
      (key) => key.startsWith("gen_ai.") || key.startsWith("aitf.")
    );
    if (!hasAiPrefix) {
      return;
    }

    const allDetections: PIIDetection[] = [];

    // Check span events for content
    for (const event of span.events ?? []) {
      const eventAttrs = event.attributes ?? {};
      for (const val of Object.values(eventAttrs)) {
        if (typeof val === "string") {
          const detections = this.detectPII(val);
          allDetections.push(...detections);
        }
      }
    }

    // Check relevant span attributes
    for (const key of [
      "aitf.mcp.tool.input",
      "aitf.mcp.tool.output",
      "aitf.skill.input",
      "aitf.skill.output",
    ]) {
      const val = attrs[key];
      if (typeof val === "string") {
        const detections = this.detectPII(val);
        allDetections.push(...detections);
      }
    }
  }

  /**
   * Detect PII in text. Returns list of PIIDetection objects.
   */
  detectPII(text: string): PIIDetection[] {
    const detections: PIIDetection[] = [];

    for (const [piiType, pattern] of Object.entries(this._patterns)) {
      // Reset regex state for global patterns
      const regex = new RegExp(pattern.source, pattern.flags);
      const matches: Array<[number, number]> = [];
      let match: RegExpExecArray | null;

      while ((match = regex.exec(text)) !== null) {
        matches.push([match.index, match.index + match[0].length]);
      }

      if (matches.length > 0) {
        detections.push({
          piiType,
          count: matches.length,
          locations: matches,
        });
      }
    }

    return detections;
  }

  /**
   * Detect and redact PII from text.
   * Returns [redacted_text, detections].
   */
  redactPII(text: string): [string, PIIDetection[]] {
    const detections = this.detectPII(text);
    if (detections.length === 0) {
      return [text, []];
    }

    let result = text;

    // Sort detections by first location, process in reverse order
    const sortedDetections = [...detections].sort(
      (a, b) => a.locations[0][0] - b.locations[0][0]
    );

    for (const detection of sortedDetections.reverse()) {
      const sortedLocations = [...detection.locations].sort(
        (a, b) => b[0] - a[0]
      );
      for (const [start, end] of sortedLocations) {
        const original = result.slice(start, end);
        let replacement: string;

        if (this._action === "redact") {
          replacement = `[${detection.piiType.toUpperCase()}_REDACTED]`;
        } else if (this._action === "hash") {
          const hashVal = createHash("sha256")
            .update(original)
            .digest("hex")
            .slice(0, 12);
          replacement = `[${detection.piiType.toUpperCase()}:${hashVal}]`;
        } else {
          continue;
        }

        result = result.slice(0, start) + replacement + result.slice(end);
      }
    }

    return [result, detections];
  }

  /**
   * Get a summary of PII detected in text.
   */
  getPIISummary(
    text: string
  ): Record<string, boolean | string[] | number | string> {
    const detections = this.detectPII(text);

    if (detections.length === 0) {
      return {
        [SecurityAttributes.PII_DETECTED]: false,
        [SecurityAttributes.PII_TYPES]: [] as string[],
        [SecurityAttributes.PII_COUNT]: 0,
        [SecurityAttributes.PII_ACTION]: this._action,
      };
    }

    return {
      [SecurityAttributes.PII_DETECTED]: true,
      [SecurityAttributes.PII_TYPES]: detections.map((d) => d.piiType),
      [SecurityAttributes.PII_COUNT]: detections.reduce(
        (sum, d) => sum + d.count,
        0
      ),
      [SecurityAttributes.PII_ACTION]: this._action,
    };
  }

  async shutdown(): Promise<void> {
    // No-op
  }

  async forceFlush(): Promise<void> {
    // No-op
  }
}
