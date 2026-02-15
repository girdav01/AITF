"""AITF PII Detection Processor.

OTel SpanProcessor that detects and optionally redacts Personally
Identifiable Information (PII) in AI inputs and outputs.

Based on PII detection patterns from AITelemetry project.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from aitf.semantic_conventions.attributes import SecurityAttributes

# PII detection patterns
PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    "api_key": re.compile(
        r"\b(?:sk-|pk-|ak-|key-)[A-Za-z0-9]{20,}\b"
    ),
    "jwt": re.compile(
        r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
    ),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "password": re.compile(
        r"(?:password|passwd|pwd)\s*[=:]\s*\S+", re.IGNORECASE
    ),
    "aws_key": re.compile(
        r"\bAKIA[0-9A-Z]{16}\b"
    ),
}


@dataclass
class PIIDetection:
    """A PII detection result."""

    pii_type: str
    count: int
    locations: list[tuple[int, int]]  # (start, end) positions


class PIIProcessor(SpanProcessor):
    """OTel SpanProcessor that detects PII in AI span content.

    Can operate in three modes:
    - "flag": Detect PII and add span attributes (default)
    - "redact": Replace PII with [REDACTED] placeholder
    - "hash": Replace PII with SHA-256 hash

    Usage:
        provider.add_span_processor(PIIProcessor(
            detect_types=["email", "ssn", "credit_card", "api_key"],
            action="redact",
        ))
    """

    def __init__(
        self,
        detect_types: list[str] | None = None,
        action: str = "flag",
        custom_patterns: dict[str, re.Pattern] | None = None,
    ):
        self._detect_types = detect_types or list(PII_PATTERNS.keys())
        self._action = action
        self._patterns: dict[str, re.Pattern] = {}

        # Build active pattern set
        for pii_type in self._detect_types:
            if pii_type in PII_PATTERNS:
                self._patterns[pii_type] = PII_PATTERNS[pii_type]

        # Add custom patterns
        if custom_patterns:
            self._patterns.update(custom_patterns)

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Detect PII in span content."""
        attrs = span.attributes or {}
        if not any(key.startswith(("gen_ai.", "aitf.")) for key in attrs.keys()):
            return

        all_detections: list[PIIDetection] = []

        # Check span events for content
        for event in span.events or []:
            event_attrs = event.attributes or {}
            for val in event_attrs.values():
                if isinstance(val, str):
                    detections = self.detect_pii(val)
                    all_detections.extend(detections)

        # Check relevant span attributes
        for key in (
            "aitf.mcp.tool.input",
            "aitf.mcp.tool.output",
            "aitf.skill.input",
            "aitf.skill.output",
        ):
            val = attrs.get(key)
            if isinstance(val, str):
                detections = self.detect_pii(val)
                all_detections.extend(detections)

    def detect_pii(self, text: str) -> list[PIIDetection]:
        """Detect PII in text. Returns list of PIIDetection objects."""
        detections: list[PIIDetection] = []
        for pii_type, pattern in self._patterns.items():
            matches = list(pattern.finditer(text))
            if matches:
                detections.append(PIIDetection(
                    pii_type=pii_type,
                    count=len(matches),
                    locations=[(m.start(), m.end()) for m in matches],
                ))
        return detections

    def redact_pii(self, text: str) -> tuple[str, list[PIIDetection]]:
        """Detect and redact PII from text.

        Returns (redacted_text, detections).
        """
        detections = self.detect_pii(text)
        if not detections:
            return text, []

        result = text
        for detection in sorted(detections, key=lambda d: d.locations[0][0], reverse=True):
            for start, end in sorted(detection.locations, reverse=True):
                original = result[start:end]
                if self._action == "redact":
                    replacement = f"[{detection.pii_type.upper()}_REDACTED]"
                elif self._action == "hash":
                    hash_val = hashlib.sha256(original.encode()).hexdigest()[:12]
                    replacement = f"[{detection.pii_type.upper()}:{hash_val}]"
                else:
                    continue
                result = result[:start] + replacement + result[end:]

        return result, detections

    def get_pii_summary(self, text: str) -> dict[str, Any]:
        """Get a summary of PII detected in text."""
        detections = self.detect_pii(text)
        if not detections:
            return {
                SecurityAttributes.PII_DETECTED: False,
                SecurityAttributes.PII_TYPES: [],
                SecurityAttributes.PII_COUNT: 0,
                SecurityAttributes.PII_ACTION: self._action,
            }
        return {
            SecurityAttributes.PII_DETECTED: True,
            SecurityAttributes.PII_TYPES: [d.pii_type for d in detections],
            SecurityAttributes.PII_COUNT: sum(d.count for d in detections),
            SecurityAttributes.PII_ACTION: self._action,
        }

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
