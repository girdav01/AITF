"""AITF OCSF Exporter.

OTel SpanExporter that converts AI spans to OCSF Category 7 events
and exports them to SIEM/XDR endpoints, S3, or local files.

Based on forwarder architecture from the AITelemetry project.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from aitf.ocsf.compliance_mapper import ComplianceMapper
from aitf.ocsf.mapper import OCSFMapper
from aitf.ocsf.schema import AIBaseEvent

logger = logging.getLogger(__name__)


class OCSFExporter(SpanExporter):
    """Exports OTel spans as OCSF Category 7 AI events.

    Converts AI-related spans to OCSF events using OCSFMapper,
    enriches with compliance metadata, and exports to configured
    destinations.

    Usage:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OCSFExporter(
            output_file="/var/log/aitf/ocsf_events.jsonl",
            compliance_frameworks=["nist_ai_rmf", "eu_ai_act", "mitre_atlas"],
        )
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
    """

    def __init__(
        self,
        endpoint: str | None = None,
        output_file: str | None = None,
        compliance_frameworks: list[str] | None = None,
        include_raw_span: bool = False,
        api_key: str | None = None,
    ):
        self._endpoint = endpoint
        self._output_file = output_file
        self._include_raw_span = include_raw_span
        self._api_key = api_key
        self._mapper = OCSFMapper()
        self._compliance_mapper = ComplianceMapper(frameworks=compliance_frameworks)
        self._event_count = 0

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans as OCSF events."""
        events: list[dict[str, Any]] = []

        for span in spans:
            ocsf_event = self._mapper.map_span(span)
            if ocsf_event is None:
                continue

            # Enrich with compliance metadata
            event_type = self._classify_event(ocsf_event)
            if event_type:
                self._compliance_mapper.enrich_event(ocsf_event, event_type)

            event_dict = ocsf_event.model_dump(exclude_none=True)
            events.append(event_dict)
            self._event_count += 1

        if not events:
            return SpanExportResult.SUCCESS

        # Export to configured destinations
        try:
            if self._output_file:
                self._export_to_file(events)
            if self._endpoint:
                self._export_to_endpoint(events)
            return SpanExportResult.SUCCESS
        except Exception as exc:
            logger.error("OCSF export failed: %s", exc)
            return SpanExportResult.FAILURE

    def _export_to_file(self, events: list[dict[str, Any]]) -> None:
        """Write OCSF events to JSONL file."""
        with open(self._output_file, "a") as f:
            for event in events:
                f.write(json.dumps(event, default=str) + "\n")

    def _export_to_endpoint(self, events: list[dict[str, Any]]) -> None:
        """Send OCSF events to HTTP endpoint."""
        try:
            import urllib.request

            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            payload = json.dumps(events, default=str).encode("utf-8")
            req = urllib.request.Request(
                self._endpoint,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status >= 400:
                    logger.error("OCSF endpoint returned %d", resp.status)
        except Exception as exc:
            logger.error("Failed to send to OCSF endpoint: %s", exc)
            raise

    def _classify_event(self, event: AIBaseEvent) -> str | None:
        """Classify an OCSF event for compliance mapping."""
        class_uid = event.class_uid
        mapping = {
            7001: "model_inference",
            7002: "agent_activity",
            7003: "tool_execution",
            7004: "data_retrieval",
            7005: "security_finding",
            7006: "supply_chain",
            7007: "governance",
            7008: "identity",
        }
        return mapping.get(class_uid)

    @property
    def event_count(self) -> int:
        return self._event_count

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
