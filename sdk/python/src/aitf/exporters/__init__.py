"""AITF Exporters.

Provides OTel SpanExporters for both OCSF-formatted security output (SIEM,
XDR, data lakes) and standard OpenTelemetry export (OTLP to Jaeger, Grafana
Tempo, Datadog, etc.).

AITF supports dual-pipeline export where the same spans flow to both:
- **OTLP** → Standard OTel backends for observability
- **OCSF** → SIEM/XDR for security monitoring

See :mod:`aitf.pipeline` for the recommended ``DualPipelineProvider``.
"""

from aitf.exporters.ocsf_exporter import OCSFExporter

__all__ = ["OCSFExporter"]
