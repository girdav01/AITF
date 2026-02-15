"""AITF Exporters.

Provides OTel SpanExporters for OCSF-formatted output to SIEM, XDR, and data lakes.
"""

from aitf.exporters.ocsf_exporter import OCSFExporter

__all__ = ["OCSFExporter"]
