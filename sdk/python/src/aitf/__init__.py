"""AITF - AI Telemetry Framework.

A comprehensive, security-first telemetry framework for AI systems
built on OpenTelemetry and OCSF.
"""

__version__ = "1.0.0"

from aitf.instrumentation import AITFInstrumentor

__all__ = ["AITFInstrumentor", "__version__"]
