"""AITF Span Processors.

Provides OTel SpanProcessors for security detection, PII redaction,
compliance mapping, and cost attribution.
"""

from aitf.processors.compliance_processor import ComplianceProcessor
from aitf.processors.cost_processor import CostProcessor
from aitf.processors.pii_processor import PIIProcessor
from aitf.processors.security_processor import SecurityProcessor

__all__ = [
    "SecurityProcessor",
    "PIIProcessor",
    "ComplianceProcessor",
    "CostProcessor",
]
