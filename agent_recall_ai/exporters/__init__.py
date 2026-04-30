"""
agent_recall_ai.exporters — telemetry exporters for external observability.

Available exporters:
    OTLPExporter     — OpenTelemetry OTLP traces → Datadog, Jaeger, Grafana
    DatadogExporter  — Convenience wrapper around OTLPExporter for Datadog
"""
from .otlp import OTLPExporter

try:
    from .datadog import DatadogExporter
    __all__ = ["OTLPExporter", "DatadogExporter"]
except ImportError:
    __all__ = ["OTLPExporter"]
