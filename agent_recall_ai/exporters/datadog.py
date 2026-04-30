"""
DatadogExporter — convenience wrapper around OTLPExporter for Datadog APM.

Usage:
    from agent_recall_ai.exporters import DatadogExporter

    # Reads DD_API_KEY and DD_SITE from environment
    exporter = DatadogExporter()

    # Or explicit:
    exporter = DatadogExporter(
        api_key="dd-api-key-here",
        site="datadoghq.com",        # or datadoghq.eu, us3.datadoghq.com, etc.
        service="my-agent",
        env="production",
    )

    with Checkpoint("my-task") as cp:
        exporter.attach(cp)
        ...
"""
from __future__ import annotations

import os
from typing import Optional

from .otlp import OTLPExporter


class DatadogExporter(OTLPExporter):
    """
    Pre-configured OTLPExporter that targets Datadog's OTLP intake.

    Reads ``DD_API_KEY`` and ``DD_SITE`` from environment when not explicitly
    provided.  The ``env`` and ``version`` tags are added as span attributes so
    Datadog's APM UI can filter by deployment environment.
    """

    _DATADOG_OTLP_PORT = 4317

    def __init__(
        self,
        api_key: Optional[str] = None,
        site: Optional[str] = None,
        service: str = "agent-recall-ai",
        env: str = "production",
        version: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("DD_API_KEY", "")
        resolved_site = site or os.environ.get("DD_SITE", "datadoghq.com")

        endpoint = f"https://trace.agent.{resolved_site}:{self._DATADOG_OTLP_PORT}"
        headers = {}
        if resolved_key:
            headers["DD-API-KEY"] = resolved_key

        super().__init__(
            endpoint=endpoint,
            headers=headers,
            service_name=service,
            use_grpc=True,
            insecure=False,
        )

        # Stamp Datadog unified service tagging attributes
        self._env = env
        self._version = version

    def _session_attrs(self, state: Any) -> dict:
        attrs = super()._session_attrs(state)
        attrs["deployment.environment"] = self._env
        if self._version:
            attrs["service.version"] = self._version
        return attrs


from typing import Any  # noqa: E402 — needed after class definition
