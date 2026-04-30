"""
OTLPExporter — export agent-recall-ai sessions as OpenTelemetry traces.

Sends one OTLP span per checkpoint event (decision, tool call, alert) to
any OTLP-compatible backend: Datadog, Jaeger, Grafana Tempo, Honeycomb, etc.

Why OpenTelemetry?
------------------
Distributed systems teams already have OTLP pipelines.  By emitting standard
spans, agent-recall-ai sessions show up alongside API calls, database queries,
and service mesh traffic in existing dashboards — no new tooling required.

Span hierarchy produced per checkpoint save
-------------------------------------------
  session:{session_id}                   (root span)
  ├── checkpoint:{seq}                   (one per save)
  │   ├── decision:{decision.id}         (one per decision)
  │   ├── tool_call:{tool_name}          (one per tool invocation)
  │   └── alert:{alert_type}            (one per alert raised)

Token usage, cost, and cache stats appear as span attributes so they are
queryable in any OTLP backend.

Installation
------------
    pip install 'agent-recall-ai[grpc]'
    # — or for HTTP export —
    pip install opentelemetry-exporter-otlp-proto-http

Usage
-----
    from agent_recall_ai import Checkpoint
    from agent_recall_ai.exporters import OTLPExporter

    exporter = OTLPExporter(endpoint="http://localhost:4317")

    with Checkpoint("my-task") as cp:
        cp.set_goal("Refactor auth module")
        exporter.attach(cp)          # hooks into every save

    # Export the whole session after-the-fact
    exporter.export_session(cp.state)

Datadog
-------
    exporter = OTLPExporter(
        endpoint="https://trace.agent.datadoghq.com:4317",
        headers={"DD-API-KEY": os.environ["DD_API_KEY"]},
    )

Jaeger (local dev)
------------------
    exporter = OTLPExporter(endpoint="http://localhost:4317")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.trace import StatusCode
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as _GrpcExporter
    _GRPC_EXPORTER_AVAILABLE = True
except ImportError:
    _GRPC_EXPORTER_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as _HttpExporter
    _HTTP_EXPORTER_AVAILABLE = True
except ImportError:
    _HTTP_EXPORTER_AVAILABLE = False


_EPOCH = datetime(1970, 1, 1, tzinfo=None)

def _to_ns(dt: datetime) -> int:
    """Convert a UTC datetime to nanoseconds since Unix epoch."""
    delta = dt - _EPOCH
    return int(delta.total_seconds() * 1_000_000_000)


class OTLPExporter:
    """
    Exports agent-recall-ai sessions as OpenTelemetry traces via OTLP.

    Parameters
    ----------
    endpoint:
        OTLP collector endpoint.  Defaults to the OTEL_EXPORTER_OTLP_ENDPOINT
        env var or ``http://localhost:4317``.
    headers:
        Extra HTTP/gRPC headers (e.g. API keys for managed collectors).
    service_name:
        OTel service.name attribute (default: ``agent-recall-ai``).
    use_grpc:
        True → gRPC transport (default if grpc deps installed).
        False → HTTP/Protobuf transport.
    insecure:
        Skip TLS verification (handy for local Jaeger).
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        service_name: str = "agent-recall-ai",
        use_grpc: bool = True,
        insecure: bool = False,
    ) -> None:
        if not _OTEL_AVAILABLE:
            raise ImportError(
                "opentelemetry-sdk is required: pip install opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-grpc"
            )
        self._service_name = service_name
        self._endpoint = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        self._headers = headers or {}
        self._insecure = insecure

        # Build span exporter
        self._span_exporter: SpanExporter = self._build_exporter(use_grpc)

        # Build tracer provider
        from opentelemetry.sdk.resources import Resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": _pkg_version(),
            "telemetry.sdk.name": "agent-recall-ai",
        })
        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(BatchSpanProcessor(self._span_exporter))
        self._tracer = self._provider.get_tracer(service_name)

    def _build_exporter(self, use_grpc: bool) -> "SpanExporter":
        if use_grpc and _GRPC_EXPORTER_AVAILABLE:
            return _GrpcExporter(
                endpoint=self._endpoint,
                headers=self._headers,
                insecure=self._insecure,
            )
        if _HTTP_EXPORTER_AVAILABLE:
            # Strip port for HTTP (default 4318 for HTTP OTLP)
            http_endpoint = self._endpoint
            if ":4317" in http_endpoint:
                http_endpoint = http_endpoint.replace(":4317", ":4318")
            return _HttpExporter(
                endpoint=f"{http_endpoint}/v1/traces",
                headers=self._headers,
            )
        raise ImportError(
            "No OTLP exporter found. Install one of:\n"
            "  pip install opentelemetry-exporter-otlp-proto-grpc\n"
            "  pip install opentelemetry-exporter-otlp-proto-http"
        )

    # ── Attachment API ────────────────────────────────────────────────────────

    def attach(self, checkpoint_instance: Any) -> None:
        """
        Hook into a Checkpoint instance so every call to ``.save()``
        automatically exports a span.

        Example::

            with Checkpoint("task") as cp:
                exporter.attach(cp)
                cp.set_goal("...")
            # span exported on __exit__
        """
        original_save = checkpoint_instance.save

        def _patched_save(*args: Any, **kwargs: Any) -> Any:
            result = original_save(*args, **kwargs)
            try:
                self.export_session(checkpoint_instance.state)
            except Exception as exc:
                logger.debug("OTLPExporter.attach: export failed (non-fatal): %s", exc)
            return result

        checkpoint_instance.save = _patched_save
        logger.debug("OTLPExporter attached to checkpoint %s", checkpoint_instance.state.session_id)

    # ── Export API ────────────────────────────────────────────────────────────

    def export_session(self, state: Any) -> None:
        """
        Export a full TaskState as a tree of OTel spans.

        This is safe to call multiple times — each call exports the current
        snapshot and is idempotent from the backend's perspective (checkpoint
        sequence numbers keep spans unique).
        """
        if not _OTEL_AVAILABLE:
            return

        session_id = state.session_id
        seq = state.checkpoint_seq

        with self._tracer.start_as_current_span(
            f"checkpoint:{seq}",
            attributes=self._session_attrs(state),
        ) as root:
            root.set_status(StatusCode.OK)

            # Decisions
            for decision in state.decisions:
                with self._tracer.start_as_current_span(
                    f"decision:{decision.id}",
                    attributes={
                        "decision.summary": decision.summary[:256],
                        "decision.reasoning": (decision.reasoning or "")[:256],
                        "decision.tags": ",".join(decision.tags),
                        "decision.alternatives_rejected": ",".join(
                            decision.alternatives_rejected
                        ),
                        "session.id": session_id,
                        "checkpoint.seq": seq,
                    },
                ) as span:
                    span.set_status(StatusCode.OK)

            # Tool calls
            for tc in state.tool_calls:
                with self._tracer.start_as_current_span(
                    f"tool:{tc.tool_name}",
                    attributes={
                        "tool.name": tc.tool_name,
                        "tool.input_summary": (tc.input_summary or "")[:256],
                        "tool.output_tokens": tc.output_tokens,
                        "tool.compressed": tc.compressed,
                        "session.id": session_id,
                        "checkpoint.seq": seq,
                    },
                ) as span:
                    span.set_status(StatusCode.OK)

            # Alerts
            for alert in state.alerts:
                with self._tracer.start_as_current_span(
                    f"alert:{alert.alert_type}",
                    attributes={
                        "alert.type": str(alert.alert_type),
                        "alert.severity": str(alert.severity),
                        "alert.message": alert.message[:512],
                        "session.id": session_id,
                        "checkpoint.seq": seq,
                    },
                ) as span:
                    if str(alert.severity) in ("error", "critical"):
                        span.set_status(StatusCode.ERROR, alert.message)
                    else:
                        span.set_status(StatusCode.OK)

        logger.debug(
            "OTLPExporter: exported session=%s checkpoint=%d decisions=%d tools=%d alerts=%d",
            session_id, seq, len(state.decisions), len(state.tool_calls), len(state.alerts),
        )

    def flush(self) -> None:
        """Force-flush pending spans to the collector (call before process exit)."""
        self._provider.force_flush()

    def shutdown(self) -> None:
        """Shut down the tracer provider and flush all pending spans."""
        self._provider.shutdown()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _session_attrs(self, state: Any) -> dict[str, Any]:
        token_usage = state.token_usage
        cache_savings = state.metadata.get("cache_savings", [])
        total_cache_read = sum(c.get("cache_read_tokens", 0) for c in cache_savings)
        total_cache_write = sum(c.get("cache_write_tokens", 0) for c in cache_savings)

        return {
            "session.id": state.session_id,
            "session.status": str(state.status),
            "session.goals": ",".join(state.goals[:5]),
            "checkpoint.seq": state.checkpoint_seq,
            "checkpoint.schema_version": state.schema_version or "unknown",
            "token.prompt": token_usage.prompt,
            "token.completion": token_usage.completion,
            "token.cached": token_usage.cached,
            "token.total": token_usage.total,
            "cost.usd": round(state.cost_usd, 6),
            "cache.read_tokens": total_cache_read,
            "cache.write_tokens": total_cache_write,
            "context.utilization": round(state.context_utilization, 4),
            "decisions.count": len(state.decisions),
            "files.count": len(state.files_modified),
            "tool_calls.count": len(state.tool_calls),
            "alerts.count": len(state.alerts),
        }


def _pkg_version() -> str:
    try:
        from importlib.metadata import version
        return version("agent-recall-ai")
    except Exception:
        return "0.0.0"
