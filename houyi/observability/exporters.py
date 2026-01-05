"""Exporters for trace data."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class ExporterConfig:
    """Configuration for an exporter."""

    def __init__(self, type: str, **kwargs: Any):
        self.type = type
        self.config = kwargs


class Exporter(ABC):
    """Base class for exporters."""

    @abstractmethod
    def export(self, span_data: dict[str, Any]) -> None:
        """Export span data.

        Args:
            span_data: Span data to export
        """
        pass

    def flush(self) -> None:
        """Flush any buffered data.

        Default implementation does nothing.
        Subclasses can override if they need to flush buffered data.
        """
        return  # Default: no-op


class ConsoleExporter(Exporter):
    """Export traces to console."""

    def __init__(self, verbose: bool = False):
        """Initialize console exporter.

        Args:
            verbose: Whether to print full span details
        """
        self.verbose = verbose

    def export(self, span_data: dict[str, Any]) -> None:
        """Export span to console."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Trace: {span_data['name']}")
            print(f"{'='*60}")
            self._print_span(span_data, indent=0)
            print(f"{'='*60}\n")
        else:
            # Compact format
            duration_ms = span_data['duration'] * 1000
            status = span_data['status']
            status_icon = "✅" if status == "ok" else "❌"
            print(f"{status_icon} {span_data['name']} ({duration_ms:.2f}ms)")

    def _print_span(self, span: dict, indent: int = 0) -> None:
        """Print span recursively."""
        prefix = "  " * indent
        duration_ms = span['duration'] * 1000

        print(f"{prefix}📍 {span['name']} ({duration_ms:.2f}ms)")

        if span['attributes']:
            print(f"{prefix}   Attributes:")
            for key, value in span['attributes'].items():
                print(f"{prefix}     - {key}: {value}")

        if span['events']:
            print(f"{prefix}   Events:")
            for event in span['events']:
                print(f"{prefix}     - {event['name']}")

        for child in span['children']:
            self._print_span(child, indent + 1)


class JSONExporter(Exporter):
    """Export traces to JSON file."""

    def __init__(self, filepath: str = "traces.json"):
        """Initialize JSON exporter.

        Args:
            filepath: Path to JSON file
        """
        self.filepath = filepath
        self.traces: list[dict] = []

    def export(self, span_data: dict[str, Any]) -> None:
        """Export span to JSON file."""
        self.traces.append(span_data)

    def flush(self) -> None:
        """Write all traces to file."""
        if self.traces:
            with open(self.filepath, 'w') as f:
                json.dump(self.traces, f, indent=2)
            self.traces = []


class JaegerExporter(Exporter):
    """Export traces to Jaeger using OTLP/HTTP protocol.

    Supports Jaeger v1.35+ with OTLP receiver.
    Default endpoint: http://localhost:4318/v1/traces
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        service_name: str = "houyi-agent",
        timeout: int = 5,
        batch_size: int = 10
    ):
        """Initialize Jaeger exporter.

        Args:
            endpoint: Jaeger OTLP endpoint
            service_name: Service name
            timeout: Request timeout
            batch_size: Batch size
        """
        self.endpoint = endpoint.rstrip('/') + '/v1/traces'
        self.service_name = service_name
        self.timeout = timeout
        self.batch_size = batch_size
        self.span_batch: list[dict] = []
        print(f"✅ JaegerExporter initialized - {self.endpoint}")

    def export(self, span_data: dict[str, Any]) -> None:
        """Export span to Jaeger."""
        otlp_span = self._convert_to_otlp(span_data)
        self.span_batch.append(otlp_span)

        if len(self.span_batch) >= self.batch_size:
            self.flush()

    def _convert_to_otlp(self, span: dict[str, Any]) -> dict:
        """Convert HouYi span to OTLP format."""
        start_time_ns = int(span['start_time'] * 1e9)
        end_time_ns = int(span['end_time'] * 1e9)

        return {
            "traceId": span['trace_id'],
            "spanId": span['span_id'],
            "parentSpanId": span.get('parent_id', ''),
            "name": span['name'],
            "kind": 1,
            "startTimeUnixNano": str(start_time_ns),
            "endTimeUnixNano": str(end_time_ns),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in span.get('attributes', {}).items()
            ],
            "status": {"code": 1 if span['status'] == 'ok' else 2}
        }

    def flush(self) -> None:
        """Send batched spans to Jaeger."""
        if not self.span_batch:
            return

        payload = {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": self.service_name}}
                    ]
                },
                "scopeSpans": [{
                    "spans": self.span_batch
                }]
            }]
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    print(f"✅ Exported {len(self.span_batch)} spans to Jaeger")
                else:
                    print(f"⚠️  Jaeger export failed: {response.status}")
        except urllib.error.URLError as e:
            print(f"⚠️  Jaeger export error: {e}")
        except Exception as e:
            print(f"⚠️  Jaeger export error: {e}")
        finally:
            self.span_batch = []


class DatadogExporter(Exporter):
    """Export traces to Datadog APM.

    Uses Datadog Agent API (default: http://localhost:8126/v0.4/traces)
    """

    def __init__(
        self,
        agent_url: str = "http://localhost:8126",
        service_name: str = "houyi-agent",
        env: str = "production",
        timeout: int = 5,
        batch_size: int = 10
    ):
        """Initialize Datadog exporter.

        Args:
            agent_url: Datadog Agent URL
            service_name: Service name
            env: Environment (production/staging/dev)
            timeout: Request timeout
            batch_size: Batch size
        """
        self.agent_url = agent_url.rstrip('/') + '/v0.4/traces'
        self.service_name = service_name
        self.env = env
        self.timeout = timeout
        self.batch_size = batch_size
        self.trace_batch: list[list[dict]] = []
        print(f"✅ DatadogExporter initialized - {self.agent_url}")

    def export(self, span_data: dict[str, Any]) -> None:
        """Export span to Datadog."""
        dd_trace = self._convert_to_datadog(span_data)
        self.trace_batch.append(dd_trace)

        if len(self.trace_batch) >= self.batch_size:
            self.flush()

    def _convert_to_datadog(self, span: dict[str, Any]) -> list[dict]:
        """Convert HouYi span to Datadog format."""
        trace_id = int(span['trace_id'][:16], 16)
        span_id = int(span['span_id'][:16], 16)
        parent_id = int(span.get('parent_id', '0')[:16] or '0', 16)

        start_ns = int(span['start_time'] * 1e9)
        duration_ns = int(span['duration'] * 1e9)

        dd_span = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": parent_id if parent_id else None,
            "name": span['name'],
            "resource": span['name'],
            "service": self.service_name,
            "type": "custom",
            "start": start_ns,
            "duration": duration_ns,
            "error": 1 if span['status'] != 'ok' else 0,
            "meta": {
                "env": self.env,
                **{k: str(v) for k, v in span.get('attributes', {}).items()}
            }
        }

        # Flatten children into trace
        trace = [dd_span]
        for child in span.get('children', []):
            trace.extend(self._convert_to_datadog(child))

        return trace

    def flush(self) -> None:
        """Send batched traces to Datadog."""
        if not self.trace_batch:
            return

        try:
            data = json.dumps(self.trace_batch).encode('utf-8')
            req = urllib.request.Request(
                self.agent_url,
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'Datadog-Meta-Tracer-Version': '1.0',
                    'Datadog-Meta-Lang': 'python'
                }
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    print(f"✅ Exported {len(self.trace_batch)} traces to Datadog")
                else:
                    print(f"⚠️  Datadog export failed: {response.status}")
        except urllib.error.URLError as e:
            print(f"⚠️  Datadog export error: {e}")
        except Exception as e:
            print(f"⚠️  Datadog export error: {e}")
        finally:
            self.trace_batch = []


def create_exporter(config: dict | ExporterConfig) -> Exporter:
    """Create exporter from configuration.

    Args:
        config: Exporter configuration

    Returns:
        Exporter instance
    """
    if isinstance(config, ExporterConfig):
        exporter_type = config.type
        kwargs = config.config
    else:
        exporter_type = config.get("type", "console")
        kwargs = {k: v for k, v in config.items() if k != "type"}

    if exporter_type == "console":
        return ConsoleExporter(**kwargs)
    elif exporter_type == "json":
        return JSONExporter(**kwargs)
    elif exporter_type == "jaeger":
        return JaegerExporter(**kwargs)
    elif exporter_type == "datadog":
        return DatadogExporter(**kwargs)
    else:
        raise ValueError(f"Unknown exporter type: {exporter_type}")
