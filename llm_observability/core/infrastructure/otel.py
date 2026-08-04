"""
OpenTelemetry setup: tracer, meter, and exporters.
Console by default; OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set.

Lazy initialization so tests can swap in their own providers before
any globals are created.
"""

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

from llm_observability.config.settings import OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME

_RESOURCE = Resource.create({"service.name": OTEL_SERVICE_NAME})

_tracer = None
_meter = None


def get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer

    provider = TracerProvider(resource=_RESOURCE)

    if OTEL_EXPORTER_OTLP_ENDPOINT:
        exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(OTEL_SERVICE_NAME)
    return _tracer


def get_meter():
    global _meter
    if _meter is not None:
        return _meter

    if OTEL_EXPORTER_OTLP_ENDPOINT:
        exporter = OTLPMetricExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
    else:
        exporter = ConsoleMetricExporter()

    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
    provider = MeterProvider(resource=_RESOURCE, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    _meter = metrics.get_meter(OTEL_SERVICE_NAME)
    return _meter
