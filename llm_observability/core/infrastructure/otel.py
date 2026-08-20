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

from llm_observability.config.settings import (
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_SERVICE_NAME,
)

_RESOURCE = Resource.create({"service.name": OTEL_SERVICE_NAME})

_tracer = None
_meter = None


def _sending_to_a_real_backend() -> bool:
    return bool(OTEL_EXPORTER_OTLP_ENDPOINT)


def get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer

    if _sending_to_a_real_backend():
        exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                                    insecure=True)
    else:
        exporter = ConsoleSpanExporter()

    provider = TracerProvider(resource=_RESOURCE)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(OTEL_SERVICE_NAME)
    return _tracer


def get_meter():
    global _meter
    if _meter is not None:
        return _meter

    if _sending_to_a_real_backend():
        exporter = OTLPMetricExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                                      insecure=True)
    else:
        exporter = ConsoleMetricExporter()

    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
    provider = MeterProvider(resource=_RESOURCE, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    _meter = metrics.get_meter(OTEL_SERVICE_NAME)
    return _meter
