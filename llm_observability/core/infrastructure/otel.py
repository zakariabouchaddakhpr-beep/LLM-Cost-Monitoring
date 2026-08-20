"""
Setting up OpenTelemetry.

WHAT IS OPENTELEMETRY?
It is an agreed-upon way for programs to report what they are doing, so that
monitoring tools can understand them without any special setup. Everyone
shortens the name to "OTel".

It gives us two things, and we use both:

  A TRACER makes "spans". A span is a record of one thing that happened and
  how long it took: "this chat call took 812 milliseconds and used 340
  tokens". Think of it as a stopwatch that also carries notes.

  A METER makes counters and histograms, which add numbers up across many
  calls: "45,000 tokens used today". A counter only ever goes up. A histogram
  remembers the spread of the values, not just the total.

WHERE DOES THE DATA GO?
An "exporter" decides that:

  - by default we use a Console exporter, which just prints to your terminal
  - if you set OTEL_EXPORTER_OTLP_ENDPOINT, we send it to a real monitoring
    tool instead (Jaeger, Grafana, Datadog and others all accept this format)

Both are built the first time something asks for them, rather than when this
file is first loaded. That is why the functions below start by checking
"have I built this yet?". Waiting until the last moment lets a test put its
own fake tracer in place before the real one ever gets created.
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

from llm_observability.config.settings import (
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_SERVICE_NAME,
)

# A name tag stuck onto everything we send out. A monitoring tool collecting
# data from twenty different programs uses this to tell them apart.
_RESOURCE = Resource.create({"service.name": OTEL_SERVICE_NAME})

# Built once, the first time they are needed, then reused every time after.
# None means "we haven't built this one yet".
_tracer = None
_meter = None


def _sending_to_a_real_backend() -> bool:
    """
    True if the user pointed us at a real monitoring tool. False if we should
    just print to the terminal.

    bool(...) turns the address into a plain yes or no: any address at all
    counts as True, and a missing or empty one counts as False.
    """
    return bool(OTEL_EXPORTER_OTLP_ENDPOINT)


def get_tracer():
    """
    Get the tracer - the thing we use to time and record each call.

    Builds it the first time, and hands back that same one ever after.
    """
    global _tracer
    # Already built it? Hand it straight back and skip all the setup below.
    if _tracer is not None:
        return _tracer

    if _sending_to_a_real_backend():
        # insecure=True means "don't encrypt this connection". That is normal
        # and fine when the monitoring tool is running on your own machine or
        # inside your own private network.
        exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                                    insecure=True)
    else:
        exporter = ConsoleSpanExporter()

    provider = TracerProvider(resource=_RESOURCE)
    # "Batch" means: gather the records up in memory and send them off in
    # groups, using a separate thread, instead of one at a time. This matters
    # because sending them one at a time would slow down the very calls we
    # are trying to measure.
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Save it, so the next call to this function skips everything above.
    _tracer = trace.get_tracer(OTEL_SERVICE_NAME)
    return _tracer


def get_meter():
    """
    Get the meter - the thing we use to add up totals across many calls.

    Built on first use, exactly like the tracer above.
    """
    global _meter
    if _meter is not None:
        return _meter

    if _sending_to_a_real_backend():
        exporter = OTLPMetricExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                                      insecure=True)
    else:
        exporter = ConsoleMetricExporter()

    # Totals get sent on a timer - every 5000 milliseconds, so every 5
    # seconds - rather than after every single call. Sending a running total
    # after each call would be a lot of network traffic telling the monitoring
    # tool almost nothing new.
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
    provider = MeterProvider(resource=_RESOURCE, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    _meter = metrics.get_meter(OTEL_SERVICE_NAME)
    return _meter
