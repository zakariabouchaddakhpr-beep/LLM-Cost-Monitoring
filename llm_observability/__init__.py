"""
Cost and latency observability for LLM and embedding APIs.

    import llm_observability as obs

    obs.init(name="my-app", model="cerebras/gpt-oss-120b",
             api_key=os.getenv("CEREBRAS_API_KEY"))
    result = obs.call("Summarise this ticket", tenant_id="acme", use_case="support")
    obs.dashboard()

Embedding calls are instrumented the same way, and land in the same table
behind a call_type column so cost queries still span everything:

    result = obs.embed(["doc one", "doc two"], tenant_id="acme")
    vectors = result.embeddings          # not persisted; the trace is

Each project gets its own ClickHouse database, named after the project, or
whatever CLICKHOUSE_DATABASE points to if that is set. An existing database is
reused, never replaced.

The lower-level pieces stay available for anything the three calls above do not
cover:

    from llm_observability.core.application.monitor import LLMMonitor
    from llm_observability.core.infrastructure.persistence import TraceRepository
"""

from llm_observability.api import (
    Observer,
    call,
    dashboard,
    database,
    embed,
    init,
    summary,
)

__all__ = ["Observer", "call", "dashboard", "database", "embed", "init",
           "summary"]

# NOTE: api.py imports litellm lazily, inside Observer.__init__ and the call
# path, so `import llm_observability` does not pay for a heavy dependency it may
# never use. Keep it that way — a top-level `import litellm` here would undo it.
