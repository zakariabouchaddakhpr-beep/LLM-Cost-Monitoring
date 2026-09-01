"""Cost and latency observability for LLM and embedding APIs.

    import llm_observability as obs

    obs.init(name="my-app", model="cerebras/gpt-oss-120b", api_key=...)
    result = obs.call("Summarise this ticket", tenant_id="acme")
    obs.dashboard()

Each project gets its own ClickHouse database, reused rather than replaced.
The classes underneath are all importable directly. See CODE_NOTES.md.
"""

from llm_observability.api import (
    Observer,
    call,
    dashboard,
    database,
    embed,
    init,
    review,
    summary,
)

__all__ = ["Observer", "call", "dashboard", "database", "embed", "init", "review",
           "summary"]
