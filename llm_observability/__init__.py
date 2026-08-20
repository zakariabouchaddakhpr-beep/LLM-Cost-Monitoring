"""
Tracking the cost and speed of AI API calls.

WHAT THIS DOES
Every time your app talks to an AI model, this library writes down how long
it took, how much text went in and out, what it cost, and who it was for.
Those records go into a database, and a web dashboard turns them into charts.

The problem it solves: AI APIs charge per use, and it is very easy to end up
with a large bill and no idea which feature or which customer caused it.

HOW TO USE IT

    import llm_observability as obs

    obs.init(name="my-app", model="cerebras/gpt-oss-120b",
             api_key=os.getenv("CEREBRAS_API_KEY"))
    result = obs.call("Summarise this ticket", tenant_id="acme", use_case="support")
    obs.dashboard()

Embedding calls work the same way and land in the same table, marked with a
call_type column, so one question about cost covers both kinds:

    result = obs.embed(["doc one", "doc two"], tenant_id="acme")
    vectors = result.embeddings     # handed back to you, not saved

WHERE THE DATA GOES
Each project gets its own database, named after the project (or after the
CLICKHOUSE_DATABASE setting, if you set one). An existing database is reused,
never wiped and replaced.

GOING DEEPER
If those three calls aren't enough, the classes underneath are all available
to use directly. Nothing is hidden from you:

    from llm_observability.core.application.monitor import LLMMonitor
    from llm_observability.core.infrastructure.persistence import TraceRepository

HOW THE FOLDERS ARE ORGANISED

    config/       settings, read from environment variables
    core/
      domain/     plain data and maths. No database, no internet.
      application/the actual logic: make a call, measure it, mark it
      infrastructure/  everything that talks to the outside world:
                       AI providers, the database, OpenTelemetry
    dashboard/    the Streamlit web app
    api.py        the friendly wrapper this file is describing

The split exists so that the interesting logic never gets tangled up with
database code, which makes both easier to read and much easier to test.
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

# __all__ lists what you get from `from llm_observability import *`. It also
# serves as a plain statement of what counts as the official, supported set of
# things to use - anything not on this list may change without warning.
__all__ = ["Observer", "call", "dashboard", "database", "embed", "init",
           "summary"]

# A NOTE FOR ANYONE EDITING THIS FILE:
# api.py deliberately delays importing litellm until someone actually creates
# an Observer, because loading litellm is slow. Adding `import litellm` here
# would undo all of that and make every `import llm_observability` slow again.
# Please don't.
