"""
LLMCallResult: one object holding everything we know about one LLM call.

This is the most important thing in the project. Every other part either
fills one in or reads one:

    monitor.py      makes one, and fills it in as the call goes along
    repository.py   turns one into a row in the database
    dashboard/      reads those rows back and draws the charts

It is a "dataclass". That is just a normal Python class where you list the
things it holds and Python writes the boring setup code for you. Instead of
writing an __init__ that copies ten arguments onto self, you list the ten
names and their types, and you are done.

Notice what this file does NOT import: no database, no web server, no LLM
library. That is on purpose. It means you can make one of these in a test
without a database running, without internet, and without an API key.

Nearly every field says `Optional[...] = None`. Optional means "this might
be empty". They are all optional because a call can go wrong at any point.
If the API call fails, we still know the prompt, the model and the error -
but there were never any tokens or any cost to record.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class LLMCallResult:
    """Everything we know about one call to a chat or embedding API."""

    # --- Always filled in. These are set the moment the call starts ---
    provider: str              # who answered: "openai", "groq", "cerebras"...
    trace_id: str              # our own id for this call, a random unique string
    timestamp: str             # when the call started, written as text
    model: str                 # the full model name, e.g. "groq/llama-3.3-70b"
    prompt: str                # the text we sent, kept whole
    response: Optional[str] = None   # the text that came back
    # "chat" or "embedding". Both kinds of call go in the same table, and this
    # column is how you tell them apart.
    call_type: str = "chat"

    # --- How much text went in and out ---
    # A "token" is roughly a word-piece. Providers charge you per token, so
    # these numbers are what your bill is actually based on.
    # None here means the provider never told us the count.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # Some providers remember a prompt you sent before and charge much less
    # for repeating it. Those repeated tokens are counted here. If this number
    # is high, you are getting a discount.
    cached_tokens: Optional[int] = None
    # Some models "think" before answering. You pay for that thinking, but it
    # doesn't appear in the input or output counts above, so it gets its own.
    reasoning_tokens: Optional[int] = None

    # --- Speed, and how the call ended ---
    # Why the model stopped writing. "stop" means it finished its answer
    # normally. "length" means it hit the maximum size and got cut off
    # mid-sentence - a sign you should raise the limit.
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None       # how long the call took, in milliseconds
    tokens_per_second: Optional[float] = None
    # How many tries it took. 1 means it worked first time. 3 means it failed
    # twice and only worked on the third go.
    attempts_used: Optional[int] = None

    # --- Money and quality ---
    cost_usd: Optional[float] = None
    quality_score: Optional[int] = None      # 1 to 10, given by the judge model
    quality_reasoning: Optional[str] = None  # one sentence saying why
    cost_efficiency: Optional[float] = None  # quality divided by cost

    # --- Ids, for tracking down one specific call later ---
    request_id: Optional[str] = None         # the id the provider gave it
    error_type: Optional[str] = None         # the kind of error, e.g. "RateLimitError"
    error: Optional[str] = None              # the error message itself

    # --- Ids from OpenTelemetry ---
    # OpenTelemetry is a standard way of recording what a program did. If you
    # already use a tool like Jaeger or Datadog, saving these ids lets you
    # match a row in our database to the same call in that tool.
    span_id: Optional[str] = None
    trace_otel_id: Optional[str] = None

    # --- Labels saying who and what the money was spent on ---
    # Without these you know what you spent but not why, which makes the whole
    # cost report useless. Fill them in.
    tenant_id: Optional[str] = None    # which of your customers
    use_case: Optional[str] = None     # which feature of your app
    query_type: Optional[str] = None   # what kind of request, decided by the judge

    # --- Only used by embedding calls ---
    # An embedding turns text into a long list of numbers that represents its
    # meaning. This is how many numbers are in that list, usually 384, 768
    # or 1536 depending on the model.
    vector_dimensions: Optional[int] = None
    input_count: Optional[int] = None        # how many texts we sent in one go
    # The lists of numbers themselves. These are NOT saved to the database -
    # they are enormous, and you asked for them so you could use them, not so
    # they could sit in a table.
    embeddings: Optional[list] = None

    @property
    def succeeded(self) -> bool:
        """
        True if the call worked.

        We only ever set `error` when something went wrong, so "no error
        message" and "it worked" are the same thing.

        @property means you write `result.succeeded` with no brackets, as if
        it were a stored value, even though it's worked out on the spot.
        """
        return self.error is None

    def to_dict(self) -> dict:
        """
        Turn this object into a plain dictionary.

        So instead of `result.provider` you get `{"provider": "groq", ...}`.
        Useful for saving it or sending it somewhere.
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """
        Turn this object into JSON text, laid out over several lines.

        `indent=2` is what spreads it out and indents it, instead of cramming
        everything onto one very long line.
        """
        # ensure_ascii=False keeps letters like é and 日 as themselves. With
        # the default (True) they would be mangled into codes like é,
        # which is still correct but nobody can read it.
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> None:
        """Write this result to a file on disk, as JSON."""
        # "w" means create the file, or wipe it if it already exists.
        # encoding="utf-8" makes sure non-English characters survive.
        # The `with` block closes the file for you, even if writing fails.
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
