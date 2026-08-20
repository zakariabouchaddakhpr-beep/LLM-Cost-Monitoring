"""
All the SQL that sets up our storage: the database, the table, and the views.

This file only ever builds TEXT. It never connects to anything - it just puts
SQL commands together and hands them back as strings. Someone else runs them.
"""

# Fields on LLMCallResult that we deliberately never save. The number-lists
# from an embedding call are enormous, and you asked for them so you could
# use them - not so they could sit in a table making it slow.
NON_PERSISTED = {"embeddings"}

# The columns we write, in an order that reads nicely.
#
# The order here doesn't actually have to match anything. When we insert rows
# we send this list of names along with them, so the driver matches values to
# columns by NAME. That means adding a field in the middle can never silently
# shift every value one column to the left.
COLUMNS = [
    "provider", "trace_id", "timestamp", "model", "prompt",
    "response", "call_type", "input_tokens", "output_tokens", "total_tokens",
    "cached_tokens", "reasoning_tokens", "finish_reason", "latency_ms",
    "tokens_per_second", "attempts_used", "cost_usd", "quality_score",
    "quality_reasoning", "cost_efficiency", "request_id", "error_type",
    "error", "span_id", "trace_otel_id", "tenant_id", "use_case",
    "query_type", "vector_dimensions", "input_count",
]

# WHAT THE TYPES BELOW MEAN
#
#   String                  Text.
#   UInt32                  A whole number that can't be negative.
#   Float64                 A number with a decimal point.
#   Nullable(X)             An X, but allowed to be empty.
#
#   LowCardinality(String)  Text where there are only a handful of different
#                           values - like a provider name, which is always one
#                           of five or six things. ClickHouse quietly stores
#                           these as small numbers behind the scenes, which
#                           makes counting and grouping them much faster.
#
#   DateTime64(3, 'UTC')    A date and time, accurate to thousandths of a
#                           second. The 3 is how many decimal places. This
#                           whole tool measures in milliseconds, so 3 is
#                           exactly enough.
#
# AND THE SETTINGS AT THE BOTTOM
#
#   ENGINE = MergeTree      ClickHouse's normal table type. Built for tables
#                           you add to constantly and read from a lot.
#
#   PARTITION BY month      Rows get physically filed by month. So a question
#                           about last week never has to read through last
#                           year's data to answer.
#
#   ORDER BY (...)          How rows are sorted on disk. Questions that filter
#                           on these columns can jump straight to the right
#                           place instead of scanning everything.
#
#   TTL on prompt/response  TTL means "time to live". After 90 days the TEXT
#                           in these two columns gets wiped - but the row
#                           itself stays. So you keep your cost history for
#                           ever, without hanging on to users' text for ever.
TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {database}.llm_traces (
    provider          LowCardinality(String),
    trace_id          String,
    timestamp         DateTime64(3, 'UTC'),
    model             LowCardinality(String),
    prompt            String TTL timestamp + INTERVAL 90 DAY,
    response          Nullable(String) TTL timestamp + INTERVAL 90 DAY,
    call_type         LowCardinality(String) DEFAULT 'chat',
    input_tokens      Nullable(UInt32),
    output_tokens     Nullable(UInt32),
    total_tokens      Nullable(UInt32),
    cached_tokens     Nullable(UInt32),
    reasoning_tokens  Nullable(UInt32),
    finish_reason     LowCardinality(Nullable(String)),
    latency_ms        Nullable(Float64),
    tokens_per_second Nullable(Float64),
    attempts_used     Nullable(UInt8),
    cost_usd          Nullable(Float64),
    quality_score     Nullable(UInt8),
    quality_reasoning Nullable(String),
    cost_efficiency   Nullable(Float64),
    request_id        Nullable(String),
    error_type        LowCardinality(Nullable(String)),
    error             Nullable(String),
    span_id           Nullable(String),
    trace_otel_id     Nullable(String),
    tenant_id         LowCardinality(Nullable(String)),
    use_case          LowCardinality(Nullable(String)),
    query_type        LowCardinality(Nullable(String)),
    vector_dimensions Nullable(UInt32),
    input_count       Nullable(UInt32)
) ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, provider, model)
"""

# A VIEW is a saved question that behaves like a table. It holds no data of
# its own - every time you read from it, the question underneath gets asked
# again, fresh. So a view is never out of date.
#
# These five are the questions the dashboard asks over and over.
#
# A NOTE ON quantile():
# A "percentile" answers "what value is X% of my calls below?". So p90 = 900ms
# means 90% of calls finished in under 900ms, and the slowest 10% took longer.
#
# There are two ways to work a percentile out, and they give different answers:
#
#   quantile()       estimates it. If the true answer falls between two of your
#                    measurements, it picks a value in between them - so it can
#                    return 850ms even though no call took exactly 850ms.
#   quantileExact()  always returns a number that really happened.
#
# We use quantile(). On small amounts of data the two disagree badly - measured
# 527ms apart at p99 with only 15 rows to work from - and quantile() is the one
# the dashboard's numbers were built around. It gets a little rough on very
# large tables, so it's worth revisiting if these figures ever start jumping
# around for no obvious reason.
VIEW_DDL = [
    # 1. Spend per customer. Answers "who do I bill, and how much?"
    """
CREATE OR REPLACE VIEW {database}.v_cost_by_tenant AS
SELECT tenant_id,
       count(*)           AS call_count,
       sum(cost_usd)      AS total_cost_usd,
       avg(cost_usd)      AS avg_cost_usd,
       sum(input_tokens)  AS total_input_tokens,
       sum(output_tokens) AS total_output_tokens
FROM {database}.llm_traces
WHERE tenant_id IS NOT NULL
GROUP BY tenant_id
""",
    # 2. Spend per feature. Answers "which part of my app is expensive?"
    """
CREATE OR REPLACE VIEW {database}.v_cost_by_use_case AS
SELECT use_case,
       count(*)           AS call_count,
       sum(cost_usd)      AS total_cost_usd,
       avg(cost_usd)      AS avg_cost_usd,
       avg(quality_score) AS avg_quality_score
FROM {database}.llm_traces
WHERE use_case IS NOT NULL
GROUP BY use_case
""",
    # 3. Speed per model. p50 is the middle - half your calls were faster than
    #    this. p99 is the slow tail: only 1 call in 100 was worse. That tail is
    #    what users actually complain about, and a plain average hides it
    #    completely.
    """
CREATE OR REPLACE VIEW {database}.v_latency_percentiles AS
SELECT provider,
       model,
       count(*)                   AS call_count,
       quantile(0.50)(latency_ms) AS p50_ms,
       quantile(0.90)(latency_ms) AS p90_ms,
       quantile(0.99)(latency_ms) AS p99_ms,
       avg(latency_ms)            AS avg_ms
FROM {database}.llm_traces
WHERE latency_ms IS NOT NULL
GROUP BY provider, model
""",
    # 4. Grand totals across everything.
    #
    #    This exists for simple tools that connect to the database over the
    #    web and can't send along their own filter values. TraceRepository has
    #    its own copy of this question in Python, because that version CAN
    #    take filters, and a saved view can't. If you change one, change the
    #    other - with no filters applied, the two must give the same numbers.
    """
CREATE OR REPLACE VIEW {database}.v_summary AS
SELECT count(*)           AS total_calls,
       sum(cost_usd)      AS total_cost_usd,
       avg(latency_ms)    AS avg_latency_ms,
       avg(quality_score) AS avg_quality_score,
       sum(input_tokens)  AS total_input_tokens,
       sum(output_tokens) AS total_output_tokens
FROM {database}.llm_traces
""",
    # 5. Which model is the best value for which kind of job.
    #
    #    This is the view that lets you say: "for summarising, the cheap model
    #    scores nearly as well as the expensive one - send summaries there."
    #
    #    nullIf(avg(cost_usd), 0) protects the division. It means "give me the
    #    average cost, unless that average is 0, in which case give me nothing".
    #    Without it, a zero average cost would produce infinity, and sorting
    #    the table by this column would put a meaningless row at the top for
    #    ever.
    #
    #    Embedding calls are left out, because we never mark them. They have
    #    no quality score to divide, and no query type to group by.
    """
CREATE OR REPLACE VIEW {database}.v_model_by_query_type AS
SELECT query_type,
       provider,
       model,
       count(*)                   AS call_count,
       avg(cost_usd)              AS avg_cost_usd,
       avg(quality_score)         AS avg_quality,
       quantile(0.90)(latency_ms) AS p90_ms,
       avg(quality_score) / nullIf(avg(cost_usd), 0) AS quality_per_dollar
FROM {database}.llm_traces
WHERE query_type IS NOT NULL AND call_type = 'chat'
GROUP BY query_type, provider, model
""",
]


# MIGRATIONS: commands that upgrade a table built by an OLDER version of this
# code, so it matches what the current code expects.
#
# Why are these needed? Because "CREATE TABLE IF NOT EXISTS" does absolutely
# nothing when the table already exists - including nothing about columns
# added since. Without these commands, a database set up months ago would
# quietly keep its old columns, and every attempt to save a record would fail
# on the new ones.
#
# These run EVERY time we connect, so each one has to be safe to run over and
# over. That is what "IF NOT EXISTS" and "IF EXISTS" are doing - they mean
# "only if this hasn't been done already". Adding an empty column only changes
# the table's description, not its contents, so this stays fast even on a
# table with millions of rows.
_MIGRATIONS = [
    # Older versions only saved a shortened preview of the text, under a
    # different column name. Rename rather than replace, so old data survives.
    "ALTER TABLE {database}.llm_traces RENAME COLUMN IF EXISTS prompt_preview TO prompt",
    "ALTER TABLE {database}.llm_traces RENAME COLUMN IF EXISTS response_preview TO response",
    # Columns added as the tool grew new features.
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS call_type LowCardinality(String) DEFAULT 'chat'",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS cached_tokens Nullable(UInt32)",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS reasoning_tokens Nullable(UInt32)",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS query_type LowCardinality(Nullable(String))",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS vector_dimensions Nullable(UInt32)",
    "ALTER TABLE {database}.llm_traces ADD COLUMN IF NOT EXISTS input_count Nullable(UInt32)",
    # Apply the 90-day text wipe to tables created before that rule existed.
    "ALTER TABLE {database}.llm_traces MODIFY COLUMN prompt String TTL timestamp + INTERVAL 90 DAY",
    "ALTER TABLE {database}.llm_traces MODIFY COLUMN response Nullable(String) TTL timestamp + INTERVAL 90 DAY",
]


# The four functions below all do the same small job. All the SQL above has
# {database} written in it as a placeholder, and .format(database=...) swaps
# that placeholder for the real database name.

def database_ddl(database: str) -> str:
    """The command to create the database itself. Safe if it already exists."""
    return f"CREATE DATABASE IF NOT EXISTS {database}"


def table_ddl(database: str) -> str:
    """The command to create the llm_traces table."""
    # .strip() removes the blank line at the start of the text above, which is
    # only there because the quotes had to go on their own line.
    return TABLE_DDL.format(database=database).strip()


def migrations(database: str) -> list[str]:
    """The commands that bring an older table up to date with this code."""
    return [statement.format(database=database) for statement in _MIGRATIONS]


def view_ddl(database: str) -> list[str]:
    """
    The commands that create the five views - one command per list item.

    They are kept as five separate pieces of text rather than one big block
    because ClickHouse only accepts one command per request when you talk to
    it over the web. Sending all five at once would be rejected.
    """
    return [view.format(database=database).strip() for view in VIEW_DDL]
