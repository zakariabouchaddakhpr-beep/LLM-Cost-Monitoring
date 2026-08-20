"""
Storage: saving records to ClickHouse and reading them back.

("Persistence" is just a formal word for "saving things so they survive after
the program stops running.")

    schema.py      the SQL that describes the table and views - text only
    database.py    opening connections to the server
    repository.py  TraceRepository - save a record, ask questions about them

TraceRepository is the only one of these that most code ever needs, so it's
listed here to keep the import short:

    from llm_observability.core.infrastructure.persistence import TraceRepository
"""

from llm_observability.core.infrastructure.persistence.repository import (
    TraceRepository,
)

__all__ = ["TraceRepository"]
