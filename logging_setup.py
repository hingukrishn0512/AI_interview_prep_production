"""JSON logging for the whole app.

Every log line (ours, LangGraph's, uvicorn's errors) becomes one JSON object
with a timestamp, level, logger name, and any bound context such as
request_id and user_id. Call setup_logging() once, before anything logs.
"""
import logging
import sys

import structlog


def setup_logging(level: int = logging.INFO) -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared = [
        structlog.contextvars.merge_contextvars,   # adds request_id / user_id
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
    ]

    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,   # applies the same fields to plain logging calls
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # project_raw.py adds its own plain-text handler when "interview_coach" has
    # none, which would print every line twice. A NullHandler stops that.
    logging.getLogger("interview_coach").addHandler(logging.NullHandler())

    # We log one line per request ourselves, so silence uvicorn's access log.
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False

    # One INFO line per Groq call is too noisy.
    logging.getLogger("httpx").setLevel(logging.WARNING)