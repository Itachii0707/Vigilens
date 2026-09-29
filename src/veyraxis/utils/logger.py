"""Structured logging system for Veyraxis Sentinel."""

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format logs as structured JSON strings for machine ingestion and cloud monitoring."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Include standard extra attributes if supplied
        for attr in ("request_id", "latency_ms", "client_ip", "event", "detections_count", "model_name"):
            if hasattr(record, attr):
                log_record[attr] = getattr(record, attr)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


class TextFormatter(logging.Formatter):
    """High-contrast ANSI colored text formatter for local development console."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"{color}[{record.levelname:<8}]{self.RESET} {time_str} [{record.name}]"
        msg = f"{prefix} {record.getMessage()}"
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        return msg


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure the root logger with either JSON or console formatter."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicates
    while root_logger.handlers:
        root_logger.handlers.pop()

    handler = logging.StreamHandler(sys.stdout)
    use_json = json_format or os.getenv("SENTINEL_LOG_JSON", "false").lower() in ("1", "true", "yes")

    if use_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get or create a named logger instance."""
    return logging.getLogger(name)
