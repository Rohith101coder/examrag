"""Logging configuration with secret redaction for ExamRAG."""

import logging
import re
import sys
from typing import Any


class SensitiveDataFilter(logging.Filter):
    """Filter that masks API keys, authorization tokens, and long text from log records."""

    PATTERNS = [
        (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
        (re.compile(r"(sk-or-v1-[A-Za-z0-9]{8,})", re.IGNORECASE), r"[REDACTED_OPENROUTER_KEY]"),
        (re.compile(r"(eyJ[A-Za-z0-9_\-]{15,}\.[A-Za-z0-9_\-]{15,}\.[A-Za-z0-9_\-]{15,})"), r"[REDACTED_JWT]"),
        (re.compile(r"(api[_\-\s]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9_\-]{8,}", re.IGNORECASE), r"\1[REDACTED_API_KEY]"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, replacement in self.PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)
        if record.args:
            sanitized_args: list[Any] = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.PATTERNS:
                        arg = pattern.sub(replacement, arg)
                sanitized_args.append(arg)
            record.args = tuple(sanitized_args)
        return True


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure root logger with sensitive data filter and standard formatter."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers on multiple setup calls
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataFilter())
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(level)
            handler.addFilter(SensitiveDataFilter())

    # Quiet overly chatty loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger("examrag")

