"""Structured logging for the associative memory system.

JSON-formatted logs with key metrics for observability.
Lightweight — no external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import Any

# Context variable for request tracing
_request_id: ContextVar[str] = ContextVar("request_id", default="")


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context if available
        req_id = _request_id.get("")
        if req_id:
            log_entry["request_id"] = req_id

        # Add extra structured fields
        if hasattr(record, "data"):
            log_entry["data"] = record.data

        # Add exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO", json_output: bool = True) -> logging.Logger:
    """Configure structured logging for the application."""
    logger = logging.getLogger("amem")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
    logger.addHandler(handler)

    return logger


def get_logger(name: str = "amem") -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)


def set_request_id(request_id: str):
    """Set the current request ID for trace context."""
    _request_id.set(request_id)


class MetricsCollector:
    """Lightweight in-memory metrics collector."""

    def __init__(self):
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, amount: int = 1):
        self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, value: float):
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
        # Keep last 1000 observations
        if len(self._histograms[name]) > 1000:
            self._histograms[name] = self._histograms[name][-1000:]

    def gauge(self, name: str, value: float):
        self._gauges[name] = value

    def get_all(self) -> dict:
        result = {"counters": dict(self._counters), "gauges": dict(self._gauges)}
        for name, values in self._histograms.items():
            if values:
                result.setdefault("histograms", {})[name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "p50": sorted(values)[len(values) // 2],
                    "p99": sorted(values)[int(len(values) * 0.99)],
                    "max": max(values),
                }
        return result

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for name, value in self._counters.items():
            lines.append(f"amem_{name}_total {value}")
        for name, value in self._gauges.items():
            lines.append(f"amem_{name} {value}")
        for name, values in self._histograms.items():
            if values:
                lines.append(f"amem_{name}_count {len(values)}")
                lines.append(f"amem_{name}_sum {sum(values):.6f}")
        return "\n".join(lines)


# Global metrics instance
metrics = MetricsCollector()


def timed(metric_name: str):
    """Decorator to time async functions and record to metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                metrics.increment(f"{metric_name}_success")
                return result
            except Exception:
                metrics.increment(f"{metric_name}_error")
                raise
            finally:
                elapsed = time.monotonic() - start
                metrics.observe(f"{metric_name}_duration_seconds", elapsed)
        return wrapper
    return decorator
