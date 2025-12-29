"""Telemetry utilities for measuring inference performance."""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import resource


@dataclass
class TelemetryRecord:
    """Metrics captured for one execution block."""

    latency_ms: float
    throughput_fps: float
    peak_memory_mb: float


class TelemetryLogger(contextlib.AbstractContextManager):
    """Context manager that records latency, throughput, and memory usage."""

    def __init__(self, items_processed: int = 1, logger: Optional[Callable[[TelemetryRecord], None]] = None) -> None:
        self.items_processed = items_processed
        self.logger = logger or self._default_logger
        self._start_time: Optional[float] = None
        self._start_rss: Optional[int] = None

    def __enter__(self) -> "TelemetryLogger":
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self._start_rss = usage.ru_maxrss
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        end_time = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        end_rss = usage.ru_maxrss

        latency_ms = (end_time - (self._start_time or end_time)) * 1000.0
        peak_mem_kb = max(0, end_rss - (self._start_rss or end_rss))
        peak_memory_mb = peak_mem_kb / 1024.0
        throughput_fps = (self.items_processed / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

        record = TelemetryRecord(
            latency_ms=latency_ms,
            throughput_fps=throughput_fps,
            peak_memory_mb=peak_memory_mb,
        )
        self.logger(record)
        return None

    @staticmethod
    def _default_logger(record: TelemetryRecord) -> None:
        print(
            f"Telemetry | Latency: {record.latency_ms:.2f} ms | "
            f"Throughput: {record.throughput_fps:.2f} FPS | "
            f"Peak Memory: {record.peak_memory_mb:.2f} MB"
        )


def telemetry_decorator(items_processed: int = 1, logger: Optional[Callable[[TelemetryRecord], None]] = None):
    """Decorator version of :class:`TelemetryLogger`."""

    def wrapper(fn: Callable):
        def inner(*args, **kwargs):
            with TelemetryLogger(items_processed=items_processed, logger=logger):
                return fn(*args, **kwargs)

        return inner

    return wrapper
