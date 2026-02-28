import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from contextlib import contextmanager
from .base_report_module import BaseReportModule

logger = logging.getLogger(__name__)

class PerformanceStatistics:
    """
    Collects and reports performance metrics across the NudeNix pipeline.
    Acts as a data store and a registry for reporting modules.
    """
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.total_start_time = 0.0
        self.total_end_time = 0.0

        # Generic metric storage
        self.counters = defaultdict(int)
        self.sums = defaultdict(float)
        self.categorized_counters = defaultdict(lambda: defaultdict(int))

        # Registry for reporting modules: List of (priority, module)
        self._modules: List[Tuple[int, BaseReportModule]] = []

    def start(self):
        if self.enabled:
            self.total_start_time = time.perf_counter()

    def stop(self):
        if self.enabled:
            self.total_end_time = time.perf_counter()

    def register_module(self, module: BaseReportModule, priority: int = 100):
        """Registers a reporting module with an optional priority (lower is earlier)."""
        if self.enabled:
            self._modules.append((priority, module))

    def increment_counter(self, key: str, amount: int = 1):
        if self.enabled:
            self.counters[key] += amount

    def record_sum(self, key: str, value: float):
        if self.enabled:
            self.sums[key] += value

    def record_categorized_counter(self, category_key: str, item_key: str, amount: int = 1):
        if self.enabled:
            self.categorized_counters[category_key][item_key] += amount

    def get_total_duration(self) -> float:
        """Calculates total duration based on start/stop or sum of stages."""
        duration = self.total_end_time - self.total_start_time
        if duration <= 0:
            # Fallback: sum of major pipeline stages
            duration = sum(self.sums.get(k, 0) for k in ['reader', 'hashing', 'detection', 'censors_total', 'output'])
        return duration

    @contextmanager
    def timer(self, stage_key: str):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.record_sum(stage_key, (end - start))
            self.increment_counter(f"{stage_key}_calls")

    @contextmanager
    def time_censor(self, censor_type: str):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.record_sum(f"censor_{censor_type}_time", (end - start))
            self.record_categorized_counter("censor_calls_by_type", censor_type)

    def print_report(self):
        if not self.enabled or self.counters.get('frame_count', 0) == 0:
            return

        print("\n" + "-"*50)
        print("NudeNix Performance & Analysis Report")
        print("-"*50)

        # Sort modules by priority
        sorted_modules = sorted(self._modules, key=lambda x: x[0])

        section_num = SectionCounter()
        for _, module in sorted_modules:
            title = module.get_title()
            lines = module.render(self)

            if title:
                print(f"{section_num.next()}. {title}")

            for line in lines:
                prefix = "   - " if title else ""
                print(f"{prefix}{line}")

            if not title: # Add space after summary
                print("")

        print("-" * 50)

class SectionCounter:
    """Helper to handle numbered sections in the report."""
    def __init__(self):
        self.count = 0
    def next(self):
        self.count += 1
        return self.count
